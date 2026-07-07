"""DQN baseline — adaptation of CleanRL's dqn_atari.py with two Rainbow
components: double DQN targets and a dueling architecture.

NOT full Rainbow: no prioritized replay, no noisy nets, no distributional
(C51) head, no n-step returns — "DQN with selected improvements".

Protocol notes (same as PPO): Phase 0 wrapper chain via baselines/common.py,
grayscale + 4-frame stack, no reward clipping, env step = post-action-repeat
step. Replay stores uint8 frames to keep 100k transitions in a few GB.
Bootstrapping masks on ``terminated`` only — a time-limit truncation is not
a terminal state (consistent with the Dreamer continue head from Phase 1).

Usage:
    python baselines/dqn_rainbow.py baselines.total_env_steps=100000
"""

from __future__ import annotations

import time

import gymnasium as gym
import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig

from baselines.common import make_baseline_env
from train.common_logger import BenchmarkLogger
from train.train_world_model import resolve_device


class DuelingQNetwork(nn.Module):
    """Nature CNN (64x64) with dueling value/advantage streams."""

    def __init__(self, in_channels: int, num_actions: int, dueling: bool = True):
        super().__init__()
        self.dueling = dueling
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 512), nn.ReLU(),
        )
        self.advantage = nn.Linear(512, num_actions)
        self.value = nn.Linear(512, 1) if dueling else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        adv = self.advantage(h)
        if not self.dueling:
            return adv
        return self.value(h) + adv - adv.mean(dim=1, keepdim=True)


class Uint8ReplayBuffer:
    """Circular buffer storing normalized CHW float obs as uint8."""

    def __init__(self, capacity: int, obs_shape, rng: np.random.Generator):
        self.capacity = int(capacity)
        self.rng = rng
        self.obs = np.zeros((self.capacity, *obs_shape), dtype=np.uint8)
        self.next_obs = np.zeros((self.capacity, *obs_shape), dtype=np.uint8)
        self.action = np.zeros(self.capacity, dtype=np.int64)
        self.reward = np.zeros(self.capacity, dtype=np.float32)
        self.terminated = np.zeros(self.capacity, dtype=np.float32)
        self.pos = 0
        self.full = False

    @staticmethod
    def _to_uint8(obs: np.ndarray) -> np.ndarray:
        return np.clip((obs + 0.5) * 255.0, 0, 255).astype(np.uint8)

    @staticmethod
    def _to_float(obs: np.ndarray) -> np.ndarray:
        return obs.astype(np.float32) / 255.0 - 0.5

    def add(self, obs, next_obs, action, reward, terminated) -> None:
        i = self.pos
        self.obs[i] = self._to_uint8(obs)
        self.next_obs[i] = self._to_uint8(next_obs)
        self.action[i] = action
        self.reward[i] = reward
        self.terminated[i] = float(terminated)
        self.pos = (self.pos + 1) % self.capacity
        self.full = self.full or self.pos == 0

    def __len__(self) -> int:
        return self.capacity if self.full else self.pos

    def sample(self, batch_size: int, device):
        idx = self.rng.integers(0, len(self), size=batch_size)
        to = lambda x, dtype=torch.float32: torch.as_tensor(x, dtype=dtype, device=device)
        return (
            to(self._to_float(self.obs[idx])),
            to(self._to_float(self.next_obs[idx])),
            to(self.action[idx], torch.int64),
            to(self.reward[idx]),
            to(self.terminated[idx]),
        )


def linear_schedule(start_e: float, end_e: float, duration: int, t: int) -> float:
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def train_dqn(cfg: DictConfig, seed: int | None = None) -> None:
    b = cfg.baselines
    d = b.dqn
    seed = cfg.seed if seed is None else seed
    device = resolve_device(b.device)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    env = make_baseline_env(cfg.env, b.frame_stack, b.grayscale)
    assert isinstance(env.action_space, gym.spaces.Discrete), "DQN needs discrete actions"
    num_actions = int(env.action_space.n)

    q_net = DuelingQNetwork(
        env.observation_space.shape[0], num_actions, dueling=bool(d.dueling)
    ).to(device)
    target_net = DuelingQNetwork(
        env.observation_space.shape[0], num_actions, dueling=bool(d.dueling)
    ).to(device)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = torch.optim.Adam(q_net.parameters(), lr=d.lr)
    buffer = Uint8ReplayBuffer(d.buffer_size, env.observation_space.shape, rng)
    bench = BenchmarkLogger(b.benchmark_dir, "dqn", cfg.env.name, seed)

    total_steps = int(b.total_env_steps)
    explore_steps = int(d.exploration_fraction * total_steps)
    start_time = time.time()
    n_episodes = 0

    obs, _ = env.reset(seed=seed)
    print(f"[dqn] device={device} total_steps={total_steps} "
          f"double={bool(d.double)} dueling={bool(d.dueling)}")
    for global_step in range(1, total_steps + 1):
        eps = linear_schedule(d.start_e, d.end_e, explore_steps, global_step)
        if rng.random() < eps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q = q_net(torch.as_tensor(obs, dtype=torch.float32, device=device)[None])
                action = int(q.argmax(dim=1).item())

        next_obs, reward, terminated, truncated, info = env.step(action)
        buffer.add(obs, next_obs, action, reward, terminated)
        obs = next_obs

        if terminated or truncated:
            ep = info["episode"]
            n_episodes += 1
            bench.log_episode(global_step, float(ep["r"]), int(ep["l"]))
            print(f"[dqn] step {global_step:8d} | episode {n_episodes:4d} | "
                  f"return {float(ep['r']):8.2f} | len {int(ep['l'])} | eps {eps:.3f}")
            obs, _ = env.reset()

        if global_step > int(d.learning_starts) and global_step % int(d.train_frequency) == 0:
            s, s2, a, r, term = buffer.sample(int(d.batch_size), device)
            with torch.no_grad():
                if d.double:
                    best = q_net(s2).argmax(dim=1, keepdim=True)
                    next_q = target_net(s2).gather(1, best).squeeze(1)
                else:
                    next_q = target_net(s2).max(dim=1).values
                td_target = r + float(d.gamma) * (1.0 - term) * next_q
            q_pred = q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
            loss = nn.functional.mse_loss(q_pred, td_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if global_step % int(d.target_update_interval) == 0:
                for tp, qp in zip(target_net.parameters(), q_net.parameters()):
                    tp.data.copy_(float(d.tau) * qp.data + (1 - float(d.tau)) * tp.data)

        if global_step % 10000 == 0:
            sps = int(global_step / (time.time() - start_time))
            print(f"[dqn] step {global_step}/{total_steps} | {sps} steps/s")

    env.close()
    bench.close()
    print(f"[dqn] done: {total_steps} env steps, {n_episodes} episodes "
          f"in {(time.time() - start_time) / 60:.1f} min -> {bench.path}")


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    train_dqn(cfg)


if __name__ == "__main__":
    main()
