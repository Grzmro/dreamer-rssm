"""SAC baseline — adaptation of CleanRL's sac_continuous_action.py.

Extended for pixel observations (the project protocol is pixel-based): a
Nature-CNN encoder is owned by the critic and trained ONLY by the Q loss;
the actor consumes detached encoder features (SAC-AE/DrQ convention).
Vector observations fall back to the original state-based MLP layout.
Twin Q networks, EMA targets, tanh-squashed Normal policy with rescaling
to the env bounds, optional entropy auto-tuning. Hyperparameters are
CleanRL defaults except the replay capacity / learning starts, which are
shrunk to the benchmark budget (configs/baselines/default.yaml).

Usage:
    python baselines/sac.py env=carracing baselines.total_env_steps=100000
"""

from __future__ import annotations

import time

import gymnasium as gym
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from baselines.common import make_baseline_env
from train.common_logger import BenchmarkLogger
from train.train_world_model import resolve_device

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


class PixelEncoder(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 512), nn.ReLU(),
        )
        self.out_dim = 512

    def forward(self, x):
        return self.net(x)


class QNetwork(nn.Module):
    def __init__(self, feat_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim + act_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, feat, action):
        return self.net(torch.cat([feat, action], dim=-1)).squeeze(-1)


class Actor(nn.Module):
    def __init__(self, feat_dim: int, action_space: gym.spaces.Box):
        super().__init__()
        act_dim = int(np.prod(action_space.shape))
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )
        self.mean = nn.Linear(256, act_dim)
        self.log_std = nn.Linear(256, act_dim)
        scale = (action_space.high - action_space.low) / 2.0
        bias = (action_space.high + action_space.low) / 2.0
        self.register_buffer("action_scale", torch.as_tensor(scale, dtype=torch.float32))
        self.register_buffer("action_bias", torch.as_tensor(bias, dtype=torch.float32))

    def forward(self, feat):
        h = self.net(feat)
        mean = self.mean(h)
        log_std = torch.tanh(self.log_std(h))
        # CleanRL's smooth clamp of log-std into [LOG_STD_MIN, LOG_STD_MAX].
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, feat):
        mean, log_std = self(feat)
        normal = torch.distributions.Normal(mean, log_std.exp())
        u = normal.rsample()
        y = torch.tanh(u)
        action = y * self.action_scale + self.action_bias
        log_prob = normal.log_prob(u) - torch.log(
            self.action_scale * (1 - y.pow(2)) + 1e-6
        )
        return action, log_prob.sum(-1), torch.tanh(mean) * self.action_scale + self.action_bias


class ReplayBuffer:
    """Circular buffer; pixel obs stored uint8, vector obs float32."""

    def __init__(self, capacity, obs_shape, act_dim, rng, pixels: bool):
        self.capacity = int(capacity)
        self.rng = rng
        self.pixels = pixels
        dtype = np.uint8 if pixels else np.float32
        self.obs = np.zeros((self.capacity, *obs_shape), dtype=dtype)
        self.next_obs = np.zeros((self.capacity, *obs_shape), dtype=dtype)
        self.action = np.zeros((self.capacity, act_dim), dtype=np.float32)
        self.reward = np.zeros(self.capacity, dtype=np.float32)
        self.terminated = np.zeros(self.capacity, dtype=np.float32)
        self.pos, self.full = 0, False

    def _enc(self, o):
        return np.clip((o + 0.5) * 255.0, 0, 255).astype(np.uint8) if self.pixels else o

    def _dec(self, o):
        return o.astype(np.float32) / 255.0 - 0.5 if self.pixels else o

    def add(self, obs, next_obs, action, reward, terminated):
        i = self.pos
        self.obs[i], self.next_obs[i] = self._enc(obs), self._enc(next_obs)
        self.action[i], self.reward[i] = action, reward
        self.terminated[i] = float(terminated)
        self.pos = (self.pos + 1) % self.capacity
        self.full = self.full or self.pos == 0

    def __len__(self):
        return self.capacity if self.full else self.pos

    def sample(self, batch_size, device):
        idx = self.rng.integers(0, len(self), size=batch_size)
        to = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
        return (
            to(self._dec(self.obs[idx])), to(self._dec(self.next_obs[idx])),
            to(self.action[idx]), to(self.reward[idx]), to(self.terminated[idx]),
        )


def train_sac(cfg: DictConfig, seed: int | None = None) -> None:
    b = cfg.baselines
    s = b.sac
    seed = cfg.seed if seed is None else seed
    device = resolve_device(b.device)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    env = make_baseline_env(cfg.env, b.frame_stack, b.grayscale)
    assert isinstance(env.action_space, gym.spaces.Box), "SAC needs continuous actions"
    obs_shape = env.observation_space.shape
    act_dim = int(np.prod(env.action_space.shape))
    pixels = len(obs_shape) == 3

    if pixels:
        encoder = PixelEncoder(obs_shape[0]).to(device)
        target_encoder = PixelEncoder(obs_shape[0]).to(device)
        target_encoder.load_state_dict(encoder.state_dict())
        feat_dim = encoder.out_dim
    else:
        encoder = target_encoder = None
        feat_dim = int(np.prod(obs_shape))

    def feats(module, x, detach=False):
        f = module(x) if module is not None else x.flatten(1)
        return f.detach() if detach else f

    actor = Actor(feat_dim, env.action_space).to(device)
    qf1, qf2 = QNetwork(feat_dim, act_dim).to(device), QNetwork(feat_dim, act_dim).to(device)
    qf1_t, qf2_t = QNetwork(feat_dim, act_dim).to(device), QNetwork(feat_dim, act_dim).to(device)
    qf1_t.load_state_dict(qf1.state_dict())
    qf2_t.load_state_dict(qf2.state_dict())

    critic_params = list(qf1.parameters()) + list(qf2.parameters())
    if encoder is not None:
        critic_params += list(encoder.parameters())  # encoder trains with the Q loss only
    q_optim = torch.optim.Adam(critic_params, lr=s.q_lr)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=s.policy_lr)

    if s.autotune:
        target_entropy = -float(act_dim)
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=s.q_lr)
        alpha = log_alpha.exp().item()
    else:
        alpha = float(s.alpha)

    buffer = ReplayBuffer(s.buffer_size, obs_shape, act_dim, rng, pixels)
    bench = BenchmarkLogger(b.benchmark_dir, "sac", cfg.env.name, seed)
    total_steps = int(b.total_env_steps)
    start_time = time.time()
    n_episodes = 0

    obs, _ = env.reset(seed=seed)
    print(f"[sac] device={device} pixels={pixels} total_steps={total_steps}")
    for global_step in range(1, total_steps + 1):
        if global_step <= int(s.learning_starts):
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                f = feats(encoder, torch.as_tensor(
                    obs, dtype=torch.float32, device=device)[None])
                action = actor.get_action(f)[0][0].cpu().numpy()

        next_obs, reward, terminated, truncated, info = env.step(action)
        buffer.add(obs, next_obs, action, reward, terminated)
        obs = next_obs

        if terminated or truncated:
            ep = info["episode"]
            n_episodes += 1
            bench.log_episode(global_step, float(ep["r"]), int(ep["l"]))
            print(f"[sac] step {global_step:8d} | episode {n_episodes:4d} | "
                  f"return {float(ep['r']):8.2f} | len {int(ep['l'])} | alpha {alpha:.3f}")
            obs, _ = env.reset()

        if global_step > int(s.learning_starts):
            o, o2, a, r, term = buffer.sample(int(s.batch_size), device)
            with torch.no_grad():
                f2_t = feats(target_encoder, o2)
                next_a, next_logp, _ = actor.get_action(
                    feats(encoder, o2) if pixels else f2_t
                )
                q_next = torch.min(qf1_t(f2_t, next_a), qf2_t(f2_t, next_a))
                target = r + float(s.gamma) * (1 - term) * (q_next - alpha * next_logp)
            f = feats(encoder, o)
            q_loss = F.mse_loss(qf1(f, a), target) + F.mse_loss(qf2(f, a), target)
            q_optim.zero_grad()
            q_loss.backward()
            q_optim.step()

            if global_step % int(s.policy_frequency) == 0:
                for _ in range(int(s.policy_frequency)):  # compensate delay (CleanRL)
                    f_a = feats(encoder, o, detach=True)  # actor never trains the encoder
                    pi, logp, _ = actor.get_action(f_a)
                    q_pi = torch.min(qf1(f_a, pi), qf2(f_a, pi))
                    actor_loss = (alpha * logp - q_pi).mean()
                    actor_optim.zero_grad()
                    actor_loss.backward()
                    actor_optim.step()
                    if s.autotune:
                        with torch.no_grad():
                            _, logp2, _ = actor.get_action(f_a)
                        alpha_loss = (-log_alpha.exp() * (logp2 + target_entropy)).mean()
                        alpha_optim.zero_grad()
                        alpha_loss.backward()
                        alpha_optim.step()
                        alpha = log_alpha.exp().item()

            if global_step % int(s.target_network_frequency) == 0:
                tau = float(s.tau)
                for tp, p in zip(qf1_t.parameters(), qf1.parameters()):
                    tp.data.lerp_(p.data, tau)
                for tp, p in zip(qf2_t.parameters(), qf2.parameters()):
                    tp.data.lerp_(p.data, tau)
                if encoder is not None:
                    for tp, p in zip(target_encoder.parameters(), encoder.parameters()):
                        tp.data.lerp_(p.data, tau)

        if global_step % 10000 == 0:
            sps = int(global_step / (time.time() - start_time))
            print(f"[sac] step {global_step}/{total_steps} | {sps} steps/s")

    env.close()
    bench.close()
    print(f"[sac] done: {total_steps} env steps, {n_episodes} episodes "
          f"in {(time.time() - start_time) / 60:.1f} min -> {bench.path}")


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    train_sac(cfg)


if __name__ == "__main__":
    main()
