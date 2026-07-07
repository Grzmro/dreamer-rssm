"""PPO baseline — adaptation of CleanRL's ppo_atari.py / ppo_continuous_action.py.

Single-file PPO (GAE, clipped surrogate, value clipping, entropy bonus)
wired to the Phase 0 wrapper chain via baselines/common.py and to the
shared benchmark logger. Discrete actions -> categorical head; continuous
actions (Box) -> diagonal Normal with a state-independent log-std.
Hyperparameters are CleanRL defaults (configs/baselines/default.yaml).

Usage:
    python baselines/ppo.py baselines.total_env_steps=100000
"""

from __future__ import annotations

import time
from pathlib import Path

import gymnasium as gym
import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.distributions import Categorical, Normal

from baselines.common import make_baseline_env
from train.common_logger import BenchmarkLogger
from train.train_world_model import resolve_device


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias: float = 0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class NatureEncoder(nn.Module):
    """CleanRL's Nature CNN, adjusted for 64x64 inputs (outputs 512)."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Conv2d(in_channels, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 4 * 4, 512)),
            nn.ReLU(),
        )
        self.out_dim = 512

    def forward(self, x):
        return self.net(x)


class MlpEncoder(nn.Module):
    """CleanRL's continuous-control trunk (for non-image observations)."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(in_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
        )
        self.out_dim = 64

    def forward(self, x):
        return self.net(x)


class Agent(nn.Module):
    def __init__(self, obs_space, action_space):
        super().__init__()
        if len(obs_space.shape) == 3:
            self.encoder = NatureEncoder(obs_space.shape[0])
        else:
            self.encoder = MlpEncoder(int(np.prod(obs_space.shape)))
        self.discrete = isinstance(action_space, gym.spaces.Discrete)
        act_dim = action_space.n if self.discrete else int(np.prod(action_space.shape))
        self.actor = layer_init(nn.Linear(self.encoder.out_dim, act_dim), std=0.01)
        self.critic = layer_init(nn.Linear(self.encoder.out_dim, 1), std=1.0)
        if not self.discrete:
            self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    def get_value(self, x):
        return self.critic(self.encoder(x)).squeeze(-1)

    def get_action_and_value(self, x, action=None):
        hidden = self.encoder(x)
        if self.discrete:
            dist = Categorical(logits=self.actor(hidden))
            if action is None:
                action = dist.sample()
            logprob, entropy = dist.log_prob(action), dist.entropy()
        else:
            mean = self.actor(hidden)
            dist = Normal(mean, self.actor_logstd.expand_as(mean).exp())
            if action is None:
                action = dist.sample()
            logprob, entropy = dist.log_prob(action).sum(-1), dist.entropy().sum(-1)
        return action, logprob, entropy, self.critic(hidden).squeeze(-1)


def train_ppo(cfg: DictConfig, seed: int | None = None) -> None:
    b = cfg.baselines
    p = b.ppo
    seed = cfg.seed if seed is None else seed
    device = resolve_device(b.device)
    torch.manual_seed(seed)
    np.random.seed(seed)

    envs = gym.vector.SyncVectorEnv(
        [
            lambda: make_baseline_env(cfg.env, b.frame_stack, b.grayscale)
            for _ in range(p.num_envs)
        ]
    )
    agent = Agent(envs.single_observation_space, envs.single_action_space).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=p.lr, eps=1e-5)
    bench = BenchmarkLogger(b.benchmark_dir, "ppo", cfg.env.name, seed)

    num_steps, num_envs = int(p.num_steps), int(p.num_envs)
    batch_size = num_steps * num_envs
    minibatch_size = batch_size // int(p.num_minibatches)
    total_steps = int(b.total_env_steps)
    num_iterations = max(1, total_steps // batch_size)

    obs_shape = envs.single_observation_space.shape
    act_shape = envs.single_action_space.shape
    obs = torch.zeros((num_steps, num_envs, *obs_shape), device=device)
    actions = torch.zeros((num_steps, num_envs, *act_shape), device=device)
    logprobs = torch.zeros((num_steps, num_envs), device=device)
    rewards = torch.zeros((num_steps, num_envs), device=device)
    dones = torch.zeros((num_steps, num_envs), device=device)
    values = torch.zeros((num_steps, num_envs), device=device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=seed)
    next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
    next_done = torch.zeros(num_envs, device=device)
    n_episodes = 0

    print(f"[ppo] device={device} iterations={num_iterations} batch={batch_size}")
    for iteration in range(1, num_iterations + 1):
        if p.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / num_iterations
            optimizer.param_groups[0]["lr"] = frac * p.lr

        for step in range(num_steps):
            global_step += num_envs
            obs[step] = next_obs
            dones[step] = next_done
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
            actions[step] = action
            logprobs[step] = logprob
            values[step] = value

            env_action = action.cpu().numpy()
            if not agent.discrete:
                env_action = np.clip(
                    env_action,
                    envs.single_action_space.low,
                    envs.single_action_space.high,
                )
            next_obs_np, reward, term, trunc, infos = envs.step(env_action)
            rewards[step] = torch.as_tensor(reward, dtype=torch.float32, device=device)
            next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
            next_done = torch.as_tensor(
                np.logical_or(term, trunc), dtype=torch.float32, device=device
            )

            if "episode" in infos:
                for i in range(num_envs):
                    if infos["_episode"][i]:
                        r = float(infos["episode"]["r"][i])
                        l = int(infos["episode"]["l"][i])
                        bench.log_episode(global_step, r, l)
                        n_episodes += 1
                        print(f"[ppo] step {global_step:8d} | episode {n_episodes:4d} "
                              f"| return {r:8.2f} | len {l}")

        # GAE (CleanRL bootstrap-on-truncation-free variant).
        with torch.no_grad():
            next_value = agent.get_value(next_obs)
            advantages = torch.zeros_like(rewards)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + p.gamma * nextvalues * nextnonterminal - values[t]
                lastgaelam = (
                    delta + p.gamma * p.gae_lambda * nextnonterminal * lastgaelam
                )
                advantages[t] = lastgaelam
            returns = advantages + values

        b_obs = obs.reshape(-1, *obs_shape)
        b_actions = actions.reshape(-1, *act_shape)
        b_logprobs = logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        inds = np.arange(batch_size)
        for _ in range(int(p.update_epochs)):
            np.random.shuffle(inds)
            for start in range(0, batch_size, minibatch_size):
                mb = inds[start:start + minibatch_size]
                mb_actions = b_actions[mb]
                if agent.discrete:
                    mb_actions = mb_actions.long()
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb], mb_actions
                )
                logratio = newlogprob - b_logprobs[mb]
                ratio = logratio.exp()

                mb_adv = b_advantages[mb]
                if p.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg_loss = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * torch.clamp(ratio, 1 - p.clip_coef, 1 + p.clip_coef),
                ).mean()

                if p.clip_vloss:
                    v_clipped = b_values[mb] + torch.clamp(
                        newvalue - b_values[mb], -p.clip_coef, p.clip_coef
                    )
                    v_loss = 0.5 * torch.max(
                        (newvalue - b_returns[mb]) ** 2,
                        (v_clipped - b_returns[mb]) ** 2,
                    ).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb]) ** 2).mean()

                loss = pg_loss - p.ent_coef * entropy.mean() + p.vf_coef * v_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), p.max_grad_norm)
                optimizer.step()

        sps = int(global_step / (time.time() - start_time))
        print(f"[ppo] iter {iteration}/{num_iterations} | step {global_step} | {sps} steps/s")

    envs.close()
    bench.close()
    print(f"[ppo] done: {global_step} env steps, {n_episodes} episodes "
          f"in {(time.time() - start_time) / 60:.1f} min -> {bench.path}")


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    train_ppo(cfg)


if __name__ == "__main__":
    main()
