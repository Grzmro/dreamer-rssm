"""Shared environment construction for the model-free baselines (Phase 3).

The baselines consume the EXACT Phase 0 wrapper chain (``envs.make_env``:
action repeat, 64x64 resize, time limit, [-0.5, 0.5] float32 CHW
normalization) so that one env step means the same thing for every agent.

Two DELIBERATE, documented deviations for the feedforward baselines only:
  * grayscale=True and a 4-frame channel stack — the conventional
    model-free Atari input (a memoryless policy cannot infer velocity from
    a single frame; Dreamer builds temporal state in the RSSM instead);
  * nothing else: no NoopReset/FireReset/EpisodicLife, and NO reward
    clipping anywhere (Pong rewards are already in {-1, 0, 1}), so episode
    returns on the benchmark plots share the same scale as Dreamer's.
"""

from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np
from omegaconf import OmegaConf

from envs import make_env


class ChannelFrameStack(gym.Wrapper):
    """Stack the last k CHW frames along the channel axis."""

    def __init__(self, env: gym.Env, k: int):
        super().__init__(env)
        self.k = k
        self._frames: deque[np.ndarray] = deque(maxlen=k)
        c, h, w = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=-0.5, high=0.5, shape=(c * k, h, w), dtype=np.float32
        )

    def _obs(self) -> np.ndarray:
        return np.concatenate(list(self._frames), axis=0)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.k):
            self._frames.append(obs)
        return self._obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(obs)
        return self._obs(), reward, terminated, truncated, info


def make_baseline_env(
    env_cfg, frame_stack: int = 4, grayscale: bool = True, record_stats: bool = True
) -> gym.Env:
    """Phase 0 chain + (grayscale, frame stack) for feedforward agents."""
    cfg = OmegaConf.merge(env_cfg, {"grayscale": bool(grayscale)})
    env = make_env(cfg)
    if frame_stack > 1:
        env = ChannelFrameStack(env, frame_stack)
    if record_stats:
        env = gym.wrappers.RecordEpisodeStatistics(env)
    return env
