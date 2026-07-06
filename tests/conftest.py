"""Shared test fixtures: a fast, deterministic image environment."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest


class DummyImageEnv(gym.Env):
    """Deterministic RGB-image env for wrapper tests.

    Observation encodes the internal step counter (all pixels == t % 256),
    reward is always 1.0 per inner step, episode terminates after
    ``episode_len`` inner steps.
    """

    def __init__(self, shape=(48, 32, 3), episode_len=100):
        self.observation_space = gym.spaces.Box(0, 255, shape=shape, dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(4)
        self._episode_len = episode_len
        self.t = 0

    def _obs(self):
        return np.full(
            self.observation_space.shape, self.t % 256, dtype=np.uint8
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        return self._obs(), {}

    def step(self, action):
        self.t += 1
        terminated = self.t >= self._episode_len
        return self._obs(), 1.0, terminated, False, {}


@pytest.fixture
def dummy_env():
    return DummyImageEnv()


def make_synthetic_episode(length: int, obs_shape=(64, 64, 3), fill: int | None = None):
    """Build a valid episode dict of a given length for buffer tests."""
    rng = np.random.default_rng(length)
    obs = (
        np.full((length, *obs_shape), fill, dtype=np.uint8)
        if fill is not None
        else rng.integers(0, 256, size=(length, *obs_shape), dtype=np.uint8)
    )
    terminated = np.zeros(length, dtype=bool)
    terminated[-1] = True
    return {
        "obs": obs,
        "action": rng.integers(0, 4, size=length),
        "reward": rng.normal(size=length).astype(np.float32),
        "terminated": terminated,
        "truncated": np.zeros(length, dtype=bool),
    }
