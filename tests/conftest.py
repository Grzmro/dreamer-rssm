"""Shared test fixtures: a fast, deterministic image environment."""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

EPISODE_LEN = 6  # length of a dummy-env episode, in effective steps


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


class DummyBoxImageEnv(DummyImageEnv):
    """Same as DummyImageEnv but with a CarRacing-shaped action space.

    ``Box([-1,0,0], [1,1,1])``: asymmetric bounds, so an agent emitting raw
    tanh output in [-1,1] would step outside the space unless the factory's
    RescaleAction wrapper is in the chain.
    """

    def __init__(self, shape=(48, 32, 3), episode_len=100):
        super().__init__(shape=shape, episode_len=episode_len)
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )


@pytest.fixture
def dummy_env():
    return DummyImageEnv()


def dummy_env_factory(_env_cfg=None):
    """Phase 0 chain on the dummy base env: resize to 64x64, then normalize.

    Signature matches ``envs.make_env`` so it can be monkeypatched in.
    """
    from envs.wrappers import NormalizeObservation, ResizeObservation

    env = DummyImageEnv(episode_len=EPISODE_LEN)
    env = ResizeObservation(env, (64, 64))
    return NormalizeObservation(env)


def dummy_box_env_factory(_env_cfg=None):
    """Same chain on a continuous base env, including the rescale wrapper."""
    from envs.factory import _rescale_continuous_actions
    from envs.wrappers import NormalizeObservation, ResizeObservation

    env = DummyBoxImageEnv(episode_len=EPISODE_LEN)
    env = _rescale_continuous_actions(env)
    env = ResizeObservation(env, (64, 64))
    return NormalizeObservation(env)


def tiny_dreamer_cfg(total_env_steps: int, **overrides):
    """Compose the real Hydra config with a model small enough for CPU tests."""
    from hydra import compose, initialize_config_dir

    base = {
        "train_dreamer.device": "cpu",
        "train_dreamer.prefill_steps": 0,
        "train_dreamer.total_env_steps": total_env_steps,
        "train_dreamer.train_ratio": 0,  # collection only unless overridden
        "train_dreamer.checkpoint_interval": 1000000,
        "buffer.batch_size": 4,
        "buffer.seq_len": 8,
        "model.cnn_depth": 4,
        "model.deter_dim": 16,
        "model.hidden_dim": 16,
        "model.stoch_groups": 4,
        "model.stoch_classes": 4,
        "model.head_hidden_dim": 16,
        "model.head_layers": 1,
        "agent.actor.hidden_dim": 16,
        "agent.actor.num_layers": 1,
        "agent.critic.hidden_dim": 16,
        "agent.critic.num_layers": 1,
    }
    base.update(overrides)
    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        return compose(
            config_name="config",
            overrides=[f"{k}={v}" for k, v in base.items()],
        )


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
