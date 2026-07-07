"""Protocol regression tests: baselines and Dreamer must see the same env.

Guards against accidental protocol drift (different action repeat, obs
scaling, action space...) between the Phase 0 chain used by Dreamer and
the baseline wrappers built on top of it. Uses CartPole rendering-free?
No — uses ALE/Pong-v5 like the benchmark itself (ROMs ship with ale-py).
"""

import numpy as np
import pytest
from omegaconf import OmegaConf

from baselines.common import ChannelFrameStack, make_baseline_env
from envs import make_env


def pong_cfg(**overrides):
    cfg = {
        "name": "ALE/Pong-v5",
        "size": [64, 64],
        "grayscale": False,
        "action_repeat": 4,
        "time_limit": 50,
        "normalize": True,
        "env_kwargs": {"frameskip": 1, "repeat_action_probability": 0.0},
    }
    cfg.update(overrides)
    return OmegaConf.create(cfg)


@pytest.fixture(scope="module")
def envs():
    dreamer_env = make_env(pong_cfg())
    baseline_env = make_baseline_env(pong_cfg(), frame_stack=4, grayscale=True)
    yield dreamer_env, baseline_env
    dreamer_env.close()
    baseline_env.close()


def test_identical_action_space(envs):
    dreamer_env, baseline_env = envs
    assert dreamer_env.action_space == baseline_env.action_space


def test_same_obs_range_and_dtype(envs):
    dreamer_env, baseline_env = envs
    obs_d, _ = dreamer_env.reset(seed=0)
    obs_b, _ = baseline_env.reset(seed=0)
    for obs in (obs_d, obs_b):
        assert obs.dtype == np.float32
        assert obs.min() >= -0.5 and obs.max() <= 0.5


def test_documented_shape_deviation_only(envs):
    """Baselines differ ONLY by grayscale + stacking: (3,64,64) vs (12? no — 4,64,64)."""
    dreamer_env, baseline_env = envs
    assert dreamer_env.observation_space.shape == (3, 64, 64)
    assert baseline_env.observation_space.shape == (4, 64, 64)  # 4 x 1 gray channel


def test_identical_env_without_deviations():
    """With frame_stack=1 and grayscale off, the baseline env IS the Dreamer env."""
    d = make_env(pong_cfg())
    b = make_baseline_env(pong_cfg(), frame_stack=1, grayscale=False, record_stats=False)
    obs_d, _ = d.reset(seed=123)
    obs_b, _ = b.reset(seed=123)
    assert obs_d.shape == obs_b.shape
    assert np.allclose(obs_d, obs_b)
    # Same action -> same reward and same next frame (same repeat, same ALE).
    for action in (2, 3, 1):
        od, rd, td_, trd, _ = d.step(action)
        ob, rb, tb, trb, _ = b.step(action)
        assert rd == rb and td_ == tb and trd == trb
        assert np.allclose(od, ob)
    d.close()
    b.close()


def test_frame_stack_semantics():
    import gymnasium as gym

    class _Counter(gym.Env):
        """Fake CHW env emitting frame i at step i."""

        def __init__(self):
            self.observation_space = gym.spaces.Box(-0.5, 0.5, (1, 2, 2), np.float32)
            self.action_space = gym.spaces.Discrete(2)
            self.i = 0

        def reset(self, **kw):
            self.i = 0
            return np.full((1, 2, 2), 0.0, np.float32), {}

        def step(self, a):
            self.i += 1
            return np.full((1, 2, 2), self.i / 10, np.float32), 0.0, False, False, {}

    env = ChannelFrameStack(_Counter(), k=3)
    obs, _ = env.reset()
    assert obs.shape == (3, 2, 2)
    assert np.allclose(obs, 0.0)  # reset frame repeated k times
    obs, *_ = env.step(0)
    assert np.allclose(obs[:2], 0.0) and np.allclose(obs[2], 0.1)  # oldest first
    obs, *_ = env.step(0)
    assert np.allclose(obs[0], 0.0) and np.allclose(obs[1], 0.1) and np.allclose(obs[2], 0.2)
