import importlib.util

import numpy as np
import pytest

from envs import wrap_env
from envs.wrappers import (
    ActionRepeat,
    GrayscaleObservation,
    NormalizeObservation,
    ResizeObservation,
)
from tests.conftest import DummyImageEnv


def _wrap_dummy(**kwargs):
    defaults = dict(size=(64, 64), action_repeat=1, time_limit=1000)
    defaults.update(kwargs)
    env = DummyImageEnv()
    # Build the same chain as envs.factory.wrap_env, but on the dummy base env.
    if defaults["action_repeat"] > 1:
        env = ActionRepeat(env, defaults["action_repeat"])
    env = ResizeObservation(env, defaults["size"])
    if defaults.get("grayscale"):
        env = GrayscaleObservation(env)
    import gymnasium as gym

    env = gym.wrappers.TimeLimit(env, max_episode_steps=defaults["time_limit"])
    if defaults.get("normalize", True):
        env = NormalizeObservation(env)
    return env


def test_obs_shape_and_range_rgb():
    env = _wrap_dummy()
    obs, info = env.reset(seed=0)
    assert obs.shape == (3, 64, 64)
    assert obs.dtype == np.float32
    assert obs.min() >= -0.5 and obs.max() <= 0.5
    obs, *_ = env.step(env.action_space.sample())
    assert obs.shape == (3, 64, 64)
    assert obs.min() >= -0.5 and obs.max() <= 0.5


def test_grayscale_shape():
    env = _wrap_dummy(grayscale=True)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (1, 64, 64)


def test_raw_obs_in_info_is_uint8_hwc():
    env = _wrap_dummy()
    _, info = env.reset(seed=0)
    assert info["raw_obs"].shape == (64, 64, 3)
    assert info["raw_obs"].dtype == np.uint8
    _, _, _, _, info = env.step(env.action_space.sample())
    assert info["raw_obs"].dtype == np.uint8


def test_normalization_is_exact():
    env = _wrap_dummy()
    env.reset(seed=0)
    obs, *_ = env.step(0)
    # DummyImageEnv pixels are all t=1 after one step: 1/255 - 0.5 everywhere.
    assert np.allclose(obs, 1 / 255 - 0.5)


def test_action_repeat_steps_and_reward_sum():
    env = _wrap_dummy(action_repeat=4)
    env.reset(seed=0)
    _, reward, *_ = env.step(0)
    assert env.unwrapped.t == 4  # 4 inner env steps per wrapper step
    assert reward == pytest.approx(4.0)  # rewards summed (1.0 per inner step)


def test_action_repeat_stops_at_episode_end():
    env = ActionRepeat(DummyImageEnv(episode_len=3), repeat=10)
    env.reset(seed=0)
    _, reward, terminated, _, _ = env.step(0)
    assert terminated
    assert reward == pytest.approx(3.0)  # only 3 inner steps happened


def test_time_limit_truncates_effective_steps():
    env = _wrap_dummy(action_repeat=2, time_limit=5)
    # Base episode is 100 inner steps; limit of 5 effective steps hits first.
    env.reset(seed=0)
    truncated = False
    for i in range(5):
        _, _, terminated, truncated, _ = env.step(0)
    assert truncated and not terminated
    assert env.unwrapped.t == 10  # 5 effective steps x action_repeat 2


@pytest.mark.skipif(
    importlib.util.find_spec("ale_py") is None, reason="ale-py not installed"
)
def test_atari_pong_end_to_end():
    env = wrap_env(
        "ALE/Pong-v5", size=(64, 64), action_repeat=4, time_limit=100
    )
    obs, info = env.reset(seed=0)
    assert obs.shape == (3, 64, 64)
    assert obs.dtype == np.float32
    assert info["raw_obs"].shape == (64, 64, 3)
    for _ in range(3):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert obs.shape == (3, 64, 64)
        assert -0.5 <= obs.min() and obs.max() <= 0.5
    env.close()
