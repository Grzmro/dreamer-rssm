"""Gymnasium wrappers for image-based world-model RL.

The wrapper chain produces normalized float32 CHW observations in [-0.5, 0.5]
while exposing the raw uint8 HWC frame in ``info["raw_obs"]`` so the replay
buffer can store memory-efficient uint8 and normalize only at sampling time.
"""

from __future__ import annotations

import cv2
import gymnasium as gym
import numpy as np


class ActionRepeat(gym.Wrapper):
    """Repeat the same action for ``repeat`` env steps, summing rewards.

    Stops early if the episode terminates or truncates mid-repeat.
    """

    def __init__(self, env: gym.Env, repeat: int):
        super().__init__(env)
        if repeat < 1:
            raise ValueError(f"repeat must be >= 1, got {repeat}")
        self._repeat = repeat

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        obs, info = None, {}
        for _ in range(self._repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


class ResizeObservation(gym.ObservationWrapper):
    """Resize image observations to ``size`` (H, W) using area interpolation."""

    def __init__(self, env: gym.Env, size: tuple[int, int] = (64, 64)):
        super().__init__(env)
        obs_space = env.observation_space
        if not (isinstance(obs_space, gym.spaces.Box) and len(obs_space.shape) == 3):
            raise ValueError(
                f"ResizeObservation requires image observations (H, W, C), "
                f"got space {obs_space}"
            )
        self._size = tuple(size)
        channels = obs_space.shape[-1]
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(*self._size, channels), dtype=np.uint8
        )

    def observation(self, obs):
        h, w = self._size
        resized = cv2.resize(obs, (w, h), interpolation=cv2.INTER_AREA)
        if resized.ndim == 2:  # cv2 drops the channel dim for single-channel input
            resized = resized[..., None]
        return resized.astype(np.uint8)


class GrayscaleObservation(gym.ObservationWrapper):
    """Convert RGB observations (H, W, 3) to grayscale (H, W, 1)."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        h, w, c = env.observation_space.shape
        if c != 3:
            raise ValueError(f"GrayscaleObservation expects 3 channels, got {c}")
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(h, w, 1), dtype=np.uint8
        )

    def observation(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        return gray[..., None].astype(np.uint8)


class NormalizeObservation(gym.Wrapper):
    """Convert uint8 HWC frames to float32 CHW in [-0.5, 0.5].

    The pre-normalization uint8 HWC frame is passed through in
    ``info["raw_obs"]`` for uint8 storage in the replay buffer.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        h, w, c = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=-0.5, high=0.5, shape=(c, h, w), dtype=np.float32
        )

    @staticmethod
    def normalize(obs: np.ndarray) -> np.ndarray:
        return (obs.astype(np.float32) / 255.0 - 0.5).transpose(2, 0, 1)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info["raw_obs"] = obs
        return self.normalize(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["raw_obs"] = obs
        return self.normalize(obs), reward, terminated, truncated, info
