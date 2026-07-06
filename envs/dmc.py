"""Optional dm_control adapter exposing the gymnasium API with pixel observations.

Requires ``pip install dm_control`` (pulls in MuJoCo), which is NOT installed
by default in this repo — see README. The adapter renders camera pixels as
uint8 HWC observations so the standard wrapper chain from
:mod:`envs.wrappers` applies unchanged.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np


class DMCEnv(gym.Env):
    """Wrap a ``dm_control.suite`` task as a gymnasium env with image observations."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        domain: str,
        task: str,
        render_size: tuple[int, int] = (210, 160),
        camera_id: int = 0,
        seed: int | None = None,
    ):
        try:
            from dm_control import suite
        except ImportError as e:
            raise ImportError(
                "dm_control is not installed. Install it with "
                "`pip install dreamer-rssm[dmc]` or `pip install dm_control` "
                "(requires MuJoCo; see README)."
            ) from e

        self._env = suite.load(
            domain, task, task_kwargs={"random": seed} if seed is not None else None
        )
        self._render_size = render_size
        self._camera_id = camera_id

        spec = self._env.action_spec()
        self.action_space = gym.spaces.Box(
            low=spec.minimum.astype(np.float32),
            high=spec.maximum.astype(np.float32),
            dtype=np.float32,
        )
        h, w = render_size
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(h, w, 3), dtype=np.uint8
        )

    def _pixels(self) -> np.ndarray:
        h, w = self._render_size
        return self._env.physics.render(height=h, width=w, camera_id=self._camera_id)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._env.reset()
        return self._pixels(), {}

    def step(self, action):
        time_step = self._env.step(np.asarray(action, dtype=np.float32))
        reward = float(time_step.reward or 0.0)
        # dm_control episodes end only by time limit (truncation), except for
        # explicit termination signaled via discount == 0.
        terminated = time_step.last() and time_step.discount == 0.0
        truncated = time_step.last() and not terminated
        return self._pixels(), reward, terminated, truncated, {}

    def render(self):
        return self._pixels()

    def close(self):
        self._env.close()
