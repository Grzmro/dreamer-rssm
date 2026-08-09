"""Environment factory: base env construction + standard wrapper chain."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from envs.wrappers import (
    ActionRepeat,
    GrayscaleObservation,
    NormalizeObservation,
    ResizeObservation,
)


def _make_base_env(env_name: str, **env_kwargs) -> gym.Env:
    if env_name.startswith("ALE/"):
        import ale_py

        gym.register_envs(ale_py)
        # Disable ALE's built-in frameskip and sticky actions by default so
        # that action repeat is controlled explicitly by our wrapper.
        kwargs = {"frameskip": 1, "repeat_action_probability": 0.0}
        kwargs.update(env_kwargs)
        return gym.make(env_name, **kwargs)
    if env_name.startswith("dmc/"):
        from envs.dmc import DMCEnv

        # Format: "dmc/<domain>/<task>", e.g. "dmc/walker/walk".
        _, domain, task = env_name.split("/")
        return DMCEnv(domain, task, **env_kwargs)
    return gym.make(env_name, **env_kwargs)


def _rescale_continuous_actions(env: gym.Env) -> gym.Env:
    """Normalize a continuous action space to [-1, 1]^A.

    The Dreamer actor emits tanh-squashed actions in [-1, 1] and the replay
    buffer stores exactly what the actor produced. Without this wrapper an env
    with different bounds (CarRacing: Box([-1,0,0], [1,1,1]) — gas and brake
    live in [0,1]) receives out-of-range actions: gymnasium clips gas
    internally while the buffer keeps the unclipped value, so the world model
    would be trained on an action the environment never applied.

    Normalizing here rather than inside each agent keeps one action space for
    the actor, the buffer, the world model and the baselines alike. PPO's clip
    to the action-space bounds and SAC's own scale/bias become identities.
    """
    space = env.action_space
    if not isinstance(space, gym.spaces.Box):
        return env  # discrete: nothing to rescale
    return gym.wrappers.RescaleAction(
        env,
        min_action=np.full(space.shape, -1.0, dtype=np.float32),
        max_action=np.full(space.shape, 1.0, dtype=np.float32),
    )


def wrap_env(
    env_name: str,
    *,
    size: tuple[int, int] = (64, 64),
    grayscale: bool = False,
    action_repeat: int = 4,
    time_limit: int = 1000,
    normalize: bool = True,
    **env_kwargs,
) -> gym.Env:
    """Build an environment with the standard Dreamer-style wrapper chain.

    Chain: base env -> [RescaleAction to [-1,1] if continuous] -> ActionRepeat
    -> Resize(64x64) -> [Grayscale] -> TimeLimit (in effective, post-repeat
    steps) -> Normalize to [-0.5, 0.5].
    """
    env = _make_base_env(env_name, **env_kwargs)
    env = _rescale_continuous_actions(env)
    if action_repeat > 1:
        env = ActionRepeat(env, action_repeat)
    env = ResizeObservation(env, tuple(size))
    if grayscale:
        env = GrayscaleObservation(env)
    if time_limit and time_limit > 0:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=time_limit)
    if normalize:
        env = NormalizeObservation(env)
    return env


def make_env(env_cfg) -> gym.Env:
    """Build an environment from a Hydra/OmegaConf env config node."""
    env_kwargs = dict(env_cfg.get("env_kwargs") or {})
    return wrap_env(
        env_cfg.name,
        size=tuple(env_cfg.get("size", (64, 64))),
        grayscale=bool(env_cfg.get("grayscale", False)),
        action_repeat=int(env_cfg.get("action_repeat", 4)),
        time_limit=int(env_cfg.get("time_limit", 1000)),
        normalize=bool(env_cfg.get("normalize", True)),
        **env_kwargs,
    )
