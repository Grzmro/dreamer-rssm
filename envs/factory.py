"""Environment factory: base env construction + standard wrapper chain."""

from __future__ import annotations

import gymnasium as gym

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

    Chain: base env -> ActionRepeat -> Resize(64x64) -> [Grayscale] ->
    TimeLimit (in effective, post-repeat steps) -> Normalize to [-0.5, 0.5].
    """
    env = _make_base_env(env_name, **env_kwargs)
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
