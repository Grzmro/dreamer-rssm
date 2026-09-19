"""Save and replay trained baseline policies (Phase 4 visualization).

The benchmark loops only ever wrote episode returns to CSV, which is enough
for a learning curve but not to *show* a baseline playing: there was no way
to get a trained PPO/DQN policy back after the run. This module stores the
weights next to the benchmark CSVs and hands them back as a uniform
callable, so ``viz/gameplay_gif.py`` can put Dreamer and the baselines in
the same frame.

Interface (shared with the Dreamer wrapper in viz/gameplay_gif.py):
    policy.reset()        # start of episode
    action = policy(obs)  # obs as the agent's own env produces it
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Callable

import gymnasium as gym
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf


class Policy:
    """A stateless-or-not action selector with an explicit episode reset."""

    def __init__(self, fn: Callable[[np.ndarray], object], meta: dict,
                 reset_fn: Callable[[], None] | None = None):
        self._fn = fn
        self._reset = reset_fn
        self.meta = meta

    def reset(self) -> None:
        if self._reset is not None:
            self._reset()

    def __call__(self, obs: np.ndarray):
        return self._fn(obs)


def save_baseline_checkpoint(
    path: str | Path,
    *,
    kind: str,
    model: torch.nn.Module,
    cfg: DictConfig,
    seed: int,
    env_step: int,
    obs_shape,
    action_dim: int,
    discrete: bool = True,
    extra: dict | None = None,
) -> Path:
    """Write ``<kind>`` weights plus everything needed to rebuild the net."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": kind,
        "model_state": model.state_dict(),
        "cfg": OmegaConf.to_container(cfg, resolve=True),
        "seed": int(seed),
        "env_step": int(env_step),
        "obs_shape": tuple(int(s) for s in obs_shape),
        "action_dim": int(action_dim),
        "discrete": bool(discrete),
    }
    payload.update(extra or {})
    torch.save(payload, path)
    return path


def _spaces(ckpt: dict):
    obs_space = gym.spaces.Box(
        low=-0.5, high=0.5, shape=tuple(ckpt["obs_shape"]), dtype=np.float32
    )
    if ckpt["discrete"]:
        action_space = gym.spaces.Discrete(int(ckpt["action_dim"]))
    else:
        action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(int(ckpt["action_dim"]),), dtype=np.float32
        )
    return obs_space, action_space


def load_baseline_policy(
    path: str | Path, device, epsilon: float = 0.0, seed: int = 0
) -> Policy:
    """Rebuild a saved PPO/DQN policy as a ``Policy``.

    ``epsilon`` adds uniform-random actions at evaluation time. DQN's greedy
    argmax policy can otherwise deadlock on Atari (the same state maps to the
    same action forever); a small epsilon is the conventional eval fix and is
    recorded in the returned metadata so a demo can state it.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    kind = ckpt["kind"]
    obs_space, action_space = _spaces(ckpt)
    rng = np.random.default_rng(seed)
    meta = {
        "kind": kind,
        "env_step": int(ckpt["env_step"]),
        "seed": int(ckpt["seed"]),
        "eval_epsilon": float(epsilon),
        "checkpoint": str(path),
    }

    if kind == "ppo":
        from baselines.ppo import Agent

        agent = Agent(obs_space, action_space).to(device)
        agent.load_state_dict(ckpt["model_state"])
        agent.eval()

        @torch.no_grad()
        def act(obs: np.ndarray):
            x = torch.as_tensor(obs, dtype=torch.float32, device=device)[None]
            action, _, _, _ = agent.get_action_and_value(x)  # on-policy sample
            if ckpt["discrete"]:
                if epsilon > 0 and rng.random() < epsilon:
                    return int(action_space.sample())
                return int(action.item())
            return action[0].cpu().numpy().astype(np.float32)

        return Policy(act, meta)

    if kind == "dqn":
        from baselines.dqn_rainbow import DuelingQNetwork

        q_net = DuelingQNetwork(
            int(ckpt["obs_shape"][0]), int(ckpt["action_dim"]),
            dueling=bool(ckpt.get("dueling", True)),
        ).to(device)
        q_net.load_state_dict(ckpt["model_state"])
        q_net.eval()

        @torch.no_grad()
        def act(obs: np.ndarray):
            if epsilon > 0 and rng.random() < epsilon:
                return int(action_space.sample())
            x = torch.as_tensor(obs, dtype=torch.float32, device=device)[None]
            return int(q_net(x).argmax(dim=1).item())

        return Policy(act, meta)

    raise ValueError(f"unknown baseline checkpoint kind: {kind!r}")


def random_policy(action_space: gym.Space, seed: int = 0) -> Policy:
    """The honest floor every learning curve starts from."""
    space = copy.deepcopy(action_space)  # own RNG, independent of the env's
    space.seed(seed)
    return Policy(lambda obs: space.sample(), {"kind": "random", "seed": int(seed)})
