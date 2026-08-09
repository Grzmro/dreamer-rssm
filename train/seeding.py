"""One place that makes a run reproducible from ``cfg.seed``.

Gymnasium's ``env.reset(seed=...)`` seeds the environment's own RNG but NOT
``env.action_space``, which has a separate generator. Every agent here draws
its warm-up actions from ``env.action_space.sample()`` — random prefill in
train/collect.py and train/dreamer_loop.py, epsilon-greedy in
baselines/dqn_rainbow.py, learning-starts in baselines/sac.py — so without
seeding that space explicitly two runs with the same ``seed`` consume
different action sequences and cannot be reproduced.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int, env=None) -> None:
    """Seed Python, NumPy, torch and (when given) the env's action space.

    ``env`` may be a plain or a vector environment; both expose an
    ``action_space`` whose generator needs its own seed.
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if env is not None:
        env.action_space.seed(seed)
