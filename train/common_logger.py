"""Shared benchmark logging protocol (Phase 3).

Every agent (Dreamer + model-free baselines) reports finished episodes
through the same interface with the same x-axis definition:

    env_step  = cumulative environment interactions AFTER action repeat
                (one ``env.step()`` on the Phase 0 wrapper chain), including
                random prefill/warm-up steps — samples consumed are samples
                consumed regardless of the policy that consumed them;
    wall_time = seconds since the agent's training loop started.

One CSV per run: ``<root>/<env>/<agent>_seed<seed>.csv`` with columns
``env_step,wall_time_s,episode_return,episode_length``. CSV was chosen over
per-agent TensorBoard dirs so viz/benchmark_comparison.py can overlay curves
without any conversion step.
"""

from __future__ import annotations

import csv
import re
import time
from pathlib import Path

import numpy as np


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


class BenchmarkLogger:
    """Appends one CSV row per finished episode; flushes on every write."""

    FIELDS = ("env_step", "wall_time_s", "episode_return", "episode_length")

    def __init__(self, root: str | Path, agent_name: str, env_name: str, seed: int):
        self.path = (
            Path(root) / _sanitize(env_name) / f"{_sanitize(agent_name)}_seed{seed}.csv"
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not self.path.exists()
        self._file = open(self.path, "a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        if new_file:
            self._writer.writerow(self.FIELDS)
        self._start = time.time()

    def log_episode(
        self,
        env_step: int,
        episode_return: float,
        episode_length: int,
        wall_time_s: float | None = None,
    ) -> None:
        if wall_time_s is None:
            wall_time_s = time.time() - self._start
        self._writer.writerow(
            [int(env_step), f"{wall_time_s:.3f}", float(episode_return), int(episode_length)]
        )
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def load_run(path: str | Path) -> dict[str, np.ndarray]:
    """Read one run CSV back into arrays (round-trip of BenchmarkLogger)."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {
        "env_step": np.array([int(r["env_step"]) for r in rows]),
        "wall_time_s": np.array([float(r["wall_time_s"]) for r in rows]),
        "episode_return": np.array([float(r["episode_return"]) for r in rows]),
        "episode_length": np.array([int(r["episode_length"]) for r in rows]),
    }


def load_benchmark(root: str | Path) -> dict[str, dict[str, list[dict[str, np.ndarray]]]]:
    """Load ``<root>/<env>/<agent>_seed<k>.csv`` -> {env: {agent: [runs]}}."""
    root = Path(root)
    out: dict[str, dict[str, list[dict[str, np.ndarray]]]] = {}
    for env_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if env_dir.name == "plots":
            continue
        agents: dict[str, list[dict[str, np.ndarray]]] = {}
        for csv_path in sorted(env_dir.glob("*_seed*.csv")):
            agent = csv_path.stem.rsplit("_seed", 1)[0]
            agents.setdefault(agent, []).append(load_run(csv_path))
        if agents:
            out[env_dir.name] = agents
    return out
