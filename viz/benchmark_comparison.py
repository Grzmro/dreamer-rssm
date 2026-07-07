"""Benchmark comparison plots (Phase 3).

For every environment under ``experiments/benchmark/`` produces two PNGs:
  1. rolling-mean episode return vs ENV STEPS (post action-repeat) — the
     sample-efficiency plot; the x-axis is environment interactions, NOT
     gradient steps (Dreamer performs train_ratio gradient updates per env
     step — that difference is the point of the comparison);
  2. rolling-mean episode return vs WALL-CLOCK time — the compute
     trade-off plot.
Multiple seeds of one agent are drawn as individual thin lines plus their
mean. Output: experiments/benchmark/plots/<env>_{env_steps,wall_clock}.png

Usage:
    python viz/benchmark_comparison.py [viz.benchmark_root=experiments/benchmark]
"""

from __future__ import annotations

from pathlib import Path

import hydra
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig

from train.common_logger import load_benchmark

AGENT_COLORS = {
    "dreamer": "tab:red",
    "dreamer-warmstart": "tab:purple",
    "ppo": "tab:blue",
    "dqn": "tab:green",
    "sac": "tab:orange",
}


def agent_color(agent: str):
    if agent in AGENT_COLORS:
        return AGENT_COLORS[agent]
    for prefix, color in AGENT_COLORS.items():  # e.g. future "dreamer-*" variants
        if agent.startswith(prefix):
            return color
    return None


def rolling(x: np.ndarray, window: int) -> np.ndarray:
    if len(x) < 2:
        return x
    w = min(window, len(x))
    return np.convolve(x, np.ones(w) / w, mode="valid")


def _plot_axis(ax, runs_by_agent, x_key: str, window: int) -> None:
    for agent, runs in sorted(runs_by_agent.items()):
        color = agent_color(agent)
        for i, run in enumerate(runs):
            y = rolling(run["episode_return"], window)
            x = run[x_key][len(run[x_key]) - len(y):]
            ax.plot(
                x, y, color=color, alpha=0.9 if len(runs) == 1 else 0.45,
                lw=1.8 if len(runs) == 1 else 1.0,
                label=agent if i == 0 else None,
            )
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylabel(f"episode return (rolling mean, {window})")


def make_plots(root: Path, window: int = 10) -> list[Path]:
    bench = load_benchmark(root)
    if not bench:
        raise RuntimeError(f"no benchmark CSVs found under {root}")
    plots_dir = root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    written = []

    for env_name, agents in bench.items():
        fig, ax = plt.subplots(figsize=(9, 5))
        _plot_axis(ax, agents, "env_step", window)
        ax.set_xlabel("Environment steps (post action-repeat)")
        ax.set_title(f"{env_name}: sample efficiency")
        fig.tight_layout()
        p1 = plots_dir / f"{env_name}_env_steps.png"
        fig.savefig(p1, dpi=130)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 5))
        _plot_axis(ax, agents, "wall_time_s", window)
        ax.set_xlabel("Wall-clock time [s]")
        ax.set_title(f"{env_name}: compute trade-off")
        fig.tight_layout()
        p2 = plots_dir / f"{env_name}_wall_clock.png"
        fig.savefig(p2, dpi=130)
        plt.close(fig)

        written += [p1, p2]
        print(f"[benchmark] {env_name}: {p1.name}, {p2.name}")
    return written


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    root = Path(cfg.viz.get("benchmark_root") or "experiments/benchmark")
    make_plots(root, window=int(cfg.viz.get("dream_window", 10)))


if __name__ == "__main__":
    main()
