"""Multi-seed learning curves: mean +- std per agent, per environment.

Reads the shared benchmark CSV tree (train/common_logger.py), interpolates
every seed's rolling-mean return onto a common env-step grid, and plots the
across-seed mean with a shaded +-1 std band — one figure per environment,
all agents overlaid. Output: <root>/plots/<env>_learning_curves.png

Usage:
    python viz/learning_curves.py
    python viz/learning_curves.py viz.benchmark_root=experiments/benchmark
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
from viz.benchmark_comparison import agent_color, rolling


def aggregate_seeds(
    runs: list[dict[str, np.ndarray]], window: int = 10, grid_points: int = 200
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(grid, mean, std) of rolling returns across seeds.

    Each seed's rolling-mean curve is linearly interpolated onto a shared
    env-step grid spanning the range covered by ALL seeds (so the band never
    extrapolates beyond a shorter run).
    """
    curves = []
    for run in runs:
        y = rolling(run["episode_return"], window)
        x = run["env_step"][len(run["env_step"]) - len(y):]
        if len(x) >= 2:
            curves.append((x, y))
    if not curves:
        raise ValueError("no runs with >= 2 episodes to aggregate")
    lo = max(float(x[0]) for x, _ in curves)
    hi = min(float(x[-1]) for x, _ in curves)
    grid = np.linspace(lo, hi, grid_points)
    interp = np.stack([np.interp(grid, x, y) for x, y in curves])
    return grid, interp.mean(axis=0), interp.std(axis=0)


def make_learning_curves(root: Path, window: int = 10) -> list[Path]:
    bench = load_benchmark(root)
    if not bench:
        raise RuntimeError(f"no benchmark CSVs under {root}")
    plots_dir = root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    written = []

    for env_name, agents in bench.items():
        fig, ax = plt.subplots(figsize=(9, 5))
        for agent, runs in sorted(agents.items()):
            color = agent_color(agent)
            grid, mean, std = aggregate_seeds(runs, window)
            ax.plot(grid, mean, color=color, lw=2,
                    label=f"{agent} (n={len(runs)} seed{'s' if len(runs) != 1 else ''})")
            ax.fill_between(grid, mean - std, mean + std, color=color, alpha=0.2)
        ax.set_xlabel("Environment steps (post action-repeat)")
        ax.set_ylabel(f"episode return (rolling mean {window}, mean ± std)")
        ax.set_title(f"{env_name}: learning curves")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        path = plots_dir / f"{env_name}_learning_curves.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        written.append(path)
        print(f"[learning_curves] {path}")
    return written


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    root = Path(cfg.viz.get("benchmark_root") or "experiments/benchmark")
    make_learning_curves(root, window=int(cfg.viz.get("dream_window", 10)))


if __name__ == "__main__":
    main()
