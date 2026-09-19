"""One-slide summary: Dreamer against the model-free baselines.

`viz/benchmark_comparison.py` draws the raw curves and
`viz/learning_curves.py` the across-seed bands; this puts the whole claim on
a single figure a reader can take in at once:

  (1) sample efficiency — return vs env steps, mean ± std across seeds,
  (2) final return per agent — bars with every seed drawn as a dot, so a
      one-seed bar cannot pass for a measured effect,
  (3) wall-clock — time to spend the same env-step budget (log scale),
      which is where the model-based agent pays for its sample efficiency.

Also writes the same numbers as markdown + CSV, ready to paste into
RESULTS.md. Output: <root>/plots/<env>_summary.png / _summary.{md,csv}

Usage:
    python viz/summary_figure.py
    python viz/summary_figure.py viz.benchmark_root=experiments/benchmark
"""

from __future__ import annotations

import csv
from pathlib import Path

import hydra
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig

from train.common_logger import load_benchmark
from viz.ablation_summary import final_return
from viz.benchmark_comparison import agent_color, rolling
from viz.learning_curves import aggregate_seeds


def summarize_agents(agents: dict, window: int = 10, last_k: int = 10) -> list[dict]:
    """Per-agent summary rows, best final return first."""
    rows = []
    for agent, runs in agents.items():
        mean, std = final_return(runs, last_k)
        finals = [float(np.mean(r["episode_return"][-last_k:])) for r in runs]
        rows.append({
            "agent": agent,
            "seeds": len(runs),
            "env_steps": int(max(int(r["env_step"][-1]) for r in runs)),
            "final_return_mean": round(mean, 2),
            "final_return_std": round(std, 2),
            "best_episode": round(float(max(r["episode_return"].max() for r in runs)), 2),
            "wall_clock_min": round(
                float(np.mean([r["wall_time_s"][-1] for r in runs])) / 60.0, 1
            ),
            "_finals": finals,
        })
    return sorted(rows, key=lambda r: r["final_return_mean"], reverse=True)


def _write_table(rows: list[dict], base: Path) -> tuple[Path, Path]:
    keys = [k for k in rows[0] if not k.startswith("_")]
    with open(base.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows([{k: r[k] for k in keys} for r in rows])
    md = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    md += ["| " + " | ".join(str(r[k]) for k in keys) + " |" for r in rows]
    md.append("")
    md.append("final_return = mean over seeds of each run's last-10-episode mean; "
              "wall_clock = mean seconds of each seed's run, in minutes.")
    base.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return base.with_suffix(".csv"), base.with_suffix(".md")


def make_summary(root: Path, window: int = 10, last_k: int = 10) -> list[Path]:
    bench = load_benchmark(root)
    if not bench:
        raise RuntimeError(f"no benchmark CSVs under {root}")
    plots_dir = root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    written = []

    for env_name, agents in bench.items():
        rows = summarize_agents(agents, window, last_k)
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

        ax = axes[0]
        for agent, runs in sorted(agents.items()):
            try:
                grid, mean, std = aggregate_seeds(runs, window)
            except ValueError:  # a run with a single episode has no curve
                continue
            color = agent_color(agent)
            ax.plot(grid, mean, color=color, lw=2, label=f"{agent} (n={len(runs)})")
            ax.fill_between(grid, mean - std, mean + std, color=color, alpha=0.2)
        ax.set_xlabel("Environment steps (post action-repeat)")
        ax.set_ylabel(f"episode return (rolling {window})")
        ax.set_title("Sample efficiency")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        ax = axes[1]
        names = [r["agent"] for r in rows]
        xs = np.arange(len(rows))
        ax.bar(xs, [r["final_return_mean"] for r in rows],
               yerr=[r["final_return_std"] for r in rows], capsize=4,
               color=[agent_color(n) or "tab:gray" for n in names], alpha=0.85)
        for x, r in zip(xs, rows):  # every seed visible: 1 dot = 1 seed
            ax.scatter(np.full(len(r["_finals"]), x), r["_finals"], s=18,
                       color="black", zorder=3)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{r['agent']}\nn={r['seeds']}" for r in rows], fontsize=8)
        ax.set_ylabel(f"final return (last {last_k} episodes)")
        ax.set_title("Final performance")
        ax.grid(alpha=0.3, axis="y")

        ax = axes[2]
        ax.bar(xs, [max(r["wall_clock_min"], 1e-3) for r in rows],
               color=[agent_color(n) or "tab:gray" for n in names], alpha=0.85)
        ax.set_yscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels(names, fontsize=8, rotation=15, ha="right")
        ax.set_ylabel("wall-clock [min]")
        ax.set_title("Cost of the same env-step budget")
        ax.grid(alpha=0.3, axis="y")

        fig.suptitle(f"{env_name}: Dreamer vs model-free baselines", fontsize=13)
        fig.tight_layout()
        png = plots_dir / f"{env_name}_summary.png"
        fig.savefig(png, dpi=130)
        plt.close(fig)
        csv_path, md_path = _write_table(rows, plots_dir / f"{env_name}_summary")
        written += [png, md_path, csv_path]
        print(f"[summary] {env_name}: {png.name}, {md_path.name}, {csv_path.name}")
    return written


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    root = Path(cfg.viz.get("benchmark_root") or "experiments/benchmark")
    make_summary(root, window=int(cfg.viz.get("dream_window", 10)))


if __name__ == "__main__":
    main()
