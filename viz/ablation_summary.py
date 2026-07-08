"""Ablation summary: per-group curves + a markdown/CSV results table.

Groups Dreamer variants by their benchmark labels (base run = "dreamer",
ablation presets label themselves "dreamer-H5", "dreamer-nofreenats", ...).
For each requested group, draws mean+-std curves of the member variants and
emits a table with:
  * final return  — mean +- std over seeds of each run's last-K episodes,
  * steps to threshold — env steps until the rolling return first reaches
    the given fraction of the best variant's final return (n/a if never).

Outputs: <root>/plots/ablation_<group>.png, ablation_<group>.md / .csv.

Usage:
    python viz/ablation_summary.py                       # all groups it can find
    python viz/ablation_summary.py viz.ablation_env=ALE_Pong-v5
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
from viz.learning_curves import aggregate_seeds
from viz.benchmark_comparison import rolling

# Group name -> variant labels (base always included for reference).
GROUPS = {
    "horizon": ["dreamer", "dreamer-H5", "dreamer-H10", "dreamer-H20"],
    "latent_size": [
        "dreamer", "dreamer-deter128", "dreamer-deter256",
        "dreamer-stoch16x16", "dreamer-stoch32x64",
    ],
    "loss_variants": [
        "dreamer", "dreamer-noklbal", "dreamer-nofreenats", "dreamer-norecon",
    ],
    "latent_type": ["dreamer", "dreamer-gaussian"],
    # Training-time parametric studies (Phase 4E / Cyfronet):
    "train_ratio": ["dreamer", "dreamer-tr0.1", "dreamer-tr1.0"],
    "entropy_coef": ["dreamer", "dreamer-ent1e-4", "dreamer-ent1e-3"],
}


def final_return(runs, last_k: int = 10) -> tuple[float, float]:
    """Mean and std over seeds of each run's mean last-K episode returns."""
    finals = [float(np.mean(r["episode_return"][-last_k:])) for r in runs]
    return float(np.mean(finals)), float(np.std(finals))


def steps_to_threshold(runs, threshold: float, window: int = 10) -> float | None:
    """Mean env steps at which the rolling return first reaches ``threshold``."""
    hits = []
    for run in runs:
        y = rolling(run["episode_return"], window)
        x = run["env_step"][len(run["env_step"]) - len(y):]
        idx = np.argmax(y >= threshold)
        if y[idx] >= threshold:
            hits.append(float(x[idx]))
    return float(np.mean(hits)) if hits else None


def summarize_group(
    group: str, variants: list[str], agents: dict, plots_dir: Path,
    window: int = 10, threshold_frac: float = 0.9,
) -> list[dict] | None:
    present = {v: agents[v] for v in variants if v in agents}
    if len(present) < 2:
        return None  # nothing to compare yet

    fig, ax = plt.subplots(figsize=(9, 5))
    rows = []
    for variant, runs in present.items():
        grid, mean, std = aggregate_seeds(runs, window)
        ax.plot(grid, mean, lw=2, label=f"{variant} (n={len(runs)})")
        ax.fill_between(grid, mean - std, mean + std, alpha=0.2)
        f_mean, f_std = final_return(runs)
        rows.append({"variant": variant, "seeds": len(runs),
                     "final_return_mean": round(f_mean, 2),
                     "final_return_std": round(f_std, 2)})

    best = max(r["final_return_mean"] for r in rows)
    # Threshold in absolute units: base-floor + frac * (best - floor), using
    # the worst variant's final as the floor so negative returns work too.
    floor = min(r["final_return_mean"] for r in rows)
    threshold = floor + threshold_frac * (best - floor)
    for row in rows:
        s = steps_to_threshold(present[row["variant"]], threshold, window)
        row[f"steps_to_{int(threshold_frac * 100)}pct"] = (
            int(s) if s is not None else "n/a"
        )

    ax.set_xlabel("Environment steps (post action-repeat)")
    ax.set_ylabel(f"episode return (rolling mean {window}, mean ± std)")
    ax.set_title(f"Ablation: {group}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / f"ablation_{group}.png", dpi=130)
    plt.close(fig)

    keys = list(rows[0].keys())
    with open(plots_dir / f"ablation_{group}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    md = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    md += ["| " + " | ".join(str(r[k]) for k in keys) + " |" for r in rows]
    (plots_dir / f"ablation_{group}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[ablation] {group}: {len(rows)} variants -> plots/ablation_{group}.{{png,md,csv}}")
    return rows


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    root = Path(cfg.viz.get("benchmark_root") or "experiments/benchmark")
    bench = load_benchmark(root)
    env_filter = cfg.viz.get("ablation_env")
    plots_dir = root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    any_done = False
    for env_name, agents in bench.items():
        if env_filter and env_name != env_filter:
            continue
        for group, variants in GROUPS.items():
            if summarize_group(group, variants, agents, plots_dir):
                any_done = True
    if not any_done:
        print("[ablation] no group has >= 2 variants logged yet — run the "
              "ablation presets first (see configs/ablation/)")


if __name__ == "__main__":
    main()
