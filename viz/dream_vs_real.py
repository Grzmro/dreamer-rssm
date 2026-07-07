"""Dream-vs-real validation plot (Phase 2 acceptance criterion 2).

Reads ``metrics.jsonl`` written by train/dreamer_loop.py and plots, on a
shared env-step axis:
  * real episode returns (scatter + rolling mean), and
  * the mean imagined lambda-return R^lambda at rollout start.

The two curves should share the trend/turning points. If imagined returns
climb while real returns stay flat, the actor is exploiting world-model
errors — that is a diagnosis, not a rendering artifact; see README.

Also prints the Pearson correlation between the imagined-return series and
the real rolling-average return interpolated onto the update timestamps.

Usage:
    python viz/dream_vs_real.py viz.run_dir=experiments/<dreamer_run>
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig


def load_metrics(run_dir: Path) -> tuple[list[dict], list[dict]]:
    episodes, updates = [], []
    with open(run_dir / "metrics.jsonl", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            (episodes if rec.get("kind") == "episode" else updates).append(rec)
    return episodes, updates


def rolling(x: np.ndarray, window: int) -> np.ndarray:
    if len(x) < 2:
        return x
    w = min(window, len(x))
    return np.convolve(x, np.ones(w) / w, mode="valid")


def dream_vs_real(run_dir: Path, out_path: Path, window: int = 10) -> None:
    episodes, updates = load_metrics(run_dir)
    if not episodes or not updates:
        raise RuntimeError(f"metrics.jsonl in {run_dir} lacks episode or update records")

    ep_steps = np.array([e["env_step"] for e in episodes], dtype=float)
    ep_returns = np.array([e["return"] for e in episodes], dtype=float)
    up_steps = np.array([u["env_step"] for u in updates], dtype=float)
    dream = np.array([u["dream_return"] for u in updates], dtype=float)

    roll = rolling(ep_returns, window)
    roll_steps = ep_steps[len(ep_steps) - len(roll):]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.scatter(ep_steps, ep_returns, s=8, alpha=0.3, color="tab:blue",
                label="real episode return")
    ax1.plot(roll_steps, roll, color="tab:blue", lw=2,
             label=f"real return (rolling {window})")
    ax1.set_xlabel("env steps")
    ax1.set_ylabel("real episode return", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(up_steps, dream, color="tab:red", lw=1.5, alpha=0.8,
             label="imagined $R^\\lambda$ (rollout start)")
    ax2.set_ylabel("imagined lambda-return", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.set_title("Dream vs real returns")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)

    # Correlation: real rolling return interpolated at the update timestamps.
    if len(roll) >= 2:
        real_at_updates = np.interp(up_steps, roll_steps, roll)
        if np.std(real_at_updates) > 1e-8 and np.std(dream) > 1e-8:
            r = float(np.corrcoef(real_at_updates, dream)[0, 1])
            print(f"[dream_vs_real] Pearson r(real rolling, dream) = {r:.4f} "
                  f"over {len(dream)} update records")
        else:
            print("[dream_vs_real] correlation undefined (constant series)")
    print(f"[dream_vs_real] saved {out_path}")


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    run_dir = Path(cfg.viz.run_dir)
    out = Path(cfg.viz.get("dream_out") or (run_dir / "dream_vs_real.png"))
    dream_vs_real(run_dir, out, window=int(cfg.viz.get("dream_window", 10)))


if __name__ == "__main__":
    main()
