"""Sample-efficiency ratio: how much data does each agent need for one return?

The sample-matched benchmark says who is ahead at a fixed budget; when a
baseline is still at the random floor there, that says little more than "it
needs more data". This table asks the quantitative version: pick a target
return (default: the reference agent's final return, i.e. Dreamer's), then
for every agent report the env steps and the wall-clock it took to first
reach it, and how many times more samples that is than the reference.

One definition for everybody: the first env step at which the rolling mean
(``viz.dream_window`` episodes) of episode return reaches the target, averaged
over the seeds that got there, with the seed coverage printed next to it
(see ``steps_to_threshold``). Agents never reaching the target get "n/a" —
a truncated run is a lower bound, not a number, and the ``steps_run`` column
says how far it got.

Put the runs to compare under one ``<root>/<env>/`` directory, e.g. the
Dreamer 400k CSVs next to the long-budget baseline CSVs (RESULTS.md §F).

Outputs: <root>/plots/sample_efficiency_<env>.{md,csv}.

Usage:
    python viz/sample_efficiency.py viz.benchmark_root=experiments/benchmark_pong_time3h
    python viz/sample_efficiency.py viz.efficiency_threshold=0.0
"""

from __future__ import annotations

import csv
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

from train.common_logger import load_benchmark
from viz.ablation_summary import final_return, steps_to_threshold


def sample_efficiency_table(
    agents: dict, reference: str = "dreamer", threshold: float | None = None,
    window: int = 10,
) -> tuple[list[dict], float]:
    """Rows of the table plus the target return that was used."""
    if threshold is None:
        if reference not in agents:
            raise ValueError(
                f"no runs for reference agent {reference!r} (have {sorted(agents)}); "
                "set viz.efficiency_threshold explicitly or copy its CSVs in"
            )
        threshold = final_return(agents[reference])[0]

    ref_steps = None
    if reference in agents:
        ref_steps = steps_to_threshold(agents[reference], threshold, window)[0]

    rows = []
    for agent, runs in sorted(agents.items()):
        f_mean, f_std = final_return(runs)
        steps, reached, total = steps_to_threshold(runs, threshold, window)
        secs = steps_to_threshold(runs, threshold, window, x_key="wall_time_s")[0]
        ends = [r["env_step"][-1] for r in runs if len(r["env_step"])]
        rows.append({
            "agent": agent,
            "seeds": len(runs),
            "steps_run": int(np.mean(ends)) if ends else 0,
            "final_return": f"{f_mean:+.1f} ± {f_std:.1f}",
            "steps_to_target": int(steps) if steps is not None else "n/a",
            "seeds_reaching": f"{reached}/{total}",
            "hours_to_target": round(secs / 3600, 2) if secs is not None else "n/a",
            f"samples_vs_{reference}": (
                f"{steps / ref_steps:.1f}x" if steps is not None and ref_steps else "n/a"
            ),
        })
    return rows, float(threshold)


def write_table(rows: list[dict], threshold: float, out_stem: Path) -> None:
    keys = list(rows[0].keys())
    with open(out_stem.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    md = [f"Target return: {threshold:+.2f}", "",
          "| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    md += ["| " + " | ".join(str(r[k]) for k in keys) + " |" for r in rows]
    out_stem.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    root = Path(cfg.viz.get("benchmark_root") or "experiments/benchmark")
    bench = load_benchmark(root)
    if not bench:
        raise RuntimeError(f"no benchmark CSVs found under {root}")
    plots_dir = root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    for env_name, agents in bench.items():
        rows, threshold = sample_efficiency_table(
            agents,
            reference=cfg.viz.get("efficiency_reference") or "dreamer",
            threshold=cfg.viz.get("efficiency_threshold"),
            window=int(cfg.viz.get("dream_window", 10)),
        )
        out = plots_dir / f"sample_efficiency_{env_name}"
        write_table(rows, threshold, out)
        print(out.with_suffix(".md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
