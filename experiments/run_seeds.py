"""Multi-seed experiment driver (Phase 4).

Runs N seeds of the currently composed config — including any ablation
preset — through the shared benchmark protocol. Defaults to Dreamer only
(ablations are Dreamer variants); pass benchmark.agents to include
baselines. Seeds run SEQUENTIALLY: parallel runs would contend for the
single sandbox GPU — a wall-clock limitation, not a correctness one.

Usage:
    python experiments/run_seeds.py "benchmark.seeds=[0,1,2]" \
        benchmark.total_env_steps=30000
    python experiments/run_seeds.py ablation=no_free_nats "benchmark.seeds=[0,1]"
    python experiments/run_seeds.py ablation=horizon_5 "benchmark.agents=[dreamer,dqn]"

Each variant labels its own CSV rows (train_dreamer.benchmark_agent set by
the ablation preset), so viz/learning_curves.py and viz/ablation_summary.py
can overlay variants without bookkeeping.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    from train.run_benchmark import run_benchmark

    # Default to Dreamer-only unless benchmark.agents was overridden on the
    # command line (ablations are Dreamer variants; baselines by request).
    overrides = hydra.core.hydra_config.HydraConfig.get().overrides.task
    if not any(o.startswith("benchmark.agents") for o in overrides):
        cfg.benchmark.agents = ["dreamer"]
    # Ablation presets only affect Dreamer; baselines run their defaults.
    variant = cfg.train_dreamer.get("benchmark_agent", "dreamer")
    print(f"[run_seeds] variant={variant} seeds={list(cfg.benchmark.seeds)} "
          f"budget={cfg.benchmark.total_env_steps} env steps")
    run_benchmark(cfg)


if __name__ == "__main__":
    main()
