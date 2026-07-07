"""Benchmark driver (Phase 3): run every agent on one env with one budget.

Sequentially trains the requested agents (dreamer, ppo, dqn, sac) for
``baselines.total_env_steps`` env steps each, for every seed in
``benchmark.seeds``, logging through the shared CSV protocol, then renders
the comparison plots. Agents incompatible with the env's action space are
skipped with a notice (dqn needs discrete, sac needs continuous).

The Dreamer entry runs FROM SCRATCH (no warm start, no preloaded buffer):
its random prefill counts on the shared env-step axis like every other
agent's warm-up — samples consumed are samples consumed.

Usage:
    python train/run_benchmark.py benchmark.total_env_steps=100000
    python train/run_benchmark.py env=carracing "benchmark.agents=[dreamer,ppo,sac]"
"""

from __future__ import annotations

import time
import traceback
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf


def run_benchmark(cfg: DictConfig) -> None:
    from envs import make_env

    bench = cfg.benchmark
    root = Path(cfg.baselines.benchmark_dir)
    probe = make_env(cfg.env)
    discrete = hasattr(probe.action_space, "n")
    probe.close()

    for seed in bench.seeds:
        for agent in bench.agents:
            if agent == "dqn" and not discrete:
                print(f"[benchmark] skip dqn (continuous action space)")
                continue
            if agent == "sac" and discrete:
                print(f"[benchmark] skip sac (discrete action space)")
                continue
            run_cfg = OmegaConf.merge(cfg, {"seed": int(seed)})
            run_cfg.baselines.total_env_steps = int(bench.total_env_steps)
            print(f"\n[benchmark] === {agent} seed {seed} on {cfg.env.name} "
                  f"({bench.total_env_steps} env steps) ===")
            start = time.time()
            try:
                _run_agent(agent, run_cfg, root, seed)
            except Exception:
                traceback.print_exc()
                print(f"[benchmark] {agent} seed {seed} FAILED — continuing")
            print(f"[benchmark] {agent} seed {seed}: {(time.time() - start) / 60:.1f} min")

    from viz.benchmark_comparison import make_plots

    make_plots(root)


def _run_agent(agent: str, cfg: DictConfig, root: Path, seed: int) -> None:
    out_dir = root / "runs" / f"{agent}_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if agent == "dreamer":
        from train.dreamer_loop import train_dreamer

        cfg.train_dreamer.benchmark_dir = str(root)
        cfg.train_dreamer.buffer_dir = None
        cfg.train_dreamer.init_wm_ckpt = None
        # Shared budget covers prefill + policy steps.
        prefill = min(int(cfg.train_dreamer.prefill_steps), int(cfg.baselines.total_env_steps))
        cfg.train_dreamer.prefill_steps = prefill
        cfg.train_dreamer.total_env_steps = int(cfg.baselines.total_env_steps) - prefill
        train_dreamer(cfg, output_dir=out_dir)
    elif agent == "ppo":
        from baselines.ppo import train_ppo

        train_ppo(cfg, seed=seed)
    elif agent == "dqn":
        from baselines.dqn_rainbow import train_dqn

        train_dqn(cfg, seed=seed)
    elif agent == "sac":
        from baselines.sac import train_sac

        train_sac(cfg, seed=seed)
    else:
        raise ValueError(f"unknown agent: {agent!r}")


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    run_benchmark(cfg)


if __name__ == "__main__":
    main()
