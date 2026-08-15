"""Benchmark driver (Phase 3): run every agent on one env with one budget.

Sequentially trains the requested agents (dreamer, ppo, dqn, sac) for
``baselines.total_env_steps`` env steps each, for every seed in
``benchmark.seeds``, logging through the shared CSV protocol, then renders
the comparison plots. Agents incompatible with the env's action space are
skipped with a notice (dqn needs discrete, sac needs continuous).

The Dreamer entry runs FROM SCRATCH (no warm start, no preloaded buffer):
its random prefill counts on the shared env-step axis like every other
agent's warm-up — samples consumed are samples consumed.

Two comparisons are supported and they answer different questions:

* **sample-matched** (default) — one shared ``total_env_steps`` for everybody.
  Dreamer is expected to win; this is the claim the README makes.
* **wall-clock-matched** — set ``benchmark.time_budget_s`` and give each agent
  a step target it can plausibly reach in that time via
  ``benchmark.agent_env_steps``. Every agent then gets the same seconds on the
  same GPU, and on Pong the baselines are expected to win because they step
  the env ~35-80x faster. Report both; neither one alone is the whole story.

Usage:
    python train/run_benchmark.py benchmark.total_env_steps=100000
    python train/run_benchmark.py env=carracing "benchmark.agents=[dreamer,ppo,sac]"
    python train/run_benchmark.py benchmark.time_budget_s=33840 \
        "benchmark.agent_env_steps={dreamer: 400000, dqn: 13400000, ppo: 33000000}"
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
            steps = _agent_steps(bench, agent)
            run_cfg.baselines.total_env_steps = steps
            budget_s = bench.get("time_budget_s")
            run_cfg.baselines.time_budget_s = budget_s
            run_cfg.train_dreamer.time_budget_s = budget_s
            budget_note = (
                "" if budget_s is None
                else f", wall-clock cap {_fmt_duration(float(budget_s))}"
            )
            print(f"\n[benchmark] === {agent} seed {seed} on {cfg.env.name} "
                  f"({steps} env steps{budget_note}) ===")
            start = time.time()
            try:
                _run_agent(agent, run_cfg, root, seed)
            except Exception:
                traceback.print_exc()
                print(f"[benchmark] {agent} seed {seed} FAILED — continuing")
            print(f"[benchmark] {agent} seed {seed}: {(time.time() - start) / 60:.1f} min")

    from viz.benchmark_comparison import make_plots

    make_plots(root)


def _fmt_duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} h"


def _agent_steps(bench: DictConfig, agent: str) -> int:
    """Planned step horizon for ``agent`` — per-agent override or the shared one.

    Only meaningful together with ``time_budget_s``: in a wall-clock-matched
    sweep each agent needs a different step target, because the point is that
    they consume env steps at wildly different rates. Without a time budget
    this returns the one shared budget for everybody, i.e. the sample-matched
    protocol is unchanged.
    """
    per_agent = bench.get("agent_env_steps")
    if per_agent is not None and agent in per_agent:
        return int(per_agent[agent])
    return int(bench.total_env_steps)


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
