"""End-to-end pipeline verification driver ("czy wszystko działa").

Runs the whole repo as a sequence of subprocess steps, grouped in four tiers,
and writes a PASS/FAIL report. Every step reuses EXISTING code, scripts and
checkpoints — the training-flavoured steps are tiny smoke runs (minutes each)
whose only purpose is to prove the pipeline executes and stays numerically
sane. They are NOT the deferred Cyfronet training and none of their numbers
are scientific results.

    Tier 1  unit tests            pytest -q  (78 tests)
    Tier 2  import + Hydra compose integrity (every entry point + ablation preset)
    Tier 3  training smoke runs   dreamer / ppo / dqn / ppo-cont / sac / run_seeds
    Tier 4  visualization + notebook on existing checkpoints/data

Usage:
    python scripts/verify_pipeline.py                 # confidence profile (~1-2 h)
    python scripts/verify_pipeline.py --quick         # minimal budgets (~10 min)
    python scripts/verify_pipeline.py --tier 1,2      # only some tiers
    python scripts/verify_pipeline.py --skip-notebook # skip the slow notebook re-exec
    python scripts/verify_pipeline.py --stop-on-fail

Reusable on Athena as a pre-flight: `python scripts/verify_pipeline.py --quick`.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY = sys.executable
NAN_RE = re.compile(r"(?<![A-Za-z])(nan|inf)(?![A-Za-z])", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Budgets
# --------------------------------------------------------------------------- #
def budgets(quick: bool) -> dict:
    """Tiny 'does-it-run' budgets. quick = seconds-scale, confidence = minutes."""
    if quick:
        return dict(
            dr_prefill=100, dr_total=200, dr_ratio=0.1, dr_batch=8, dr_seq=20,
            ppo_steps=600, dqn_steps=800, ppo_cont_steps=300, sac_steps=400,
            seeds_steps=600, sanity_steps=500, video_max=200,
            t_train=900, t_viz=600, t_notebook=1800,
        )
    return dict(
        dr_prefill=500, dr_total=3000, dr_ratio=0.3, dr_batch=16, dr_seq=50,
        ppo_steps=5000, dqn_steps=3000, ppo_cont_steps=800, sac_steps=800,
        seeds_steps=2000, sanity_steps=3000, video_max=300,
        t_train=3600, t_viz=1200, t_notebook=2400,
    )


# --------------------------------------------------------------------------- #
# Step model
# --------------------------------------------------------------------------- #
@dataclass
class Step:
    name: str
    tier: int
    cmd: list[str]
    artifacts: list[str] = field(default_factory=list)  # globs, relative to REPO
    timeout: int = 600
    check_nan: bool = False
    # filled in after running:
    status: str = "PENDING"
    seconds: float = 0.0
    detail: str = ""
    log_path: Path | None = None


def build_steps(vdir: Path, b: dict, skip_notebook: bool) -> list[Step]:
    dreamer_dir = vdir / "dreamer"
    bench_dir = vdir / "bench"
    viz_out = vdir / "viz"
    ckpt = dreamer_dir / "checkpoints" / "dreamer_final.pt"

    wm_ckpt = "experiments/wm_final_twohot/checkpoints/wm_final.pt"
    val_buf = "experiments/data/val_5k/buffer"
    pong_run = "experiments/dreamer_pong"
    pong_ckpt = "experiments/dreamer_pong/checkpoints/dreamer_final.pt"

    import_probe = (
        "import importlib, importlib.util, sys;"
        "mods=['train.dreamer_loop','train.run_benchmark','baselines.ppo',"
        "'baselines.dqn_rainbow','baselines.sac','viz.reconstruction',"
        "'viz.open_loop_rollout','viz.dream_vs_real','viz.real_vs_imagined_video',"
        "'viz.benchmark_comparison','viz.learning_curves','viz.ablation_summary',"
        "'viz.sanity_checks'];"
        "[print('ok',m) or importlib.import_module(m) for m in mods];"
        "spec=importlib.util.spec_from_file_location('run_seeds','experiments/run_seeds.py');"
        "mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod);"
        "print('ok experiments/run_seeds.py')"
    )
    compose_probe = (
        "import glob, os; from pathlib import Path;"
        "from hydra import initialize_config_dir, compose;"
        "cfgdir=os.path.join(os.getcwd(),'configs');"
        "presets=[Path(p).stem for p in glob.glob('configs/ablation/*.yaml')];"
        "import contextlib;"
        "ctx=initialize_config_dir(config_dir=cfgdir, version_base='1.3');"
        "ctx.__enter__();"
        "compose(config_name='config');"
        "[compose(config_name='config', overrides=['ablation='+p]) or print('composed',p) for p in presets];"
        "print('OK composed base +',len(presets),'presets')"
    )

    steps: list[Step] = [
        # ---- Tier 1 : unit tests -------------------------------------------
        Step("pytest-suite", 1, [PY, "-m", "pytest", "-q"],
             timeout=1200),

        # ---- Tier 2 : import + config integrity ----------------------------
        Step("imports", 2, [PY, "-c", import_probe], timeout=300),
        Step("hydra-compose-ablations", 2, [PY, "-c", compose_probe], timeout=300),

        # ---- Tier 3 : training smoke runs ----------------------------------
        Step("dreamer-discrete-pong", 3, [
            PY, "train/dreamer_loop.py",
            f"train_dreamer.prefill_steps={b['dr_prefill']}",
            f"train_dreamer.total_env_steps={b['dr_total']}",
            f"train_dreamer.train_ratio={b['dr_ratio']}",
            "train_dreamer.log_interval=10",
            "train_dreamer.checkpoint_interval=100000",
            f"train_dreamer.benchmark_dir={bench_dir.as_posix()}",
            f"buffer.batch_size={b['dr_batch']}",
            f"buffer.seq_len={b['dr_seq']}",
            "env.time_limit=200",
            f"hydra.run.dir={dreamer_dir.as_posix()}",
        ], artifacts=[str(ckpt), str(dreamer_dir / "metrics.jsonl")],
            timeout=b["t_train"], check_nan=True),

        Step("ppo-discrete-pong", 3, [
            PY, "baselines/ppo.py",
            f"baselines.total_env_steps={b['ppo_steps']}",
            "baselines.ppo.num_envs=2", "baselines.device=cpu",
            "env.time_limit=200",
            f"baselines.benchmark_dir={bench_dir.as_posix()}",
        ], artifacts=[str(bench_dir / "ALE_Pong-v5" / "ppo_seed*.csv")],
            timeout=b["t_train"], check_nan=True),

        Step("dqn-pong", 3, [
            PY, "baselines/dqn_rainbow.py",
            f"baselines.total_env_steps={b['dqn_steps']}",
            "baselines.dqn.learning_starts=300",
            "baselines.dqn.buffer_size=5000", "baselines.device=cpu",
            "env.time_limit=200",
            f"baselines.benchmark_dir={bench_dir.as_posix()}",
        ], artifacts=[str(bench_dir / "ALE_Pong-v5" / "dqn_seed*.csv")],
            timeout=b["t_train"], check_nan=True),

        Step("ppo-continuous-carracing", 3, [
            PY, "baselines/ppo.py", "env=carracing",
            f"baselines.total_env_steps={b['ppo_cont_steps']}",
            "baselines.ppo.num_envs=2", "baselines.device=cpu",
            "env.time_limit=200",
            f"baselines.benchmark_dir={bench_dir.as_posix()}",
        ], artifacts=[str(bench_dir / "CarRacing-v3" / "ppo_seed*.csv")],
            timeout=b["t_train"], check_nan=True),

        Step("sac-carracing", 3, [
            PY, "baselines/sac.py", "env=carracing",
            f"baselines.total_env_steps={b['sac_steps']}",
            "baselines.sac.learning_starts=200",
            "baselines.sac.buffer_size=5000", "baselines.device=cpu",
            "env.time_limit=200",
            f"baselines.benchmark_dir={bench_dir.as_posix()}",
        ], artifacts=[str(bench_dir / "CarRacing-v3" / "sac_seed*.csv")],
            timeout=b["t_train"], check_nan=True),

        Step("run_seeds-wrapper", 3, [
            PY, "experiments/run_seeds.py",
            "benchmark.seeds=[0]", "benchmark.agents=[ppo]",
            f"benchmark.total_env_steps={b['seeds_steps']}",
            "baselines.ppo.num_envs=2", "baselines.device=cpu",
            "env.time_limit=200",
            f"baselines.benchmark_dir={(vdir / 'bench_seeds').as_posix()}",
            f"hydra.run.dir={(vdir / 'run_seeds').as_posix()}",
        ], artifacts=[str(vdir / "bench_seeds" / "ALE_Pong-v5" / "ppo_seed*.csv")],
            timeout=b["t_train"], check_nan=True),

        # ---- Tier 4 : visualization on existing artefacts ------------------
        Step("viz-reconstruction", 4, [
            PY, "viz/reconstruction.py",
            f"viz.ckpt={wm_ckpt}", f"viz.buffer_dir={val_buf}",
            f"viz.wm_out_dir={viz_out.as_posix()}",
        ], artifacts=[str(viz_out / "reconstruction_*.png")], timeout=b["t_viz"]),

        Step("viz-open-loop", 4, [
            PY, "viz/open_loop_rollout.py",
            f"viz.ckpt={wm_ckpt}", f"viz.buffer_dir={val_buf}",
            f"viz.wm_out_dir={viz_out.as_posix()}",
        ], artifacts=[str(viz_out / "open_loop_*.png")], timeout=b["t_viz"]),

        Step("viz-dream-vs-real", 4, [
            PY, "viz/dream_vs_real.py",
            f"viz.run_dir={pong_run}",
            f"viz.dream_out={(viz_out / 'dream_vs_real.png').as_posix()}",
        ], artifacts=[str(viz_out / "dream_vs_real.png")], timeout=b["t_viz"]),

        Step("viz-real-vs-imagined-video", 4, [
            PY, "viz/real_vs_imagined_video.py",
            f"viz.video_ckpt={pong_ckpt}",
            f"viz.video_out_dir={viz_out.as_posix()}",
            "viz.video_branch_points=[30]", "viz.video_horizon=20",
            f"viz.video_max_steps={b['video_max']}",
        ], artifacts=[str(viz_out / "*branch0030_H20.gif"),
                      str(viz_out / "*branch0030_H20.mp4")], timeout=b["t_viz"]),

        Step("viz-benchmark-comparison", 4, [
            PY, "viz/benchmark_comparison.py",
        ], artifacts=["experiments/benchmark/plots/*.png"], timeout=b["t_viz"]),

        Step("viz-learning-curves", 4, [
            PY, "viz/learning_curves.py",
        ], timeout=b["t_viz"]),

        Step("viz-ablation-summary", 4, [
            PY, "viz/ablation_summary.py",
        ], timeout=b["t_viz"]),

        Step("viz-sanity-checks", 4, [
            PY, "viz/sanity_checks.py",
            f"viz.collect_steps={b['sanity_steps']}",
            f"viz.out_dir={(viz_out / 'sanity').as_posix()}",
        ], artifacts=[str(viz_out / "sanity" / "*.png")], timeout=b["t_viz"]),
    ]

    if not skip_notebook:
        steps.append(Step("notebook-reexec", 4, [
            PY, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
            "--ExecutePreprocessor.timeout=1200",
            "--output", (vdir / "method_validation_reexec.ipynb").as_posix(),
            "notebooks/method_validation.ipynb",
        ], artifacts=[str(vdir / "method_validation_reexec.ipynb")],
            timeout=b["t_notebook"]))

    return steps


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def run_step(step: Step, env: dict, logs_dir: Path) -> None:
    step.log_path = logs_dir / f"{step.name}.log"
    start = time.time()
    try:
        with open(step.log_path, "w", encoding="utf-8", errors="replace") as log:
            log.write("$ " + " ".join(step.cmd) + "\n\n")
            log.flush()
            proc = subprocess.run(
                step.cmd, cwd=REPO, env=env, stdout=log,
                stderr=subprocess.STDOUT, timeout=step.timeout,
            )
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        step.seconds = time.time() - start
        step.status, step.detail = "FAIL", f"timeout after {step.timeout}s"
        return
    step.seconds = time.time() - start

    if rc != 0:
        step.status, step.detail = "FAIL", f"exit code {rc}"
        return

    # artifact existence
    missing = [pat for pat in step.artifacts
               if not glob.glob(str(REPO / pat)) and not glob.glob(pat)]
    if missing:
        step.status, step.detail = "FAIL", "missing artifact(s): " + ", ".join(missing)
        return

    # numeric sanity
    if step.check_nan and step.log_path.exists():
        text = step.log_path.read_text(encoding="utf-8", errors="replace")
        tail = "\n".join(text.splitlines()[-80:])
        m = NAN_RE.search(tail)
        if m:
            bad = next((ln for ln in tail.splitlines() if NAN_RE.search(ln)), "")
            step.status, step.detail = "FAIL", f"nan/inf in log: {bad.strip()[:120]}"
            return

    step.status = "PASS"


def log_tail(step: Step, n: int = 25) -> str:
    if not step.log_path or not step.log_path.exists():
        return "(no log)"
    lines = step.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-n:])


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def write_report(steps: list[Step], vdir: Path, profile: str,
                 started: datetime, total_s: float) -> Path:
    n_pass = sum(s.status == "PASS" for s in steps)
    n_fail = sum(s.status == "FAIL" for s in steps)
    n_skip = sum(s.status == "SKIP" for s in steps)
    verdict = "✅ ALL PASS" if n_fail == 0 else f"❌ {n_fail} FAILED"

    lines = [
        f"# Pipeline verification report — {verdict}",
        "",
        f"- Profile: **{profile}**",
        f"- Started: {started:%Y-%m-%d %H:%M:%S}",
        f"- Duration: {total_s / 60:.1f} min",
        f"- Result: {n_pass} pass / {n_fail} fail / {n_skip} skip "
        f"({len(steps)} steps)",
        "",
        "> Training-flavoured steps are tiny smoke runs (minutes each) that only",
        "> prove the pipeline executes and stays numerically sane. They are NOT",
        "> the deferred Cyfronet training and their numbers are not results.",
        "",
        "| tier | step | status | time (s) | detail |",
        "|---|---|---|---|---|",
    ]
    for s in steps:
        icon = {"PASS": "✅", "FAIL": "❌", "SKIP": "⏭️"}.get(s.status, "•")
        lines.append(
            f"| {s.tier} | {s.name} | {icon} {s.status} | {s.seconds:.0f} | "
            f"{s.detail or '-'} |"
        )

    fails = [s for s in steps if s.status == "FAIL"]
    if fails:
        lines += ["", "## Failure logs (tail)"]
        for s in fails:
            lines += [f"\n### {s.name}", "```", log_tail(s), "```"]

    report = vdir / "report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true",
                    help="minimal budgets (~10 min) — pipeline smoke only")
    ap.add_argument("--tier", default="1,2,3,4",
                    help="comma list of tiers to run (default all)")
    ap.add_argument("--skip-notebook", action="store_true",
                    help="skip the slow notebook re-execution")
    ap.add_argument("--stop-on-fail", action="store_true",
                    help="abort at the first failing step")
    args = ap.parse_args()

    tiers = {int(t) for t in args.tier.split(",") if t.strip()}
    profile = "quick" if args.quick else "confidence"
    started = datetime.now()
    ts = started.strftime("%Y%m%d_%H%M%S")
    vdir = REPO / "experiments" / "verification" / f"run_{ts}"
    logs_dir = vdir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["SDL_VIDEODRIVER"] = "dummy"
    env["PYTHONUTF8"] = "1"
    env["HYDRA_FULL_ERROR"] = "1"
    env["TQDM_DISABLE"] = "1"

    steps = build_steps(vdir, budgets(args.quick), args.skip_notebook)
    steps = [s for s in steps if s.tier in tiers]

    print(f"[verify] profile={profile}  tiers={sorted(tiers)}  "
          f"steps={len(steps)}  out={vdir}")
    t0 = time.time()
    for i, s in enumerate(steps, 1):
        print(f"[verify] ({i}/{len(steps)}) tier{s.tier} {s.name} ... ",
              end="", flush=True)
        run_step(s, env, logs_dir)
        print(f"{s.status} ({s.seconds:.0f}s) {s.detail}")
        if s.status == "FAIL" and args.stop_on_fail:
            for rest in steps[i:]:
                rest.status = "SKIP"
            break
    total_s = time.time() - t0

    report = write_report(steps, vdir, profile, started, total_s)
    n_fail = sum(s.status == "FAIL" for s in steps)
    print(f"\n[verify] {'ALL PASS' if n_fail == 0 else str(n_fail) + ' FAILED'} "
          f"in {total_s / 60:.1f} min")
    print(f"[verify] report: {report}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
