# Results & experiment status (Phase 4)

Status date: 2026-08-09. **Local training is paused at the user's
request**; the one training-time study run so far — the Phase 4E
parametric sweep (§E) — was executed remotely on Cyfronet Athena on
2026-07-20. This file records what has been run, what every deferred
experiment costs, and the exact commands to finish the plan (locally or on
Cyfronet Athena via `slurm/`). Nothing below is silently skipped: every gap
is listed in [Deferred work](#deferred-work).

Revised 2026-08-09 after an audit of the recorded runs, with no new training:
the dream-vs-real correlation is now reported detrended as well as raw (§E,
and README "Phase 2 results" §2), the seed-dependent failure mode is stated
explicitly (§E), and two correctness fixes landed — continuous action
rescaling and reproducible seeding — both noted where they affect the
results below.

## A. Environment validation ladder

| # | environment | actions | obs | status | approx. cost (GTX 1660 Ti) |
|---|---|---|---|---|---|
| 1 | CartPole-v1 (vector) | discrete | state | **skipped — decision**: Phases 0–2 were validated directly on Pong; a vector-obs sanity env would bypass the pixel encoder/decoder entirely, testing little of this architecture | — |
| 2 | ALE/Pong-v5 | discrete | pixels 64×64 | **validated** (Phases 1–2 full runs: WM criteria + reward −21 → −1.7) | WM-only 10k updates ≈ 75 min; full Dreamer 60k steps ≈ 4.2 h |
| 3a | Atari Breakout / MsPacman | discrete | pixels | **deferred** (no training allowed now); configs exist (`env=atari_breakout`) or are one YAML away | ≈ 4–6 h each per seed |
| 3b | CarRacing-v3 | continuous | pixels | **deferred**; env verified working (Box2D present), Dreamer continuous branch + SAC/PPO smoke-tested on it; run LAST (long episodes) | ≈ 6–10 h per seed |
| — | dm_control (walker, cartpole) | continuous | pixels | **unavailable in this sandbox** (`dm_control` not installed); adapter exists (`envs/dmc.py`), untested | — |

Ladder rule respected: nothing was launched on step-3 environments; step 2
passed all sanity checks first (KL 1.2–1.6 nats — no collapse; reward rises).

## B. Learning curves (multi-seed)

Infrastructure ready: `experiments/run_seeds.py` (sequential seeds — the
single sandbox GPU rules out parallel runs; wall-clock limitation, not
correctness), `viz/learning_curves.py` (mean ± std across seeds on a shared
env-step grid).

**What exists today (single seed, from Phase 3 — see README "Benchmark
results")**: Dreamer-warmstart reaches −1.7 on Pong while PPO/DQN stay at
the −21 random floor for the whole 100k-step budget; wall-clock order is
reversed (2.7 / 9.2 min vs 249 min). 1 seed = methodologically weak;
treat as direction, not effect size.

**To finish (exact commands)**:

```bash
# Pong, 3 seeds, Dreamer + both discrete baselines (~14 h GPU on 1660 Ti,
# ~2-3 h on an A100):
python experiments/run_seeds.py "benchmark.seeds=[0,1,2]" \
    "benchmark.agents=[dreamer,ppo,dqn]" benchmark.total_env_steps=100000
python viz/learning_curves.py
```

Priority if budget is short (per the Phase 4 spec): Dreamer full seeds
first, baselines at ≥ 2 seeds, annotate any reduction here.

## C. Ablations

All ablations are single-Hydra-override presets (`configs/ablation/`),
each self-labels its benchmark CSV (`dreamer-H5`, `dreamer-nofreenats`,
...). Representative env: **Pong** (ladder step 2). Suggested budget:
30k env steps, 2 seeds per variant (a deliberate compromise vs 3–5 in
Part B — more variants, fewer seeds). ~2 h/run on the 1660 Ti, ~25 min on
an A100 ⇒ full matrix ≈ 22 GPU-h (1660 Ti) / ≈ 5 GPU-h (A100).

| group | variants (preset) | run status | expected / hypothesis (to verify) |
|---|---|---|---|
| horizon | `horizon_5`, `horizon_10`, base H=15, `horizon_20` | **deferred** | short H → weaker credit assignment; H=20 ≈ 15 given open-loop stays coherent ≥ 40 steps |
| latent size | `deter_128`, `deter_256`, base 512; `stoch_16x16`, base 32×32, `stoch_32x64` | **deferred** | Pong is simple: expect mild degradation only at 128 |
| loss variants | `no_kl_balance`, `no_free_nats`, `no_reconstruction`, base | **deferred** (mandatory: `no_reconstruction`) | no free nats → KL→0 collapse early; reconstruction-free → unstable/much worse given the Phase 1 finding that reward gradient alone failed to shape features (probe AUC 0.49) |
| latent type | base categorical vs `gaussian_latent` | **deferred** | V2 finding: categorical better on Atari |

```bash
# One line per variant, e.g.:
python experiments/run_seeds.py ablation=no_reconstruction \
    "benchmark.seeds=[0,1]" benchmark.total_env_steps=30000
python viz/ablation_summary.py        # plots + md/csv tables per group
# On Athena: sbatch slurm/ablations.sbatch   (job array over all presets)
```

Note: the reconstruction-free preset genuinely removes the decoder
(no parameters, no forward, no gradient — `models/world_model.py`), so it
is a compute ablation, not a loss-weight-zero imitation. Verified by unit
tests (`tests/test_ablations.py`).

## D. Visual artifacts (generated from the Phase 2 checkpoint, inference only)

Checkpoint: `experiments/dreamer_pong/checkpoints/dreamer_final.pt`
(60k env steps, 18k updates).

1. **Reconstructions** — `experiments/phase4_final/reconstruction_*.png`:
   posterior recon MSE/px on held-out data: mean **0.00002** (min 0.00001,
   max 0.00004) — improved ~2.5× over the Phase 1 checkpoint (0.00005).
2. **Dreams (open-loop prior, H=40 — 2.7× the training horizon)** —
   `experiments/phase4_final/open_loop_*.png|gif`: per-step MSE/px
   0.00002 (step 1) → 0.00010 (step 40), **no degeneration point found**
   within 40 steps; paddles and playfield stay coherent. This partially
   answers the horizon ablation: the world model is not the binding
   constraint at H=20 vs 15 on Pong.
3. **Real-vs-imagined side-by-side videos** — `experiments/videos/`:
   `pong-v5_branch{0030,0150,0400}_H20.{gif,mp4}` (early/middle/late
   branch, red border + held frame at the branch moment) and
   `pong-v5_branch0150_H50.{gif,mp4}` (degeneration probe: H=50 from a
   model trained with H=15 — still visually coherent, consistent with the
   open-loop MSE curve). Second environment (CarRacing) deferred — no
   trained continuous checkpoint yet (no training allowed at present).

## E. Method-validation notebook (inference-only evidence)

`notebooks/method_validation.ipynb` (executed, outputs embedded; figures
also in `experiments/analysis/`). Key numbers, all on the Phase 2
checkpoint + held-out data, no training involved:

| evidence | result |
|---|---|
| trained vs random policy (12 eval episodes each) | **−2.00 ± 1.58 vs −20.17 ± 1.86**, Mann-Whitney one-sided **p = 8.5e-6**, rank-biserial effect **1.00** (zero overlap) |
| dream-vs-real learning correlation (training logs) | r = **0.96 raw**, **0.72** after removing the trend in env_step, **−0.01** on first differences — mostly a shared upward trend, not step-to-step tracking (see README "Phase 2 results" §2; all three printed by `viz/dream_vs_real.py`) |
| reward head, held-out | Pearson r = 0.87; scoring-event detection **ROC-AUC = 0.987** |
| continue head | cont prob 1.000 at non-terminal vs 0.699 at terminal steps |
| parametric A: open-loop error vs horizon K=1..60 | model beats the repeat-last-frame baseline at **every** K (×5.7 avg over first 15 steps); error grows smoothly, no degeneration cliff by K=60 |
| parametric B: posterior context sweep | monotone: 1 frame → 2.5e-4 MSE/px, 12 frames → 6.8e-5; most of the gain within ~5 frames |
| critic vs empirical discounted return-to-go | Pearson r = 0.52 (moderate by construction — G_t is a single-sample, high-variance target) |

The training-time parametric studies proposed there have been run on
Cyfronet Athena (Slurm array 2806135, 15×A100, 2026-07-20; one GPU per
(variant, seed), shared 60k env-step budget). Final Pong return,
mean ± std over 3 seeds; `steps_to_90pct` = env steps to first reach 90%
of the way from the group's worst to the group's best final return (a
group-relative threshold, per `viz/ablation_summary.py` — so the base run,
which sits in both groups, crosses a different absolute threshold in each:
55.5k in the train_ratio group, 56.5k in the entropy_coef group).
`seeds reaching` is how many of the three seeds crossed that threshold at
all: the step count averages **only those**, so it is reported for
completeness and is not a ranking metric here.

| variant | final return | steps_to_90pct | seeds reaching |
|---|---|---|---|
| base (train_ratio 0.3, entropy 3e-4) | −11.9 ± 6.4 | ~55.5k / 56.5k | 1/3 in both groups |
| train_ratio 0.1 | −20.5 ± 0.2 | n/a (no learning) | 0/3 |
| **train_ratio 1.0** | **−6.1 ± 3.6** | ~47.3k | 2/3 |
| **entropy_coef 1e-4** | **−6.4 ± 1.9** | ~43.8k | 1/3 |
| entropy_coef 1e-3 | −14.5 ± 6.8 | ~47.3k | 1/3 |

Reading — every conclusion below rests on `final return`, which is computed
from all three seeds; apart from train_ratio 1.0 each step count comes from
a single seed, so the gaps between them (43.8k vs 47.3k) are noise rather
than an effect. train_ratio is a pure compute-for-return knob at fixed
sample count — 0.1 never leaves the random-policy floor (std 0.24: all three
seeds pinned at −21), while 1.0 buys the best return at ~8× the
wall-clock of 0.1 (~4 h vs ~0.5 h per seed on an A100). For entropy,
*lower* is better at this budget: 1e-4 matches tr=1.0's return with the
smallest seed variance of any variant, whereas 1e-3 over-explores —
worse mean than base and the largest spread. Artifacts:
`experiments/benchmark/plots/ablation_{train_ratio,entropy_coef}.{png,md,csv}`
and per-run CSVs under `experiments/benchmark/ALE_Pong-v5/` (generated
by `viz/ablation_summary.py` + `viz/learning_curves.py`).

**Known failure mode: from scratch, roughly one seed in three never leaves
the floor.** In the base configuration seed 1 finishes at −20.8 while seeds
0 and 2 reach −8.7 and −6.1; the same seed also fails at entropy 1e-3
(−20.8) and is the worst of its group at train_ratio 1.0 (−10.6). That is a
reproducible seed-dependent failure, not spread around a mean, and it is why
the base row reads −11.9 ± 6.4. Raising train_ratio to 1.0 rescues it only
partially. The warm-started Phase 2 run does not show it — that run starts
from a Phase 1 world model and a preloaded buffer, i.e. 110k unique
interactions in total, not 60k.

**This sweep is the strongest internal-validity evidence in the repo.**
train_ratio changes only how many imagination-based gradient updates are
taken per environment step, at a fixed sample budget: 0.1 leaves all three
seeds at the random floor, 0.3 gets −11.9, 1.0 gets −6.1. A monotone
dose-response on exactly that knob is what distinguishes "the policy learns
from imagined rollouts" from "the policy learns from something else".

Reproducibility caveat: every run above predates the seeding fix in
`train/seeding.py` (Gymnasium does not seed `env.action_space` from
`reset(seed=...)`, so random prefill and exploration differed between runs
with the same seed). The numbers remain valid as samples of across-seed
variance, but those specific runs cannot be reproduced step for step. Runs
started after the fix can.

Still prepared but not yet run:

| study | presets | script |
|---|---|---|
| architecture/loss ablations | `configs/ablation/*` (11 presets) | `slurm/ablations.sbatch` |

## Deferred work

In priority order, everything runnable as-is. Position 1 has been promoted
above the multi-seed curves: the train_ratio dose-response (§E) already shows
that imagination training is what drives learning, and `no_reconstruction` is
the one remaining experiment that tests *why* — whether the pixel loss is
load-bearing for the representation the policy learns from.

1. **`no_reconstruction` ablation (C)** — the single most informative run
   left. Pre-registered hypothesis, written down before launching:

   > Removing the decoder collapses Pong learning to near the random floor
   > (final return worse than −18 on 2 seeds at 30k steps), because reward
   > and continue signals alone are too sparse to shape the latent. The
   > Phase 1 evidence for this is direct: with the plain MSE reward head the
   > representation never encoded scoring events (probe AUC 0.49) until the
   > reconstruction gradient was rebalanced.
   >
   > Decisive either way. If it collapses, the pixel loss is load-bearing
   > and the Phase 1 finding generalises to policy learning. If it learns
   > anyway, the reconstruction term is not what makes this work and the
   > README's account of the architecture needs revising.

2. Horizon ablations (C) — §D found no open-loop degeneration through H=40,
   so H=20 ≈ H=15 is the prediction; H=5 should hurt credit assignment.
3. Multi-seed Pong learning curves (B) — commands above.
4. Remaining ablation groups (C).
5. Ladder step 3: Breakout/MsPacman seeds, then CarRacing (Dreamer
   continuous + SAC + PPO) with fewer seeds; CarRacing videos afterwards.
   Note CarRacing is now unblocked on the action-space side — continuous
   actions are rescaled to [-1,1] in the shared wrapper chain, so the buffer
   stores the action the env actually applied (see README, protocol
   deviation 6).

Suggested venue: Cyfronet Athena (A100) — `slurm/setup_athena.sh`,
`slurm/benchmark.sbatch`, `slurm/benchmark_array.sbatch`,
`slurm/ablations.sbatch`. Full plan ≈ 1–2 A100 GPU-days total.

The ablation array indexes `configs/ablation/` presets in the order listed in
`slurm/ablations.sbatch`, so single studies can be launched without the whole
matrix (~2 h/run on a 1660 Ti, ~25 min on an A100; 2 seeds per preset, or set
`SEEDS` to override — see below):

```bash
# Task 0 is the base run. Launch it WITH no_reconstruction, not separately:
# the only base data on record is 60k steps / 3 seeds and predates the fixes
# below, so it is not a control for a 30k reconstruction-free run.
SEEDS='[0,1,2]' sbatch --export=ALL,SEEDS --array=0,10 slurm/ablations.sbatch
sbatch --array=1-3  slurm/ablations.sbatch   # horizon_5 / _10 / _20   (~3 A100-h)
sbatch --array=0-10 slurm/ablations.sbatch   # the full matrix         (~5 A100-h)

# After the array finishes:
python viz/ablation_summary.py && python viz/learning_curves.py
```

Two seeds is the default because the full matrix is priced at two. For the
decisive presets use three: the one-in-three floor failure documented in §E
is otherwise indistinguishable from the collapse the `no_reconstruction`
hypothesis predicts. Three seeds at 30k steps is ~2 h per array task on an
A100, still inside the script's 6 h walltime.

**Before launching anything on Athena, sync `$SCRATCH/dreamer-rssm` to a
revision that contains the 2026-08-09 correctness fixes** — in particular
`fix(dreamer): act on the reset observation, not the terminal frame` and
`fix(seed): make a run reproducible from cfg.seed`. Runs started on older
code carry the reset-observation bug and are not comparable with anything
produced after it.
