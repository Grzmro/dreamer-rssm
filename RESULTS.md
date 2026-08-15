# Results & experiment status (Phase 4)

Status date: 2026-08-15. **Local training is paused at the user's request**;
every training-time study was executed remotely on Cyfronet Athena. Three
campaigns have run: the Phase 4E parametric sweep (§E, 2026-07-20), the
full 400k benchmark (§B, 2026-08-10) and the `no_reconstruction` ablation
with its base control (§C, 2026-08-15). This file records what has been
run, what every deferred experiment costs, and the exact commands to finish
the plan (locally or on Cyfronet Athena via `slurm/`). Nothing below is
silently skipped: every gap is listed in [Deferred work](#deferred-work).

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
| 2 | ALE/Pong-v5 | discrete | pixels 64×64 | **validated** (Phases 1–2 full runs; benchmarked to 400k × 3 seeds, final +9.6 ± 1.0 — §B) | WM-only 10k updates ≈ 75 min; full Dreamer 60k steps ≈ 4.2 h; 400k ≈ 9.4 h on an A100 |
| 3a | Atari Breakout / MsPacman | discrete | pixels | **ready to launch** on Athena (`ENV=atari_breakout`); no local training | ≈ 4–6 h each per seed |
| 3b | CarRacing-v3 | continuous | pixels | **ready to launch** on Athena — Box2D installed and `CarRacing-v3` verified there 2026-08-15; Dreamer continuous branch + SAC/PPO smoke-tested; run LAST (long episodes) | ≈ 6–10 h per seed |
| — | dm_control (walker, cartpole) | continuous | pixels | **unavailable in this sandbox** (`dm_control` not installed); adapter exists (`envs/dmc.py`), untested | — |

Ladder rule respected: nothing was launched on step-3 environments; step 2
passed all sanity checks first (KL 1.2–1.6 nats — no collapse; reward rises).

## B. Learning curves (multi-seed)

Infrastructure ready: `experiments/run_seeds.py` (sequential seeds — the
single sandbox GPU rules out parallel runs; wall-clock limitation, not
correctness), `viz/learning_curves.py` (mean ± std across seeds on a shared
env-step grid).

**Done — 400k env steps, 3 seeds, all three discrete agents.** Cyfronet
Athena, Slurm array 2887185 (3×A100, one seed per task, 2026-08-10),
revision `ef90d98` — i.e. *after* the reset-observation and seeding fixes.
Final return is the mean over each run's last 10 episodes, then mean ± std
across seeds.

| agent | final return | per seed | wall-clock (mean/seed) |
|---|---|---|---|
| **dreamer** | **+9.6 ± 1.0** | 10.4 / 8.2 / 10.2 | 9.4 h |
| dqn (double+dueling) | −7.9 ± 2.2 | −11.0 / −6.9 / −5.9 | 17 min |
| ppo | −19.9 ± 0.8 | −18.8 / −20.3 / −20.7 | 7 min |

Dreamer is the only agent that reaches a positive score, and it does so on
all three seeds; PPO never leaves the random floor within the budget. The
wall-clock order is reversed by roughly the same factor it always was —
Dreamer costs ~33× DQN and ~83× PPO. Both axes are the point: sample
efficiency is won, wall-clock is lost, and averaging the two together would
erase the finding.

**The one-in-three floor failure documented in §E does not appear at this
budget.** At 60k the base configuration had one seed stuck at −20.8; here
the worst seed finishes at +8.2. Read together with the truncation table
below, the failure at 60k looks like *not yet started* rather than
*permanently stuck* — learning on Pong begins somewhere between 30k and
60k steps and the seeds reach it at different times.

Base run truncated at increasing budgets (same three seeds, same run):

| budget | return | per seed |
|---|---|---|
| 20–30k | −20.4 ± 0.4 | −20.5 / −19.9 / −20.8 |
| 50–60k | −9.5 ± 2.7 | −5.6 / −11.3 / −11.5 |
| 90–100k | −3.8 ± 1.8 | −4.4 / −1.4 / −5.7 |
| 140–150k | +4.4 ± 5.1 | 11.6 / 0.9 / 0.7 |
| 190–200k | +7.0 ± 3.1 | 11.2 / 3.6 / 6.3 |

This table is load-bearing for §C: **nothing has happened yet at 30k**, so
that budget cannot discriminate between architecture variants.

Artifacts: `experiments/benchmark/ALE_Pong-v5_bench400k_20260810/*.csv`
(9 runs) and `experiments/benchmark/plots/ALE_Pong-v5_bench400k_20260810_*.png`
(learning curves, env-step and wall-clock axes).

## C. Ablations

All ablations are single-Hydra-override presets (`configs/ablation/`),
each self-labels its benchmark CSV (`dreamer-H5`, `dreamer-nofreenats`,
...). Representative env: **Pong** (ladder step 2).

**The 30k budget originally suggested here is dead — see the result below.
Use ≥ 100k.** Base Dreamer is still at the random floor at 30k (§B), so at
that budget every variant scores −20-something and the comparison measures
nothing. The revised figure is ~1.1 min per 1k env steps per seed on an
A100, i.e. ≈ 2.8 h per seed at 150k, ≈ 8 h for three seeds — which does not
fit the 6 h walltime in `slurm/ablations.sbatch`, so pass
`--time=12:00:00` or split seeds across array tasks.

| group | variants (preset) | run status | expected / hypothesis (to verify) |
|---|---|---|---|
| horizon | `horizon_5`, `horizon_10`, base H=15, `horizon_20` | **deferred** | short H → weaker credit assignment; H=20 ≈ 15 given open-loop stays coherent ≥ 40 steps |
| latent size | `deter_128`, `deter_256`, base 512; `stoch_16x16`, base 32×32, `stoch_32x64` | **deferred** | Pong is simple: expect mild degradation only at 128 |
| loss variants | `no_kl_balance`, `no_free_nats`, `no_reconstruction`, base | `no_reconstruction` **run at 30k — inconclusive** (below); `no_kl_balance`/`no_free_nats` deferred | no free nats → KL→0 collapse early; reconstruction-free → unstable/much worse given the Phase 1 finding that reward gradient alone failed to shape features (probe AUC 0.49) |
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

### C.1 `no_reconstruction` at 30k — inconclusive, and why that matters

Cyfronet Athena, Slurm array 2909563 (2×A100, 2026-08-15), 30k env steps,
3 seeds, revision `4ee3da4`. The base run (array task 0) was launched in
the same array, on the same budget, seeds and revision, precisely so the
ablation had a control.

| variant | final return | per seed | wall-clock (mean/seed) |
|---|---|---|---|
| base (decoder present) | −20.8 ± 0.1 | −20.7 / −20.9 / −20.9 | 33 min |
| `no_reconstruction` | −20.2 ± 0.3 | −19.8 / −20.4 / −20.3 | 32 min |

**Both arms are at the random floor, so the experiment does not
discriminate.** Removing the decoder cannot be shown to hurt a run that has
not started learning either way — §B puts the onset of learning on Pong
between 30k and 60k steps, and an independent measurement agrees: the base
curve from the 400k benchmark, truncated to 20–30k, gives −20.4 ± 0.4
against −20.8 ± 0.1 measured here in a separate run.

The pre-registered criterion was *"final return worse than −18 on 2 seeds
at 30k"*. `no_reconstruction` satisfies it at −20.2 — **and so does the
base at −20.8.** Had the ablation been launched alone, as this file
previously instructed (`sbatch --array=10`), the recorded outcome would
have been a confirmed hypothesis and a false claim that the pixel loss is
load-bearing. The control is the only reason that did not happen. The
criterion was underspecified, not merely unlucky: a threshold on absolute
return is meaningless without knowing where the control sits.

Two further cautions about this table. The `steps_to_90pct` column in
`ablation_loss_variants.md` (9.6k vs 12.6k) is noise — the group's
worst-to-best span is 0.7 points, so the 90% threshold falls inside
seed variance. And the near-identical wall-clock (32 vs 33 min) is itself
informative: at this scale the decoder is not what makes training slow, so
a reconstruction-free variant buys almost no compute back.

**Redesigned test.** No new base run is needed — the 400k benchmark base
(§B) is a valid control at any budget ≤ 400k: same revision for training
purposes, same three seeds, and verified consistent with a separate base
run at 30k. So only the ablation arm has to be re-run, at a budget where
the control is clearly learning:

```bash
# ~8 h for 3 seeds on one A100; compare against the §B base truncated to 150k
SEEDS='[0,1,2]' STEPS=150000 sbatch --time=12:00:00 \
    --export=ALL,SEEDS,STEPS --array=10 slurm/ablations.sbatch
```

Revised hypothesis, registered before that run: at 150k the base reaches
+4.4 ± 5.1, so a load-bearing pixel loss should leave `no_reconstruction`
below −15, and anything above roughly −5 falsifies the claim that
reconstruction is what shapes the representation the policy uses.

Artifacts: `experiments/benchmark/ALE_Pong-v5_norecon_20260815/*.csv`,
`experiments/benchmark/plots/ablation_loss_variants.{png,md,csv}`.

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
| architecture/loss ablations | `configs/ablation/*` (10 presets still open; `no_reconstruction` ran but was inconclusive at 30k — §C.1) | `slurm/ablations.sbatch` |

## Deferred work

In priority order, everything runnable as-is. Multi-seed Pong curves (B) are
done as of 2026-08-10 and have dropped off this list. Position 1 is still the
`no_reconstruction` question — the train_ratio dose-response (§E) shows that
imagination training drives learning, and this is the experiment that tests
*why* — but it now needs a budget at which the control is awake.

1. **Re-run `no_reconstruction` at 150k (C.1)** — the 30k attempt was
   inconclusive because base Dreamer is also at the floor there. Only the
   ablation arm needs GPU time; the §B base is the control. Command and
   revised pre-registered hypothesis in §C.1. ≈ 8 A100-h.
2. **Wall-clock-matched benchmark (B)** — the same comparison with equal
   *seconds* instead of equal samples, which is the budget that actually
   applies when one GPU is the constraint. Infrastructure landed 2026-08-15
   (`benchmark.time_budget_s` + `benchmark.agent_env_steps`,
   `train/budget.py`, README "The other budget").

   **Only the baselines need GPU time.** Dreamer's recorded 400k run already
   spans 9.4 h and its CSV carries `wall_time_s`, so its return at any budget
   T ≤ 9.4 h is obtained by truncating on that column — no re-run.

   Start at T = 2 h (≈ 12 A100-h: 2 agents × 3 seeds × 2 h), not at the full
   9.4 h (≈ 56 A100-h). At 2 h DQN gets ~2.9M steps and PPO ~7M against
   Dreamer's ~85k, which is already a 30-80× sample advantage; if the
   baselines win there, the longer budget can only widen it and need not be
   run at all.

   ```bash
   ENV=atari_pong sbatch --time=8:00:00 --export=ALL,ENV slurm/benchmark.sbatch
   # with, inside the script or as overrides:
   #   benchmark.time_budget_s=7200
   #   "benchmark.agents=[ppo,dqn]"
   #   "benchmark.agent_env_steps={dqn: 2900000, ppo: 7000000}"
   ```

   Pre-registered expectation, written before the run: **the baselines win.**
   PPO and DQN both solve Pong given millions of frames, and 2 h buys them
   that. The interesting number is not who wins but the crossover — the
   wall-clock at which Dreamer's curve stops being ahead — and whether
   Dreamer's advantage survives at all once the axis is seconds. A result
   where Dreamer still leads at equal time would be surprising and would
   need checking for a throughput bug before being believed.
3. Horizon ablations (C) — §D found no open-loop degeneration through H=40,
   so H=20 ≈ H=15 is the prediction; H=5 should hurt credit assignment.
   Same budget correction applies: run at ≥ 100k, not 30k.
4. Remaining ablation groups (C), likewise at ≥ 100k. Note this raises the
   full-matrix cost well above the ≈ 5 A100-h quoted below, which assumed
   30k — budget ≈ 8 A100-h per preset at 150k × 3 seeds.
5. Ladder step 3: Breakout/MsPacman, then CarRacing (Dreamer continuous +
   SAC + PPO) with fewer seeds; CarRacing videos afterwards. Both are
   unblocked as of 2026-08-15:
   - the benchmark scripts take `ENV` from the environment, so a second game
     needs no edit: `ENV=atari_breakout sbatch --export=ALL,ENV slurm/benchmark.sbatch`;
   - each env writes to its own `experiments/benchmark/<env>/` tree, so a new
     game cannot append into Pong's CSVs;
   - continuous actions are rescaled to [-1,1] in the shared wrapper chain,
     so the buffer stores the action the env actually applied (see README,
     protocol deviation 6);
   - Box2D is installed in the Athena venv and `CarRacing-v3` was verified
     there on 2026-08-15 (`swig` + `gymnasium[box2d]`).
   Breakout first: it reuses the exact discrete path the 400k benchmark just
   validated, so it answers "is this tuned to Pong?" for ~5 A100-h. CarRacing
   last — it is the only study that exercises the continuous actor loss
   (gradient through the dynamics rather than REINFORCE), and its long
   episodes make it the most expensive at ≈ 6–10 h per seed.

Suggested venue: Cyfronet Athena (A100) — `slurm/setup_athena.sh`,
`slurm/benchmark.sbatch`, `slurm/benchmark_array.sbatch`,
`slurm/ablations.sbatch`. Full plan ≈ 1–2 A100 GPU-days total.

The ablation array indexes `configs/ablation/` presets in the order listed in
`slurm/ablations.sbatch`, so single studies can be launched without the whole
matrix. `SEEDS` and `STEPS` are environment overrides:

```bash
# Re-run of the decisive preset at a budget where the control is learning.
# No base task this time: the 400k benchmark base (section B) is the control.
SEEDS='[0,1,2]' STEPS=150000 sbatch --time=12:00:00 \
    --export=ALL,SEEDS,STEPS --array=10 slurm/ablations.sbatch

# Horizons, same correction (~24 A100-h for the three presets):
SEEDS='[0,1,2]' STEPS=150000 sbatch --time=12:00:00 \
    --export=ALL,SEEDS,STEPS --array=1-3 slurm/ablations.sbatch

# After the array finishes:
python viz/ablation_summary.py && python viz/learning_curves.py
```

The 30k default that the array still ships with is a smoke-test budget, kept
so `--array=0-10` stays cheap enough to prove the matrix executes. It is not
a research budget: section C.1 is the worked example of a 30k ablation
answering nothing.

Two operational rules learned the hard way on 2026-08-10 and -15:

- **Sync `$SCRATCH/dreamer-rssm` before launching.** Runs started before the
  2026-08-09 fixes — `fix(dreamer): act on the reset observation, not the
  terminal frame` and `fix(seed): make a run reproducible from cfg.seed` —
  carry the reset-observation bug and are not comparable with anything
  produced after it.
- **Archive the previous campaign's CSV directory before starting a new one
  on the same env.** `BenchmarkLogger` appends, and a preset that reuses a
  label (task 0 writes plain `dreamer_seed*.csv`) will append a fresh
  low-step curve onto a finished high-step one, which `load_run()` then
  rejects outright. The convention in this repo is a dated sibling:
  `ALE_Pong-v5_bench400k_20260810/`, `ALE_Pong-v5_norecon_20260815/`.
  Different *environments* need no such care — they get their own tree.
