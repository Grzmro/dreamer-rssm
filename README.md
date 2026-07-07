# dreamer-rssm

A from-scratch **DreamerV2/V3**-style world-model RL implementation (simplified, with ablations), built in phases.

**Status: Phase 2 — policy learning in imagination.** On top of the Phase 0
data pipeline and the Phase 1 world model (CNN encoder/decoder, RSSM with
categorical/gaussian latents, reward/continue heads), the repo now contains
the full Dreamer loop: an actor and critic trained purely on imagination
rollouts through the frozen world-model prior, lambda-returns with an EMA
target critic, and interleaved collection <-> world-model <-> actor-critic
training driven by a train ratio (see [Roadmap](#roadmap)).

## Repo structure

```
envs/          # environment wrappers (64x64 resize, grayscale, action repeat, time limit, normalization)
data/          # sequential replay buffer (whole episodes, uint8, FIFO eviction)
train/         # collection, world-model training, lambda-returns, imagination rollout, Dreamer loop
models/        # world model (encoder/decoder/RSSM/heads) + actor, critic, AC losses, return normalizer
viz/           # sanity checks, reconstruction, open-loop rollout, dream-vs-real returns
configs/       # Hydra configs (groups: env, buffer, collect, model, train_wm, agent, train_dreamer)
experiments/   # run outputs (gitignored)
tests/         # pytest
```

## Installation

Requires Python 3.10+ (tested on 3.12, Windows).

```bash
pip install -e .          # Atari environments (ale-py, ROMs bundled with the package)
pip install -e .[dev]     # + pytest
pip install -e .[dmc]     # + dm_control (optional, requires MuJoCo — see below)
```

## Data collection (random policy)

```bash
python train/collect.py                                        # default: ALE/Pong-v5, 10k steps
python train/collect.py env=atari_breakout collect.num_steps=20000
python train/collect.py buffer.capacity=100000 env.action_repeat=2 env.grayscale=true
```

Output goes to `experiments/runs/<timestamp>/`:
- `tb/` — TensorBoard logs (`tensorboard --logdir experiments/runs`): episode return and length, throughput (steps/s),
- `buffer/` — episodes saved as `episode_*.npz` (disable with `collect.save_buffer=false`).

## Batch sampling (programmatic)

```python
from data import SequenceReplayBuffer

buffer = SequenceReplayBuffer.load("experiments/runs/<ts>/buffer", capacity=100_000)
batch = buffer.sample(batch_size=16, seq_len=50)
batch["obs"]     # [16, 50, 3, 64, 64] float32 in [-0.5, 0.5]
batch["action"]  # [16, 50] int64 (discrete) / [16, 50, A] float32 (continuous)
batch["reward"], batch["terminated"], batch["truncated"]  # [16, 50]
batch["is_first"], batch["mask"]  # [16, 50]; mask=0 for padding on episodes shorter than L
```

Conventions:
- episodes are stored whole; observations as **uint8 HWC** (memory-efficient), normalized to `[-0.5, 0.5]` and transposed to CHW only in `sample()`;
- episodes are sampled with probability proportional to their length (avoids bias against long episodes);
- step layout follows the Dreamer convention: `obs[0]` is the reset observation, `action[0]` is a zero dummy, `reward[0] == 0`, `is_first[0] == True`;
- capacity is in steps; oldest episodes are FIFO-evicted once it's exceeded.

## Sanity checks

```bash
python viz/sanity_checks.py                                           # collects 3k fresh steps
python viz/sanity_checks.py viz.buffer_dir=experiments/runs/<ts>/buffer  # uses a saved buffer
```

Writes to `experiments/sanity/`: episode return histogram, episode length histogram,
a frame grid from a sampled sequence; prints per-step reward stats and the pixel value
range of a batch (must be within `[-0.5, 0.5]`).

## World model (Phase 1)

Architecture (see `models/README.md` for file-level details):

- **Encoder**: 4x stride-2 conv (32-64-128-256 channels, kernel 4), channel
  LayerNorm + SiLU, flattened to a 4096-dim embedding.
- **RSSM**: GRU deterministic state `h` (512), stochastic latent `z` —
  default **categorical** 32 variables x 32 classes with straight-through
  gradients and 1% unimix; **gaussian** (30-dim, reparameterized) available as
  an ablation via `model=rssm_gaussian`.
- **Decoder**: mirror transposed CNN from `[h, z]` (1536-dim features); MSE
  reconstruction (unit-variance Gaussian — simplified vs full V3 likelihood).
- **Heads**: reward (MSE; symlog + two-hot left as a documented TODO) and
  continue (BCE on `1 - terminated`; truncation is not treated as death).
- **Loss**: `recon + reward + cont + kl`, all scales 1.0 by default;
  KL balancing 0.8/0.2 (V2) with free nats 1.0 (V3), applied per timestep
  after summing over latent groups.

Training (world model only — no policy learning):

```bash
# collect data first (or let the script collect fresh data itself)
python train/collect.py collect.num_steps=50000 hydra.run.dir=experiments/data/train_50k
python train/collect.py collect.num_steps=5000 seed=100 hydra.run.dir=experiments/data/val_5k

python train/train_world_model.py \
    train_wm.buffer_dir=experiments/data/train_50k/buffer \
    train_wm.val_buffer_dir=experiments/data/val_5k/buffer
```

Logs (TensorBoard): every loss component, raw KL vs post-free-nats KL, grad
norm, held-out reward Pearson correlation. Checkpoints land in
`<run_dir>/checkpoints/`.

Validation visualizations:

```bash
python viz/reconstruction.py    viz.ckpt=<run>/checkpoints/wm_final.pt viz.buffer_dir=<data>/buffer
python viz/open_loop_rollout.py viz.ckpt=<run>/checkpoints/wm_final.pt viz.buffer_dir=<data>/buffer
```

`open_loop_rollout` is the key check: 5 posterior warm-up steps, then 15
prior-only steps (`imagine()`, no encoder) replaying true actions, decoded and
compared frame-by-frame against ground truth (PNG grids + GIFs + per-step MSE).

## Policy learning in imagination (Phase 2)

Components (all on RSSM features `s_t = [h_t, z_t]`):

- **Actor** (`models/actor.py`): MLP -> categorical logits with 1% unimix
  (discrete, REINFORCE) or tanh-squashed Normal (continuous, reparameterized
  dynamics backprop). `act()` returns RSSM-ready actions + log-prob + entropy.
- **Critic** (`models/critic.py`): MLP value head (`mse` or `symlog_twohot`,
  two-hot default) plus an **EMA target critic** (`tau = 0.98` per gradient
  step) used only for lambda-return bootstraps.
- **Imagination rollout** (`train/imagine_rollout.py`): starts from
  **detached** posterior states of the replayed batch, rolls the prior for
  `H = 15` steps with actor-chosen actions, world-model **weights frozen**
  (the graph stays differentiable w.r.t. actions for the continuous branch;
  the discrete branch runs dynamics under `no_grad` entirely).
- **Lambda-returns** (`train/lambda_returns.py`): TD(lambda) with
  `lambda = 0.95`, `gamma = 0.99`, discount `gamma * cont_prob` from the
  continue head — an imagined episode end cuts future rewards and
  down-weights later loss terms (cumulative-product weights).
- **Losses** (`models/losses.py`): critic regresses toward `sg(R^lambda)`;
  actor = REINFORCE with a no-grad critic baseline (discrete) or direct
  return maximization (continuous), entropy bonus `3e-4`, V3 return-range
  normalization (`models/return_normalizer.py`, ablation switch).
- **Full loop** (`train/dreamer_loop.py`): random prefill -> collect with
  the actor (online posterior belief tracking) while performing
  `train_ratio` gradient updates per env step (fractional accumulator);
  three separate Adam optimizers (WM `3e-4`, actor `4e-5`, critic `1e-4`).

```bash
# From scratch (prefill 5k random steps, then 100k env steps):
python train/dreamer_loop.py

# Warm-started from Phase 1 (recommended on a small GPU):
python train/dreamer_loop.py \
    train_dreamer.buffer_dir=experiments/data/train_50k/buffer \
    train_dreamer.init_wm_ckpt=experiments/wm_final_twohot/checkpoints/wm_final.pt

# Dream-vs-real validation plot from the run's metrics.jsonl:
python viz/dream_vs_real.py viz.run_dir=experiments/<run_dir>
```

Gradient-separation invariants (unit-tested in `tests/test_losses.py` and
`tests/test_imagine_rollout.py`): a full actor-critic step leaves every
world-model parameter without gradient; the actor loss never trains the
critic (baseline under `no_grad`, detached advantage); the critic loss never
reaches the actor (detached input features, detached targets).

## Model-free baselines & benchmark (Phase 3)

Single-file adaptations of CleanRL's reference implementations (no CleanRL
dependency — the point of CleanRL is that its files are self-contained),
wired to the exact Phase 0 wrapper chain and a shared measurement protocol:

- `baselines/ppo.py` — PPO (GAE, clipped surrogate; categorical or
  diagonal-Normal head by action space; Nature CNN or MLP trunk by obs space).
- `baselines/dqn_rainbow.py` — DQN + double targets + dueling heads.
  **Not** full Rainbow: no prioritized replay, noisy nets, C51 or n-step.
- `baselines/sac.py` — SAC with twin Q, EMA targets, entropy auto-tuning;
  pixel branch trains a CNN encoder with the Q loss only and feeds the
  actor detached features (SAC-AE/DrQ convention).

Hyperparameters are CleanRL defaults, deliberately untuned, except replay
capacity / learning-starts shrunk to the benchmark budget (documented in
`configs/baselines/default.yaml`).

### Measurement protocol

**The x-axis of every sample-efficiency comparison is environment
interactions (env steps, counted after action repeat) — not gradient steps
and not wall-clock time.** All agents step the same Phase 0 wrapper chain,
count steps identically (including random prefill/warm-up), report raw
(unclipped) episode returns through the same CSV logger
(`train/common_logger.py`), and every curve also records wall-clock time.

Dreamer typically wins sample-efficiency (fewer env steps to a given reward
level) because it performs `train_ratio` gradient updates per env step on
imagined data; model-free baselines typically win wall-clock time (less
real time per env step) and/or the asymptote (higher final reward with an
unconstrained step budget). **That difference is what this benchmark is
designed to show — it is not something to hide or average away.** The
reward-vs-wall-clock plot exists precisely to show the other side of the
trade-off.

Deliberate protocol deviations (all documented, none silent):

1. Feedforward baselines get **grayscale + 4-frame channel stack** (the
   conventional model-free Atari input — a memoryless policy cannot infer
   velocity from one frame); Dreamer keeps single RGB frames because the
   RSSM builds temporal state. Same emulator, action repeat, time limit
   and rewards (regression-tested in `tests/test_baseline_protocol.py`).
2. **No reward clipping anywhere** (CleanRL's Atari scripts clip to
   [-1, 1]; Pong rewards already are, so nothing changes on Pong, and the
   plots stay in raw-return units everywhere).
3. No NoopReset / FireReset / EpisodicLife wrappers for anyone.
4. DQN/SAC bootstrap masks use `terminated` only (truncation is not death),
   consistent with the Dreamer continue head.
5. Replay capacity and learning-starts for DQN/SAC shrunk from CleanRL's
   1M-step defaults to the benchmark budget.

```bash
# Everything on one env, shared budget, then comparison plots:
python train/run_benchmark.py benchmark.total_env_steps=100000
python train/run_benchmark.py env=carracing "benchmark.agents=[dreamer,ppo,sac]"

# Individual agents:
python baselines/ppo.py baselines.total_env_steps=100000
python baselines/dqn_rainbow.py baselines.total_env_steps=100000
python baselines/sac.py env=carracing baselines.total_env_steps=100000
python viz/benchmark_comparison.py            # plots from experiments/benchmark
```

### Benchmark results (Pong, shortened — single seed)

Shared budget 100k env steps (post action-repeat), CleanRL default
hyperparameters, 1 seed (methodologically weak — a real experiment needs
>= 3 seeds; see the suggested budget below). Plots:
`experiments/benchmark/plots/ALE_Pong-v5_{env_steps,wall_clock}.png`.

| agent | return @ 100k env steps | wall-clock |
|---|---|---|
| PPO | ~-21 .. -18, no sustained improvement | **2.7 min** |
| DQN (double+dueling) | -21 .. -20, no improvement | **9.2 min** |
| Dreamer (warm-start, x-shifted +50k) | **-1.7** (avg last 10) | 249 min (+98 min WM pretrain) |
| Dreamer (from scratch) | interrupted at 23k steps, still ~-21 | (partial) |

The expected pattern shows up clearly even in these shortened runs:
model-free agents burned the whole sample budget without leaving the
random floor (Pong typically needs ~1M+ frames for DQN/PPO) while using
25-90x less wall-clock time; Dreamer converted the same number of
interactions into near-parity play at ~25x the compute per step. The
from-scratch Dreamer curve was interrupted early (23k steps: world model
already at recon 2.5e-4/px, policy still at the floor, entropy notably
lower than in the warm-started run — worth watching for premature entropy
collapse when training from scratch on sparse rewards).

Suggested full experiment (outside this sandbox): 3+ seeds x 400k env
steps for the model-free agents (DQN/PPO reach non-trivial Pong play
around 1-2M frames = 250-500k post-repeat steps), 3 seeds x 100-150k steps
for scratch Dreamer, identical protocol — roughly one GPU-day on a single
A100-class card.

## Tests

```bash
pytest tests/
```

Coverage: wrappers (observation shape/range/dtype, `raw_obs` in info, exact normalization,
action repeat with reward summing and early stop, time limit) and the replay buffer
(add/validation, sample shapes, normalization, padding + mask for short episodes,
FIFO eviction, save/load) + an end-to-end integration test on `ALE/Pong-v5`.
Phase 1 adds: pipeline shapes for both latent types, mid-sequence `is_first`
state reset, straight-through gradient flow to the encoder, closed-form KL
values + free-nats floor + balancing stop-gradient direction, padding-mask
correctness, and the continue-target/truncation distinction.

## Environments

- **Atari (ALE)** — the main target, works out-of-the-box (`ale-py` >= 0.10 bundles the ROMs).
  ALE's built-in frameskip and sticky actions are disabled in the config (`frameskip=1`,
  `repeat_action_probability=0.0`) — our wrapper controls action repeat instead.
- **dm_control** — the adapter in [envs/dmc.py](envs/dmc.py) (gymnasium API + pixel rendering) is
  implemented but **untested in this environment** — `dm_control`/MuJoCo is not installed
  by default. Requires a local `pip install -e .[dmc]`; sample config:
  `configs/env/dmc_walker_walk.yaml` (`env=dmc_walker_walk`).

## Logging

TensorBoard by default (no API keys / internet needed). The logger interface in
[train/logger.py](train/logger.py) is deliberately minimal (`log_scalar`/`close`) — wiring up
wandb means adding a `WandbLogger` in one place and setting `logger.backend=wandb`.

## Roadmap

- **Phase 0**: data pipeline — DONE.
- **Phase 1**: world model (encoder/decoder, RSSM, heads, isolated training,
  reconstruction + open-loop validation) — DONE; validation numbers in
  [Phase 1 results](#phase-1-results).
- **Phase 2 (next step)**: actor/critic trained in imagination on the prior
  (no encoder), using `RSSM.imagine()` with a policy callable; then ablations
  (categorical vs gaussian latents, KL balancing, free nats).

## Phase 1 results

Reference setup: `ALE/Pong-v5`, 50k random-policy env steps (train) + 5k
held-out (val, different seed), categorical latents, batch 16 x seq 50,
Adam 3e-4, grad clip 100, on a GTX 1660 Ti (~2.2 updates/s). Base run:
10k updates (75 min) with the MSE reward head, then 3k warm-started updates
(23 min) with the `symlog_twohot` head (now the default config).

1. **Reconstruction**: per-pixel MSE fell monotonically 0.436 -> 0.00038
   (1k updates) -> 0.00015 (10k) -> **0.00005** after the two-hot fine-tune
   (val). Reconstructions track both paddles and the score area; the 1-px
   ball and digit details stay blurry (`experiments/phase1_final/reconstruction_*.png`).
2. **Reward**: held-out Pearson r between predicted and true reward =
   **0.75** (two-hot head, 3k updates; still rising). Diagnosis worth noting:
   with the plain MSE head at loss scale 1.0 the correlation stayed at ~0 and
   a probe on frozen features scored AUC 0.49 — the pixel-summed
   reconstruction gradient dominated and the representation never encoded
   scoring events. A `loss_scales.reward=100` fine-tune reached r = 0.91,
   confirming the signal exists; the two-hot head fixes it at scale 1.0.
3. **Open-loop rollout** (5 posterior warm-up steps, then 15 prior-only
   steps replaying true actions): per-step MSE stays flat, 0.00006 (step 1)
   -> 0.00011 (step 15) — no degeneration into noise; both paddles and the
   playfield remain visually coherent over the horizon
   (`experiments/phase1_final/open_loop_*.png|gif`).
4. **KL does not collapse**: raw KL(post||prior) stabilized at 0.4–1.0 nats
   during the base run and rose to **1.3–1.7 nats** (above free_nats = 1.0)
   once the reward signal forced more information through the latent.

Known limitations: the 1-px Pong ball is below what the decoder reproduces
reliably at 64x64 with this budget — expect imagination rollouts to fuzz the
ball; more training or a larger `cnn_depth` sharpens it. The gaussian latent
variant is smoke-tested (loss decreases, all unit tests pass) but has no full
reference run.

## Phase 2 results

Reference run: `ALE/Pong-v5`, world model warm-started from the Phase 1
checkpoint, buffer preloaded with the same 50k random steps (so total
unique env interactions = 50k random + 60k collected by the actor), train
ratio 0.3 (one joint WM+AC update per ~3.3 env steps), horizon 15,
gamma 0.99, lambda 0.95, entropy coef 3e-4, return normalization on,
REINFORCE actor (discrete). 60k env steps, 18k updates, 249 min on the
GTX 1660 Ti (~1.2 joint updates/s), single seed.

1. **Real reward rises**: first actor episodes scored **-21.0** (the
   random-policy floor); after ~3.5k env steps the 10-episode average
   reached ~-6, and the final 10 episodes averaged **-1.7** with a best
   episode of **+1.0** — near-parity Pong within 1000-step (time-limited)
   episodes after 60k interactions.
2. **Imagined returns track real returns**: Pearson r between the mean
   imagined lambda-return at rollout start and the rolling real return =
   **0.96** over 180 logged checkpoints; both curves turn upward together
   (`experiments/dreamer_pong/dream_vs_real.png`). Imagination stays
   mildly optimistic (~+0.004/step imagined vs ~-0.001/step real at the
   end) — no runaway world-model exploitation observed.
3. **No gradient leaks**: unit tests assert every world-model parameter
   has zero/None grad after a full posterior -> rollout -> actor+critic
   backward step (both action types), and that the actor and critic never
   cross-train each other.
4. **Critic is stable**: critic loss fell 5.2 -> ~0.65-0.70 and plateaued
   with no divergent oscillation; the EMA target (tau 0.98) tracks the
   online critic with the expected lag. Policy entropy declined gently
   0.78 -> ~0.52 (floor 0.49, max ln 6 = 1.79) — no collapse. KL stayed
   at 1.2-1.6 nats throughout, so the world model kept learning while
   the policy trained on it.

Working hyperparameters (deviations from spec defaults: none — H=15,
gamma=0.99, lambda=0.95 as specified): train_ratio 0.3, WM lr 3e-4, actor
lr 4e-5, critic lr 1e-4, entropy 3e-4 + unimix 1%, epsilon-greedy off.
