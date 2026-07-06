# dreamer-rssm

A from-scratch **DreamerV2/V3**-style world-model RL implementation (simplified, with ablations), built in phases.

**Status: Phase 0 — data infrastructure.** This commit contains only the data pipeline:
environments → wrappers → sequential replay buffer → batch sampling `[B, L, C, H, W]`.
**There are no neural networks here** — the RSSM, encoder/decoder, heads, actor, and critic
land in Phase 1+ (see [Roadmap](#roadmap)).

## Repo structure

```
envs/          # environment wrappers (64x64 resize, grayscale, action repeat, time limit, normalization)
data/          # sequential replay buffer (whole episodes, uint8, FIFO eviction)
train/         # random-policy data collection + logger (TensorBoard)
viz/           # sanity checks (histograms, frame grid)
configs/       # Hydra configs (groups: env, buffer, collect)
models/        # EMPTY — Phase 1+ (RSSM, encoder/decoder, heads)
agents/        # EMPTY — Phase 1+ (actor/critic in imagination)
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

## Tests

```bash
pytest tests/
```

Coverage: wrappers (observation shape/range/dtype, `raw_obs` in info, exact normalization,
action repeat with reward summing and early stop, time limit) and the replay buffer
(add/validation, sample shapes, normalization, padding + mask for short episodes,
FIFO eviction, save/load) + an end-to-end integration test on `ALE/Pong-v5`.

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

- **Phase 0 (this commit)**: data pipeline — DONE.
- **Phase 1 (next step)**: world model — CNN encoder, RSSM (deterministic core GRU,
  prior `p(z|h)`, posterior `q(z|h,e)`), decoder, reward/continue heads; trained on batches
  `[B, L, ...]` from this buffer (padding mask already in place). Suggested first step: encoder +
  RSSM trained on reconstruction over the posterior, with prior/posterior KL metrics.
- **Phase 2**: actor/critic trained in imagination on the prior (no encoder), ablations.
