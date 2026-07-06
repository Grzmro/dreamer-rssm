"""Sanity checks over a filled replay buffer.

Produces PNGs in ``viz.out_dir`` (default: experiments/sanity/):
  - histogram of episode returns,
  - histogram of episode lengths,
  - a grid of frames from one sampled sequence (visual check of
    resize / normalization / action repeat),
and prints per-step reward stats plus the pixel value range of a batch.

Usage:
    python viz/sanity_checks.py                      # collects fresh data first
    python viz/sanity_checks.py viz.buffer_dir=experiments/runs/<ts>/buffer
"""

from __future__ import annotations

from pathlib import Path

import hydra
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig

from data.replay_buffer import SequenceReplayBuffer


def _get_buffer(cfg: DictConfig, out_dir: Path) -> SequenceReplayBuffer:
    if cfg.viz.buffer_dir:
        print(f"[sanity] loading buffer from {cfg.viz.buffer_dir}")
        return SequenceReplayBuffer.load(
            cfg.viz.buffer_dir, capacity=cfg.buffer.capacity, seed=cfg.seed
        )
    print(f"[sanity] no buffer_dir given -> collecting {cfg.viz.collect_steps} fresh steps")
    from train.collect import collect_random_data

    cfg.collect.num_steps = int(cfg.viz.collect_steps)
    cfg.collect.save_buffer = False
    return collect_random_data(cfg, output_dir=out_dir)


def run_sanity_checks(cfg: DictConfig) -> None:
    out_dir = Path(cfg.viz.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    buffer = _get_buffer(cfg, out_dir)

    episodes = list(buffer._episodes)  # read-only introspection for stats
    returns = np.array([ep["reward"].sum() for ep in episodes])
    lengths = np.array([len(ep["obs"]) for ep in episodes])
    all_rewards = np.concatenate([ep["reward"] for ep in episodes])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(returns, bins=30, color="steelblue", edgecolor="black")
    ax.set(title="Episode returns", xlabel="return", ylabel="count")
    fig.tight_layout()
    fig.savefig(out_dir / "episode_returns_hist.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(lengths, bins=30, color="darkorange", edgecolor="black")
    ax.set(title="Episode lengths (steps)", xlabel="length", ylabel="count")
    fig.tight_layout()
    fig.savefig(out_dir / "episode_lengths_hist.png", dpi=120)
    plt.close(fig)

    # Frame grid from one sampled sequence.
    seq_len = int(cfg.buffer.seq_len)
    batch = buffer.sample(batch_size=1, seq_len=seq_len)
    obs = batch["obs"][0].numpy()  # [L, C, H, W], float32 in [-0.5, 0.5]
    num_frames = min(int(cfg.viz.num_frames), seq_len)
    frame_idx = np.linspace(0, seq_len - 1, num_frames, dtype=int)

    fig, axes = plt.subplots(1, num_frames, figsize=(2 * num_frames, 2.4))
    for ax, t in zip(np.atleast_1d(axes), frame_idx):
        frame = obs[t].transpose(1, 2, 0) + 0.5  # denormalize to [0, 1] for display
        ax.imshow(frame.squeeze(), cmap="gray" if frame.shape[-1] == 1 else None)
        ax.set_title(f"t={t}", fontsize=8)
        ax.axis("off")
    fig.suptitle("Sampled sequence frames (after resize + action repeat)")
    fig.tight_layout()
    fig.savefig(out_dir / "sample_sequence_frames.png", dpi=120)
    plt.close(fig)

    # Pixel range check on a full batch.
    big_batch = buffer.sample(
        batch_size=min(int(cfg.buffer.batch_size), buffer.num_episodes), seq_len=seq_len
    )
    px_min, px_max = big_batch["obs"].min().item(), big_batch["obs"].max().item()

    print("\n[sanity] === buffer statistics ===")
    print(f"  episodes: {buffer.num_episodes}, total steps: {buffer.num_steps}")
    print(f"  episode return:  min={returns.min():.2f} max={returns.max():.2f} mean={returns.mean():.2f}")
    print(f"  episode length:  min={lengths.min()} max={lengths.max()} mean={lengths.mean():.1f}")
    print(f"  per-step reward: min={all_rewards.min():.3f} max={all_rewards.max():.3f} mean={all_rewards.mean():.4f}")
    print(f"  batch obs shape: {tuple(big_batch['obs'].shape)}")
    print(f"  batch pixel range: [{px_min:.3f}, {px_max:.3f}] (expected within [-0.5, 0.5])")
    assert -0.5 <= px_min and px_max <= 0.5, "pixel values outside [-0.5, 0.5]!"
    print(f"  PNGs written to {out_dir.resolve()}")


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    run_sanity_checks(cfg)


if __name__ == "__main__":
    main()
