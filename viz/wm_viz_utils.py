"""Shared helpers for world-model visualizations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data.replay_buffer import SequenceReplayBuffer


def to_image(frame: torch.Tensor) -> np.ndarray:
    """[C, H, W] in [-0.5, 0.5] -> HWC uint8 for display."""
    img = (frame.detach().cpu().float() + 0.5).clamp(0, 1)
    img = (img * 255).byte().permute(1, 2, 0).numpy()
    return img.repeat(3, axis=2) if img.shape[2] == 1 else img


def save_frame_rows(
    rows: list[list[np.ndarray]],
    path: Path,
    row_labels: list[str] | None = None,
    col_labels: list[str] | None = None,
    title: str | None = None,
) -> None:
    """Save a labelled grid of HWC uint8 frames as a PNG via matplotlib."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_rows, n_cols = len(rows), len(rows[0])
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(1.4 * n_cols, 1.6 * n_rows), squeeze=False
    )
    for r, row in enumerate(rows):
        for c, frame in enumerate(row):
            ax = axes[r][c]
            ax.imshow(frame)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0 and col_labels:
                ax.set_title(col_labels[c], fontsize=7)
            if c == 0 and row_labels:
                ax.set_ylabel(row_labels[r], fontsize=8)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_gif(frames: list[np.ndarray], path: Path, fps: int = 5, scale: int = 3) -> None:
    """Save HWC uint8 frames as an animated GIF (nearest-neighbor upscaled)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    images = [
        Image.fromarray(f).resize((f.shape[1] * scale, f.shape[0] * scale), Image.NEAREST)
        for f in frames
    ]
    images[0].save(
        path, save_all=True, append_images=images[1:], duration=1000 // fps, loop=0
    )


def load_buffer_for_viz(cfg) -> SequenceReplayBuffer:
    if not cfg.viz.buffer_dir:
        raise ValueError(
            "viz.buffer_dir is required (point it at a saved buffer, e.g. "
            "experiments/runs/<ts>/buffer)"
        )
    return SequenceReplayBuffer.load(
        cfg.viz.buffer_dir, capacity=cfg.buffer.capacity, seed=cfg.seed
    )
