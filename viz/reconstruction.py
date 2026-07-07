"""Side-by-side ground-truth vs posterior reconstructions.

Usage:
    python viz/reconstruction.py viz.ckpt=<run>/checkpoints/wm_final.pt \
        viz.buffer_dir=<run>/buffer
"""

from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from train.train_world_model import load_world_model, resolve_device
from viz.wm_viz_utils import load_buffer_for_viz, save_frame_rows, to_image


@torch.no_grad()
def run(cfg: DictConfig) -> None:
    if not cfg.viz.ckpt:
        raise ValueError("viz.ckpt is required (path to a world-model checkpoint)")
    device = resolve_device(cfg.train_wm.device)
    wm, _ = load_world_model(cfg.viz.ckpt, device)
    buffer = load_buffer_for_viz(cfg)
    out_dir = Path(cfg.viz.wm_out_dir)

    batch = buffer.sample(batch_size=4, seq_len=cfg.buffer.seq_len, device=device)
    out = wm(batch)

    num_frames = int(cfg.viz.recon_frames)
    idx = np.linspace(0, cfg.buffer.seq_len - 1, num_frames, dtype=int)
    mses = []
    for b in range(batch["obs"].shape[0]):
        gt_row = [to_image(batch["obs"][b, t]) for t in idx]
        rec_row = [to_image(out["recon"][b, t]) for t in idx]
        mse = (out["recon"][b] - batch["obs"][b]).pow(2).mean().item()
        mses.append(mse)
        save_frame_rows(
            [gt_row, rec_row],
            out_dir / f"reconstruction_{b}.png",
            row_labels=["ground truth", "reconstruction"],
            col_labels=[f"t={t}" for t in idx],
            title=f"Posterior reconstruction (seq {b}, mse/px {mse:.5f})",
        )

    print(f"[recon] per-pixel MSE over {len(mses)} sequences: "
          f"mean {np.mean(mses):.5f}, min {min(mses):.5f}, max {max(mses):.5f}")
    print(f"[recon] PNGs written to {out_dir.resolve()}")


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
