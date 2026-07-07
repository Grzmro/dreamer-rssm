"""Open-loop consistency check: posterior warm-up, then prior-only rollout.

The key Phase 1 validation: after k warm-up steps in posterior mode (seeing
observations), the RSSM continues with ``imagine()`` — prior only, no
encoder — for ~15 steps, replaying the true action sequence from data. The
decoded prior latents are compared against the real frames. A healthy world
model stays visually coherent (objects/background do not dissolve into
noise) over the horizon.

Usage:
    python viz/open_loop_rollout.py viz.ckpt=<run>/checkpoints/wm_final.pt \
        viz.buffer_dir=<run>/buffer
"""

from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from train.train_world_model import load_world_model, resolve_device
from viz.wm_viz_utils import load_buffer_for_viz, save_frame_rows, save_gif, to_image


@torch.no_grad()
def run(cfg: DictConfig) -> None:
    if not cfg.viz.ckpt:
        raise ValueError("viz.ckpt is required (path to a world-model checkpoint)")
    device = resolve_device(cfg.train_wm.device)
    wm, _ = load_world_model(cfg.viz.ckpt, device)
    buffer = load_buffer_for_viz(cfg)
    out_dir = Path(cfg.viz.wm_out_dir)

    warmup = int(cfg.viz.warmup)
    horizon = int(cfg.viz.rollout_horizon)
    total = warmup + horizon

    batch = buffer.sample(batch_size=4, seq_len=total, device=device)
    action = wm.prepare_action(batch["action"])

    # Posterior warm-up on the first `warmup` steps (sees observations).
    B = batch["obs"].shape[0]
    embed = wm.encoder(batch["obs"][:, :warmup].flatten(0, 1)).unflatten(0, (B, warmup))
    obs_out = wm.rssm.observe(embed, action[:, :warmup], batch["is_first"][:, :warmup])

    # Prior-only rollout replaying the true actions (no encoder from here on).
    # Action layout: action[t] leads INTO obs[t], so step t of the rollout
    # (predicting obs[warmup + t]) uses action[warmup + t].
    img = wm.rssm.imagine(
        obs_out["h"][:, -1], obs_out["z"][:, -1], action[:, warmup:total]
    )

    post_feat = wm.features(obs_out["h"], obs_out["z"])
    prior_feat = wm.features(img["h"], img["z"])
    post_recon = wm.decoder(post_feat.flatten(0, 1)).unflatten(0, (B, warmup))
    prior_recon = wm.decoder(prior_feat.flatten(0, 1)).unflatten(0, (B, horizon))
    pred = torch.cat([post_recon, prior_recon], dim=1)  # [B, total, C, H, W]

    # Numerical consistency: per-step MSE of the prior segment vs ground truth.
    prior_mse = (prior_recon - batch["obs"][:, warmup:total]).pow(2).mean(
        dim=(0, 2, 3, 4)
    )
    warmup_mse = (post_recon - batch["obs"][:, :warmup]).pow(2).mean().item()
    print(f"[open-loop] posterior warm-up mse/px: {warmup_mse:.5f}")
    print("[open-loop] prior rollout mse/px per step:")
    for t, v in enumerate(prior_mse.tolist()):
        print(f"    step {t + 1:2d}: {v:.5f}")
    print(f"[open-loop] prior rollout mean mse/px: {prior_mse.mean().item():.5f}")

    col_labels = [
        f"t={t}" + (" (post)" if t < warmup else " (PRIOR)") for t in range(total)
    ]
    for b in range(B):
        gt_row = [to_image(batch["obs"][b, t]) for t in range(total)]
        pred_row = [to_image(pred[b, t]) for t in range(total)]
        save_frame_rows(
            [gt_row, pred_row],
            out_dir / f"open_loop_{b}.png",
            row_labels=["ground truth", "model"],
            col_labels=col_labels,
            title=(
                f"Open-loop rollout (seq {b}): {warmup} posterior warm-up steps, "
                f"then {horizon} prior-only steps"
            ),
        )
        gif_frames = [
            np.concatenate([to_image(batch["obs"][b, t]), to_image(pred[b, t])], axis=1)
            for t in range(total)
        ]
        save_gif(gif_frames, out_dir / f"open_loop_{b}.gif", fps=5)

    print(f"[open-loop] PNGs + GIFs (GT | model) written to {out_dir.resolve()}")


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
