"""World-model training in isolation on random-policy data (Phase 1).

No actor/critic, no imagination-based policy learning — the prior is only
rolled out for open-loop validation (viz/open_loop_rollout.py).

Usage:
    python train/train_world_model.py
    python train/train_world_model.py train_wm.buffer_dir=experiments/runs/<ts>/buffer
    python train/train_world_model.py model=rssm_gaussian train_wm.num_updates=2000
"""

from __future__ import annotations

import time
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from data.replay_buffer import SequenceReplayBuffer
from train.logger import make_logger


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def get_buffer(cfg: DictConfig, output_dir: Path) -> SequenceReplayBuffer:
    """Load a saved Phase 0 buffer, or collect fresh random-policy data."""
    if cfg.train_wm.buffer_dir:
        print(f"[train_wm] loading buffer from {cfg.train_wm.buffer_dir}")
        return SequenceReplayBuffer.load(
            cfg.train_wm.buffer_dir, capacity=cfg.buffer.capacity, seed=cfg.seed
        )
    print(f"[train_wm] collecting {cfg.train_wm.collect_steps} random-policy steps")
    from train.collect import collect_random_data

    collect_cfg = OmegaConf.merge(cfg, {})  # do not mutate the caller's cfg
    collect_cfg.collect.num_steps = int(cfg.train_wm.collect_steps)
    collect_cfg.collect.save_buffer = True
    return collect_random_data(collect_cfg, output_dir=output_dir)


def build_world_model(cfg: DictConfig, device: torch.device):
    """Construct a WorldModel matching the configured environment's spaces."""
    from envs import make_env
    from models.world_model import WorldModel

    env = make_env(cfg.env)
    obs_channels = env.observation_space.shape[0]
    discrete = hasattr(env.action_space, "n")
    action_dim = env.action_space.n if discrete else env.action_space.shape[0]
    env.close()
    return WorldModel(obs_channels, action_dim, discrete, cfg.model).to(device)


@torch.no_grad()
def validate_reward_correlation(
    wm, buffer: SequenceReplayBuffer, cfg: DictConfig, device
) -> tuple[float, float]:
    """Pearson correlation between predicted and true rewards.

    ``buffer`` should be a held-out validation buffer when available
    (train_wm.val_buffer_dir); it falls back to fresh windows from the
    training buffer otherwise. Returns (pearson_r, recon_mse_per_pixel).
    """
    wm.eval()
    preds, trues, recon_mses = [], [], []
    for _ in range(int(cfg.train_wm.val_batches)):
        batch = buffer.sample(cfg.buffer.batch_size, cfg.buffer.seq_len, device=device)
        out = wm(batch)
        m = batch["mask"].bool()
        preds.append(out["reward_pred"][m].cpu())
        trues.append(batch["reward"][m].cpu())
        per_px = ((out["recon"] - batch["obs"]).pow(2).mean(dim=(-3, -2, -1)))[m]
        recon_mses.append(per_px.cpu())
    wm.train()

    pred = torch.cat(preds).numpy()
    true = torch.cat(trues).numpy()
    recon_mse = torch.cat(recon_mses).mean().item()
    if np.std(true) < 1e-8 or np.std(pred) < 1e-8:
        return float("nan"), recon_mse
    return float(np.corrcoef(pred, true)[0, 1]), recon_mse


def train_world_model(cfg: DictConfig, output_dir: str | Path | None = None):
    if output_dir is None:
        output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    output_dir = Path(output_dir)
    device = resolve_device(cfg.train_wm.device)
    torch.manual_seed(cfg.seed)

    buffer = get_buffer(cfg, output_dir)
    if cfg.train_wm.val_buffer_dir:
        print(f"[train_wm] loading VAL buffer from {cfg.train_wm.val_buffer_dir}")
        val_buffer = SequenceReplayBuffer.load(
            cfg.train_wm.val_buffer_dir, capacity=cfg.buffer.capacity, seed=cfg.seed + 1
        )
    else:
        val_buffer = buffer  # fallback: fresh windows from the training buffer
    wm = build_world_model(cfg, device)
    if cfg.train_wm.init_ckpt:
        print(f"[train_wm] initializing weights from {cfg.train_wm.init_ckpt}")
        ckpt = torch.load(cfg.train_wm.init_ckpt, map_location=device, weights_only=False)
        wm.load_state_dict(ckpt["model_state"])
    n_params = sum(p.numel() for p in wm.parameters())
    print(f"[train_wm] device={device} params={n_params / 1e6:.1f}M "
          f"latent={cfg.model.latent_type} buffer={buffer.num_steps} steps")

    optim = torch.optim.Adam(wm.parameters(), lr=cfg.train_wm.lr, eps=cfg.train_wm.adam_eps)
    logger = make_logger(cfg.logger.backend, output_dir / "tb")
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    num_updates = int(cfg.train_wm.num_updates)
    log_interval = int(cfg.train_wm.log_interval)
    start = time.time()

    for step in range(1, num_updates + 1):
        batch = buffer.sample(cfg.buffer.batch_size, cfg.buffer.seq_len, device=device)
        loss, metrics = wm.loss(batch)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(wm.parameters(), cfg.train_wm.grad_clip)
        optim.step()

        if step % log_interval == 0 or step == 1:
            for tag, value in metrics.items():
                logger.log_scalar(tag, value.item(), step)
            logger.log_scalar("train/grad_norm", grad_norm.item(), step)
            ups = step / (time.time() - start)
            print(
                f"[train_wm] {step:6d}/{num_updates} | loss {metrics['loss/total']:9.2f} | "
                f"recon/px {metrics['recon/mse_per_pixel']:.5f} | "
                f"kl {metrics['kl/value']:6.2f} | reward {metrics['loss/reward']:.4f} | "
                f"grad {grad_norm.item():7.1f} | {ups:.2f} up/s"
            )

        if step % int(cfg.train_wm.val_interval) == 0:
            r, recon_mse = validate_reward_correlation(wm, val_buffer, cfg, device)
            logger.log_scalar("val/reward_pearson_r", r, step)
            logger.log_scalar("val/recon_mse_per_pixel", recon_mse, step)
            print(f"[train_wm] {step:6d} VAL | reward pearson r = {r:.4f} | "
                  f"recon mse/px = {recon_mse:.5f}")

        if step % int(cfg.train_wm.checkpoint_interval) == 0:
            _save_checkpoint(wm, cfg, ckpt_dir / f"wm_{step:06d}.pt")

    final_path = ckpt_dir / "wm_final.pt"
    _save_checkpoint(wm, cfg, final_path)
    r, recon_mse = validate_reward_correlation(wm, val_buffer, cfg, device)
    logger.log_scalar("val/reward_pearson_r", r, num_updates)
    logger.close()
    print(f"[train_wm] done in {(time.time() - start) / 60:.1f} min | "
          f"final reward pearson r = {r:.4f} | recon mse/px = {recon_mse:.5f}")
    print(f"[train_wm] final checkpoint: {final_path}")
    return wm, buffer, final_path


def _save_checkpoint(wm, cfg: DictConfig, path: Path) -> None:
    torch.save(
        {
            "model_state": wm.state_dict(),
            "cfg": OmegaConf.to_container(cfg, resolve=True),
            "obs_channels": wm.encoder.net[0].in_channels,
            "action_dim": wm.action_dim,
            "discrete_actions": wm.discrete_actions,
        },
        path,
    )


def load_world_model(ckpt_path: str | Path, device) -> tuple[torch.nn.Module, DictConfig]:
    """Rebuild a WorldModel from a checkpoint saved by this script."""
    from models.world_model import WorldModel

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = OmegaConf.create(ckpt["cfg"])
    wm = WorldModel(
        ckpt["obs_channels"], ckpt["action_dim"], ckpt["discrete_actions"], cfg.model
    ).to(device)
    wm.load_state_dict(ckpt["model_state"])
    wm.eval()
    return wm, cfg


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    train_world_model(cfg)


if __name__ == "__main__":
    main()
