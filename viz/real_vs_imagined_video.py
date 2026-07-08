"""Side-by-side video: real episode continuation vs imagination branch.

The flagship Phase 4 artifact. Runs ONE real episode with the trained
agent (inference only — no gradient updates), recording frames and the
posterior state (h_t, z_t) at every step. At each requested branch point
it rolls the actor through the PRIOR for ``viz.video_horizon`` steps,
decodes the imagined states, and renders a two-panel video:

    left  = what actually happened next in the environment,
    right = what the world model dreamed from the same state.

Both panels show the same real frames for ``viz.video_context`` steps
before the branch; from the branch on, the right panel gets a red border
("the model starts dreaming here") and the first dream frame is repeated
for a short visual pause. Outputs GIF + MP4 per branch point into
``viz.video_out_dir``.

Usage:
    python viz/real_vs_imagined_video.py viz.video_ckpt=experiments/dreamer_pong/checkpoints/dreamer_final.pt
    python viz/real_vs_imagined_video.py viz.video_ckpt=... "viz.video_branch_points=[30,150,400]" viz.video_horizon=30
"""

from __future__ import annotations

from pathlib import Path

import cv2
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from envs import make_env
from models.actor import Actor
from models.world_model import WorldModel
from train.dreamer_loop import OnlinePolicy
from train.imagine_rollout import imagine_rollout


def load_dreamer(ckpt_path: str | Path, device):
    """Rebuild (world model, actor) from a dreamer_loop checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = OmegaConf.create(ckpt["cfg"])
    wm = WorldModel(
        ckpt["obs_channels"], ckpt["action_dim"], ckpt["discrete_actions"], cfg.model
    ).to(device)
    wm.load_state_dict(ckpt["model_state"])
    wm.eval()
    a = cfg.agent.actor
    action_type = "discrete" if ckpt["discrete_actions"] else "continuous"
    actor = Actor(
        wm.rssm.feat_dim, ckpt["action_dim"], action_type,
        hidden_dim=a.hidden_dim, num_layers=a.num_layers,
        unimix=a.unimix, min_std=a.min_std, init_std=a.init_std,
    ).to(device)
    actor.load_state_dict(ckpt["actor_state"])
    actor.eval()
    return wm, actor, cfg


@torch.no_grad()
def run_real_episode(env, wm, actor, device, max_steps: int):
    """One inference-only episode; returns frames [T,C,H,W] and states."""
    policy = OnlinePolicy(wm, actor, epsilon=0.0, env=env, device=device)
    obs, _ = env.reset()
    frames, hs, zs = [], [], []
    for _ in range(max_steps):
        action = policy(obs)  # updates policy.state to the posterior of obs
        frames.append(np.asarray(obs, dtype=np.float32))
        h, z = policy.state
        hs.append(h[0].clone())
        zs.append(z[0].clone())
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    return np.stack(frames), torch.stack(hs), torch.stack(zs)


@torch.no_grad()
def dream_from(wm, actor, h: torch.Tensor, z: torch.Tensor, horizon: int) -> np.ndarray:
    """Prior+actor rollout decoded to frames [K, C, H, W] (numpy)."""
    rollout = imagine_rollout(wm, actor, h[None], z[None], horizon=horizon)
    feat = rollout["feat"][0, 1:]  # imagined states s_1..s_K
    return wm.decoder(feat).cpu().numpy()


def to_uint8(chw: np.ndarray) -> np.ndarray:
    hwc = np.clip(chw + 0.5, 0.0, 1.0).transpose(1, 2, 0)
    img = (hwc * 255).astype(np.uint8)
    return img if img.shape[-1] == 3 else np.repeat(img, 3, axis=-1)


def compose_panels(
    real: list[np.ndarray], dream: list[np.ndarray], branch_at: int,
    scale: int = 4, pad: int = 4,
) -> list[np.ndarray]:
    """uint8 HWC frames -> list of side-by-side panels with labels/marker."""
    out = []
    for t, (l, r) in enumerate(zip(real, dream)):
        l = cv2.resize(l, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        r = cv2.resize(r, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        if t >= branch_at:  # red border: the model is dreaming
            r = cv2.copyMakeBorder(r, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=(255, 40, 40))
            r = cv2.resize(r, (l.shape[1], l.shape[0]))
        panel = cv2.copyMakeBorder(
            np.hstack([l, np.full((l.shape[0], pad, 3), 255, np.uint8), r]),
            22, 4, 4, 4, cv2.BORDER_CONSTANT, value=(0, 0, 0),
        )
        label = f"REAL      t={t}      {'DREAM' if t >= branch_at else 'real (pre-branch)'}"
        cv2.putText(panel, label, (8, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (255, 255, 255), 1, cv2.LINE_AA)
        out.append(panel)
    return out


def write_video(panels: list[np.ndarray], out_base: Path, fps: int, pause: int = 3) -> None:
    """Write GIF + MP4; the first dream frame is held ``pause`` extra ticks."""
    out_base.parent.mkdir(parents=True, exist_ok=True)
    gif = [Image.fromarray(p) for p in panels]
    gif[0].save(out_base.with_suffix(".gif"), save_all=True, append_images=gif[1:],
                duration=int(1000 / fps), loop=0)
    h, w = panels[0].shape[:2]
    vw = cv2.VideoWriter(str(out_base.with_suffix(".mp4")),
                         cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for p in panels:
        vw.write(cv2.cvtColor(p, cv2.COLOR_RGB2BGR))
    vw.release()


def make_videos(cfg: DictConfig) -> list[Path]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wm, actor, ckpt_cfg = load_dreamer(cfg.viz.video_ckpt, device)
    if wm.decoder is None:
        raise RuntimeError("checkpoint was trained reconstruction-free — nothing to decode")
    env = make_env(cfg.env)

    horizon = int(cfg.viz.video_horizon)
    context = int(cfg.viz.video_context)
    out_dir = Path(cfg.viz.video_out_dir)
    frames, hs, zs = run_real_episode(env, wm, actor, device, int(cfg.viz.video_max_steps))
    env.close()
    T = len(frames)
    print(f"[video] episode of {T} steps recorded")

    written = []
    for branch in cfg.viz.video_branch_points:
        branch = int(branch)
        if branch + horizon >= T or branch < context:
            print(f"[video] skip branch {branch} (episode too short: {T})")
            continue
        real_seq = [to_uint8(f) for f in frames[branch - context: branch + horizon]]
        dream = dream_from(wm, actor, hs[branch], zs[branch], horizon)
        dream_seq = [to_uint8(f) for f in frames[branch - context: branch]]
        dream_seq += [to_uint8(f) for f in dream]
        # Visual pause at the branch moment: hold the first dream frame.
        panels = compose_panels(real_seq, dream_seq, branch_at=context)
        panels = panels[:context] + [panels[context]] * 3 + panels[context:]
        env_tag = cfg.env.name.split("/")[-1].lower()
        base = out_dir / f"{env_tag}_branch{branch:04d}_H{horizon}"
        write_video(panels, base, fps=int(cfg.viz.video_fps))
        written += [base.with_suffix(".gif"), base.with_suffix(".mp4")]
        print(f"[video] branch @ {branch}: {base}.gif/.mp4")
    return written


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    if not cfg.viz.get("video_ckpt"):
        raise ValueError("viz.video_ckpt is required")
    make_videos(cfg)


if __name__ == "__main__":
    main()
