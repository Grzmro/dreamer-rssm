"""Gameplay GIFs: the trained agents actually playing, side by side.

The other Phase 4 videos answer "what does the world model imagine?"
(`viz/real_vs_imagined_video.py`). This one answers the first question
anyone asks — "does it play?" — and answers it *comparatively*: Dreamer and
the model-free baselines play the same environment from the same seeds, and
their episodes are rendered into one strip so the difference is visible
rather than only plottable.

Every panel shows what the agent actually receives (the 64x64 wrapper
output, upscaled), so the strip is the agents' view, not a prettier render
they never saw. Baselines run on their own (grayscale, frame-stacked) input
per `baselines/common.py`; the panel label says so.

Agents are named as ``kind[:label]=checkpoint`` (or bare ``random``):

    python viz/gameplay_gif.py \
        '+viz.gameplay_agents=["dreamer:Dreamer (100k)=experiments/dreamer_pong/checkpoints/dreamer_final.pt","ppo=experiments/baselines/ppo_seed0.pt","random"]'

Outputs into ``viz.gameplay_out_dir``:
    <env>_<label>.gif/.mp4   one strip per agent,
    <env>_compare.gif/.mp4   all agents side by side,
    gameplay.json            returns/lengths of every episode played.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import hydra
import numpy as np
from omegaconf import DictConfig
from PIL import Image

from baselines.common import make_baseline_env
from baselines.policies import load_baseline_policy, random_policy
from envs import make_env
from train.dreamer_loop import OnlinePolicy
from train.train_world_model import resolve_device

KINDS = ("dreamer", "ppo", "dqn", "random")


def parse_spec(spec: str) -> tuple[str, str, str | None]:
    """``"kind[:label]=path"`` or ``"kind[:label]"`` -> (kind, label, path)."""
    head, _, path = str(spec).partition("=")
    kind, _, label = head.partition(":")
    kind, label, path = kind.strip(), label.strip(), path.strip()
    if kind not in KINDS:
        raise ValueError(f"unknown agent kind {kind!r} in {spec!r}; expected one of {KINDS}")
    if kind != "random" and not path:
        raise ValueError(f"{kind} needs a checkpoint: '{kind}=path/to.pt'")
    return kind, (label or kind), (path or None)


def build_agent(kind: str, ckpt: str | None, cfg: DictConfig, device, seed: int):
    """(env, policy, note) for one agent spec; the env is the agent's own."""
    if kind == "dreamer":
        from viz.real_vs_imagined_video import load_dreamer

        wm, actor, _, info = load_dreamer(ckpt, device)
        env = make_env(cfg.env)
        policy = OnlinePolicy(wm, actor, epsilon=0.0, env=env, device=device)
        return env, policy, f"RSSM latent policy, trained {int(info['env_step'])} env steps"
    if kind == "random":
        env = make_env(cfg.env)
        return env, random_policy(env.action_space, seed), "uniform random actions"
    b = cfg.baselines
    env = make_baseline_env(cfg.env, int(b.frame_stack), bool(b.grayscale),
                            record_stats=False)
    policy = load_baseline_policy(
        ckpt, device, epsilon=float(cfg.viz.get("gameplay_eval_epsilon", 0.0)), seed=seed
    )
    stack = f"{int(b.frame_stack)}x{'gray' if b.grayscale else 'rgb'} stack"
    return env, policy, f"{kind.upper()} ({stack}), trained {policy.meta['env_step']} env steps"


def play_episode(env, policy, seed: int, max_steps: int) -> dict:
    """One inference-only episode; frames are the agent's own uint8 input."""
    obs, info = env.reset(seed=seed)
    if hasattr(policy, "reset"):
        policy.reset()
    frames = [np.asarray(info["raw_obs"])]
    rewards: list[float] = []
    for _ in range(max_steps):
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        frames.append(np.asarray(info["raw_obs"]))
        rewards.append(float(reward))
        if terminated or truncated:
            break
    rewards_arr = np.asarray(rewards, dtype=np.float32)
    return {
        "frames": frames,
        "rewards": rewards_arr,
        "score": np.concatenate([[0.0], np.cumsum(rewards_arr)]),
        "episode_return": float(rewards_arr.sum()),
        "length": len(rewards),
    }


def record_agent(env, policy, label: str, seed: int, episodes: int, max_steps: int) -> dict:
    """Play ``episodes`` episodes, keep the best-returning one."""
    best, returns = None, []
    for i in range(max(1, episodes)):
        ep = play_episode(env, policy, seed + i, max_steps)
        returns.append(ep["episode_return"])
        if best is None or ep["episode_return"] > best["episode_return"]:
            best = ep
        print(f"[gameplay] {label}: episode {i + 1}/{episodes} "
              f"return={ep['episode_return']:.1f} length={ep['length']}")
    assert best is not None
    best["all_returns"] = returns  # honest denominator: the kept one is the best of these
    best["label"] = label
    return best


def to_rgb(frame: np.ndarray) -> np.ndarray:
    """uint8 HWC with 1 or 3 channels -> HWC RGB."""
    img = np.asarray(frame, dtype=np.uint8)
    if img.ndim == 2:
        img = img[..., None]
    return np.repeat(img, 3, axis=-1) if img.shape[-1] == 1 else img


def draw_panel(
    frame: np.ndarray, label: str, lines: list[str], scale: int, width: int | None = None
) -> np.ndarray:
    """One agent's frame with a title bar above and status lines below."""
    img = cv2.resize(to_rgb(frame), None, fx=scale, fy=scale,
                     interpolation=cv2.INTER_NEAREST)
    if width is not None and img.shape[1] != width:
        pad = width - img.shape[1]
        left = max(pad // 2, 0)
        img = cv2.copyMakeBorder(img, 0, 0, left, max(pad - left, 0),
                                 cv2.BORDER_CONSTANT, value=(0, 0, 0))
    head = np.zeros((26, img.shape[1], 3), np.uint8)
    cv2.putText(head, label[:40], (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    foot = np.zeros((16 * len(lines) + 8, img.shape[1], 3), np.uint8)
    for i, line in enumerate(lines):
        cv2.putText(foot, line[:44], (6, 16 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (200, 220, 255), 1, cv2.LINE_AA)
    return np.vstack([head, img, foot])


def compose_strip(records: list[dict], scale: int = 4, gap: int = 6) -> list[np.ndarray]:
    """Side-by-side panels for every agent, padded to the longest episode.

    A finished episode holds its last frame, dimmed, with its final return —
    so a short episode does not silently disappear from the comparison.
    """
    total = max(len(r["frames"]) for r in records)
    width = max(r["frames"][0].shape[1] for r in records) * scale
    strip = []
    for t in range(total):
        panels = []
        for rec in records:
            last = len(rec["frames"]) - 1
            i = min(t, last)
            frame = rec["frames"][i]
            over = t > last
            if over:
                frame = (to_rgb(frame).astype(np.float32) * 0.45).astype(np.uint8)
            lines = [
                f"score {rec['score'][min(i, len(rec['score']) - 1)]:+.0f}   t={i}",
                f"final return {rec['episode_return']:+.1f}" if over else " ",
            ]
            panels.append(draw_panel(frame, rec["label"], lines, scale, width))
        height = max(p.shape[0] for p in panels)
        panels = [
            cv2.copyMakeBorder(p, 0, height - p.shape[0], 0, 0, cv2.BORDER_CONSTANT,
                               value=(0, 0, 0)) for p in panels
        ]
        spacer = np.full((height, gap, 3), 40, np.uint8)
        row = panels[0]
        for p in panels[1:]:
            row = np.hstack([row, spacer, p])
        strip.append(cv2.copyMakeBorder(row, 4, 4, 4, 4, cv2.BORDER_CONSTANT,
                                        value=(0, 0, 0)))
    return strip


def write_video(panels: list[np.ndarray], out_base: Path, fps: int, hold: int = 8) -> list[Path]:
    """GIF + MP4; the last frame is held so the final score stays readable."""
    out_base.parent.mkdir(parents=True, exist_ok=True)
    panels = list(panels) + [panels[-1]] * hold
    images = [Image.fromarray(p) for p in panels]
    gif = out_base.with_suffix(".gif")
    images[0].save(gif, save_all=True, append_images=images[1:],
                   duration=int(1000 / fps), loop=0, optimize=True)
    mp4 = out_base.with_suffix(".mp4")
    h, w = panels[0].shape[:2]
    writer = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for p in panels:
        writer.write(cv2.cvtColor(p, cv2.COLOR_RGB2BGR))
    writer.release()
    return [gif, mp4]


def make_gameplay(cfg: DictConfig) -> list[Path]:
    v = cfg.viz
    specs = list(v.get("gameplay_agents") or [])
    if not specs:
        raise ValueError(
            "viz.gameplay_agents is empty; e.g. "
            "'+viz.gameplay_agents=[\"dreamer=<ckpt>\",\"ppo=<ckpt>\",\"random\"]'"
        )
    device = resolve_device(cfg.train_dreamer.device)
    out_dir = Path(v.get("gameplay_out_dir") or "experiments/gameplay")
    env_tag = str(cfg.env.name).split("/")[-1].replace("-", "_").lower()
    seed = int(v.get("gameplay_seed", 0))
    scale, fps = int(v.get("gameplay_scale", 4)), int(v.get("gameplay_fps", 12))

    records, notes, written = [], {}, []
    for spec in specs:
        kind, label, ckpt = parse_spec(spec)
        env, policy, note = build_agent(kind, ckpt, cfg, device, seed)
        rec = record_agent(env, policy, label, seed,
                           int(v.get("gameplay_episodes", 1)),
                           int(v.get("gameplay_max_steps", 600)))
        env.close()
        records.append(rec)
        notes[label] = note
        written += write_video(compose_strip([rec], scale), out_dir / f"{env_tag}_{_slug(label)}", fps)

    if len(records) > 1:
        written += write_video(compose_strip(records, scale), out_dir / f"{env_tag}_compare", fps)

    summary = {
        "env": str(cfg.env.name),
        "action_repeat": int(cfg.env.action_repeat),
        "seed": seed,
        "episodes_per_agent": int(v.get("gameplay_episodes", 1)),
        "agents": [
            {
                "label": r["label"], "note": notes[r["label"]],
                "kept_return": r["episode_return"], "kept_length": r["length"],
                "all_returns": r["all_returns"],
            }
            for r in records
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "gameplay.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for path in written:
        print(f"[gameplay] {path}")
    return written


def _slug(label: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in label.lower()).strip("_")


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    make_gameplay(cfg)


if __name__ == "__main__":
    main()
