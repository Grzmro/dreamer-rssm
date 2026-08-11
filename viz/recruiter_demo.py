"""Record a Pong match plus the agent's own imagination, for a demo page.

Inference only, from a Phase 2 checkpoint. Plays real episodes until one
ends with the agent reaching ``demo.target_points`` first, recording every
frame it saw and every action it took. At a fixed cadence it also branches
off an ``imagine_rollout`` from the CURRENT posterior belief and decodes it
to pixels — the same prior-only path the policy was trained on, so the
"dream" strips are real model output, not an animation.

The result is a single JSON blob (base64 PNGs + metadata) meant to be
embedded in a self-contained HTML page; nothing here renders the page.

Usage:
    python viz/recruiter_demo.py \
        +demo.ckpt=experiments/dreamer_pong/checkpoints/dreamer_final.pt \
        +demo.out=experiments/demo/match.json
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from envs import make_env
from models.actor import Actor
from train.imagine_rollout import imagine_rollout
from train.train_world_model import load_world_model, resolve_device


def _png_b64(frame_hwc_uint8: np.ndarray) -> str:
    """Encode one HWC uint8 frame as a base64 PNG string."""
    buf = io.BytesIO()
    Image.fromarray(frame_hwc_uint8).save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _decode_to_uint8(obs_chw: torch.Tensor) -> np.ndarray:
    """[C,H,W] in [-0.5, 0.5] -> HWC uint8, matching the env's normalization."""
    img = (obs_chw.detach().cpu().float() + 0.5).clamp(0, 1)
    return (img.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)


def _load_agent(ckpt_path: str, device):
    """Rebuild the world model and actor saved by dreamer_loop.py."""
    wm, cfg = load_world_model(ckpt_path, device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = cfg.agent.actor
    actor = Actor(
        wm.rssm.feat_dim,
        ckpt["action_dim"],
        "discrete" if ckpt["discrete_actions"] else "continuous",
        hidden_dim=a.hidden_dim, num_layers=a.num_layers,
        unimix=a.unimix, min_std=a.min_std, init_std=a.init_std,
    ).to(device)
    actor.load_state_dict(ckpt["actor_state"])
    actor.eval()
    return wm, actor, cfg, ckpt


class Belief:
    """The posterior belief the policy carries between env steps.

    Same bookkeeping as train.dreamer_loop.OnlinePolicy, but it also hands
    back the state so a dream can be branched off the exact belief the
    action was chosen from.
    """

    def __init__(self, wm, actor, device):
        self.wm, self.actor, self.device = wm, actor, device
        self.reset()

    def reset(self) -> None:
        self.state = None
        self.prev_action = None
        self.is_first = True

    @torch.no_grad()
    def step(self, obs: np.ndarray) -> tuple[int, torch.Tensor, torch.Tensor]:
        wm = self.wm
        obs_t = torch.as_tensor(obs, device=self.device)[None]
        embed = wm.encoder(obs_t)[:, None]
        if self.prev_action is None:
            action_in = torch.zeros(1, 1, dtype=torch.long, device=self.device)
        else:
            action_in = self.prev_action.view(1, 1)
        is_first = torch.tensor([[self.is_first]], device=self.device)

        out = wm.rssm.observe(
            embed, wm.prepare_action(action_in), is_first, state=self.state
        )
        h, z = out["h"][:, -1], out["z"][:, -1]
        self.state = (h, z)
        self.is_first = False

        onehot, _, _ = self.actor.act(wm.features(h, z))
        idx = int(onehot.argmax(-1).item())
        self.prev_action = torch.tensor([idx], device=self.device)
        return idx, h, z


@torch.no_grad()
def _dream(wm, actor, h, z, horizon: int, gamma: float) -> dict:
    """Roll the actor through the prior and decode the result to pixels."""
    roll = imagine_rollout(wm, actor, h, z, horizon=horizon, gamma=gamma)
    feat = roll["feat"][0, 1:]  # s_1..s_H — the imagined future, not the start
    frames = wm.decoder(feat)
    return {
        "frames": [_png_b64(_decode_to_uint8(f)) for f in frames],
        "actions": roll["action"][0].argmax(-1).tolist(),
        "reward": [round(v, 3) for v in roll["reward"][0].tolist()],
    }


@torch.no_grad()
def record(cfg: DictConfig) -> Path:
    d = cfg.demo
    device = resolve_device(cfg.train_dreamer.device)
    wm, actor, ck_cfg, ckpt = _load_agent(d.ckpt, device)
    print(f"[demo] checkpoint: {d.ckpt}  (env_step={ckpt['env_step']}, "
          f"update={ckpt['update']}, device={device})")

    env_cfg = OmegaConf.merge(cfg.env, {"time_limit": int(d.time_limit)})
    env = make_env(env_cfg)
    names = env.unwrapped.get_action_meanings()
    belief = Belief(wm, actor, device)
    target, horizon = int(d.target_points), int(d.horizon)

    best: tuple[tuple[int, int], dict] | None = None
    all_scores: list[list[int]] = []
    for attempt in range(1, int(d.max_attempts) + 1):
        obs, info = env.reset(seed=int(d.seed) + attempt)
        belief.reset()
        frames = [_png_b64(info["raw_obs"])]
        actions: list[int] = []
        scores = [[0, 0]]
        branches: list[dict] = []
        agent = opponent = 0

        while True:
            action, h, z = belief.step(obs)
            step_i = len(actions)
            if step_i % int(d.branch_every) == 0:
                branches.append(
                    {"step": step_i, **_dream(wm, actor, h, z, horizon,
                                              float(cfg.train_dreamer.gamma))}
                )
            obs, reward, terminated, truncated, info = env.step(action)
            agent += int(reward > 0)
            opponent += int(reward < 0)
            frames.append(_png_b64(info["raw_obs"]))
            actions.append(action)
            scores.append([agent, opponent])
            if terminated or truncated or max(agent, opponent) >= target:
                break

        won = agent >= target
        all_scores.append([agent, opponent])
        # Rank by margin so a 4-5 loss beats a 1-5 one; a win outranks all.
        rank = (int(won), agent - opponent)
        if best is None or rank > best[0]:
            best = (rank, {
                "frames": frames, "actions": actions,
                "action_names": list(names), "scores": scores,
                "branches": branches,
                "meta": {
                    "agent_score": agent, "opponent_score": opponent,
                    "steps": len(actions), "won": won, "attempt": attempt,
                    "horizon": horizon, "action_repeat": int(cfg.env.action_repeat),
                    "env": cfg.env.name, "train_env_steps": int(ckpt["env_step"]),
                    "updates": int(ckpt["update"]),
                },
            })
        print(f"[demo] attempt {attempt}: {agent}-{opponent} in {len(actions)} "
              f"steps -> {'WIN, keeping' if won else 'retry'}")
        if won:
            break

    env.close()
    assert best is not None
    payload = best[1]
    # Honest denominator for the page: how many matches this one was picked from.
    payload["meta"]["attempts_played"] = attempt
    payload["meta"]["all_scores"] = all_scores
    out = Path(d.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload), encoding="utf-8")
    m = payload["meta"]
    print(f"[demo] kept attempt {m['attempt']}: {m['agent_score']}-"
          f"{m['opponent_score']} out of {attempt} played")
    print(f"[demo] wrote {out} ({out.stat().st_size / 1e6:.1f} MB, "
          f"{len(payload['frames'])} frames, {len(payload['branches'])} dreams)")
    return out


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    record(cfg)


if __name__ == "__main__":
    main()
