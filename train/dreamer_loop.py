"""Full Dreamer loop (Phase 2): collect <-> world model <-> actor-critic.

Per env step (after a random prefill), the current actor collects data via
its posterior belief state; ``train_ratio`` gradient updates are performed
per env step (fractional, accumulator-based). One update = one world-model
step on a replayed batch + one actor-critic step on imagination rollouts
started from the (detached) posterior states of that same batch.

Usage:
    python train/dreamer_loop.py
    python train/dreamer_loop.py train_dreamer.buffer_dir=... train_dreamer.init_wm_ckpt=...
"""

from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from data.replay_buffer import SequenceReplayBuffer
from envs import make_env
from models.actor import Actor
from models.critic import Critic, TargetCritic
from models.losses import actor_loss, compute_lambda_targets, critic_loss
from models.return_normalizer import ReturnNormalizer
from models.world_model import WorldModel
from train.collect import _finalize_episode, _new_episode
from train.imagine_rollout import imagine_rollout
from train.logger import make_logger
from train.train_world_model import resolve_device


def build_models(cfg: DictConfig, env, device):
    obs_channels = env.observation_space.shape[0]
    discrete = hasattr(env.action_space, "n")
    action_dim = env.action_space.n if discrete else env.action_space.shape[0]
    action_type = cfg.agent.action_type
    if action_type == "auto":
        action_type = "discrete" if discrete else "continuous"

    wm = WorldModel(obs_channels, action_dim, discrete, cfg.model).to(device)
    a = cfg.agent.actor
    actor = Actor(
        wm.rssm.feat_dim, action_dim, action_type,
        hidden_dim=a.hidden_dim, num_layers=a.num_layers,
        unimix=a.unimix, min_std=a.min_std, init_std=a.init_std,
    ).to(device)
    c = cfg.agent.critic
    critic = Critic(
        wm.rssm.feat_dim, hidden_dim=c.hidden_dim, num_layers=c.num_layers, head=c.head
    ).to(device)
    target = TargetCritic(critic).to(device)
    return wm, actor, critic, target, action_type


def warm_start_world_model(wm: WorldModel, ckpt_path: str, device) -> None:
    """Load Phase 1 weights, dropping shape-mismatched modules (head changes)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    own = wm.state_dict()
    state = {
        k: v for k, v in ckpt["model_state"].items()
        if k in own and own[k].shape == v.shape
    }
    wm.load_state_dict(state, strict=False)
    dropped = sorted({k.split(".")[0] for k in set(ckpt["model_state"]) - set(state)})
    print(f"[dreamer] warm-started WM from {ckpt_path}"
          + (f" (skipped: {dropped})" if dropped else ""))


class OnlinePolicy:
    """Tracks the posterior belief state across env steps and picks actions.

    Mirrors the Dreamer data layout: at each new observation the posterior
    is updated with the action that LED INTO it (zero dummy at is_first),
    then the actor acts on features of the fresh state.
    """

    def __init__(self, wm: WorldModel, actor: Actor, epsilon: float, env, device):
        self.wm = wm
        self.actor = actor
        self.epsilon = epsilon
        self.env = env
        self.device = device
        self.discrete = hasattr(env.action_space, "n")
        self.reset()

    def reset(self) -> None:
        self.state = None  # lazily initialized on first call
        self.prev_action = None
        self.is_first = True

    @torch.no_grad()
    def __call__(self, obs: np.ndarray):
        wm = self.wm
        obs_t = torch.as_tensor(obs, device=self.device)[None]  # [1, C, H, W]
        embed = wm.encoder(obs_t)[:, None]  # [1, 1, E]

        if self.prev_action is None:
            action_in = torch.zeros(
                1, 1, wm.action_dim, device=self.device
            ) if not self.discrete else torch.zeros(1, 1, dtype=torch.long, device=self.device)
        else:
            action_in = self.prev_action.view(1, 1, *self.prev_action.shape[1:])
        is_first = torch.tensor([[self.is_first]], device=self.device)

        out = wm.rssm.observe(
            embed, wm.prepare_action(action_in), is_first, state=self.state
        )
        h, z = out["h"][:, -1], out["z"][:, -1]
        self.state = (h, z)
        self.is_first = False

        feat = wm.features(h, z)
        action_onehot, _, _ = self.actor.act(feat)  # stochastic policy sample
        if self.discrete:
            idx = int(action_onehot.argmax(-1).item())
            if self.epsilon > 0 and np.random.rand() < self.epsilon:
                idx = self.env.action_space.sample()
            self.prev_action = torch.tensor([idx], device=self.device)
            return idx
        action = action_onehot[0].cpu().numpy().astype(np.float32)
        self.prev_action = action_onehot
        return action


def update_step(
    cfg, buffer, wm, actor, critic, target, normalizer, optims, action_type, device
):
    """One joint gradient update: world model, then actor-critic in imagination."""
    td = cfg.train_dreamer
    wm_optim, actor_optim, critic_optim = optims

    batch = buffer.sample(cfg.buffer.batch_size, cfg.buffer.seq_len, device=device)
    out = wm(batch)
    wm_total, metrics = wm.loss(batch, out)
    wm_optim.zero_grad(set_to_none=True)
    wm_total.backward()
    wm_grad = torch.nn.utils.clip_grad_norm_(wm.parameters(), td.grad_clip)
    wm_optim.step()

    # Imagination starts: every real (non-padding) posterior state, detached.
    keep = batch["mask"].flatten(0, 1) > 0.5
    h0 = out["h"].detach().flatten(0, 1)[keep]
    z0 = out["z"].detach().flatten(0, 1)[keep]
    rollout = imagine_rollout(wm, actor, h0, z0, horizon=td.horizon, gamma=td.gamma)
    returns, weights = compute_lambda_targets(rollout, target, lam=td.td_lambda)
    c_loss, c_metrics = critic_loss(critic, rollout, returns, weights)
    a_loss, a_metrics = actor_loss(
        rollout, returns, weights, critic, action_type,
        entropy_coef=td.entropy_coef, normalizer=normalizer,
    )

    critic_optim.zero_grad(set_to_none=True)
    c_loss.backward()
    critic_grad = torch.nn.utils.clip_grad_norm_(critic.parameters(), td.grad_clip)
    critic_optim.step()

    actor_optim.zero_grad(set_to_none=True)
    a_loss.backward()
    actor_grad = torch.nn.utils.clip_grad_norm_(actor.parameters(), td.grad_clip)
    actor_optim.step()

    target.update(critic, tau=cfg.agent.critic.tau)

    metrics.update(c_metrics)
    metrics.update(a_metrics)
    metrics["grad_norm/wm"] = wm_grad.detach()
    metrics["grad_norm/critic"] = critic_grad.detach()
    metrics["grad_norm/actor"] = actor_grad.detach()
    # Expected imagined return from the start states (dream-vs-real signal).
    metrics["returns/dream_start"] = returns[:, 0].mean().detach()
    return metrics


def train_dreamer(cfg: DictConfig, output_dir: str | Path | None = None):
    if output_dir is None:
        output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    output_dir = Path(output_dir)
    td = cfg.train_dreamer
    device = resolve_device(td.device)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    env = make_env(cfg.env)
    wm, actor, critic, target, action_type = build_models(cfg, env, device)
    if td.init_wm_ckpt:
        warm_start_world_model(wm, td.init_wm_ckpt, device)
    normalizer = (
        ReturnNormalizer().to(device) if td.normalize_returns else None
    )
    optims = (
        torch.optim.Adam(wm.parameters(), lr=td.wm_lr, eps=td.adam_eps),
        torch.optim.Adam(actor.parameters(), lr=td.actor_lr, eps=td.adam_eps),
        torch.optim.Adam(critic.parameters(), lr=td.critic_lr, eps=td.adam_eps),
    )

    if td.buffer_dir:
        buffer = SequenceReplayBuffer.load(
            td.buffer_dir, capacity=cfg.buffer.capacity, seed=cfg.seed
        )
        print(f"[dreamer] preloaded buffer: {buffer.num_steps} steps")
    else:
        buffer = SequenceReplayBuffer(cfg.buffer.capacity, seed=cfg.seed)

    logger = make_logger(cfg.logger.backend, output_dir / "tb")
    # Phase 3 (additive): shared benchmark protocol — one CSV row per episode.
    bench = None
    if td.get("benchmark_dir"):
        from train.common_logger import BenchmarkLogger

        bench = BenchmarkLogger(
            td.benchmark_dir, td.get("benchmark_agent", "dreamer"), cfg.env.name, cfg.seed
        )
    metrics_path = output_dir / "metrics.jsonl"
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = open(metrics_path, "a", encoding="utf-8")

    def jsonl(record: dict) -> None:
        metrics_file.write(json.dumps(record) + "\n")
        metrics_file.flush()

    n_params = sum(p.numel() for m in (wm, actor, critic) for p in m.parameters())
    print(f"[dreamer] device={device} params={n_params / 1e6:.1f}M "
          f"action_type={action_type} train_ratio={td.train_ratio}")

    policy = OnlinePolicy(wm, actor, td.epsilon, env, device)
    obs, info = env.reset(seed=cfg.seed)
    episode = _new_episode(info["raw_obs"], env.action_space)
    policy.reset()

    prefill_needed = max(0, int(td.prefill_steps) - buffer.num_steps)
    total_steps = prefill_needed + int(td.total_env_steps)
    # The random->policy switch waits for an episode boundary so the online
    # posterior state never starts cold mid-episode.
    use_policy = prefill_needed == 0
    recent_returns: deque[float] = deque(maxlen=10)
    env_step = update = 0
    accum = 0.0
    start_time = time.time()

    while env_step < total_steps:
        action = policy(obs) if use_policy else env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        episode["obs"].append(info["raw_obs"])
        episode["action"].append(np.asarray(action))
        episode["reward"].append(reward)
        episode["terminated"].append(terminated)
        episode["truncated"].append(truncated)
        env_step += 1

        if terminated or truncated:
            ep_return = float(np.sum(episode["reward"]))
            ep_length = len(episode["reward"]) - 1
            buffer.add_episode(_finalize_episode(episode))
            recent_returns.append(ep_return)
            logger.log_scalar("env/episode_return", ep_return, env_step)
            logger.log_scalar("env/episode_length", ep_length, env_step)
            jsonl({"kind": "episode", "env_step": env_step, "update": update,
                   "return": ep_return, "length": ep_length,
                   "policy": "actor" if use_policy else "random"})
            if bench is not None:
                bench.log_episode(env_step, ep_return, ep_length)
            print(f"[dreamer] step {env_step:7d} | episode return {ep_return:8.2f} "
                  f"| avg10 {np.mean(recent_returns):8.2f} | updates {update}")
            _, info = env.reset()
            episode = _new_episode(info["raw_obs"], env.action_space)
            policy.reset()
            if env_step >= prefill_needed:
                use_policy = True

        if env_step >= prefill_needed and buffer.num_episodes > 0:
            accum += float(td.train_ratio)
            while accum >= 1.0:
                accum -= 1.0
                update += 1
                metrics = update_step(
                    cfg, buffer, wm, actor, critic, target, normalizer,
                    optims, action_type, device,
                )
                if update % int(td.log_interval) == 0 or update == 1:
                    for tag, value in metrics.items():
                        logger.log_scalar(tag, value.item(), update)
                    ups = update / max(time.time() - start_time, 1e-9)
                    print(
                        f"[dreamer] update {update:7d} | recon/px "
                        f"{metrics['recon/mse_per_pixel']:.5f} | "
                        f"dream R {metrics['returns/dream_start']:7.3f} | "
                        f"entropy {metrics['actor/entropy']:.3f} | "
                        f"critic V {metrics['critic/value_mean']:7.3f} | {ups:.2f} up/s"
                    )
                    jsonl({
                        "kind": "update", "env_step": env_step, "update": update,
                        "dream_return": metrics["returns/dream_start"].item(),
                        "entropy": metrics["actor/entropy"].item(),
                        "critic_value": metrics["critic/value_mean"].item(),
                        "critic_loss": metrics["loss/critic"].item(),
                        "actor_loss": metrics["loss/actor"].item(),
                        "wm_loss": metrics["loss/total"].item(),
                        "kl": metrics["kl/value"].item(),
                        "avg10_return": float(np.mean(recent_returns)) if recent_returns else None,
                    })
                if update % int(td.checkpoint_interval) == 0:
                    _save_checkpoint(
                        wm, actor, critic, target, normalizer, cfg,
                        env_step, update, ckpt_dir / f"dreamer_{update:07d}.pt",
                    )

    final_path = ckpt_dir / "dreamer_final.pt"
    _save_checkpoint(wm, actor, critic, target, normalizer, cfg, env_step, update, final_path)
    env.close()
    logger.close()
    metrics_file.close()
    if bench is not None:
        bench.close()
    mins = (time.time() - start_time) / 60
    print(f"[dreamer] done: {env_step} env steps, {update} updates in {mins:.1f} min")
    print(f"[dreamer] final checkpoint: {final_path}")
    return final_path


def _save_checkpoint(wm, actor, critic, target, normalizer, cfg, env_step, update, path):
    """Superset of the Phase 1 format — loadable by load_world_model too."""
    torch.save(
        {
            "model_state": wm.state_dict(),
            "actor_state": actor.state_dict(),
            "critic_state": critic.state_dict(),
            "target_critic_state": target.state_dict(),
            "normalizer_state": normalizer.state_dict() if normalizer else None,
            "cfg": OmegaConf.to_container(cfg, resolve=True),
            "obs_channels": wm.encoder.net[0].in_channels,
            "action_dim": wm.action_dim,
            "discrete_actions": wm.discrete_actions,
            "env_step": env_step,
            "update": update,
        },
        path,
    )


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    train_dreamer(cfg)


if __name__ == "__main__":
    main()
