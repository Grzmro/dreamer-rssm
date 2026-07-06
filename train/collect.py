"""Random-policy data collection into the sequential replay buffer.

Usage:
    python train/collect.py
    python train/collect.py env=atari_pong buffer.capacity=100000 collect.num_steps=20000
"""

from __future__ import annotations

import time
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

from data.replay_buffer import SequenceReplayBuffer
from envs import make_env
from train.logger import make_logger


def _zero_action(action_space):
    """Dummy action stored at t=0 of each episode (Dreamer convention)."""
    sample = action_space.sample()
    return np.zeros_like(np.asarray(sample))


def collect_random_data(cfg: DictConfig, output_dir: str | Path | None = None):
    """Collect data with a random policy; returns the filled buffer."""
    if output_dir is None:
        output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    output_dir = Path(output_dir)

    env = make_env(cfg.env)
    buffer = SequenceReplayBuffer(cfg.buffer.capacity, seed=cfg.seed)
    logger = make_logger(cfg.logger.backend, output_dir / "tb")

    num_steps = int(cfg.collect.num_steps)
    log_interval = int(cfg.collect.log_interval)
    max_episodes = cfg.collect.get("num_episodes") or float("inf")

    print(f"[collect] env={cfg.env.name} target_steps={num_steps} -> {output_dir}")

    total_steps = 0
    episode_idx = 0
    next_log = log_interval
    start_time = time.time()

    _, info = env.reset(seed=cfg.seed)
    episode = _new_episode(info["raw_obs"], env.action_space)

    while total_steps < num_steps and episode_idx < max_episodes:
        action = env.action_space.sample()
        _, reward, terminated, truncated, info = env.step(action)
        episode["obs"].append(info["raw_obs"])
        episode["action"].append(np.asarray(action))
        episode["reward"].append(reward)
        episode["terminated"].append(terminated)
        episode["truncated"].append(truncated)
        total_steps += 1

        if terminated or truncated:
            ep_return = float(np.sum(episode["reward"]))
            ep_length = len(episode["reward"]) - 1  # exclude the t=0 dummy step
            buffer.add_episode(_finalize_episode(episode))
            throughput = total_steps / (time.time() - start_time)
            logger.log_scalar("collect/episode_return", ep_return, total_steps)
            logger.log_scalar("collect/episode_length", ep_length, total_steps)
            logger.log_scalar("collect/steps_per_sec", throughput, total_steps)
            episode_idx += 1
            print(
                f"[collect] episode {episode_idx:4d} | return {ep_return:8.2f} | "
                f"length {ep_length:5d} | steps {total_steps}/{num_steps} | "
                f"{throughput:.1f} steps/s"
            )
            _, info = env.reset()
            episode = _new_episode(info["raw_obs"], env.action_space)
        elif total_steps >= next_log:
            throughput = total_steps / (time.time() - start_time)
            logger.log_scalar("collect/steps_per_sec", throughput, total_steps)
            print(f"[collect] steps {total_steps}/{num_steps} | {throughput:.1f} steps/s")
            next_log += log_interval

    # Keep a partially finished episode too if it has at least one transition.
    if len(episode["obs"]) > 1:
        buffer.add_episode(_finalize_episode(episode))

    env.close()
    logger.close()

    if cfg.collect.save_buffer:
        buffer_dir = output_dir / "buffer"
        buffer.save(buffer_dir)
        print(f"[collect] saved {buffer.num_episodes} episodes to {buffer_dir}")

    print(
        f"[collect] done: {buffer.num_steps} steps in {buffer.num_episodes} episodes "
        f"({time.time() - start_time:.1f}s)"
    )
    return buffer


def _new_episode(raw_obs: np.ndarray, action_space) -> dict[str, list]:
    return {
        "obs": [raw_obs],
        "action": [_zero_action(action_space)],
        "reward": [0.0],
        "terminated": [False],
        "truncated": [False],
    }


def _finalize_episode(episode: dict[str, list]) -> dict[str, np.ndarray]:
    return {
        "obs": np.stack(episode["obs"]).astype(np.uint8),
        "action": np.stack(episode["action"]),
        "reward": np.asarray(episode["reward"], dtype=np.float32),
        "terminated": np.asarray(episode["terminated"], dtype=bool),
        "truncated": np.asarray(episode["truncated"], dtype=bool),
    }


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    collect_random_data(cfg)


if __name__ == "__main__":
    main()
