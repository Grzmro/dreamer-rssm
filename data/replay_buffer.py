"""Sequential (episode-based) replay buffer for world-model training.

Episodes are stored whole, observations as uint8 HWC to save RAM.
Normalization to [-0.5, 0.5] float32 CHW happens at sampling time.

Episode record layout (Dreamer convention, all arrays share length T):
    obs[t]        observation received at step t (obs[0] is the reset obs)
    action[t]     action that *led to* obs[t] (action[0] is a zero dummy)
    reward[t]     reward received together with obs[t] (reward[0] == 0)
    terminated[t] / truncated[t]  episode-end flags (True only at t == T-1)
    is_first[t]   True only at t == 0
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import torch

REQUIRED_KEYS = ("obs", "action", "reward", "terminated", "truncated")


class SequenceReplayBuffer:
    """FIFO buffer of whole episodes with fixed-length sequence sampling.

    Args:
        capacity: maximum total number of stored steps; the oldest episodes
            are evicted (FIFO) once the limit is exceeded.
        seed: RNG seed for sampling.
    """

    def __init__(self, capacity: int, seed: int | None = None):
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self.capacity = int(capacity)
        self._episodes: deque[dict[str, np.ndarray]] = deque()
        self._num_steps = 0
        self._rng = np.random.default_rng(seed)

    @property
    def num_episodes(self) -> int:
        return len(self._episodes)

    @property
    def num_steps(self) -> int:
        return self._num_steps

    def add_episode(self, episode: dict[str, np.ndarray]) -> None:
        """Add a whole episode; evicts oldest episodes beyond capacity."""
        missing = [k for k in REQUIRED_KEYS if k not in episode]
        if missing:
            raise KeyError(f"episode is missing keys: {missing}")

        episode = {k: np.asarray(v) for k, v in episode.items()}
        length = len(episode["obs"])
        if length < 1:
            raise ValueError("cannot add an empty episode")
        for key, arr in episode.items():
            if len(arr) != length:
                raise ValueError(
                    f"all episode arrays must share length {length}, "
                    f"but '{key}' has length {len(arr)}"
                )
        if episode["obs"].dtype != np.uint8:
            raise ValueError(
                f"observations must be stored as uint8, got {episode['obs'].dtype}"
            )
        if "is_first" not in episode:
            is_first = np.zeros(length, dtype=bool)
            is_first[0] = True
            episode["is_first"] = is_first

        self._episodes.append(episode)
        self._num_steps += length
        while self._num_steps > self.capacity and len(self._episodes) > 1:
            evicted = self._episodes.popleft()
            self._num_steps -= len(evicted["obs"])

    def sample(
        self,
        batch_size: int,
        seq_len: int = 50,
        device: str | torch.device | None = None,
    ) -> dict[str, torch.Tensor]:
        """Sample a batch of fixed-length subsequences.

        Episodes are drawn with probability proportional to their length
        (longer episodes contain more windows). Episodes shorter than
        ``seq_len`` are zero-padded at the end; ``mask`` is 1.0 for real
        steps and 0.0 for padding.

        Returns a dict of tensors:
            obs        [B, L, C, H, W] float32 in [-0.5, 0.5]
            action     [B, L, ...]     (int64 for discrete, float32 otherwise)
            reward     [B, L]          float32
            terminated [B, L]          bool
            truncated  [B, L]          bool
            is_first   [B, L]          bool
            mask       [B, L]          float32
        """
        if not self._episodes:
            raise RuntimeError("cannot sample from an empty buffer")

        episodes = list(self._episodes)
        lengths = np.array([len(ep["obs"]) for ep in episodes], dtype=np.float64)
        probs = lengths / lengths.sum()
        indices = self._rng.choice(len(episodes), size=batch_size, p=probs)

        keys = ("obs", "action", "reward", "terminated", "truncated", "is_first")
        chunks: dict[str, list[np.ndarray]] = {k: [] for k in keys}
        masks = []
        for idx in indices:
            ep = episodes[idx]
            length = len(ep["obs"])
            if length >= seq_len:
                start = int(self._rng.integers(0, length - seq_len + 1))
                sl = slice(start, start + seq_len)
                for k in keys:
                    chunks[k].append(ep[k][sl])
                masks.append(np.ones(seq_len, dtype=np.float32))
            else:
                pad = seq_len - length
                for k in keys:
                    arr = ep[k]
                    pad_width = [(0, pad)] + [(0, 0)] * (arr.ndim - 1)
                    chunks[k].append(np.pad(arr, pad_width))
                mask = np.zeros(seq_len, dtype=np.float32)
                mask[:length] = 1.0
                masks.append(mask)

        obs = np.stack(chunks["obs"])  # [B, L, H, W, C] uint8
        obs = obs.astype(np.float32) / 255.0 - 0.5
        obs = obs.transpose(0, 1, 4, 2, 3)  # -> [B, L, C, H, W]

        action = np.stack(chunks["action"])
        action = action.astype(
            np.int64 if np.issubdtype(action.dtype, np.integer) else np.float32
        )

        batch = {
            "obs": torch.from_numpy(obs),
            "action": torch.from_numpy(action),
            "reward": torch.from_numpy(
                np.stack(chunks["reward"]).astype(np.float32)
            ),
            "terminated": torch.from_numpy(np.stack(chunks["terminated"]).astype(bool)),
            "truncated": torch.from_numpy(np.stack(chunks["truncated"]).astype(bool)),
            "is_first": torch.from_numpy(np.stack(chunks["is_first"]).astype(bool)),
            "mask": torch.from_numpy(np.stack(masks)),
        }
        if device is not None:
            batch = {k: v.to(device) for k, v in batch.items()}
        return batch

    def save(self, directory: str | Path) -> None:
        """Save each episode as a compressed .npz file in ``directory``."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for i, ep in enumerate(self._episodes):
            np.savez_compressed(directory / f"episode_{i:06d}.npz", **ep)

    @classmethod
    def load(
        cls, directory: str | Path, capacity: int, seed: int | None = None
    ) -> "SequenceReplayBuffer":
        """Load all episode .npz files from ``directory`` (sorted by name)."""
        buffer = cls(capacity, seed=seed)
        files = sorted(Path(directory).glob("episode_*.npz"))
        if not files:
            raise FileNotFoundError(f"no episode_*.npz files found in {directory}")
        for f in files:
            with np.load(f) as npz:
                buffer.add_episode({k: npz[k] for k in npz.files})
        return buffer
