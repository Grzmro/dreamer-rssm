"""Minimal metric-logging abstraction.

TensorBoard is the default backend (no API keys / network needed). To add
wandb later, implement the same three methods in a ``WandbLogger`` and extend
``make_logger`` — call sites only depend on this interface.
"""

from __future__ import annotations

from pathlib import Path


class TensorBoardLogger:
    def __init__(self, log_dir: str | Path):
        from torch.utils.tensorboard import SummaryWriter

        self._writer = SummaryWriter(str(log_dir))

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        self._writer.add_scalar(tag, value, step)

    def close(self) -> None:
        self._writer.close()


def make_logger(backend: str, log_dir: str | Path):
    if backend == "tensorboard":
        return TensorBoardLogger(log_dir)
    if backend == "wandb":
        raise NotImplementedError(
            "wandb backend is not wired up yet; implement WandbLogger here "
            "(same log_scalar/close interface) and select it via logger.backend=wandb."
        )
    raise ValueError(f"unknown logger backend: {backend!r}")
