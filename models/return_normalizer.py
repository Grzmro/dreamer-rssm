"""Return normalization by an EMA of the return-range percentiles (DreamerV3).

The actor loss divides returns/advantages by
``max(limit, EMA(Per(R, 95) - Per(R, 5)))`` — large returns are scaled
down toward unit range, but small returns are NOT scaled up (the ``limit``
floor), so near-zero noise in sparse-reward games is not amplified.

Implemented as an nn.Module with buffers so the state is checkpointed;
kept separate so it can be disabled as an ablation
(``train_dreamer.normalize_returns=false``).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ReturnNormalizer(nn.Module):
    def __init__(
        self,
        decay: float = 0.99,
        low_percentile: float = 5.0,
        high_percentile: float = 95.0,
        limit: float = 1.0,
    ):
        super().__init__()
        self.decay = decay
        self.low_percentile = low_percentile
        self.high_percentile = high_percentile
        self.limit = limit
        self.register_buffer("range_ema", torch.tensor(0.0))
        self.register_buffer("initialized", torch.tensor(False))

    @torch.no_grad()
    def update(self, returns: torch.Tensor) -> None:
        """Update the EMA range from a batch of (detached) returns."""
        x = returns.detach().float().flatten()
        low = torch.quantile(x, self.low_percentile / 100.0)
        high = torch.quantile(x, self.high_percentile / 100.0)
        batch_range = high - low
        if self.initialized:
            self.range_ema.mul_(self.decay).add_((1 - self.decay) * batch_range)
        else:
            self.range_ema.copy_(batch_range)
            self.initialized.fill_(True)

    @property
    def scale(self) -> torch.Tensor:
        """Divisor: max(limit, EMA range) — never scales small returns up."""
        return self.range_ema.clamp_min(self.limit)

    def forward(self, returns: torch.Tensor) -> torch.Tensor:
        return returns / self.scale
