"""Prediction heads operating on RSSM features [h_t, z_t].

Reward head: scalar regression trained with MSE.
TODO(V3): add symlog + two-hot discretized regression as a switchable
upgrade (config ``reward_head: mse | symlog_twohot``). Not implemented in
Phase 1 — plain MSE is sufficient for Pong-scale rewards in {-1, 0, 1};
revisit for environments with wide reward magnitudes.

Continue head: Bernoulli logit for the continuation flag. The target is
``1 - terminated`` — truncation (time limit) is NOT an environment death and
must not be learned as one; Phase 0 already stores the two flags separately.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _mlp(in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 2) -> nn.Sequential:
    layers: list[nn.Module] = []
    dim = in_dim
    for _ in range(num_layers):
        layers += [nn.Linear(dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()]
        dim = hidden_dim
    layers.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*layers)


class RewardHead(nn.Module):
    def __init__(self, feat_dim: int, hidden_dim: int = 512, num_layers: int = 2):
        super().__init__()
        self.net = _mlp(feat_dim, hidden_dim, 1, num_layers)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """[..., feat_dim] -> [...] predicted reward."""
        return self.net(feat).squeeze(-1)


class ContinueHead(nn.Module):
    def __init__(self, feat_dim: int, hidden_dim: int = 512, num_layers: int = 2):
        super().__init__()
        self.net = _mlp(feat_dim, hidden_dim, 1, num_layers)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """[..., feat_dim] -> [...] continuation logit (target: 1 - terminated)."""
        return self.net(feat).squeeze(-1)
