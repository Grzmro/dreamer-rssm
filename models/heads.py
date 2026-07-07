"""Prediction heads operating on RSSM features [h_t, z_t].

Reward head, two variants behind config ``reward_head: mse | symlog_twohot``:
  * ``mse`` — plain scalar regression. With sparse rewards (Pong: ~2% of
    steps) its gradient is easily dominated by the pixel-summed
    reconstruction term and the head tends to predict the mean.
  * ``symlog_twohot`` — DreamerV3: regress a categorical distribution over
    exponentially-spaced bins in symlog space, target encoded as two-hot
    over the two nearest bins, trained with cross-entropy. Much stronger
    learning signal for rare/large-magnitude rewards.

Continue head: Bernoulli logit for the continuation flag. The target is
``1 - terminated`` — truncation (time limit) is NOT an environment death and
must not be learned as one; Phase 0 already stores the two flags separately.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.exp(x.abs()) - 1)


def _mlp(in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 2) -> nn.Sequential:
    layers: list[nn.Module] = []
    dim = in_dim
    for _ in range(num_layers):
        layers += [nn.Linear(dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()]
        dim = hidden_dim
    layers.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*layers)


class RewardHead(nn.Module):
    """MSE variant: predicts the scalar reward directly."""

    def __init__(self, feat_dim: int, hidden_dim: int = 512, num_layers: int = 2):
        super().__init__()
        self.net = _mlp(feat_dim, hidden_dim, 1, num_layers)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """[..., feat_dim] -> [...] predicted reward."""
        return self.net(feat).squeeze(-1)

    def loss(self, feat: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Per-element loss [...] (caller applies the padding mask)."""
        return (self(feat) - target).pow(2)

    def prediction(self, feat: torch.Tensor) -> torch.Tensor:
        return self(feat)


class TwoHotSymlogRewardHead(nn.Module):
    """DreamerV3 variant: categorical over bins in symlog space, two-hot CE.

    The final linear layer is zero-initialized (V3) so the initial
    prediction is exactly 0 reward, avoiding a large early loss spike.
    """

    def __init__(
        self,
        feat_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 2,
        num_bins: int = 255,
        low: float = -20.0,
        high: float = 20.0,
    ):
        super().__init__()
        self.net = _mlp(feat_dim, hidden_dim, num_bins, num_layers)
        final: nn.Linear = self.net[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        self.register_buffer("bins", torch.linspace(low, high, num_bins))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """[..., feat_dim] -> [..., num_bins] logits."""
        return self.net(feat)

    def _two_hot(self, target: torch.Tensor) -> torch.Tensor:
        """Encode symlog(target) as weights over the two nearest bins."""
        x = symlog(target).clamp(self.bins[0], self.bins[-1])
        idx_hi = torch.searchsorted(self.bins, x, right=False).clamp(1, len(self.bins) - 1)
        idx_lo = idx_hi - 1
        lo, hi = self.bins[idx_lo], self.bins[idx_hi]
        w_hi = ((x - lo) / (hi - lo)).clamp(0, 1)
        two_hot = torch.zeros(*x.shape, len(self.bins), device=x.device)
        two_hot.scatter_(-1, idx_lo.unsqueeze(-1), (1 - w_hi).unsqueeze(-1))
        two_hot.scatter_add_(-1, idx_hi.unsqueeze(-1), w_hi.unsqueeze(-1))
        return two_hot

    def loss(self, feat: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Per-element cross-entropy [...] against the two-hot target."""
        logits = self(feat)
        return -(self._two_hot(target) * F.log_softmax(logits, dim=-1)).sum(-1)

    def prediction(self, feat: torch.Tensor) -> torch.Tensor:
        """Expected value under the categorical, mapped back via symexp."""
        probs = F.softmax(self(feat), dim=-1)
        return symexp((probs * self.bins).sum(-1))


def make_reward_head(kind: str, feat_dim: int, hidden_dim: int, num_layers: int):
    if kind == "mse":
        return RewardHead(feat_dim, hidden_dim, num_layers)
    if kind == "symlog_twohot":
        return TwoHotSymlogRewardHead(feat_dim, hidden_dim, num_layers)
    raise ValueError(f"unknown reward_head: {kind!r}")


class ContinueHead(nn.Module):
    def __init__(self, feat_dim: int, hidden_dim: int = 512, num_layers: int = 2):
        super().__init__()
        self.net = _mlp(feat_dim, hidden_dim, 1, num_layers)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """[..., feat_dim] -> [...] continuation logit (target: 1 - terminated)."""
        return self.net(feat).squeeze(-1)
