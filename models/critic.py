"""Critic V(s_t) on RSSM features, plus the EMA target critic.

The value head reuses the reward-head machinery from Phase 1
(``make_reward_head``): either a plain scalar MSE regressor or the
DreamerV3 symlog + two-hot categorical head (default — same rationale as
for rewards: lambda-return targets in sparse-reward games are dominated by
near-zero values, and the two-hot head keeps a usable learning signal).

Target critic (DreamerV2/V3 convention): a frozen copy of the critic whose
weights track the online critic by exponential moving average, updated once
per gradient step (``tau`` close to 1 == slow target). It is used ONLY to
compute bootstrap values for the lambda-return targets; it is never trained
by a gradient of its own.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from models.heads import make_reward_head


class Critic(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 3,
        head: str = "symlog_twohot",
    ):
        super().__init__()
        self.head = make_reward_head(head, feat_dim, hidden_dim, num_layers)

    def value(self, feat: torch.Tensor) -> torch.Tensor:
        """[..., F] -> [...] state value."""
        return self.head.prediction(feat)

    def loss(self, feat: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Per-element regression loss [...] against (detached) targets."""
        return self.head.loss(feat, target)

    forward = value


class TargetCritic(nn.Module):
    """EMA copy of the critic; provides bootstrap values only."""

    def __init__(self, critic: Critic):
        super().__init__()
        self.critic = copy.deepcopy(critic)
        for p in self.critic.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def value(self, feat: torch.Tensor) -> torch.Tensor:
        return self.critic.value(feat)

    @torch.no_grad()
    def update(self, online: Critic, tau: float) -> None:
        """target <- tau * target + (1 - tau) * online, once per grad step."""
        for p_t, p_o in zip(self.critic.parameters(), online.parameters()):
            p_t.lerp_(p_o, 1.0 - tau)
        for b_t, b_o in zip(self.critic.buffers(), online.buffers()):
            b_t.copy_(b_o)
