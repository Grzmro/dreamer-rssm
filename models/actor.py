"""Actor: policy pi(a | s_t) over RSSM features s_t = concat(h_t, z_t).

Two action-space branches (config ``action_type``):
  * ``discrete`` (Atari): categorical logits with 1% unimix (same stability
    trick as the RSSM latents). Trained with REINFORCE — the sample itself is
    non-differentiable and the gradient comes from ``log_prob * advantage``
    (see models/losses.py), NOT from a straight-through relaxation.
  * ``continuous`` (DMC/CarRacing): Tanh-squashed diagonal Normal with
    reparameterized sampling, so the actor loss can backpropagate directly
    through imagined dynamics (world-model weights frozen).

``act()`` always returns the action in RSSM-ready form: one-hot float [B, A]
for discrete, squashed float [B, A] for continuous. For env stepping the
discrete integer index is ``action.argmax(-1)``.
"""

from __future__ import annotations

import torch
import torch.distributions as td
import torch.nn as nn
import torch.nn.functional as F

from models.heads import _mlp


class Actor(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        action_dim: int,
        action_type: str = "discrete",
        hidden_dim: int = 512,
        num_layers: int = 3,
        unimix: float = 0.01,
        min_std: float = 0.1,
        init_std: float = 1.0,
    ):
        super().__init__()
        if action_type not in ("discrete", "continuous"):
            raise ValueError(f"unknown action_type: {action_type!r}")
        self.action_type = action_type
        self.action_dim = action_dim
        self.unimix = unimix
        self.min_std = min_std
        # softplus(raw + bias) + min_std == init_std at raw == 0.
        self._std_bias = float(
            torch.log(torch.expm1(torch.tensor(max(init_std - min_std, 1e-4)))).item()
        )
        out_dim = action_dim if action_type == "discrete" else 2 * action_dim
        self.net = _mlp(feat_dim, hidden_dim, out_dim, num_layers)

    # ---------------------------------------------------------------- dists

    def _discrete_logits(self, feat: torch.Tensor) -> torch.Tensor:
        logits = self.net(feat)
        if self.unimix > 0:
            probs = F.softmax(logits, dim=-1)
            probs = (1 - self.unimix) * probs + self.unimix / self.action_dim
            logits = probs.log()
        return logits

    def _continuous_params(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, raw_std = self.net(feat).chunk(2, dim=-1)
        std = F.softplus(raw_std + self._std_bias) + self.min_std
        return mean, std

    # ------------------------------------------------------------------ act

    def act(
        self, feat: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action for features [..., F].

        Returns (action [..., A], log_prob [...], entropy [...]).
        Continuous entropy is the sample estimate ``-log_prob`` (the tanh-
        squashed Normal has no closed form); discrete entropy is exact.
        """
        if self.action_type == "discrete":
            logits = self._discrete_logits(feat)
            dist = td.Categorical(logits=logits)
            if deterministic:
                idx = logits.argmax(dim=-1)
            else:
                idx = dist.sample()
            action = F.one_hot(idx, self.action_dim).float()
            return action, dist.log_prob(idx), dist.entropy()

        mean, std = self._continuous_params(feat)
        normal = td.Normal(mean, std)
        u = mean if deterministic else normal.rsample()  # reparameterized
        action = torch.tanh(u)
        # Change of variables for the tanh squash (as in SAC).
        log_prob = normal.log_prob(u).sum(-1)
        log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-6).sum(-1)
        return action, log_prob, -log_prob

    def log_prob_entropy(
        self, feat: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Re-evaluate log pi(a|s) and entropy for a given (one-hot) action.

        Discrete only — used when the rollout stores actions and the loss
        needs fresh log-probs from the current policy parameters.
        """
        if self.action_type != "discrete":
            raise NotImplementedError("re-evaluation is only used for discrete actions")
        dist = td.Categorical(logits=self._discrete_logits(feat))
        return dist.log_prob(action.argmax(dim=-1)), dist.entropy()
