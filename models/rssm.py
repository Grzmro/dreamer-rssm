"""Recurrent State-Space Model (RSSM) core, DreamerV2/V3 style.

State = (h_t, z_t):
    h_t  deterministic GRU state, updated from [z_{t-1}, a_{t-1}] and h_{t-1}
    z_t  stochastic latent, sampled from either
         - the prior      p(z_t | h_t)        (imagination; no observation), or
         - the posterior  q(z_t | h_t, e_t)   (training; sees encoder embedding)

The two paths are deliberately kept separate:
    * ``observe()``  runs the POSTERIOR over a real sequence (world-model training),
    * ``imagine()``  rolls out the PRIOR only, never touching the encoder
      (open-loop validation now; policy learning in imagination in Phase 2).

Latent variants (config ``latent_type``):
    * ``categorical`` (default): ``stoch_groups`` categorical variables with
      ``stoch_classes`` classes each, straight-through gradients
      (one-hot sample forward, softmax gradient backward).
    * ``gaussian``: diagonal Gaussian with standard reparameterization —
      simpler baseline / ablation, not the main target.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.distributions as td
import torch.nn as nn
import torch.nn.functional as F

State = tuple[torch.Tensor, torch.Tensor]  # (h, z_flat)


class RSSM(nn.Module):
    def __init__(
        self,
        action_dim: int,
        embed_dim: int,
        deter_dim: int = 512,
        hidden_dim: int = 512,
        latent_type: str = "categorical",
        stoch_groups: int = 32,
        stoch_classes: int = 32,
        stoch_dim: int = 30,
        min_std: float = 0.1,
        unimix: float = 0.01,
    ):
        super().__init__()
        if latent_type not in ("categorical", "gaussian"):
            raise ValueError(f"unknown latent_type: {latent_type!r}")
        self.latent_type = latent_type
        self.deter_dim = deter_dim
        self.stoch_groups = stoch_groups
        self.stoch_classes = stoch_classes
        self.stoch_dim = stoch_dim
        self.min_std = min_std
        # DreamerV3 trick: mix 1% uniform into categorical probs so neither
        # KL side can saturate at infinite log-ratios.
        self.unimix = unimix

        if latent_type == "categorical":
            self.z_flat_dim = stoch_groups * stoch_classes
            stats_dim = stoch_groups * stoch_classes
        else:
            self.z_flat_dim = stoch_dim
            stats_dim = 2 * stoch_dim
        self.feat_dim = deter_dim + self.z_flat_dim

        # Deterministic path: [z_{t-1}, a_{t-1}] -> GRU input, then GRUCell.
        self.gru_in = nn.Sequential(
            nn.Linear(self.z_flat_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.gru = nn.GRUCell(hidden_dim, deter_dim)

        # Prior p(z|h) and posterior q(z|h,e) stat networks.
        self.prior_net = nn.Sequential(
            nn.Linear(deter_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, stats_dim),
        )
        self.posterior_net = nn.Sequential(
            nn.Linear(deter_dim + embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, stats_dim),
        )

    # ------------------------------------------------------------------ state

    def initial_state(self, batch_size: int, device) -> State:
        h = torch.zeros(batch_size, self.deter_dim, device=device)
        z = torch.zeros(batch_size, self.z_flat_dim, device=device)
        return h, z

    def _deter_step(
        self, h: torch.Tensor, z: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        return self.gru(self.gru_in(torch.cat([z, action], dim=-1)), h)

    # ------------------------------------------------- distributions / samples

    def _stats(self, raw: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.latent_type == "categorical":
            logits = raw.view(-1, self.stoch_groups, self.stoch_classes)
            if self.unimix > 0:
                probs = F.softmax(logits, dim=-1)
                probs = (1 - self.unimix) * probs + self.unimix / self.stoch_classes
                logits = probs.log()
            return {"logits": logits}
        mean, raw_std = raw.chunk(2, dim=-1)
        std = F.softplus(raw_std) + self.min_std
        return {"mean": mean, "std": std}

    def _dist(self, stats: dict[str, torch.Tensor]) -> td.Distribution:
        if self.latent_type == "categorical":
            return td.Independent(
                td.OneHotCategoricalStraightThrough(logits=stats["logits"]), 1
            )
        return td.Independent(td.Normal(stats["mean"], stats["std"]), 1)

    def sample(self, stats: dict[str, torch.Tensor]) -> torch.Tensor:
        """Sample z (flattened) with gradients flowing to the stats.

        Categorical: straight-through — one-hot forward, softmax grad backward
        (rsample of OneHotCategoricalStraightThrough == hard + probs - sg(probs)).
        Gaussian: standard reparameterization.
        """
        z = self._dist(stats).rsample()
        return z.flatten(start_dim=-2) if self.latent_type == "categorical" else z

    # ------------------------------------------------------------------ paths

    def observe(
        self,
        embed: torch.Tensor,  # [B, L, E]
        action: torch.Tensor,  # [B, L, A] — action[t] led INTO obs[t] (Dreamer layout)
        is_first: torch.Tensor,  # [B, L] bool/float
        state: State | None = None,
    ) -> dict[str, torch.Tensor]:
        """Posterior pass over a real sequence (world-model training).

        At steps flagged ``is_first`` the recurrent state and the incoming
        action are reset to zeros, so episodes never leak across boundaries.
        Returns stacked [B, L, ...] tensors: h, z, and prior/posterior stats.
        """
        B, L = embed.shape[:2]
        h, z = state if state is not None else self.initial_state(B, embed.device)
        is_first = is_first.float()

        hs, zs, prior_stats, post_stats = [], [], [], []
        for t in range(L):
            reset = 1.0 - is_first[:, t : t + 1]
            h, z = h * reset, z * reset
            a = action[:, t] * reset  # action[0] of an episode is a zero dummy anyway
            h = self._deter_step(h, z, a)
            prior = self._stats(self.prior_net(h))
            post = self._stats(self.posterior_net(torch.cat([h, embed[:, t]], dim=-1)))
            z = self.sample(post)
            hs.append(h)
            zs.append(z)
            prior_stats.append(prior)
            post_stats.append(post)

        return {
            "h": torch.stack(hs, dim=1),
            "z": torch.stack(zs, dim=1),
            "prior": self._stack_stats(prior_stats),
            "post": self._stack_stats(post_stats),
        }

    def imagine(
        self,
        h0: torch.Tensor,
        z0: torch.Tensor,
        actions: torch.Tensor | Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        horizon: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Prior-only rollout — no encoder, no observations.

        ``actions`` is either a precomputed [B, H, A] tensor (this phase:
        replayed actions from data) or a callable ``(h, z) -> action [B, A]``
        (Phase 2: an actor choosing actions in imagination).
        Returns h, z and prior stats stacked over the horizon: state after
        step t corresponds to timestep t+1 relative to (h0, z0).
        """
        if callable(actions):
            if horizon is None:
                raise ValueError("horizon is required when actions is a callable")
        else:
            horizon = actions.shape[1] if horizon is None else horizon

        h, z = h0, z0
        hs, zs, prior_stats = [], [], []
        for t in range(horizon):
            a = actions(h, z) if callable(actions) else actions[:, t]
            h = self._deter_step(h, z, a)
            prior = self._stats(self.prior_net(h))
            z = self.sample(prior)
            hs.append(h)
            zs.append(z)
            prior_stats.append(prior)

        return {
            "h": torch.stack(hs, dim=1),
            "z": torch.stack(zs, dim=1),
            "prior": self._stack_stats(prior_stats),
        }

    # -------------------------------------------------------------------- kl

    def kl_loss(
        self,
        post: dict[str, torch.Tensor],
        prior: dict[str, torch.Tensor],
        balance: float = 0.8,
        free_nats: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """KL(posterior || prior) with KL balancing and free nats.

        Two KL terms with opposite stop-gradients (DreamerV2):
            balance       * KL(sg(post) || prior)   — trains the prior (dynamics)
            (1 - balance) * KL(post || sg(prior))   — regularizes the posterior

        Free nats are applied per timestep AFTER summing over latent
        dimensions/groups: max(KL, free_nats), preventing latent collapse.

        Returns (kl_loss, kl_value), both [B, L]; kl_value is the raw
        (unclipped, unbalanced) KL for logging.
        """
        detached_post = {k: v.detach() for k, v in post.items()}
        detached_prior = {k: v.detach() for k, v in prior.items()}
        kl_dyn = self._kl(detached_post, prior)  # [B, L]
        kl_rep = self._kl(post, detached_prior)
        kl_value = self._kl(detached_post, detached_prior)

        loss = balance * kl_dyn.clamp_min(free_nats) + (1 - balance) * kl_rep.clamp_min(
            free_nats
        )
        return loss, kl_value

    def _kl(
        self, q: dict[str, torch.Tensor], p: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """KL per element, summed over latent groups/dims -> [...]."""
        if self.latent_type == "categorical":
            kl = td.kl_divergence(
                td.Categorical(logits=q["logits"]), td.Categorical(logits=p["logits"])
            )
            return kl.sum(dim=-1)  # sum over groups
        kl = td.kl_divergence(
            td.Normal(q["mean"], q["std"]), td.Normal(p["mean"], p["std"])
        )
        return kl.sum(dim=-1)  # sum over dims

    @staticmethod
    def _stack_stats(stats: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        return {k: torch.stack([s[k] for s in stats], dim=1) for k in stats[0]}
