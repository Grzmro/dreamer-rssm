"""WorldModel: encoder + RSSM + decoder + heads, with the combined loss.

Loss = loss_scales.recon * MSE(decoded, obs)          (unit-variance Gaussian NLL)
     + loss_scales.reward * MSE(reward_pred, reward)
     + loss_scales.cont  * BCE(cont_logit, 1 - terminated)
     + loss_scales.kl    * [balance * max(KL(sg(q)||p), free_nats)
                            + (1-balance) * max(KL(q||sg(p)), free_nats)]

All per-step terms are weighted by the padding mask from the Phase 0 buffer,
so zero-padded tails of short episodes contribute nothing.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.decoder import ConvDecoder
from models.encoder import ConvEncoder
from models.heads import ContinueHead, make_reward_head
from models.rssm import RSSM


class WorldModel(nn.Module):
    def __init__(self, obs_channels: int, action_dim: int, discrete_actions: bool, cfg):
        """``cfg`` is the ``model`` config node (see configs/model/*.yaml)."""
        super().__init__()
        self.action_dim = action_dim
        self.discrete_actions = discrete_actions
        self.cfg = cfg

        self.encoder = ConvEncoder(
            in_channels=obs_channels,
            depth=cfg.cnn_depth,
            num_layers=cfg.cnn_layers,
        )
        self.rssm = RSSM(
            action_dim=action_dim,
            embed_dim=self.encoder.embed_dim,
            deter_dim=cfg.deter_dim,
            hidden_dim=cfg.hidden_dim,
            latent_type=cfg.latent_type,
            stoch_groups=cfg.stoch_groups,
            stoch_classes=cfg.stoch_classes,
            stoch_dim=cfg.stoch_dim,
            unimix=cfg.unimix,
        )
        feat_dim = self.rssm.feat_dim
        self.decoder = ConvDecoder(
            feat_dim=feat_dim,
            out_channels=obs_channels,
            depth=cfg.cnn_depth,
            num_layers=cfg.cnn_layers,
        )
        self.reward_head = make_reward_head(
            cfg.get("reward_head", "mse"), feat_dim, cfg.head_hidden_dim, cfg.head_layers
        )
        self.continue_head = ContinueHead(feat_dim, cfg.head_hidden_dim, cfg.head_layers)

    # ---------------------------------------------------------------- helpers

    def prepare_action(self, action: torch.Tensor) -> torch.Tensor:
        """Discrete int actions [B, L] -> one-hot [B, L, A]; float pass through."""
        if self.discrete_actions:
            return F.one_hot(action.long(), self.action_dim).float()
        return action.float()

    @staticmethod
    def features(h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return torch.cat([h, z], dim=-1)

    # ---------------------------------------------------------------- forward

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Posterior pass over a batch from the Phase 0 buffer.

        Returns everything needed for the loss and for visualization:
        recon [B,L,C,H,W], reward_pred [B,L], cont_logit [B,L], prior/post
        stats, h [B,L,D], z [B,L,Z].
        """
        obs = batch["obs"]  # [B, L, C, H, W] in [-0.5, 0.5]
        B, L = obs.shape[:2]
        action = self.prepare_action(batch["action"])

        embed = self.encoder(obs.flatten(0, 1)).unflatten(0, (B, L))
        out = self.rssm.observe(embed, action, batch["is_first"])

        feat = self.features(out["h"], out["z"])  # [B, L, F]
        recon = self.decoder(feat.flatten(0, 1)).unflatten(0, (B, L))
        out["feat"] = feat
        out["recon"] = recon
        out["reward_pred"] = self.reward_head.prediction(feat)
        out["cont_logit"] = self.continue_head(feat)
        return out

    # ------------------------------------------------------------------ loss

    def loss(
        self, batch: dict[str, torch.Tensor], out: dict[str, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the total loss and a dict of components for logging."""
        if out is None:
            out = self.forward(batch)
        cfg = self.cfg
        mask = batch["mask"]  # [B, L] 1.0 for real steps, 0.0 for padding
        denom = mask.sum().clamp_min(1.0)

        # Reconstruction: unit-variance Gaussian NLL == MSE summed over pixels.
        recon_err = (out["recon"] - batch["obs"]).pow(2).sum(dim=(-3, -2, -1))  # [B,L]
        recon_loss = (recon_err * mask).sum() / denom

        # MSE or two-hot cross-entropy depending on the configured head.
        reward_loss = (self.reward_head.loss(out["feat"], batch["reward"]) * mask).sum() / denom

        cont_target = 1.0 - batch["terminated"].float()  # truncation is not death
        cont_loss = (
            F.binary_cross_entropy_with_logits(
                out["cont_logit"], cont_target, reduction="none"
            )
            * mask
        ).sum() / denom

        kl_loss, kl_value = self.rssm.kl_loss(
            out["post"], out["prior"], balance=cfg.kl_balance, free_nats=cfg.free_nats
        )
        kl_loss = (kl_loss * mask).sum() / denom
        kl_value = (kl_value * mask).sum() / denom

        s = cfg.loss_scales
        total = (
            s.recon * recon_loss
            + s.reward * reward_loss
            + s.cont * cont_loss
            + s.kl * kl_loss
        )
        metrics = {
            "loss/total": total.detach(),
            "loss/recon": recon_loss.detach(),
            "loss/reward": reward_loss.detach(),
            "loss/cont": cont_loss.detach(),
            "loss/kl": kl_loss.detach(),  # after balancing + free nats
            "kl/value": kl_value.detach(),  # raw KL(post || prior)
            # Per-pixel MSE is easier to interpret across image sizes.
            "recon/mse_per_pixel": (recon_loss / batch["obs"][0, 0].numel()).detach(),
        }
        return total, metrics
