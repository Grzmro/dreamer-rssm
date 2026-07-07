"""Actor and critic losses on imagination rollouts (Phase 2).

Gradient separation rules (complementing train/imagine_rollout.py):
  * the critic trains ONLY through its own regression loss — its input
    features are detached (no path back to the actor through the imagined
    graph in the continuous branch) and its targets are detached
    lambda-returns computed from the TARGET critic;
  * the actor loss never backpropagates into the critic: the discrete
    baseline V(s_t) is computed under no_grad, the advantage is detached,
    and the continuous branch touches only the frozen target critic;
  * loss weights (cumulative imagined discounts) are always detached.
"""

from __future__ import annotations

import torch

from models.critic import Critic, TargetCritic
from models.return_normalizer import ReturnNormalizer
from train.lambda_returns import discount_weights, lambda_returns


def compute_lambda_targets(
    rollout: dict[str, torch.Tensor],
    target_critic: TargetCritic,
    lam: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lambda-returns [N, H] (targets for s_0..s_{H-1}) and loss weights [N, H].

    Bootstrap values come from the target critic. Its parameters never
    require grad, but the graph THROUGH the features stays alive — in the
    continuous branch the actor differentiates the returns w.r.t. its
    actions, which includes the value term.
    """
    value_next = target_critic.critic.value(rollout["feat"][:, 1:])  # V(s_1..s_H)
    returns = lambda_returns(rollout["reward"], rollout["discount"], value_next, lam=lam)
    weights = discount_weights(rollout["discount"])
    return returns, weights


def critic_loss(
    critic: Critic,
    rollout: dict[str, torch.Tensor],
    returns: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Weighted regression of V(s_t) toward sg(lambda-returns), t = 0..H-1."""
    feat = rollout["feat"][:, :-1].detach()  # no gradient to actor/WM via input
    loss = (critic.loss(feat, returns.detach()) * weights).mean()
    with torch.no_grad():
        value = critic.value(feat)
    metrics = {
        "loss/critic": loss.detach(),
        "critic/value_mean": value.mean(),
        "critic/return_mean": returns.mean().detach(),
    }
    return loss, metrics


def actor_loss(
    rollout: dict[str, torch.Tensor],
    returns: torch.Tensor,
    weights: torch.Tensor,
    critic: Critic,
    action_type: str,
    entropy_coef: float = 3e-4,
    normalizer: ReturnNormalizer | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Policy-gradient loss for the imagined trajectory.

    Discrete: REINFORCE with the online critic as baseline —
        -log pi(a_t|s_t) * sg((R_t - V(s_t)) / max(1, S)).
    Continuous: reparameterized dynamics backprop — maximize the (scaled)
    lambda-returns directly through the frozen world model.
    Both minus an entropy bonus, all terms weighted by the imagined
    discount weights.
    """
    if normalizer is not None:
        normalizer.update(returns)
        scale = normalizer.scale
    else:
        scale = torch.ones((), device=returns.device)

    if action_type == "discrete":
        with torch.no_grad():  # baseline must not train the critic from here
            baseline = critic.value(rollout["feat"][:, :-1])
        advantage = ((returns - baseline) / scale).detach()
        pg = -(weights * rollout["log_prob"] * advantage).mean()
    else:
        pg = -(weights * returns / scale).mean()
        advantage = returns.detach()

    entropy = (weights * rollout["entropy"]).mean()
    loss = pg - entropy_coef * entropy
    metrics = {
        "loss/actor": loss.detach(),
        "actor/entropy": rollout["entropy"].mean().detach(),
        "actor/advantage_mean": advantage.mean(),
        "actor/return_scale": scale.detach(),
        "returns/imagined_mean": returns.mean().detach(),
    }
    return loss, metrics
