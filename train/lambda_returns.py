"""TD(lambda) returns for imagination rollouts (Dreamer convention).

Tensor convention used throughout Phase 2: time is the LAST batch axis,
shapes are [B, H] (batch first, horizon second).

Indexing matches the rollout produced by train/imagine_rollout.py: for
imagined states s_1..s_H (s_0 is the detached posterior start),
``reward[t]`` and ``discount[t]`` belong to state s_{t+1} — the reward
received when ENTERING that state — and ``bootstrap`` is V_target(s_H).

    R_t = r_{t+1} + gamma_{t+1} * ((1 - lam) * v_{t+1} + lam * R_{t+1})

computed backward from R_{H-1} = r_H + gamma_H * v_H, where
gamma_t = gamma * continue_prob_t is the model-predicted discount.
"""

from __future__ import annotations

import torch


def lambda_returns(
    reward: torch.Tensor,  # [B, H]  r_{t+1} for t = 0..H-1
    discount: torch.Tensor,  # [B, H]  gamma * cont_prob_{t+1}
    value: torch.Tensor,  # [B, H]  V_target(s_{t+1})
    bootstrap: torch.Tensor | None = None,  # [B]  V_target(s_H); default value[:, -1]
    lam: float = 0.95,
) -> torch.Tensor:
    """Lambda-returns [B, H]: returns[:, t] is the target for state s_t.

    ``value`` must come from the (frozen) TARGET critic; the result is a
    regression target and should be treated as a constant (detach) by both
    the critic and the actor loss.
    """
    if bootstrap is None:
        bootstrap = value[:, -1]
    H = reward.shape[1]
    returns = torch.empty_like(reward)
    nxt = bootstrap
    for t in reversed(range(H)):
        if t == H - 1:
            # No blending at the boundary: R_{H-1} = r_H + gamma_H * v_H.
            nxt = reward[:, t] + discount[:, t] * nxt
        else:
            nxt = reward[:, t] + discount[:, t] * (
                (1 - lam) * value[:, t] + lam * nxt
            )
        returns[:, t] = nxt
    return returns


def discount_weights(discount: torch.Tensor) -> torch.Tensor:
    """Cumulative loss weights [B, H] from per-step discounts [B, H].

    weight[:, 0] = 1 and weight[:, t] = prod_{i < t} discount[:, i]: a step
    only counts as much as the probability that the imagined episode was
    still running when it was reached. Detached — weights are never a
    gradient path.
    """
    ones = torch.ones_like(discount[:, :1])
    return torch.cumprod(torch.cat([ones, discount[:, :-1]], dim=1), dim=1).detach()
