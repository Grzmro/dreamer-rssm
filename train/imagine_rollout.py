"""Imagination rollout: actor + frozen world model, prior dynamics only.

Gradient rules (the whole point of this module — see tests/test_gradient_leak.py):

1. START DETACH: the posterior start states (h_0, z_0) are detached, so no
   actor/critic gradient can ever reach world-model parameters through the
   starting point of imagination.
2. FROZEN WEIGHTS, NOT FROZEN GRAPH: during the rollout every world-model
   parameter has ``requires_grad == False`` (``freeze_parameters`` context
   manager). For the continuous branch the computation graph itself stays
   alive, because the reparameterized actor needs
   d(reward/value)/d(action) to flow back through the RSSM prior and the
   heads INTO THE ACTOR — freezing weights blocks gradients into the world
   model while keeping the graph differentiable w.r.t. actions.
3. DISCRETE = REINFORCE: no gradient is needed through the dynamics at all,
   so the rollout runs under ``torch.no_grad()`` and only the actor's
   log-probs/entropies (recomputed outside no_grad on detached features)
   carry gradient — cheaper, and structurally leak-proof.

Rollout layout (H = horizon):
    s_0            detached posterior start
    a_t ~ pi(s_t)  for t = 0..H-1
    s_{t+1}        prior step from (s_t, a_t)
    reward[t], cont_prob[t]  predictions AT s_{t+1} (reward on entering)
so ``feat`` is [B, H+1, F] over s_0..s_H and everything else is [B, H].
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext

import torch
import torch.nn as nn


@contextmanager
def freeze_parameters(*modules: nn.Module):
    """Temporarily set requires_grad=False on all parameters of ``modules``."""
    params = [p for m in modules for p in m.parameters()]
    prev = [p.requires_grad for p in params]
    try:
        for p in params:
            p.requires_grad_(False)
        yield
    finally:
        for p, r in zip(params, prev):
            p.requires_grad_(r)


def imagine_rollout(
    wm,
    actor,
    h0: torch.Tensor,  # [N, D] posterior deterministic states (any grad state)
    z0: torch.Tensor,  # [N, Z]
    horizon: int,
    gamma: float = 0.99,
) -> dict[str, torch.Tensor]:
    """Roll the actor through the world model's prior for ``horizon`` steps.

    Returns a dict of:
        feat      [N, H+1, F]  features of s_0..s_H (s_0 detached start)
        action    [N, H, A]    a_0..a_{H-1} (one-hot for discrete)
        log_prob  [N, H]       log pi(a_t | s_t)      (grad -> actor only)
        entropy   [N, H]       policy entropy at s_t   (grad -> actor only)
        reward    [N, H]       predicted reward entering s_1..s_H
        cont_prob [N, H]       predicted continue prob at s_1..s_H
        discount  [N, H]       gamma * cont_prob (input to lambda-returns)

    For ``actor.action_type == "continuous"`` feat/reward/discount stay
    connected to the actions in the graph (world-model weights frozen); for
    ``"discrete"`` they are plain grad-free tensors.
    """
    h, z = h0.detach(), z0.detach()  # rule 1: never backprop into the start
    discrete = actor.action_type == "discrete"

    with freeze_parameters(wm):  # rule 2: no gradient into WM weights
        # Rule 3: for discrete/REINFORCE the dynamics run under no_grad;
        # the actor is always called outside it, so its log-probs and
        # entropies keep gradients to the actor parameters.
        feats, actions, log_probs, entropies = [wm.features(h, z)], [], [], []
        for _ in range(horizon):
            a, logp, ent = actor.act(feats[-1])
            with torch.no_grad() if discrete else nullcontext():
                h, z, _ = wm.rssm.imagine_step(h, z, a)
            feats.append(wm.features(h, z))
            actions.append(a)
            log_probs.append(logp)
            entropies.append(ent)

        feat = torch.stack(feats, dim=1)  # [N, H+1, F]
        pred_ctx = torch.no_grad() if discrete else nullcontext()
        with pred_ctx:
            imagined = feat[:, 1:]  # s_1..s_H
            reward = wm.reward_head.prediction(imagined)
            cont_prob = torch.sigmoid(wm.continue_head(imagined))

    return {
        "feat": feat,
        "action": torch.stack(actions, dim=1),
        "log_prob": torch.stack(log_probs, dim=1),
        "entropy": torch.stack(entropies, dim=1),
        "reward": reward,
        "cont_prob": cont_prob,
        "discount": gamma * cont_prob,
    }

