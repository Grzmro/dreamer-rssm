"""Gradient-separation and masking tests for the actor/critic losses.

These are the definitive Phase 2 leak tests: a FULL actor-critic step
(posterior -> rollout -> lambda-targets -> both losses -> both backwards)
must leave the world model untouched, and neither loss may cross-train
the other network.
"""

import pytest
import torch
from omegaconf import OmegaConf

from models.actor import Actor
from models.critic import Critic, TargetCritic
from models.losses import actor_loss, compute_lambda_targets, critic_loss
from models.return_normalizer import ReturnNormalizer
from models.world_model import WorldModel
from train.imagine_rollout import imagine_rollout

B, L, A, H = 2, 6, 4, 5


def small_cfg():
    return OmegaConf.create(
        {
            "latent_type": "categorical",
            "stoch_groups": 4,
            "stoch_classes": 5,
            "stoch_dim": 6,
            "unimix": 0.01,
            "deter_dim": 16,
            "hidden_dim": 16,
            "cnn_depth": 8,
            "cnn_layers": 4,
            "head_hidden_dim": 16,
            "head_layers": 1,
            "reward_head": "symlog_twohot",
            "kl_balance": 0.8,
            "free_nats": 1.0,
            "loss_scales": {"recon": 1.0, "reward": 1.0, "cont": 1.0, "kl": 1.0},
        }
    )


def make_batch(discrete=True):
    action = (
        torch.randint(0, A, (B, L))
        if discrete
        else torch.rand(B, L, A) * 2 - 1
    )
    batch = {
        "obs": torch.rand(B, L, 3, 64, 64) - 0.5,
        "action": action,
        "reward": torch.randn(B, L),
        "terminated": torch.zeros(B, L, dtype=torch.bool),
        "truncated": torch.zeros(B, L, dtype=torch.bool),
        "is_first": torch.zeros(B, L, dtype=torch.bool),
        "mask": torch.ones(B, L),
    }
    batch["is_first"][:, 0] = True
    return batch


def full_ac_step(action_type):
    """Posterior -> rollout -> targets -> losses, exactly like the real loop."""
    wm = WorldModel(3, A, discrete_actions=action_type == "discrete", cfg=small_cfg())
    actor = Actor(wm.rssm.feat_dim, A, action_type, hidden_dim=32, num_layers=2)
    critic = Critic(wm.rssm.feat_dim, hidden_dim=32, num_layers=2, head="mse")
    target = TargetCritic(critic)

    out = wm(make_batch(discrete=action_type == "discrete"))  # posterior WITH a live graph into the WM
    h0 = out["h"].flatten(0, 1)
    z0 = out["z"].flatten(0, 1)
    rollout = imagine_rollout(wm, actor, h0, z0, horizon=H)
    returns, weights = compute_lambda_targets(rollout, target)
    c_loss, _ = critic_loss(critic, rollout, returns, weights)
    a_loss, _ = actor_loss(
        rollout, returns, weights, critic, action_type,
        normalizer=ReturnNormalizer(),
    )
    return wm, actor, critic, a_loss, c_loss


def grad_sum(module):
    return sum(
        p.grad.abs().sum().item() for p in module.parameters() if p.grad is not None
    )


@pytest.mark.parametrize("action_type", ["discrete", "continuous"])
def test_full_step_leaves_world_model_clean(action_type):
    wm, actor, critic, a_loss, c_loss = full_ac_step(action_type)
    (a_loss + c_loss).backward()
    for name, p in wm.named_parameters():
        assert p.grad is None or p.grad.abs().sum() == 0, f"WM leak via {name}"
    assert grad_sum(actor) > 0
    assert grad_sum(critic) > 0


@pytest.mark.parametrize("action_type", ["discrete", "continuous"])
def test_actor_loss_does_not_train_critic(action_type):
    _, actor, critic, a_loss, _ = full_ac_step(action_type)
    a_loss.backward()
    assert grad_sum(critic) == 0
    assert grad_sum(actor) > 0


@pytest.mark.parametrize("action_type", ["discrete", "continuous"])
def test_critic_loss_does_not_train_actor(action_type):
    _, actor, critic, _, c_loss = full_ac_step(action_type)
    c_loss.backward()
    assert grad_sum(actor) == 0
    assert grad_sum(critic) > 0


def test_continue_prob_masks_later_steps_in_critic_loss():
    """Steps after an imagined episode end must not contribute to the loss."""
    critic = Critic(8, hidden_dim=16, num_layers=1, head="mse")
    feat = torch.randn(3, H + 1, 8)
    discount = torch.full((3, H), 0.99)
    discount[:, 1] = 0.0  # imagined death entering s_2 -> weights 0 from t=2 on
    rollout = {"feat": feat, "discount": discount}
    from train.lambda_returns import discount_weights

    weights = discount_weights(discount)
    assert (weights[:, 2:] == 0).all()

    returns_a = torch.randn(3, H)
    returns_b = returns_a.clone()
    returns_b[:, 2:] = 999.0  # garbage beyond the imagined death
    loss_a, _ = critic_loss(critic, rollout, returns_a, weights)
    loss_b, _ = critic_loss(critic, rollout, returns_b, weights)
    assert torch.allclose(loss_a, loss_b)


def test_normalizer_shrinks_large_advantages():
    critic = Critic(8, hidden_dim=16, num_layers=1, head="mse")
    feat = torch.zeros(2, H + 1, 8)
    rollout = {
        "feat": feat,
        "log_prob": torch.zeros(2, H, requires_grad=True) + 0.1,
        "entropy": torch.ones(2, H),
        "discount": torch.full((2, H), 0.99),
    }
    weights = torch.ones(2, H)
    returns = torch.linspace(0.0, 100.0, 2 * H).view(2, H)  # huge, spread out
    norm = ReturnNormalizer(decay=0.0)
    _, metrics = actor_loss(
        rollout, returns, weights, critic, "discrete", normalizer=norm
    )
    assert metrics["actor/return_scale"] > 1.0  # 5-95 percentile range ~ 90
    # Raw advantages are ~50 on average; after scaling they are ~unit-sized.
    assert metrics["actor/advantage_mean"].abs() < 5.0
