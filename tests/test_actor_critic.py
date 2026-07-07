import torch

from models.actor import Actor
from models.critic import Critic, TargetCritic

FEAT = 32


def test_discrete_actor_shapes_and_onehot():
    actor = Actor(FEAT, action_dim=6, action_type="discrete", hidden_dim=64, num_layers=2)
    feat = torch.randn(4, FEAT)
    action, log_prob, entropy = actor.act(feat)
    assert action.shape == (4, 6)
    assert log_prob.shape == (4,)
    assert entropy.shape == (4,)
    # One-hot: exactly one 1.0 per row.
    assert torch.allclose(action.sum(-1), torch.ones(4))
    assert set(action.unique().tolist()) <= {0.0, 1.0}
    assert (entropy > 0).all()


def test_discrete_actor_deterministic_is_argmax():
    actor = Actor(FEAT, action_dim=6, action_type="discrete", hidden_dim=64, num_layers=2)
    feat = torch.randn(4, FEAT)
    action, _, _ = actor.act(feat, deterministic=True)
    logits = actor._discrete_logits(feat)
    assert (action.argmax(-1) == logits.argmax(-1)).all()


def test_discrete_actor_batched_time_dims():
    actor = Actor(FEAT, action_dim=4, action_type="discrete", hidden_dim=64, num_layers=2)
    feat = torch.randn(3, 5, FEAT)  # [B, H, F]
    action, log_prob, entropy = actor.act(feat)
    assert action.shape == (3, 5, 4)
    assert log_prob.shape == (3, 5)
    assert entropy.shape == (3, 5)


def test_continuous_actor_bounded_and_reparameterized():
    actor = Actor(FEAT, action_dim=3, action_type="continuous", hidden_dim=64, num_layers=2)
    feat = torch.randn(8, FEAT, requires_grad=True)
    action, log_prob, entropy = actor.act(feat)
    assert action.shape == (8, 3)
    assert log_prob.shape == (8,)
    assert (action.abs() < 1.0).all()
    # Reparameterized: gradient flows from the action back to the input.
    action.sum().backward()
    assert feat.grad is not None and feat.grad.abs().sum() > 0


def test_continuous_log_prob_finite():
    actor = Actor(FEAT, action_dim=2, action_type="continuous", hidden_dim=64, num_layers=2)
    _, log_prob, entropy = actor.act(torch.randn(16, FEAT))
    assert torch.isfinite(log_prob).all()
    assert torch.isfinite(entropy).all()


def test_critic_value_shape_and_loss():
    critic = Critic(FEAT, hidden_dim=64, num_layers=2, head="symlog_twohot")
    feat = torch.randn(4, 7, FEAT)
    value = critic.value(feat)
    assert value.shape == (4, 7)
    loss = critic.loss(feat, torch.randn(4, 7))
    assert loss.shape == (4, 7)


def test_target_critic_ema_moves_slowly():
    critic = Critic(FEAT, hidden_dim=64, num_layers=2, head="mse")
    target = TargetCritic(critic)
    # Perturb the online critic, then EMA-update the target.
    with torch.no_grad():
        for p in critic.parameters():
            p.add_(1.0)
    before = [p.clone() for p in target.critic.parameters()]
    target.update(critic, tau=0.98)
    for b, t, o in zip(before, target.critic.parameters(), critic.parameters()):
        # Target moved toward online by exactly 2% of the gap.
        assert torch.allclose(t, b + 0.02 * (o - b), atol=1e-6)
        assert not torch.allclose(t, o)  # ... but is not equal to online


def test_target_critic_has_no_grads():
    critic = Critic(FEAT, hidden_dim=64, num_layers=2, head="mse")
    target = TargetCritic(critic)
    value = target.value(torch.randn(4, FEAT))
    assert not value.requires_grad
    assert all(not p.requires_grad for p in target.critic.parameters())
