import math

import pytest
import torch

from models.rssm import RSSM

B, L, A, E = 3, 6, 4, 32


def make_rssm(latent_type="categorical", **kwargs):
    defaults = dict(
        action_dim=A,
        embed_dim=E,
        deter_dim=16,
        hidden_dim=16,
        latent_type=latent_type,
        stoch_groups=4,
        stoch_classes=5,
        stoch_dim=6,
        unimix=0.01,
    )
    defaults.update(kwargs)
    return RSSM(**defaults)


@pytest.mark.parametrize("latent_type,z_dim", [("categorical", 20), ("gaussian", 6)])
def test_observe_shapes(latent_type, z_dim):
    rssm = make_rssm(latent_type)
    out = rssm.observe(
        torch.randn(B, L, E), torch.randn(B, L, A), torch.zeros(B, L, dtype=torch.bool)
    )
    assert out["h"].shape == (B, L, 16)
    assert out["z"].shape == (B, L, z_dim)
    if latent_type == "categorical":
        assert out["post"]["logits"].shape == (B, L, 4, 5)
        # z is one-hot per group in the forward pass
        z = out["z"].view(B, L, 4, 5)
        assert torch.allclose(z.sum(-1), torch.ones(B, L, 4))
    else:
        assert out["post"]["mean"].shape == (B, L, 6)
        assert (out["post"]["std"] > 0).all()


@pytest.mark.parametrize("latent_type", ["categorical", "gaussian"])
def test_imagine_shapes_and_policy_callable(latent_type):
    rssm = make_rssm(latent_type)
    h0, z0 = rssm.initial_state(B, "cpu")
    horizon = 15

    out = rssm.imagine(h0, z0, torch.randn(B, horizon, A))
    assert out["h"].shape == (B, horizon, 16)
    assert out["z"].shape[:2] == (B, horizon)

    # Phase 2 interface: actions from a callable(h, z) -> [B, A].
    calls = []

    def policy(h, z):
        calls.append((h.shape, z.shape))
        return torch.randn(h.shape[0], A)

    out = rssm.imagine(h0, z0, policy, horizon=5)
    assert out["h"].shape == (B, 5, 16)
    assert len(calls) == 5

    with pytest.raises(ValueError):
        rssm.imagine(h0, z0, policy)  # callable requires explicit horizon


def test_is_first_resets_state():
    """A mid-sequence is_first must make the step identical to a fresh start.

    h_t is computed before z_t is sampled, so h at the reset step is a
    deterministic function of (embed-independent) zeroed inputs — it must
    equal h[0] of a fresh observe() on the suffix.
    """
    rssm = make_rssm()
    embed = torch.randn(B, L, E)
    action = torch.randn(B, L, A)
    t_reset = 3
    is_first = torch.zeros(B, L, dtype=torch.bool)
    is_first[:, 0] = True
    is_first[:, t_reset] = True

    full = rssm.observe(embed, action, is_first)
    fresh = rssm.observe(
        embed[:, t_reset:], action[:, t_reset:], is_first[:, t_reset:]
    )
    assert torch.allclose(full["h"][:, t_reset], fresh["h"][:, 0], atol=1e-6)
    assert torch.allclose(
        full["prior"]["logits"][:, t_reset], fresh["prior"]["logits"][:, 0], atol=1e-6
    )
    # Sanity: without the reset the state would differ.
    no_reset = rssm.observe(embed, action, torch.zeros(B, L, dtype=torch.bool))
    assert not torch.allclose(no_reset["h"][:, t_reset], fresh["h"][:, 0], atol=1e-4)


@pytest.mark.parametrize("latent_type", ["categorical", "gaussian"])
def test_straight_through_gradient_flows(latent_type):
    """Gradients must flow from sampled z back to the stat networks' inputs."""
    rssm = make_rssm(latent_type)
    embed = torch.randn(B, L, E, requires_grad=True)
    out = rssm.observe(embed, torch.randn(B, L, A), torch.zeros(B, L, dtype=torch.bool))
    out["z"].sum().backward()
    assert embed.grad is not None
    assert embed.grad.abs().sum() > 0


def test_kl_identical_distributions_gives_free_nats():
    """KL(q||q) = 0, so after free-nats clamping the loss equals free_nats."""
    rssm = make_rssm()
    logits = torch.randn(B, L, 4, 5)
    stats = {"logits": logits}
    loss, value = rssm.kl_loss(stats, stats, balance=0.8, free_nats=1.0)
    assert torch.allclose(value, torch.zeros(B, L), atol=1e-6)
    assert torch.allclose(loss, torch.full((B, L), 1.0))


def test_kl_gaussian_matches_closed_form():
    rssm = make_rssm("gaussian")
    q = {"mean": torch.full((1, 1, 6), 1.0), "std": torch.full((1, 1, 6), 2.0)}
    p = {"mean": torch.zeros(1, 1, 6), "std": torch.ones(1, 1, 6)}
    # Per dim: log(s_p/s_q) + (s_q^2 + (m_q-m_p)^2) / (2 s_p^2) - 0.5
    per_dim = math.log(1 / 2) + (4 + 1) / 2 - 0.5
    assert torch.allclose(rssm._kl(q, p), torch.tensor(6 * per_dim), atol=1e-5)


def test_kl_categorical_matches_manual():
    rssm = make_rssm(stoch_groups=1, stoch_classes=2)
    q_probs = torch.tensor([0.9, 0.1])
    p_probs = torch.tensor([0.5, 0.5])
    q = {"logits": q_probs.log().view(1, 1, 1, 2)}
    p = {"logits": p_probs.log().view(1, 1, 1, 2)}
    expected = float((q_probs * (q_probs.log() - p_probs.log())).sum())
    assert torch.allclose(rssm._kl(q, p), torch.tensor(expected), atol=1e-6)


def test_kl_balancing_stop_gradients():
    """balance=1 -> gradient reaches only the prior; balance=0 -> only the posterior."""
    rssm = make_rssm()
    post_logits = torch.randn(1, 1, 4, 5, requires_grad=True)
    prior_logits = torch.randn(1, 1, 4, 5, requires_grad=True)

    loss, _ = rssm.kl_loss(
        {"logits": post_logits}, {"logits": prior_logits}, balance=1.0, free_nats=0.0
    )
    loss.sum().backward()
    assert prior_logits.grad.abs().sum() > 0
    assert post_logits.grad is None or post_logits.grad.abs().sum() == 0

    post_logits.grad = prior_logits.grad = None
    loss, _ = rssm.kl_loss(
        {"logits": post_logits}, {"logits": prior_logits}, balance=0.0, free_nats=0.0
    )
    loss.sum().backward()
    assert post_logits.grad.abs().sum() > 0
    assert prior_logits.grad is None or prior_logits.grad.abs().sum() == 0
