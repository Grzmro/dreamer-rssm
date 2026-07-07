import pytest
import torch
from omegaconf import OmegaConf

from models.actor import Actor
from models.return_normalizer import ReturnNormalizer
from models.world_model import WorldModel
from train.imagine_rollout import freeze_parameters, imagine_rollout

N, H, A = 3, 4, 5


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


def make_wm(discrete=True):
    return WorldModel(3, A, discrete_actions=discrete, cfg=small_cfg())


def make_actor(wm, action_type):
    return Actor(
        wm.rssm.feat_dim, A, action_type=action_type, hidden_dim=32, num_layers=2
    )


def start_states(wm):
    h, z = wm.rssm.initial_state(N, "cpu")
    return h + torch.randn_like(h), z


@pytest.mark.parametrize("action_type", ["discrete", "continuous"])
def test_rollout_shapes(action_type):
    wm = make_wm(discrete=action_type == "discrete")
    actor = make_actor(wm, action_type)
    h, z = start_states(wm)
    out = imagine_rollout(wm, actor, h, z, horizon=H, gamma=0.99)
    F = wm.rssm.feat_dim
    assert out["feat"].shape == (N, H + 1, F)
    assert out["action"].shape == (N, H, A)
    for key in ("log_prob", "entropy", "reward", "cont_prob", "discount"):
        assert out[key].shape == (N, H), key
    assert ((out["cont_prob"] >= 0) & (out["cont_prob"] <= 1)).all()
    assert torch.allclose(out["discount"], 0.99 * out["cont_prob"])


@pytest.mark.parametrize("action_type", ["discrete", "continuous"])
def test_no_gradient_into_world_model(action_type):
    """The core Phase 2 guarantee: actor-loss-like backward leaves WM clean."""
    wm = make_wm(discrete=action_type == "discrete")
    actor = make_actor(wm, action_type)
    h, z = start_states(wm)
    h.requires_grad_(True)  # simulate posterior states still tied to the WM graph

    out = imagine_rollout(wm, actor, h, z, horizon=H)
    if action_type == "discrete":
        surrogate = -(out["log_prob"] * torch.randn(N, H)).mean() - out["entropy"].mean()
    else:
        # Continuous: backprop straight through imagined rewards (dynamics path).
        surrogate = -out["reward"].mean() - 1e-3 * out["entropy"].mean()
    surrogate.backward()

    for name, p in wm.named_parameters():
        assert p.grad is None or p.grad.abs().sum() == 0, f"WM leak via {name}"
    assert h.grad is None  # start states were detached inside the rollout
    actor_grad = sum(
        p.grad.abs().sum().item() for p in actor.parameters() if p.grad is not None
    )
    assert actor_grad > 0  # ... while the actor DID receive gradient


def test_continuous_reward_is_differentiable_wrt_actor():
    """Frozen WM weights must not sever the action -> reward graph."""
    wm = make_wm(discrete=False)
    actor = make_actor(wm, "continuous")
    h, z = start_states(wm)
    out = imagine_rollout(wm, actor, h, z, horizon=H)
    assert out["reward"].requires_grad
    assert out["feat"].requires_grad


def test_discrete_rollout_graph_free():
    """REINFORCE branch: dynamics tensors must carry no graph at all."""
    wm = make_wm(discrete=True)
    actor = make_actor(wm, "discrete")
    h, z = start_states(wm)
    out = imagine_rollout(wm, actor, h, z, horizon=H)
    assert not out["feat"].requires_grad
    assert not out["reward"].requires_grad
    assert out["log_prob"].requires_grad
    assert out["entropy"].requires_grad


def test_freeze_parameters_restores_flags():
    wm = make_wm()
    wm.decoder.net[0].weight.requires_grad_(False)  # one already-frozen param
    flags = [p.requires_grad for p in wm.parameters()]
    with freeze_parameters(wm):
        assert all(not p.requires_grad for p in wm.parameters())
    assert [p.requires_grad for p in wm.parameters()] == flags


def test_return_normalizer_scales_down_not_up():
    norm = ReturnNormalizer(decay=0.0, limit=1.0)  # decay 0 -> track batch exactly
    big = torch.linspace(-50, 50, 200)
    norm.update(big)
    assert norm.scale > 1.0
    assert norm(big).abs().max() < big.abs().max()

    small = torch.linspace(-0.01, 0.01, 200)
    norm2 = ReturnNormalizer(decay=0.0, limit=1.0)
    norm2.update(small)
    assert norm2.scale == 1.0  # floor: small returns are NOT amplified
    assert torch.allclose(norm2(small), small)


def test_return_normalizer_ema():
    norm = ReturnNormalizer(decay=0.9, limit=1.0)
    norm.update(torch.linspace(0, 10, 100))  # first update sets range directly
    first = norm.range_ema.item()
    norm.update(torch.linspace(0, 20, 100))
    second = norm.range_ema.item()
    assert first < second < 0.9 * first + 0.1 * (20 * 0.9)  # moved, but slowly
