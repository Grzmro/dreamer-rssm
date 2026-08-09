"""Does anything actually LEARN?

Every other test in this suite checks a structural invariant — shapes,
gradient routing, closed-form values, masking. All of them pass on a model
that optimizes the wrong way: flip a sign in a loss and the shapes still
match and the gradients still reach the right parameters. These two tests
are the ones that fail in that case. They are deliberately tiny (seconds on
CPU) and assert direction, never a specific number.
"""

import torch
from omegaconf import OmegaConf

from models.critic import Critic
from models.losses import critic_loss
from models.world_model import WorldModel

B, L, A = 2, 4, 3


def small_cfg():
    return OmegaConf.create(
        {
            "latent_type": "categorical",
            "stoch_groups": 4,
            "stoch_classes": 4,
            "stoch_dim": 6,
            "unimix": 0.01,
            "deter_dim": 32,
            "hidden_dim": 32,
            "cnn_depth": 8,
            "cnn_layers": 4,
            "head_hidden_dim": 32,
            "head_layers": 1,
            "reward_head": "symlog_twohot",
            "kl_balance": 0.8,
            "free_nats": 1.0,
            "loss_scales": {"recon": 1.0, "reward": 1.0, "cont": 1.0, "kl": 1.0},
        }
    )


def memorizable_batch():
    """A batch the world model can only fit by using its recurrent state.

    Each timestep is a uniform frame with its own brightness, so the decoder
    cannot reproduce the sequence from a bias alone — it has to read which
    step it is out of [h, z].
    """
    obs = torch.zeros(B, L, 3, 64, 64)
    reward = torch.zeros(B, L)
    for b in range(B):
        for t in range(L):
            obs[b, t] = -0.4 + 0.25 * (b * L + t)  # distinct level per (b, t)
            reward[b, t] = float(t)
    batch = {
        "obs": obs,
        "action": torch.zeros(B, L, dtype=torch.long),
        "reward": reward,
        "terminated": torch.zeros(B, L, dtype=torch.bool),
        "truncated": torch.zeros(B, L, dtype=torch.bool),
        "is_first": torch.zeros(B, L, dtype=torch.bool),
        "mask": torch.ones(B, L),
    }
    batch["is_first"][:, 0] = True
    return batch


def test_world_model_overfits_a_tiny_batch():
    """Reconstruction error on a fixed batch must fall by an order of magnitude."""
    torch.manual_seed(0)
    wm = WorldModel(3, A, discrete_actions=True, cfg=small_cfg())
    batch = memorizable_batch()
    optim = torch.optim.Adam(wm.parameters(), lr=1e-3)

    _, first = wm.loss(batch)
    start = first["recon/mse_per_pixel"].item()
    for _ in range(150):
        loss, metrics = wm.loss(batch)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
    end = metrics["recon/mse_per_pixel"].item()

    assert end < start / 10.0, f"recon/px did not fall: {start:.5f} -> {end:.5f}"


def test_critic_regresses_toward_its_target():
    """critic_loss must move V(s) toward the lambda-return, not away from it."""
    torch.manual_seed(0)
    H, F = 5, 16
    critic = Critic(F, hidden_dim=32, num_layers=2, head="mse")
    feat = torch.randn(4, H + 1, F)
    rollout = {"feat": feat, "discount": torch.full((4, H), 0.99)}
    weights = torch.ones(4, H)
    returns = torch.full((4, H), 3.0)  # constant target the critic can reach

    optim = torch.optim.Adam(critic.parameters(), lr=1e-2)
    start = (critic.value(feat[:, :-1]) - 3.0).abs().mean().item()
    for _ in range(150):
        loss, _ = critic_loss(critic, rollout, returns, weights)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
    end = (critic.value(feat[:, :-1]) - 3.0).abs().mean().item()

    assert end < start / 10.0, f"critic did not converge: {start:.4f} -> {end:.4f}"
