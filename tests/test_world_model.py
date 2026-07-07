import pytest
import torch
from omegaconf import OmegaConf

from models.world_model import WorldModel

B, L, A = 2, 5, 4


def small_cfg(latent_type="categorical"):
    return OmegaConf.create(
        {
            "latent_type": latent_type,
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
            "kl_balance": 0.8,
            "free_nats": 1.0,
            "loss_scales": {"recon": 1.0, "reward": 1.0, "cont": 1.0, "kl": 1.0},
        }
    )


def make_batch(channels=3):
    batch = {
        "obs": torch.rand(B, L, channels, 64, 64) - 0.5,
        "action": torch.randint(0, A, (B, L)),
        "reward": torch.randn(B, L),
        "terminated": torch.zeros(B, L, dtype=torch.bool),
        "truncated": torch.zeros(B, L, dtype=torch.bool),
        "is_first": torch.zeros(B, L, dtype=torch.bool),
        "mask": torch.ones(B, L),
    }
    batch["is_first"][:, 0] = True
    return batch


@pytest.mark.parametrize("latent_type", ["categorical", "gaussian"])
def test_full_pipeline_shapes(latent_type):
    wm = WorldModel(3, A, discrete_actions=True, cfg=small_cfg(latent_type))
    batch = make_batch()
    out = wm(batch)
    assert out["recon"].shape == (B, L, 3, 64, 64)
    assert out["reward_pred"].shape == (B, L)
    assert out["cont_logit"].shape == (B, L)
    loss, metrics = wm.loss(batch, out)
    assert loss.ndim == 0 and torch.isfinite(loss)
    for key in ("loss/recon", "loss/reward", "loss/cont", "loss/kl", "kl/value"):
        assert key in metrics


def test_grayscale_channels():
    wm = WorldModel(1, A, discrete_actions=True, cfg=small_cfg())
    out = wm(make_batch(channels=1))
    assert out["recon"].shape == (B, L, 1, 64, 64)


def test_encoder_gets_gradient_through_straight_through():
    wm = WorldModel(3, A, discrete_actions=True, cfg=small_cfg())
    loss, _ = wm.loss(make_batch())
    loss.backward()
    enc_grad = sum(p.grad.abs().sum().item() for p in wm.encoder.parameters())
    assert enc_grad > 0


def test_padding_mask_zeroes_contribution():
    """A fully padded tail must not change the loss value."""
    torch.manual_seed(0)
    wm = WorldModel(3, A, discrete_actions=True, cfg=small_cfg())
    batch = make_batch()
    batch["mask"][:, 3:] = 0.0

    corrupted = {k: v.clone() for k, v in batch.items()}
    corrupted["obs"][:, 3:] = 0.123  # garbage in the padded region
    corrupted["reward"][:, 3:] = 99.0

    torch.manual_seed(1)
    loss_a, _ = wm.loss(batch)
    torch.manual_seed(1)
    loss_b, _ = wm.loss(corrupted)
    # Recon/reward/cont terms are masked; tiny drift can only enter through
    # the recurrent state feeding later (masked) steps, which is multiplied
    # by zero in every loss term -> values must match closely.
    assert torch.allclose(loss_a, loss_b, rtol=1e-4)


def test_continue_target_ignores_truncation():
    """cont target is 1-terminated: truncation must NOT lower the target."""
    wm = WorldModel(3, A, discrete_actions=True, cfg=small_cfg())
    batch = make_batch()
    batch["truncated"][:, -1] = True  # time-limit cut, not a real death

    torch.manual_seed(2)
    _, metrics_trunc = wm.loss(batch)
    batch2 = {k: v.clone() for k, v in batch.items()}
    batch2["truncated"][:] = False
    torch.manual_seed(2)
    _, metrics_no = wm.loss(batch2)
    assert torch.allclose(metrics_trunc["loss/cont"], metrics_no["loss/cont"])
