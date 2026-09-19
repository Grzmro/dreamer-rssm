"""Tests for the Phase 4 ablation switches, mainly reconstruction-free."""

import torch
from omegaconf import OmegaConf

from models.world_model import WorldModel

B, L, A = 2, 5, 4


def cfg(**overrides):
    base = {
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
    base.update(overrides)
    return OmegaConf.create(base)


def batch():
    b = {
        "obs": torch.rand(B, L, 3, 64, 64) - 0.5,
        "action": torch.randint(0, A, (B, L)),
        "reward": torch.randn(B, L),
        "terminated": torch.zeros(B, L, dtype=torch.bool),
        "truncated": torch.zeros(B, L, dtype=torch.bool),
        "is_first": torch.zeros(B, L, dtype=torch.bool),
        "mask": torch.ones(B, L),
    }
    b["is_first"][:, 0] = True
    return b


def test_reconstruction_free_has_no_decoder_at_all():
    wm = WorldModel(3, A, discrete_actions=True, cfg=cfg(use_decoder=False))
    assert wm.decoder is None
    # A genuinely compute-free ablation: fewer parameters, not zero weights.
    full = WorldModel(3, A, discrete_actions=True, cfg=cfg())
    n_free = sum(p.numel() for p in wm.parameters())
    n_full = sum(p.numel() for p in full.parameters())
    assert n_free < n_full


def test_reconstruction_free_forward_and_loss():
    wm = WorldModel(3, A, discrete_actions=True, cfg=cfg(use_decoder=False))
    out = wm(batch())
    assert "recon" not in out
    loss, metrics = wm.loss(batch(), out)
    assert torch.isfinite(loss)
    assert metrics["loss/recon"] == 0.0
    # Reward/cont/KL still train:
    loss.backward()
    enc_grad = sum(
        p.grad.abs().sum().item() for p in wm.encoder.parameters() if p.grad is not None
    )
    assert enc_grad > 0  # encoder still gets gradient (via posterior -> reward/KL)


def test_validation_survives_a_reconstruction_free_world_model():
    """Phase 1 validation must not assume a decoder exists.

    Regression: validate_reward_correlation read out["recon"] unconditionally,
    so `train_world_model.py ablation=no_reconstruction` died with
    KeyError: 'recon' at the first val_interval.
    """
    import math

    from data.replay_buffer import SequenceReplayBuffer
    from tests.conftest import make_synthetic_episode
    from train.train_world_model import validate_reward_correlation

    wm = WorldModel(3, A, discrete_actions=True, cfg=cfg(use_decoder=False))
    buffer = SequenceReplayBuffer(capacity=1000, seed=0)
    buffer.add_episode(make_synthetic_episode(L + 2))
    val_cfg = OmegaConf.create(
        {"train_wm": {"val_batches": 1}, "buffer": {"batch_size": B, "seq_len": L}}
    )

    r, recon_mse = validate_reward_correlation(wm, buffer, val_cfg, device="cpu")
    assert math.isnan(recon_mse)  # no decoder -> nothing to score
    assert isinstance(r, float)


def test_default_config_still_reconstructs():
    wm = WorldModel(3, A, discrete_actions=True, cfg=cfg())
    out = wm(batch())
    assert out["recon"].shape == (B, L, 3, 64, 64)


def test_ablation_presets_compose():
    """Every configs/ablation/*.yaml must compose with the main config."""
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    presets = sorted(p.stem for p in Path(config_dir, "ablation").glob("*.yaml"))
    assert "no_reconstruction" in presets and "no_free_nats" in presets
    for preset in presets:
        with initialize_config_dir(version_base="1.3", config_dir=config_dir):
            c = compose(config_name="config", overrides=[f"ablation={preset}"])
        assert c.train_dreamer.benchmark_agent.startswith("dreamer-"), preset
        if preset == "no_reconstruction":
            assert c.model.use_decoder is False
        if preset == "no_free_nats":
            assert c.model.free_nats == 0.0
        if preset.startswith("horizon_"):
            assert c.train_dreamer.horizon == int(preset.split("_")[1])


def test_slurm_ablation_array_names_only_real_presets():
    """A typo in slurm/ablations.sbatch PRESETS would only fail on the cluster,
    hours into a queue wait; the array index is also what RESULTS.md cites."""
    import re
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    script = (repo / "slurm" / "ablations.sbatch").read_text(encoding="utf-8")
    block = script.split("PRESETS=(", 1)[1].split("\n)", 1)[0]  # comments contain ")"
    names = [
        line.split("#")[0].strip().strip('"') for line in block.splitlines()
        if line.split("#")[0].strip()
    ]
    assert names[0] == ""  # index 0 is the base run, the control for every preset
    available = {p.stem for p in (repo / "configs" / "ablation").glob("*.yaml")}
    assert set(names[1:]) <= available, set(names[1:]) - available
    # Indices RESULTS.md §C/§F launch by number:
    assert names[10] == "no_reconstruction" and names[11] == "gaussian_latent"
    assert names[1] == "horizon_5" and names[3] == "horizon_20"
    array = re.search(r"#SBATCH --array=0-(\d+)", script)
    assert array and int(array.group(1)) == len(names) - 1
