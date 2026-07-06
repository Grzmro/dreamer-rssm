import numpy as np
import pytest
import torch

from data.replay_buffer import SequenceReplayBuffer
from tests.conftest import make_synthetic_episode

OBS_SHAPE = (64, 64, 3)


def test_add_episode_counts():
    buf = SequenceReplayBuffer(capacity=10_000, seed=0)
    buf.add_episode(make_synthetic_episode(80, OBS_SHAPE))
    buf.add_episode(make_synthetic_episode(120, OBS_SHAPE))
    assert buf.num_episodes == 2
    assert buf.num_steps == 200


def test_add_episode_validation():
    buf = SequenceReplayBuffer(capacity=1000)
    ep = make_synthetic_episode(10, OBS_SHAPE)
    del ep["reward"]
    with pytest.raises(KeyError):
        buf.add_episode(ep)

    ep = make_synthetic_episode(10, OBS_SHAPE)
    ep["action"] = ep["action"][:5]  # mismatched length
    with pytest.raises(ValueError):
        buf.add_episode(ep)

    ep = make_synthetic_episode(10, OBS_SHAPE)
    ep["obs"] = ep["obs"].astype(np.float32)  # must be uint8
    with pytest.raises(ValueError):
        buf.add_episode(ep)


def test_sample_shapes():
    buf = SequenceReplayBuffer(capacity=100_000, seed=0)
    for length in (80, 120, 200):
        buf.add_episode(make_synthetic_episode(length, OBS_SHAPE))
    batch = buf.sample(batch_size=16, seq_len=50)
    assert batch["obs"].shape == (16, 50, 3, 64, 64)
    assert batch["obs"].dtype == torch.float32
    assert batch["action"].shape == (16, 50)
    assert batch["action"].dtype == torch.int64
    assert batch["reward"].shape == (16, 50)
    assert batch["reward"].dtype == torch.float32
    for key in ("terminated", "truncated", "is_first"):
        assert batch[key].shape == (16, 50)
        assert batch[key].dtype == torch.bool
    assert batch["mask"].shape == (16, 50)
    assert batch["mask"].dtype == torch.float32
    assert torch.all(batch["mask"] == 1.0)  # all episodes >= seq_len


def test_sample_normalization():
    buf = SequenceReplayBuffer(capacity=1000, seed=0)
    buf.add_episode(make_synthetic_episode(60, OBS_SHAPE, fill=255))
    batch = buf.sample(batch_size=2, seq_len=10)
    assert torch.allclose(batch["obs"], torch.tensor(0.5))

    buf2 = SequenceReplayBuffer(capacity=1000, seed=0)
    buf2.add_episode(make_synthetic_episode(60, OBS_SHAPE, fill=0))
    batch = buf2.sample(batch_size=2, seq_len=10)
    assert torch.allclose(batch["obs"], torch.tensor(-0.5))


def test_short_episode_padding_and_mask():
    buf = SequenceReplayBuffer(capacity=1000, seed=0)
    buf.add_episode(make_synthetic_episode(20, OBS_SHAPE, fill=255))
    batch = buf.sample(batch_size=4, seq_len=50)
    assert batch["obs"].shape == (4, 50, 3, 64, 64)
    expected_mask = torch.zeros(50)
    expected_mask[:20] = 1.0
    assert torch.all(batch["mask"] == expected_mask.unsqueeze(0))
    # Padded observations are zeros -> normalize to -0.5; real frames -> +0.5.
    assert torch.allclose(batch["obs"][:, :20], torch.tensor(0.5))
    assert torch.allclose(batch["obs"][:, 20:], torch.tensor(-0.5))
    # Padded flags/rewards are zeros.
    assert torch.all(batch["reward"][:, 20:] == 0)
    assert not batch["is_first"][:, 20:].any()


def test_is_first_only_at_episode_start():
    buf = SequenceReplayBuffer(capacity=1000, seed=0)
    buf.add_episode(make_synthetic_episode(30, OBS_SHAPE))
    batch = buf.sample(batch_size=8, seq_len=30)
    # Window must start at 0 (episode length == seq_len), so is_first at t=0 only.
    assert torch.all(batch["is_first"][:, 0])
    assert not batch["is_first"][:, 1:].any()


def test_capacity_fifo_eviction():
    buf = SequenceReplayBuffer(capacity=250, seed=0)
    for i in range(4):  # 4 x 100 steps -> keeps only the 2 newest under cap 250
        ep = make_synthetic_episode(100, OBS_SHAPE, fill=i)
        buf.add_episode(ep)
    assert buf.num_steps == 200
    assert buf.num_episodes == 2
    # Oldest episodes (fill 0 and 1) were evicted; remaining fills are 2 and 3.
    remaining_fills = sorted(int(ep["obs"][0, 0, 0, 0]) for ep in buf._episodes)
    assert remaining_fills == [2, 3]


def test_sample_empty_buffer_raises():
    buf = SequenceReplayBuffer(capacity=100)
    with pytest.raises(RuntimeError):
        buf.sample(batch_size=1, seq_len=10)


def test_save_load_roundtrip(tmp_path):
    buf = SequenceReplayBuffer(capacity=10_000, seed=0)
    buf.add_episode(make_synthetic_episode(40, OBS_SHAPE))
    buf.add_episode(make_synthetic_episode(70, OBS_SHAPE))
    buf.save(tmp_path / "buffer")

    loaded = SequenceReplayBuffer.load(tmp_path / "buffer", capacity=10_000, seed=0)
    assert loaded.num_episodes == 2
    assert loaded.num_steps == 110
    for orig, new in zip(buf._episodes, loaded._episodes):
        for key in orig:
            np.testing.assert_array_equal(orig[key], new[key])
