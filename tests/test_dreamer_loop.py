"""Regression tests for the online collection loop (train/dreamer_loop.py).

``DummyImageEnv`` encodes its inner step counter in every pixel, so the frame
handed to the policy identifies exactly which env step it came from — enough
to pin down the observation bookkeeping around episode boundaries without
running any gradient updates (``train_ratio=0``).
"""

from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir

from envs.wrappers import NormalizeObservation, ResizeObservation
from tests.conftest import DummyImageEnv

EPISODE_LEN = 6


def _dummy_env(_env_cfg=None):
    """Phase 0 chain on the dummy base env: resize to 64x64, then normalize."""
    env = DummyImageEnv(episode_len=EPISODE_LEN)
    env = ResizeObservation(env, (64, 64))
    return NormalizeObservation(env)


def _tiny_cfg(total_env_steps: int):
    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        return compose(
            config_name="config",
            overrides=[
                "train_dreamer.device=cpu",
                "train_dreamer.prefill_steps=0",
                f"train_dreamer.total_env_steps={total_env_steps}",
                "train_dreamer.train_ratio=0",  # collection only, no updates
                "train_dreamer.checkpoint_interval=1000000",
                "model.cnn_depth=4",
                "model.deter_dim=16",
                "model.hidden_dim=16",
                "model.stoch_groups=4",
                "model.stoch_classes=4",
                "model.head_hidden_dim=16",
                "model.head_layers=1",
                "agent.actor.hidden_dim=16",
                "agent.actor.num_layers=1",
                "agent.critic.hidden_dim=16",
                "agent.critic.num_layers=1",
            ],
        )


def test_policy_acts_on_the_reset_frame_after_an_episode_ends(tmp_path, monkeypatch):
    """The first action of a new episode must come from the RESET frame.

    Regression: the loop used to discard the observation returned by
    ``env.reset()``, so the policy built its fresh belief state from the
    terminal frame of the episode that had just finished — while the replay
    buffer stored the reset frame as ``obs[0]``.
    """
    from train import dreamer_loop

    monkeypatch.setattr(dreamer_loop, "make_env", _dummy_env)

    seen: list[np.ndarray] = []
    original_call = dreamer_loop.OnlinePolicy.__call__

    def recording_call(self, obs):
        seen.append(np.asarray(obs).copy())
        return original_call(self, obs)

    monkeypatch.setattr(dreamer_loop.OnlinePolicy, "__call__", recording_call)
    dreamer_loop.train_dreamer(_tiny_cfg(EPISODE_LEN + 3), output_dir=tmp_path)

    assert len(seen) > EPISODE_LEN, "loop did not run past the episode boundary"

    # DummyImageEnv pixels are all `t`; NormalizeObservation maps them to
    # t/255 - 0.5. The reset frame is t=0, the terminal frame is t=EPISODE_LEN.
    def frame_value(t: int) -> float:
        return t / 255.0 - 0.5

    # Within the first episode the policy sees consecutive frames from t=0.
    for t in range(EPISODE_LEN):
        assert np.allclose(seen[t], frame_value(t), atol=1e-6), f"step {t}"

    first_of_new_episode = seen[EPISODE_LEN]
    assert np.allclose(first_of_new_episode, frame_value(0), atol=1e-6)
    assert not np.allclose(first_of_new_episode, frame_value(EPISODE_LEN), atol=1e-6)
