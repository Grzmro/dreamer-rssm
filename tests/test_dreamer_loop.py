"""Regression tests for the online collection loop (train/dreamer_loop.py).

``DummyImageEnv`` encodes its inner step counter in every pixel, so the frame
handed to the policy identifies exactly which env step it came from — enough
to pin down the observation bookkeeping around episode boundaries without
running any gradient updates (``train_ratio=0``).
"""

import numpy as np

from tests.conftest import (
    EPISODE_LEN,
    dummy_box_env_factory as _dummy_box_env,
    dummy_env_factory as _dummy_env,
    tiny_dreamer_cfg as _tiny_cfg,
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


def test_continuous_actions_stay_inside_the_action_space(tmp_path, monkeypatch):
    """Every emitted continuous action must be a legal action for the env.

    Regression: the actor emits tanh output in [-1, 1]^A and OnlinePolicy fed
    it straight to env.step(). On an env with asymmetric bounds (CarRacing:
    gas and brake live in [0, 1]) that is out of range — gymnasium clipped it
    internally while the replay buffer kept the unclipped value, so the world
    model learned from an action the env never applied. The factory now
    rescales continuous spaces to [-1, 1], making actor, env and buffer agree.
    """
    from train import dreamer_loop

    monkeypatch.setattr(dreamer_loop, "make_env", _dummy_box_env)

    env = _dummy_box_env()
    assert np.allclose(env.action_space.low, -1.0)
    assert np.allclose(env.action_space.high, 1.0)

    emitted: list[np.ndarray] = []
    original_call = dreamer_loop.OnlinePolicy.__call__

    def recording_call(self, obs):
        action = original_call(self, obs)
        emitted.append(np.asarray(action))
        return action

    monkeypatch.setattr(dreamer_loop.OnlinePolicy, "__call__", recording_call)
    dreamer_loop.train_dreamer(_tiny_cfg(EPISODE_LEN + 3), output_dir=tmp_path)

    assert emitted, "policy never acted"
    for action in emitted:
        assert env.action_space.contains(action.astype(np.float32)), action
