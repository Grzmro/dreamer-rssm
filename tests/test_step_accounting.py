"""The benchmark's x-axis: env steps must be counted the way the docs claim.

Every sample-efficiency claim in README/RESULTS.md rests on one promise —
each agent consumes the same number of environment interactions, counted
after action repeat, prefill included. Nothing used to test that promise:
tests/test_baseline_protocol.py checks the agents see the same *env*, not
that they count the same *steps*. These tests pin the counting itself.
"""

import re

import torch

from tests.conftest import EPISODE_LEN, dummy_env_factory, tiny_dreamer_cfg


def _run_dreamer(monkeypatch, tmp_path, total_env_steps, **overrides):
    from train import dreamer_loop

    monkeypatch.setattr(dreamer_loop, "make_env", dummy_env_factory)
    final = dreamer_loop.train_dreamer(
        tiny_dreamer_cfg(total_env_steps, **overrides), output_dir=tmp_path
    )
    return torch.load(final, map_location="cpu", weights_only=False)


def test_train_ratio_accumulator_is_exact(monkeypatch, tmp_path):
    """``train_ratio`` updates per env step, accumulated fractionally.

    Hand-computation for the dummy env: prefill is 0 and the buffer starts
    empty, so updates begin at the step where the first episode lands in the
    buffer (env_step == EPISODE_LEN) and continue to the last step. That is
    ``N - EPISODE_LEN + 1`` contributing steps, and the accumulator fires
    ``floor(contributing * train_ratio)`` times.
    """
    n = 21
    contributing = n - EPISODE_LEN + 1  # 16

    ckpt = _run_dreamer(
        monkeypatch, tmp_path / "half", n, **{"train_dreamer.train_ratio": 0.5}
    )
    assert ckpt["env_step"] == n
    assert ckpt["update"] == int(contributing * 0.5)  # 8

    ckpt = _run_dreamer(
        monkeypatch, tmp_path / "quarter", n, **{"train_dreamer.train_ratio": 0.25}
    )
    assert ckpt["env_step"] == n
    assert ckpt["update"] == int(contributing * 0.25)  # 4


def test_policy_takes_over_only_at_an_episode_boundary(monkeypatch, tmp_path):
    """The random -> actor switch waits for a reset, so belief never starts cold.

    prefill_steps=9 falls mid-episode (episodes are 6 steps long), so the
    switch must happen at the boundary at step 12, not at step 9.
    """
    from train import dreamer_loop

    monkeypatch.setattr(dreamer_loop, "make_env", dummy_env_factory)

    calls: list[int] = []
    original = dreamer_loop.OnlinePolicy.__call__
    monkeypatch.setattr(
        dreamer_loop.OnlinePolicy,
        "__call__",
        lambda self, obs: (calls.append(1), original(self, obs))[1],
    )

    prefill, after_prefill = 9, 15
    dreamer_loop.train_dreamer(
        tiny_dreamer_cfg(
            after_prefill, **{"train_dreamer.prefill_steps": prefill}
        ),
        output_dir=tmp_path,
    )

    total_steps = prefill + after_prefill  # 24
    boundary = 12  # first episode end at or after step 9
    assert len(calls) == total_steps - boundary


def test_dreamer_consumes_exactly_the_shared_benchmark_budget(monkeypatch, tmp_path):
    """run_benchmark splits the budget into prefill + policy steps, not budget + prefill.

    Dreamer's random prefill counts on the shared axis like every other
    agent's warm-up, so the total must equal the budget exactly.
    """
    import envs
    from train import dreamer_loop, run_benchmark

    monkeypatch.setattr(dreamer_loop, "make_env", dummy_env_factory)
    monkeypatch.setattr(envs, "make_env", dummy_env_factory)

    budget = 20
    cfg = tiny_dreamer_cfg(
        0,
        **{
            "train_dreamer.prefill_steps": 5,
            "baselines.total_env_steps": budget,
        },
    )
    run_benchmark._run_agent("dreamer", cfg, tmp_path, seed=0)

    ckpt = torch.load(
        tmp_path / "runs" / "dreamer_seed0" / "checkpoints" / "dreamer_final.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert ckpt["env_step"] == budget


def test_ppo_never_exceeds_the_budget(monkeypatch, tmp_path, capsys):
    """PPO rounds the budget DOWN to whole rollout batches; it must not overshoot.

    ``num_iterations = total_steps // (num_steps * num_envs)`` means PPO can
    finish up to one batch short of the shared budget — documented here as
    deliberate, and bounded, so a future change that overshoots fails loudly.
    """
    import baselines.common as baseline_common
    from baselines.ppo import train_ppo

    monkeypatch.setattr(baseline_common, "make_env", dummy_env_factory)

    num_envs, num_steps, budget = 2, 4, 30
    batch = num_envs * num_steps
    cfg = tiny_dreamer_cfg(
        0,
        **{
            "baselines.total_env_steps": budget,
            "baselines.device": "cpu",
            "baselines.frame_stack": 1,
            "baselines.ppo.num_envs": num_envs,
            "baselines.ppo.num_steps": num_steps,
            "baselines.ppo.num_minibatches": 1,
            "baselines.ppo.update_epochs": 1,
            "baselines.benchmark_dir": str(tmp_path),
        },
    )
    train_ppo(cfg, seed=0)

    out = capsys.readouterr().out
    consumed = int(re.search(r"\[ppo\] done: (\d+) env steps", out).group(1))
    assert consumed <= budget
    assert budget - consumed < batch
