"""Wall-clock budget: the second axis of the Phase 3+ benchmark protocol."""

from __future__ import annotations

import time

import pytest
from hydra import compose, initialize

from train.budget import TrainingBudget
from train.run_benchmark import _agent_steps, _fmt_duration


def test_step_limit_without_time_budget():
    b = TrainingBudget(max_env_steps=100)
    assert not b.exhausted(99)
    assert b.exhausted(100)
    assert b.stopped_on == "steps"


def test_out_of_time_is_false_without_a_budget():
    """Step-bounded loops must be untouched when no time budget is configured."""
    b = TrainingBudget(max_env_steps=10**9)
    assert b.out_of_time() is False
    assert b.remaining_s() is None


@pytest.mark.parametrize("zero", [0, 0.0, None])
def test_non_positive_budget_means_no_budget(zero):
    assert TrainingBudget(10, zero).time_budget_s is None


def test_time_limit_fires_before_the_step_limit():
    b = TrainingBudget(max_env_steps=10**9, time_budget_s=0.05)
    assert not b.out_of_time()
    # Margin well above a clock tick: time.monotonic() is ~15.6 ms coarse on
    # Windows, so a 10 ms margin made this flaky.
    time.sleep(0.15)
    assert b.out_of_time()
    assert b.stopped_on == "time"
    # A while-loop agent must see the same verdict through exhausted().
    assert b.exhausted(0)


def test_summary_flags_an_unspent_budget():
    """Stopping on steps with a time budget set means the comparison is unequal."""
    b = TrainingBudget(max_env_steps=1, time_budget_s=3600)
    b.exhausted(1)
    assert b.stopped_on == "steps"
    assert "not spent" in b.summary(1)

    b2 = TrainingBudget(max_env_steps=10**9, time_budget_s=0.01)
    time.sleep(0.1)
    b2.out_of_time()
    assert "stopped on TIME" in b2.summary(123)


def test_agent_env_steps_overrides_only_the_named_agents():
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(
            config_name="config",
            overrides=[
                "benchmark.total_env_steps=100000",
                "benchmark.agent_env_steps={dreamer: 400000, ppo: 33000000}",
            ],
        )
    assert _agent_steps(cfg.benchmark, "dreamer") == 400000
    assert _agent_steps(cfg.benchmark, "ppo") == 33000000
    assert _agent_steps(cfg.benchmark, "dqn") == 100000  # falls back


def test_fmt_duration_does_not_round_short_budgets_to_zero():
    assert _fmt_duration(25) == "25 s"
    assert _fmt_duration(600) == "10.0 min"
    assert _fmt_duration(33840) == "9.4 h"


def test_sample_matched_protocol_is_unchanged_by_default():
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="config", overrides=["benchmark.total_env_steps=12345"])
    assert cfg.benchmark.time_budget_s is None
    assert cfg.benchmark.agent_env_steps is None
    for agent in ("dreamer", "ppo", "dqn", "sac"):
        assert _agent_steps(cfg.benchmark, agent) == 12345


def test_time_3h_preset_runs_only_the_baselines_on_the_clock():
    """The long-budget baseline campaign (RESULTS.md §F), as the sbatch composes it."""
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(
            config_name="config",
            overrides=["benchmark=time_3h", "benchmark.total_env_steps=400000",
                       "benchmark.seeds=[1]"],
        )
    b = cfg.benchmark
    # Dreamer's recorded 400k run is reused (truncated on wall_time_s), not re-run.
    assert list(b.agents) == ["ppo", "dqn"]
    assert b.time_budget_s == 10800
    assert _agent_steps(b, "ppo") == 11_000_000
    assert _agent_steps(b, "dqn") == 4_500_000
    assert list(b.seeds) == [1]
