"""Tests for the Phase 4 aggregation/summary helpers on synthetic runs."""

import numpy as np
import pytest

from viz.ablation_summary import final_return, steps_to_threshold, summarize_group
from viz.learning_curves import aggregate_seeds


def fake_run(n=50, start=-21.0, end=0.0, step=1000, noise=0.0, rng=None):
    ret = np.linspace(start, end, n)
    if noise and rng is not None:
        ret = ret + rng.normal(0, noise, n)
    return {
        "env_step": np.arange(1, n + 1) * step,
        "wall_time_s": np.arange(1, n + 1, dtype=float),
        "episode_return": ret,
        "episode_length": np.full(n, 1000),
    }


def test_aggregate_seeds_mean_of_identical_runs_is_the_run():
    runs = [fake_run(), fake_run()]
    grid, mean, std = aggregate_seeds(runs, window=1)
    assert np.allclose(std, 0.0, atol=1e-9)
    assert mean[0] == pytest.approx(-21.0)
    assert mean[-1] == pytest.approx(0.0)


def test_aggregate_seeds_clips_to_shortest_run():
    runs = [fake_run(n=50), fake_run(n=25)]
    grid, mean, std = aggregate_seeds(runs, window=1)
    assert grid[-1] <= 25 * 1000  # no extrapolation beyond the shorter seed


def test_final_return_stats():
    runs = [fake_run(end=0.0), fake_run(end=-2.0)]
    mean, std = final_return(runs, last_k=1)
    assert mean == pytest.approx(-1.0)
    assert std == pytest.approx(1.0)


def test_steps_to_threshold():
    run = fake_run(n=10, start=-10, end=-1, step=100)  # rolling window 1
    s, reached, total = steps_to_threshold([run], threshold=-5.0, window=1)
    assert s is not None and 500 <= s <= 700
    assert (reached, total) == (1, 1)

    s, reached, total = steps_to_threshold([run], threshold=99.0, window=1)
    assert s is None and (reached, total) == (0, 1)


def test_steps_to_threshold_reports_partial_seed_coverage():
    """The mean covers only seeds that crossed; the counts must say so."""
    reaches = fake_run(n=10, start=-10, end=-1, step=100)
    never = [fake_run(n=10, start=-10, end=-9, step=100) for _ in range(2)]
    s, reached, total = steps_to_threshold([reaches, *never], threshold=-5.0, window=1)
    assert s is not None  # averaged over the single seed that made it
    assert (reached, total) == (1, 3)


def test_steps_to_threshold_ignores_runs_with_no_episodes():
    empty = {k: np.array([]) for k in ("env_step", "episode_return")}
    s, reached, total = steps_to_threshold([empty], threshold=0.0, window=10)
    assert s is None and (reached, total) == (0, 1)


def test_steps_to_threshold_on_wall_clock_axis():
    run = fake_run(n=10, start=-10, end=-1, step=100)  # wall_time_s = 1..10
    s, _, _ = steps_to_threshold([run], threshold=-5.0, window=1, x_key="wall_time_s")
    assert s is not None and 5 <= s <= 7


def test_sample_efficiency_ratio_against_dreamer(tmp_path):
    from viz.sample_efficiency import sample_efficiency_table, write_table

    agents = {
        # Dreamer: reaches +10 by the end of 100 x 1k = 100k steps.
        "dreamer": [fake_run(n=100, start=-21, end=10, step=1000) for _ in range(2)],
        # PPO: same shape of curve but 10x the steps per episode.
        "ppo": [fake_run(n=100, start=-21, end=10, step=10_000) for _ in range(2)],
        # DQN: never gets there within its run -> n/a, not a made-up number.
        "dqn": [fake_run(n=100, start=-21, end=0, step=10_000) for _ in range(2)],
    }
    rows, threshold = sample_efficiency_table(agents, window=1)
    # Default target = Dreamer's final return (mean of its last 10 episodes).
    assert threshold == pytest.approx(final_return(agents["dreamer"])[0])
    by = {r["agent"]: r for r in rows}
    assert by["dreamer"]["samples_vs_dreamer"] == "1.0x"
    assert by["ppo"]["samples_vs_dreamer"] == "10.0x"
    assert by["ppo"]["seeds_reaching"] == "2/2"
    assert by["dqn"]["steps_to_target"] == "n/a"
    assert by["dqn"]["seeds_reaching"] == "0/2"
    assert by["dqn"]["steps_run"] == 1_000_000

    write_table(rows, threshold, tmp_path / "eff")
    assert f"Target return: {threshold:+.2f}" in (tmp_path / "eff.md").read_text(encoding="utf-8")
    assert (tmp_path / "eff.csv").exists()


def test_sample_efficiency_needs_a_target():
    from viz.sample_efficiency import sample_efficiency_table

    with pytest.raises(ValueError, match="reference agent"):
        sample_efficiency_table({"ppo": [fake_run()]})
    rows, threshold = sample_efficiency_table({"ppo": [fake_run()]}, threshold=-5.0, window=1)
    assert threshold == -5.0 and rows[0]["samples_vs_dreamer"] == "n/a"


def test_summarize_group_writes_outputs(tmp_path):
    rng = np.random.default_rng(0)
    agents = {
        "dreamer": [fake_run(end=0.0, noise=0.3, rng=rng) for _ in range(2)],
        "dreamer-nofreenats": [fake_run(end=-15.0, noise=0.3, rng=rng) for _ in range(2)],
    }
    rows = summarize_group("loss_variants", ["dreamer", "dreamer-nofreenats"],
                           agents, tmp_path)
    assert rows is not None and len(rows) == 2
    base = next(r for r in rows if r["variant"] == "dreamer")
    ablated = next(r for r in rows if r["variant"] == "dreamer-nofreenats")
    assert base["final_return_mean"] > ablated["final_return_mean"]
    # Seed coverage of the steps-to-threshold column must be reported.
    assert base["seeds_reaching"].endswith("/2")
    assert "seeds_reaching" in (tmp_path / "ablation_loss_variants.md").read_text()
    for suffix in ("png", "md", "csv"):
        assert (tmp_path / f"ablation_loss_variants.{suffix}").exists()


def test_summarize_group_skips_single_variant(tmp_path):
    agents = {"dreamer": [fake_run()]}
    assert summarize_group("horizon", ["dreamer", "dreamer-H5"], agents, tmp_path) is None


def test_agent_colors_separate_variants_within_a_family():
    """Every "dreamer-*" ablation used to render in the same red."""
    from viz.benchmark_comparison import agent_color

    assert agent_color("dreamer") == "tab:red"
    assert agent_color("dreamer-warmstart") == "tab:purple"  # exact key wins
    assert agent_color("nonesuch") is None

    variants = ["dreamer-tr0.1", "dreamer-tr1.0", "dreamer-ent1e-4", "dreamer-ent1e-3"]
    colors = [agent_color(v) for v in variants]
    assert len({tuple(c) for c in colors}) == len(variants)
    assert all(c != "tab:red" for c in colors)
    # Stable across calls/processes (hashlib, not the randomized builtin hash).
    assert colors == [agent_color(v) for v in variants]
