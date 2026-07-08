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
    s = steps_to_threshold([run], threshold=-5.0, window=1)
    assert s is not None and 500 <= s <= 700
    assert steps_to_threshold([run], threshold=99.0, window=1) is None


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
    for suffix in ("png", "md", "csv"):
        assert (tmp_path / f"ablation_loss_variants.{suffix}").exists()


def test_summarize_group_skips_single_variant(tmp_path):
    agents = {"dreamer": [fake_run()]}
    assert summarize_group("horizon", ["dreamer", "dreamer-H5"], agents, tmp_path) is None
