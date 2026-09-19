"""Tests for the gameplay/summary/showcase visualization layer (Phase 4).

Everything here runs on synthetic frames and CSVs: no env, no checkpoint,
no training.
"""

import csv
import json

import numpy as np
import pytest
from omegaconf import OmegaConf

from viz.gameplay_gif import compose_strip, draw_panel, parse_spec, to_rgb
from viz.summary_figure import summarize_agents


def fake_record(label, n, ret, width=64, channels=3):
    frames = [
        np.full((width, width, channels), i % 255, np.uint8) for i in range(n)
    ]
    rewards = np.zeros(n - 1, np.float32)
    rewards[-1] = ret
    return {
        "frames": frames,
        "rewards": rewards,
        "score": np.concatenate([[0.0], np.cumsum(rewards)]),
        "episode_return": float(ret),
        "length": n - 1,
        "label": label,
    }


def fake_run(n=30, start=-21.0, end=-5.0, step=1000, wall=600.0):
    return {
        "env_step": np.arange(1, n + 1) * step,
        "wall_time_s": np.linspace(0, wall, n),
        "episode_return": np.linspace(start, end, n),
        "episode_length": np.full(n, 800),
    }


@pytest.mark.parametrize(
    "spec, expected",
    [
        ("random", ("random", "random", None)),
        ("ppo=a/b.pt", ("ppo", "ppo", "a/b.pt")),
        ("dreamer:Dreamer 100k=x.pt", ("dreamer", "Dreamer 100k", "x.pt")),
    ],
)
def test_parse_spec(spec, expected):
    assert parse_spec(spec) == expected


def test_parse_spec_rejects_unknown_kind_and_missing_checkpoint():
    with pytest.raises(ValueError, match="unknown agent kind"):
        parse_spec("a2c=x.pt")
    with pytest.raises(ValueError, match="needs a checkpoint"):
        parse_spec("dqn")


def test_to_rgb_expands_grayscale():
    """Baselines see grayscale frames; both inputs must compose together."""
    assert to_rgb(np.zeros((64, 64, 1), np.uint8)).shape == (64, 64, 3)
    assert to_rgb(np.zeros((64, 64), np.uint8)).shape == (64, 64, 3)
    assert to_rgb(np.zeros((64, 64, 3), np.uint8)).shape == (64, 64, 3)


def test_draw_panel_adds_header_and_footer():
    panel = draw_panel(np.zeros((64, 64, 3), np.uint8), "dreamer", ["a", "b"], scale=2)
    assert panel.shape[1] == 128
    assert panel.shape[0] > 128  # title bar + two status lines


def test_compose_strip_pads_to_the_longest_episode():
    """A baseline that dies early must stay on screen, not shorten the strip."""
    records = [fake_record("dreamer", 20, 3.0), fake_record("ppo", 8, -21.0)]
    strip = compose_strip(records, scale=2)
    assert len(strip) == 20
    assert all(f.shape == strip[0].shape for f in strip)


def test_compose_strip_mixes_grayscale_and_rgb_panels():
    records = [fake_record("dreamer", 6, 1.0), fake_record("dqn", 6, -21.0, channels=1)]
    strip = compose_strip(records, scale=2)
    assert len(strip) == 6 and strip[0].shape[2] == 3


def test_short_runs_still_get_a_curve():
    """A run with fewer episodes than the window used to vanish from the
    plot: rolling(n=8, w=10) is one point, and one point cannot interpolate."""
    from viz.learning_curves import aggregate_seeds, effective_window

    runs = [fake_run(n=8)]
    assert effective_window(runs, 10) == 7
    grid, mean, std = aggregate_seeds(runs, window=10)
    assert len(grid) > 1 and np.all(np.isfinite(mean))
    assert effective_window([fake_run(n=100)], 10) == 10  # never grows the window


def test_summarize_agents_ranks_by_final_return():
    agents = {
        "ppo": [fake_run(end=-20.0, wall=300.0)],
        "dreamer": [fake_run(end=-2.0, wall=9000.0), fake_run(end=-4.0, wall=9000.0)],
    }
    rows = summarize_agents(agents, window=1, last_k=1)
    assert [r["agent"] for r in rows] == ["dreamer", "ppo"]
    assert rows[0]["seeds"] == 2
    assert rows[0]["final_return_mean"] == pytest.approx(-3.0)
    assert rows[0]["wall_clock_min"] == pytest.approx(150.0)
    assert rows[1]["wall_clock_min"] == pytest.approx(5.0)


def _write_csv(path, run, header=("env_step", "wall_time_s", "episode_return", "episode_length")):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in range(len(run["env_step"])):
            w.writerow([int(run["env_step"][i]), f"{run['wall_time_s'][i]:.3f}",
                        run["episode_return"][i], int(run["episode_length"][i])])


def test_make_summary_writes_figure_and_table(tmp_path):
    from viz.summary_figure import make_summary

    root = tmp_path / "benchmark"
    _write_csv(root / "ALE_Pong-v5" / "dreamer_seed0.csv", fake_run(end=-2.0))
    _write_csv(root / "ALE_Pong-v5" / "ppo_seed0.csv", fake_run(end=-20.0))
    written = make_summary(root, window=2, last_k=3)
    names = {p.name for p in written}
    assert names == {"ALE_Pong-v5_summary.png", "ALE_Pong-v5_summary.md",
                     "ALE_Pong-v5_summary.csv"}
    table = (root / "plots" / "ALE_Pong-v5_summary.md").read_text()
    assert "dreamer" in table and "ppo" in table


def _showcase_cfg(tmp_path, **over):
    viz = {
        "benchmark_root": str(tmp_path / "benchmark"),
        "gameplay_out_dir": str(tmp_path / "gameplay"),
        "video_out_dir": str(tmp_path / "videos"),
        "wm_out_dir": str(tmp_path / "phase1"),
        "out_dir": str(tmp_path / "sanity"),
        "run_dir": None,
        "showcase_dir": str(tmp_path / "showcase"),
        "showcase_regenerate": False,
        "showcase_max_per_section": 6,
        "dream_window": 2,
    }
    viz.update(over)
    return OmegaConf.create({"env": {"name": "ALE/Pong-v5"}, "viz": viz})


def test_showcase_reports_missing_sections_instead_of_dropping_them(tmp_path):
    from viz.make_showcase import build

    index = build(_showcase_cfg(tmp_path))
    page = index.read_text(encoding="utf-8")
    assert "Not generated yet" in page
    assert "viz/gameplay_gif.py" in page  # the command that fills the gap
    assert "Not generated yet" in (index.parent / "SHOWCASE.md").read_text()


def test_showcase_embeds_found_artifacts_with_captions(tmp_path):
    from viz.make_showcase import build

    gameplay = tmp_path / "gameplay"
    gameplay.mkdir(parents=True)
    (gameplay / "pong_v5_compare.gif").write_bytes(b"GIF89a")
    (gameplay / "gameplay.json").write_text(json.dumps({
        "env": "ALE/Pong-v5", "action_repeat": 4, "seed": 0,
        "episodes_per_agent": 2,
        "agents": [{"label": "dreamer", "note": "n", "kept_return": 3.0,
                    "kept_length": 700, "all_returns": [3.0, -1.0]}],
    }), encoding="utf-8")

    index = build(_showcase_cfg(tmp_path))
    page = index.read_text(encoding="utf-8")
    assert "assets/pong_v5_compare.gif" in page
    assert "best of 2" in page  # the caption states the denominator
    assert (index.parent / "assets" / "pong_v5_compare.gif").exists()


def test_baseline_checkpoint_roundtrip_replays_the_policy(tmp_path):
    """A saved baseline must come back as a callable policy, or the GIFs
    have nothing but Dreamer to show."""
    import gymnasium as gym
    import torch
    from omegaconf import OmegaConf

    from baselines.dqn_rainbow import DuelingQNetwork
    from baselines.policies import load_baseline_policy, save_baseline_checkpoint

    obs_shape, num_actions = (4, 64, 64), 6
    q_net = DuelingQNetwork(obs_shape[0], num_actions, dueling=True)
    path = save_baseline_checkpoint(
        tmp_path / "dqn_seed0.pt", kind="dqn", model=q_net,
        cfg=OmegaConf.create({"seed": 0}), seed=0, env_step=1000,
        obs_shape=obs_shape, action_dim=num_actions, extra={"dueling": True},
    )
    policy = load_baseline_policy(path, torch.device("cpu"), epsilon=0.0)
    obs = np.zeros(obs_shape, np.float32)
    action = policy(obs)
    assert isinstance(action, int) and 0 <= action < num_actions
    assert policy(obs) == action  # greedy: same state -> same action
    assert policy.meta["env_step"] == 1000

    greedy = q_net(torch.as_tensor(obs)[None]).argmax(dim=1).item()
    assert action == greedy  # the loaded weights are the trained ones

    explorer = load_baseline_policy(path, torch.device("cpu"), epsilon=1.0, seed=0)
    assert {explorer(obs) for _ in range(20)} - set(range(num_actions)) == set()


def test_random_policy_does_not_consume_the_env_rng(tmp_path):
    """Two random policies seeded alike must replay identically."""
    import gymnasium as gym

    from baselines.policies import random_policy

    space = gym.spaces.Discrete(6)
    space.seed(123)
    a = [random_policy(space, seed=7)(None) for _ in range(10)]
    b = [random_policy(space, seed=7)(None) for _ in range(10)]
    assert a == b


def test_showcase_regenerate_reports_a_missing_benchmark(tmp_path):
    from viz.make_showcase import build

    index = build(_showcase_cfg(tmp_path, showcase_regenerate=True))
    assert "could not be regenerated" in index.read_text(encoding="utf-8")
