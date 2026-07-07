import numpy as np

from train.common_logger import BenchmarkLogger, load_benchmark, load_run


def test_round_trip(tmp_path):
    logger = BenchmarkLogger(tmp_path, "dreamer", "ALE/Pong-v5", seed=0)
    logger.log_episode(1000, -21.0, 824, wall_time_s=12.5)
    logger.log_episode(2000, -19.5, 900, wall_time_s=25.0)
    logger.close()

    run = load_run(logger.path)
    assert run["env_step"].tolist() == [1000, 2000]
    assert run["wall_time_s"].tolist() == [12.5, 25.0]
    assert run["episode_return"].tolist() == [-21.0, -19.5]
    assert run["episode_length"].tolist() == [824, 900]


def test_env_name_sanitized_and_appending(tmp_path):
    logger = BenchmarkLogger(tmp_path, "ppo", "ALE/Pong-v5", seed=3)
    assert "/" not in logger.path.name
    assert logger.path.parent.name == "ALE_Pong-v5"
    assert logger.path.name == "ppo_seed3.csv"
    logger.log_episode(10, 1.0, 10)
    logger.close()

    # Re-opening appends instead of clobbering (and writes no second header).
    logger2 = BenchmarkLogger(tmp_path, "ppo", "ALE/Pong-v5", seed=3)
    logger2.log_episode(20, 2.0, 10)
    logger2.close()
    run = load_run(logger2.path)
    assert run["env_step"].tolist() == [10, 20]


def test_load_benchmark_structure(tmp_path):
    for agent, seed in (("dreamer", 0), ("dreamer", 1), ("dqn", 0)):
        lg = BenchmarkLogger(tmp_path, agent, "ALE/Pong-v5", seed)
        lg.log_episode(100, float(seed), 50)
        lg.close()
    (tmp_path / "plots").mkdir()  # must be ignored by the loader

    bench = load_benchmark(tmp_path)
    assert set(bench.keys()) == {"ALE_Pong-v5"}
    assert set(bench["ALE_Pong-v5"].keys()) == {"dreamer", "dqn"}
    assert len(bench["ALE_Pong-v5"]["dreamer"]) == 2
    assert np.isclose(bench["ALE_Pong-v5"]["dqn"][0]["episode_return"][0], 0.0)
