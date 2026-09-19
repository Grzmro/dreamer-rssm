| agent | seeds | env_steps | final_return_mean | final_return_std | best_episode | wall_clock_min |
|---|---|---|---|---|---|---|
| ppo | 1 | 2624 | -1.6 | 0.0 | 0.0 | 0.1 |
| dreamer | 1 | 3000 | -1.8 | 0.0 | 0.0 | 31.3 |
| dqn | 1 | 3000 | -2.0 | 0.0 | 1.0 | 0.2 |

final_return = mean over seeds of each run's last-10-episode mean; wall_clock = mean seconds of each seed's run, in minutes.
