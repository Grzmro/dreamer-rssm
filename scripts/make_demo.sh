#!/usr/bin/env bash
# Build the whole visual demo from one command: train a shared-budget
# benchmark (Dreamer + PPO + DQN), then render every artifact and the
# showcase page around them.
#
# The defaults below are a SMOKE-SCALE budget that finishes on a CPU in
# well under an hour; they produce a working demo, not a result. For
# something worth showing as evidence, pass a real budget and seeds:
#
#   STEPS=100000 SEEDS="[0,1,2]" TIME_LIMIT=1000 scripts/make_demo.sh
#
# Everything lands under $ROOT (default experiments/demo) and nothing here
# overwrites an existing benchmark tree unless you point $ROOT at one.
set -euo pipefail

ROOT="${ROOT:-experiments/demo}"
STEPS="${STEPS:-3000}"
SEEDS="${SEEDS:-[0]}"
TIME_LIMIT="${TIME_LIMIT:-100}"
DEVICE="${DEVICE:-cpu}"
TRAIN_RATIO="${TRAIN_RATIO:-0.2}"
EPISODES="${EPISODES:-2}"          # gameplay episodes per agent (best kept)
export PYTHONPATH="${PYTHONPATH:-.}"

echo "=== 1/5 benchmark: $STEPS env steps, seeds $SEEDS, into $ROOT"
python train/run_benchmark.py \
    env.time_limit="$TIME_LIMIT" \
    benchmark.total_env_steps="$STEPS" "benchmark.seeds=$SEEDS" \
    "benchmark.agents=[dreamer,ppo,dqn]" \
    baselines.benchmark_dir="$ROOT" \
    baselines.checkpoint_dir="$ROOT/baseline_ckpts" \
    baselines.device="$DEVICE" baselines.ppo.num_envs="${PPO_ENVS:-2}" \
    train_dreamer.device="$DEVICE" train_dreamer.train_ratio="$TRAIN_RATIO"

CKPT="$ROOT/runs/dreamer_seed0/checkpoints/dreamer_final.pt"

echo "=== 2/5 gameplay GIFs (Dreamer vs baselines vs random)"
python viz/gameplay_gif.py \
    "+viz.gameplay_agents=[\"dreamer:Dreamer=$CKPT\",\"ppo:PPO=$ROOT/baseline_ckpts/ppo_seed0.pt\",\"dqn:DQN=$ROOT/baseline_ckpts/dqn_seed0.pt\",\"random:Random\"]" \
    env.time_limit="$TIME_LIMIT" viz.gameplay_episodes="$EPISODES" \
    viz.gameplay_max_steps="$TIME_LIMIT" viz.gameplay_out_dir="$ROOT/gameplay" \
    train_dreamer.device="$DEVICE"

echo "=== 3/5 real vs imagined"
python viz/real_vs_imagined_video.py viz.video_ckpt="$CKPT" \
    env.time_limit="$TIME_LIMIT" "viz.video_branch_points=[20,60]" \
    viz.video_horizon=15 viz.video_max_steps="$TIME_LIMIT" \
    viz.video_out_dir="$ROOT/videos"
python viz/dream_vs_real.py viz.run_dir="$ROOT/runs/dreamer_seed0"

echo "=== 4/5 world-model diagnostics"
python train/collect.py collect.num_steps=1200 env.time_limit="$TIME_LIMIT" \
    hydra.run.dir="$ROOT/collect"
for script in reconstruction open_loop_rollout; do
    python "viz/$script.py" viz.ckpt="$CKPT" viz.buffer_dir="$ROOT/collect/buffer" \
        viz.wm_out_dir="$ROOT/wm" train_wm.device="$DEVICE"
done

echo "=== 5/5 showcase page"
NOTE="${NOTE:-Budget: $STEPS env steps per agent, seeds $SEEDS, ${TIME_LIMIT}-step episodes on $DEVICE. Read the budget before the numbers.}"
python viz/make_showcase.py \
    "viz.showcase_note=\"$NOTE\"" \
    viz.benchmark_root="$ROOT" viz.gameplay_out_dir="$ROOT/gameplay" \
    viz.video_out_dir="$ROOT/videos" viz.wm_out_dir="$ROOT/wm" \
    viz.run_dir="$ROOT/runs/dreamer_seed0" viz.out_dir="$ROOT/sanity" \
    viz.showcase_dir="$ROOT/showcase" viz.showcase_max_per_section=8

echo "done -> $ROOT/showcase/index.html"
