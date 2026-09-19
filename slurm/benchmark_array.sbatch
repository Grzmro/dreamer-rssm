#!/bin/bash
# Same benchmark, but 3 seeds run in parallel as a Slurm job array (one GPU
# each) instead of sequentially inside a single job. Faster wall-clock at
# the cost of 3 GPUs reserved at once — use this over benchmark.sbatch when
# your grant has GPU quota to spare. Results land in the same shared CSV
# tree, so viz/benchmark_comparison.py sees all seeds regardless of which
# script produced them.
#
# Usage: sbatch slurm/benchmark_array.sbatch
#        ENV=atari_breakout sbatch --export=ALL,ENV slurm/benchmark_array.sbatch
#
# ENV is a config group name from configs/env/ (atari_pong, atari_breakout,
# carracing). Each env writes to its own <BENCH_DIR>/<env>/ tree.
#
# BENCHMARK picks a preset from configs/benchmark/ (default: sample-matched,
# all agents), e.g. BENCHMARK=time_3h for the longer, time-capped baselines.
# BENCH_DIR picks where the CSVs go (default experiments/benchmark); give each
# campaign its own directory, since BenchmarkLogger appends:
#   ENV=atari_breakout BENCH_DIR=$SCRATCH/dreamer-rssm/experiments/benchmark_breakout400k \
#       sbatch --export=ALL,ENV,BENCH_DIR slurm/benchmark_array.sbatch

#SBATCH --job-name=dreamer-benchmark
#SBATCH --account=plglscclass26-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
# 400k steps at the measured rate (60k in 1:17 on an A100, Slurm 2806135)
# is ~8.5 h for Dreamer alone, before PPO and DQN. 12 h left no margin.
#SBATCH --time=24:00:00
#SBATCH --array=0-2
#SBATCH --output=%x-%A_%a.log

set -euo pipefail

source "${SCRATCH}/venvs/dreamer-rssm/bin/activate"
export SDL_VIDEODRIVER=dummy

cd "${SCRATCH}/dreamer-rssm"

ENV="${ENV:-atari_pong}"
STEPS="${STEPS:-400000}"
BENCHMARK="${BENCHMARK:-default}"
BENCH_DIR="${BENCH_DIR:-${SCRATCH}/dreamer-rssm/experiments/benchmark}"
echo "[sbatch] env=${ENV} benchmark=${BENCHMARK} steps=${STEPS} seed=${SLURM_ARRAY_TASK_ID} bench_dir=${BENCH_DIR}"

RUN_DIR="${SCRATCH}/dreamer-rssm/experiments/athena_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
python train/run_benchmark.py \
    "env=${ENV}" \
    "benchmark=${BENCHMARK}" \
    "benchmark.total_env_steps=${STEPS}" \
    "benchmark.seeds=[${SLURM_ARRAY_TASK_ID}]" \
    baselines.benchmark_dir="${BENCH_DIR}" \
    hydra.run.dir="${RUN_DIR}"

# Only render plots once, after the last array task to finish sees every CSV.
echo "Seed ${SLURM_ARRAY_TASK_ID} done. Run this manually once all 3 array tasks finish:"
echo "  python viz/benchmark_comparison.py viz.benchmark_root=${BENCH_DIR}"
