#!/bin/bash
# Full Phase 3 benchmark on one Athena A100. Edit --account before submitting.
#
# Usage:
#   sbatch slurm/benchmark.sbatch
#   sbatch --time=24:00:00 slurm/benchmark.sbatch      # override walltime
#   ENV=atari_breakout STEPS=400000 SEEDS='[0,1,2]' \
#       sbatch --export=ALL,ENV,STEPS,SEEDS slurm/benchmark.sbatch
#
# ENV is a config group name from configs/env/ (atari_pong, atari_breakout,
# carracing). Each env writes to its own experiments/benchmark/<env>/ tree, so
# a second game cannot append into the first one's CSVs.
#
# BENCHMARK (preset in configs/benchmark/, default "default") and BENCH_DIR
# (CSV root, default experiments/benchmark) work as in benchmark_array.sbatch.
#
# Check status from a phone SSH session:
#   squeue --me
#   tail -f dreamer-benchmark-<jobid>.log

#SBATCH --job-name=dreamer-benchmark
#SBATCH --account=plglscclass26-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.log

set -euo pipefail

source "${SCRATCH}/venvs/dreamer-rssm/bin/activate"
export SDL_VIDEODRIVER=dummy   # headless Box2D (CarRacing), harmless for Atari

cd "${SCRATCH}/dreamer-rssm"

ENV="${ENV:-atari_pong}"
STEPS="${STEPS:-400000}"
SEEDS="${SEEDS:-[0]}"
BENCHMARK="${BENCHMARK:-default}"
BENCH_DIR="${BENCH_DIR:-${SCRATCH}/dreamer-rssm/experiments/benchmark}"
echo "[sbatch] env=${ENV} benchmark=${BENCHMARK} steps=${STEPS} seeds=${SEEDS} bench_dir=${BENCH_DIR}"

RUN_DIR="${SCRATCH}/dreamer-rssm/experiments/athena_${SLURM_JOB_ID}"
python train/run_benchmark.py \
    "env=${ENV}" \
    "benchmark=${BENCHMARK}" \
    "benchmark.total_env_steps=${STEPS}" \
    "benchmark.seeds=${SEEDS}" \
    baselines.benchmark_dir="${BENCH_DIR}" \
    hydra.run.dir="${RUN_DIR}"

echo "Done. Results: ${BENCH_DIR}"
echo "Copy them somewhere durable before \$SCRATCH gets cleaned, e.g.:"
echo "  cp -r ${BENCH_DIR} \$PLG_GROUPS_STORAGE/<your-group>/"
