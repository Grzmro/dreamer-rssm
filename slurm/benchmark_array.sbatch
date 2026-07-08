#!/bin/bash
# Same benchmark, but 3 seeds run in parallel as a Slurm job array (one GPU
# each) instead of sequentially inside a single job. Faster wall-clock at
# the cost of 3 GPUs reserved at once — use this over benchmark.sbatch when
# your grant has GPU quota to spare. Results land in the same shared CSV
# tree, so viz/benchmark_comparison.py sees all seeds regardless of which
# script produced them.
#
# Usage: sbatch slurm/benchmark_array.sbatch

#SBATCH --job-name=dreamer-benchmark
#SBATCH --account=plgXXXXXXX-gpu-a100      # <-- set to your PLGrid grant name
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=%x-%A_%a.log

set -euo pipefail

source "${SCRATCH}/venvs/dreamer-rssm/bin/activate"
export SDL_VIDEODRIVER=dummy

cd "${SCRATCH}/dreamer-rssm"

RUN_DIR="${SCRATCH}/dreamer-rssm/experiments/athena_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
python train/run_benchmark.py \
    benchmark.total_env_steps=400000 \
    "benchmark.seeds=[${SLURM_ARRAY_TASK_ID}]" \
    baselines.benchmark_dir="${SCRATCH}/dreamer-rssm/experiments/benchmark" \
    hydra.run.dir="${RUN_DIR}"

# Only render plots once, after the last array task to finish sees every CSV.
echo "Seed ${SLURM_ARRAY_TASK_ID} done. Run this manually once all 3 array tasks finish:"
echo "  python viz/benchmark_comparison.py"
