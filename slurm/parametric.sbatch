#!/bin/bash
# Training-time parametric studies (RESULTS.md section E) as a Slurm job
# array: {base, train_ratio 0.1/1.0, entropy 1e-4/1e-3} x 3 seeds, one
# GPU per (variant, seed) pair — 15 tasks. Shared 60k env-step budget so
# the train_ratio sweep isolates the compute knob at a fixed sample count.
#
# Wall-clock per task on an A100 (rough): tr=0.1 ~0.5 h, base ~1.5 h,
# tr=1.0 ~4 h; entropy variants ~1.5 h.
#
# Usage:
#   sbatch slurm/parametric.sbatch
# Afterwards (once ALL tasks finished):
#   python viz/ablation_summary.py && python viz/learning_curves.py

#SBATCH --job-name=dreamer-parametric
#SBATCH --account=plglscclass26-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=8:00:00
#SBATCH --array=0-14
#SBATCH --output=%x-%A_%a.log

set -euo pipefail

VARIANTS=(
    ""                # base: train_ratio 0.3, entropy 3e-4
    train_ratio_01
    train_ratio_10
    entropy_1e-4
    entropy_1e-3
)
SEEDS=(0 1 2)
VARIANT="${VARIANTS[$((SLURM_ARRAY_TASK_ID / 3))]}"
SEED="${SEEDS[$((SLURM_ARRAY_TASK_ID % 3))]}"
OVERRIDE=""
if [[ -n "$VARIANT" ]]; then OVERRIDE="ablation=${VARIANT}"; fi

source "${SCRATCH}/venvs/dreamer-rssm/bin/activate"
cd "${SCRATCH}/dreamer-rssm"

python experiments/run_seeds.py ${OVERRIDE} \
    "benchmark.seeds=[${SEED}]" \
    benchmark.total_env_steps=60000 \
    baselines.benchmark_dir="${SCRATCH}/dreamer-rssm/experiments/benchmark" \
    hydra.run.dir="${SCRATCH}/dreamer-rssm/experiments/parametric_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

echo "Variant '${VARIANT:-base}' seed ${SEED} done."
