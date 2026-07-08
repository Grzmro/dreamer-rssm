#!/bin/bash
# Phase 4 ablation matrix as a Slurm job array — one preset per GPU task,
# 2 seeds each, 30k env steps on Pong (see RESULTS.md, section C).
#
# Usage: sbatch slurm/ablations.sbatch

#SBATCH --job-name=dreamer-ablations
#SBATCH --account=plgXXXXXXX-gpu-a100      # <-- set to your PLGrid grant name
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=6:00:00
#SBATCH --array=0-10
#SBATCH --output=%x-%A_%a.log

set -euo pipefail

PRESETS=(
    ""                    # 0: base run (no ablation preset)
    horizon_5             # 1
    horizon_10            # 2
    horizon_20            # 3
    deter_128             # 4
    deter_256             # 5
    stoch_16x16           # 6
    stoch_32x64           # 7
    no_kl_balance         # 8
    no_free_nats          # 9
    no_reconstruction     # 10
)
PRESET="${PRESETS[$SLURM_ARRAY_TASK_ID]}"
OVERRIDE=""
if [[ -n "$PRESET" ]]; then OVERRIDE="ablation=${PRESET}"; fi

source "${SCRATCH}/venvs/dreamer-rssm/bin/activate"
cd "${SCRATCH}/dreamer-rssm"

python experiments/run_seeds.py ${OVERRIDE} \
    "benchmark.seeds=[0,1]" \
    benchmark.total_env_steps=30000 \
    baselines.benchmark_dir="${SCRATCH}/dreamer-rssm/experiments/benchmark" \
    hydra.run.dir="${SCRATCH}/dreamer-rssm/experiments/ablation_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

echo "Preset '${PRESET:-base}' done. After ALL array tasks finish, run:"
echo "  python viz/ablation_summary.py && python viz/learning_curves.py"
