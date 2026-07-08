#!/bin/bash
# Full Phase 3 benchmark on one Athena A100. Edit --account before submitting.
#
# Usage:
#   sbatch slurm/benchmark.sbatch
#   sbatch --time=24:00:00 slurm/benchmark.sbatch      # override walltime
#
# Check status from a phone SSH session:
#   squeue --me
#   tail -f dreamer-benchmark-<jobid>.log

#SBATCH --job-name=dreamer-benchmark
#SBATCH --account=plgXXXXXXX-gpu-a100      # <-- set to your PLGrid grant name
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

RUN_DIR="${SCRATCH}/dreamer-rssm/experiments/athena_${SLURM_JOB_ID}"
python train/run_benchmark.py \
    benchmark.total_env_steps=400000 \
    "benchmark.seeds=[0]" \
    baselines.benchmark_dir="${SCRATCH}/dreamer-rssm/experiments/benchmark" \
    hydra.run.dir="${RUN_DIR}"

echo "Done. Results: ${SCRATCH}/dreamer-rssm/experiments/benchmark"
echo "Copy them somewhere durable before \$SCRATCH gets cleaned, e.g.:"
echo "  cp -r ${SCRATCH}/dreamer-rssm/experiments/benchmark \$PLG_GROUPS_STORAGE/<your-group>/"
