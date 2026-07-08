#!/bin/bash
# One-time environment setup on a Cyfronet Athena login node.
#
# Compute nodes have NO internet access, so all pip installs must happen
# here, on the login node, before submitting any job. ALE ROMs ship inside
# ale-py, so no separate ROM download is needed.
#
# Usage (from an Athena login-node shell):
#   git clone https://github.com/Grzmro/dreamer-rssm.git $SCRATCH/dreamer-rssm
#   cd $SCRATCH/dreamer-rssm
#   bash slurm/setup_athena.sh

set -euo pipefail

module load Miniconda3 2>/dev/null || module load Python/3.11.5-GCCcore-13.2.0 2>/dev/null || {
    echo "Adjust the 'module load' line above to whatever 'module avail python' shows here." >&2
    exit 1
}

VENV_DIR="${SCRATCH}/venvs/dreamer-rssm"
python -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

pip install --upgrade pip
pip install -e ".[dev]"

echo ""
echo "Setup done. Venv: $VENV_DIR"
echo "Next: sbatch slurm/benchmark.sbatch"
