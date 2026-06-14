#!/bin/bash
#SBATCH --job-name=depresjon
#SBATCH --partition=dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/depresjon_%j.out
#SBATCH --error=logs/depresjon_%j.err

set -euo pipefail

# Adjust to your env. If using conda:
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate depresjon

DATA_DIR=${DATA_DIR:-/data/depresjon}
OUT_DIR=${OUT_DIR:-results}
TAG=${TAG:-no_smote_run1}

cd "$SLURM_SUBMIT_DIR"

# Optional second run with SMOTE for comparison:
#   sbatch --export=ALL,TAG=smote_run1,USE_SMOTE=1 scripts/submit.sbatch
SMOTE_FLAG=""
if [[ "${USE_SMOTE:-0}" == "1" ]]; then
    SMOTE_FLAG="--use-smote"
fi

python -u run.py \
    --data-dir "$DATA_DIR" \
    --dataset depresjon \
    --out-dir "$OUT_DIR" \
    --cache-dir cache \
    --tag "$TAG" \
    --tsfel-n-jobs "$SLURM_CPUS_PER_TASK" \
    $SMOTE_FLAG
