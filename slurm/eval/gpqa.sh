#!/bin/bash
#SBATCH --account=rcfd
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --job-name=gpqa
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.log

set -euo pipefail

ROOT="/p/scratch/rcfd/kowanietz1"
REPO="$ROOT/projects/reasoning-posttraining"

module load uv
cd "$REPO"

export UV_PROJECT_ENVIRONMENT="$REPO/.venv-eval"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=0,1,2,3

srun uv run \
  --project "$REPO" \
  --no-default-groups \
  --frozen \
  --offline \
  python "$REPO/scripts/eval.py" --config gpqa "$@"
