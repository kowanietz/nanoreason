#!/bin/bash
#SBATCH --account=rcfd
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --time=06:00:00
#SBATCH --signal=TERM@900
#SBATCH --job-name=dolci-sft-6k
#SBATCH --dependency=singleton
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.log

set -euo pipefail

ROOT="/p/scratch/rcfd/kowanietz1"
REPO="$ROOT/projects/reasoning-posttraining"
CONFIG="sft-6k"

module load uv

cd "$REPO"

mkdir -p logs

echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $(hostname)"
echo "Config:  $CONFIG"
echo "Started: $(date)"

nvidia-smi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

export UV_PROJECT_ENVIRONMENT="$REPO/.venv"

srun uv run \
  --group sft \
  --no-default-groups \
  --locked --offline \  torchrun \
  --standalone \
  --nproc_per_node=4 \
  "$REPO/src/train_sft.py" \
  --config-name "$CONFIG"

echo "Finished: $(date)"
