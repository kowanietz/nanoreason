#!/bin/bash
#SBATCH --account=rcfd
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --signal=TERM@900
#SBATCH --job-name=dolci-sft-6k
#SBATCH --dependency=singleton
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.log

set -euo pipefail

ROOT="${NANOREASON_ROOT:?Set NANOREASON_ROOT to your storage directory}"
REPO="${NANOREASON_REPO:?Set NANOREASON_REPO to the repository checkout}"
export NANOREASON_ROOT="$ROOT" NANOREASON_REPO="$REPO"
CONFIG="sft/packed-6k"

module load uv
cd "$REPO"
mkdir -p logs

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Config: $CONFIG"
echo "Started: $(date)"

nvidia-smi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export UV_PROJECT_ENVIRONMENT="$REPO/.venv"
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

srun uv run \
  --group sft \
  --no-default-groups \
  --locked --offline \
  torchrun --standalone --nproc_per_node=4 \
  "$REPO/src/train_sft.py" \
  --config-name "$CONFIG" "$@"

echo "Finished: $(date)"
