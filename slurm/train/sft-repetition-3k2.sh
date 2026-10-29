#!/usr/bin/env bash
#SBATCH --account=rcfd
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --time=05:00:00
#SBATCH --signal=TERM@900
#SBATCH --job-name=dolci-sft-repetition-3k2
#SBATCH --dependency=singleton
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.log

set -euo pipefail

ROOT="/p/scratch/rcfd/kowanietz1"
REPO="$ROOT/projects/reasoning-posttraining"
CONFIG="sft/repetition-3k2"

MILESTONE="${1:-}"

case "$MILESTONE" in
  4|8|12|16)
    ;;
  *)
    echo "Expected milestone 4, 8, 12, or 16" >&2
    exit 2
    ;;
esac

module load uv
cd "$REPO"
mkdir -p logs

echo "Job ID: $SLURM_JOB_ID"
echo "Config: $CONFIG"
echo "Milestone: $MILESTONE"
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
  --config-name "$CONFIG" \
  "training.stop_after_epoch=$MILESTONE"

echo "Finished: $(date)"
