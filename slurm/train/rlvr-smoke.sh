#!/usr/bin/env bash
#SBATCH --account=rcfd
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --time=00:45:00
#SBATCH --signal=TERM@900
#SBATCH --job-name=dr-grpo-smoke
#SBATCH --dependency=singleton
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.log

set -euo pipefail

ROOT="${NANOREASON_ROOT:?Set NANOREASON_ROOT to your storage directory}"
REPO="${NANOREASON_REPO:?Set NANOREASON_REPO to the repository checkout}"
export NANOREASON_ROOT="$ROOT" NANOREASON_REPO="$REPO"
STOP_STEP="${1:-4}"

if (( $# )); then
  shift
fi

module load uv
cd "$REPO"
mkdir -p logs

echo "Job ID: $SLURM_JOB_ID"
echo "Stop after optimizer step: $STOP_STEP"
nvidia-smi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export UV_PROJECT_ENVIRONMENT="$REPO/.venv-grpo"
export VLLM_CACHE_ROOT="$ROOT/.cache/vllm"
export SRUN_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"

export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1

unset PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF

srun uv run \
  --group grpo \
  --no-default-groups \
  --locked --offline \
  torchrun --standalone --nproc_per_node=4 \
  "$REPO/src/train_rlvr.py" \
  "$@" "runtime.stop_after_step=$STOP_STEP"
