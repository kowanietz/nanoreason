#!/bin/bash
#SBATCH --account=rcfd
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --job-name=math500-base
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.log

set -euo pipefail

ROOT="/p/scratch/rcfd/kowanietz1"
REPO="$ROOT/projects/reasoning-posttraining"
OLMES="$ROOT/olmes"

MODEL="$ROOT/models/Olmo-3-7B"
TOKENIZER="$MODEL"
OUTPUT="$REPO/results/base/math500-olmes-bf16"

module load uv

cd "$REPO"

mkdir -p logs
mkdir -p "$OUTPUT"

echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $(hostname)"
echo "Model:    $MODEL"
echo "Output:   $OUTPUT"
echo "Started:  $(date)"

nvidia-smi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=0

MODEL_ARGS='{"trust_remote_code":false,"dtype":"bfloat16","tokenizer":"'"$TOKENIZER"'"}'

uv run \
  --project "$OLMES" \
  --frozen \
  --offline \
  olmes \
  --model "$MODEL" \
  --model-type vllm \
  --model-args "$MODEL_ARGS" \
  --task minerva_math_500::olmes \
  --batch-size auto \
  --gpus 1 \
  --num-workers 1 \
  --output-dir "$OUTPUT"

# format results
jq . "$OUTPUT/metrics.json" > "$OUTPUT/metrics.json.tmp"
mv "$OUTPUT/metrics.json.tmp" "$OUTPUT/metrics.json"

echo "Finished: $(date)"