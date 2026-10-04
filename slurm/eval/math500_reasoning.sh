#!/bin/bash
#SBATCH --account=rcfd
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --job-name=math500-reasoning
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.log

set -euo pipefail

ROOT="/p/scratch/rcfd/kowanietz1"
REPO="$ROOT/projects/reasoning-posttraining"
OLMES="$ROOT/olmes"

MODEL="$ROOT/checkpoints/reasoning-posttraining/sft/dolci-16k-packed-1k"
TOKENIZER="$MODEL"
OUTPUT="$REPO/results/math500-reasoning-32k"

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
export CUDA_VISIBLE_DEVICES=0,1,2,3

MODEL_ARGS='{"trust_remote_code":false,"dtype":"bfloat16","tokenizer":"'"$TOKENIZER"'","max_length":32768}'
TASK_ARGS='{"generation_kwargs":{"max_gen_toks":32768,"truncate_context":false}}'

uv run \
  --project "$OLMES" \
  --frozen \
  --offline \
  olmes \
  --model "$MODEL" \
  --model-type vllm \
  --model-args "$MODEL_ARGS" \
  --task minerva_math_500::hamish_zs_reasoning \
  --task-args "$TASK_ARGS" \
  --batch-size auto \
  --gpus 4 \
  --num-workers 1 \
  --output-dir "$OUTPUT"

# format results
jq . "$OUTPUT/metrics.json" > "$OUTPUT/metrics.json.tmp"
mv "$OUTPUT/metrics.json.tmp" "$OUTPUT/metrics.json"

echo "Finished: $(date)"