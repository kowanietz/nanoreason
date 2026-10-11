#!/bin/bash
#SBATCH --account=rcfd
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --job-name=math500-6k
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.log

set -euo pipefail

ROOT="/p/scratch/rcfd/kowanietz1"
REPO="$ROOT/projects/reasoning-posttraining"
OLMES="$ROOT/olmes"

# Default preserves the original 6k benchmark. Explicit 4/8/16 select repetition exports.
MILESTONE="${1:-6k}"
case "$MILESTONE" in
    6k)
        MODEL="$ROOT/checkpoints/reasoning-posttraining/sft/dolci-16k-packed-6k-vllm011"
        EVAL_TAG="dolci-16k-packed-6k"
        ;;
    4|8)
        MODEL="$ROOT/checkpoints/reasoning-posttraining/sft/dolci-16k-repetition-3k2/epoch-$MILESTONE-model-vllm011"
        EVAL_TAG="dolci-16k-repetition-3k2-epoch-$MILESTONE"
        ;;
    16)
        MODEL="$ROOT/checkpoints/reasoning-posttraining/sft/dolci-16k-repetition-3k2-epoch-16-model-vllm011"
        EVAL_TAG="dolci-16k-repetition-3k2-epoch-$MILESTONE"
        ;;
    *)
        echo "Expected checkpoint 6k, 4, 8, or 16" >&2
        exit 2
        ;;
esac
TOKENIZER="$ROOT/models/Olmo-3-7B-Think-SFT"
OUTPUT="$REPO/results/$EVAL_TAG/math500-reasoning-32k"

module load uv

cd "$REPO"
mkdir -p logs "$OUTPUT"

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

srun uv run \
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

jq . "$OUTPUT/metrics.json" > "$OUTPUT/metrics.json.tmp"
mv "$OUTPUT/metrics.json.tmp" "$OUTPUT/metrics.json"

echo "Finished: $(date)"
