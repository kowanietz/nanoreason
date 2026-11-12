# NanoReason

Reasoning post-training of [OLMo 3 7B](https://huggingface.co/allenai/Olmo-3-1025-7B): supervised fine-tuning, [repetition](https://arxiv.org/abs/2602.11149), and reinforcement learning with verifiable rewards (group relative policy optimization).

Trained on **4× NVIDIA A100 40 GB** (single node). The main training runs account for approximately **210 GPU-hours**

## Results

| Benchmark | Base | SFT | Refined | Think |
|---|---:|---:|---:|---:|
| GPQA Main | 28.79 | 32.37 | **42.63** | 41.74 |
| MATH-500 | 63.00 | 88.40 | **91.00** | **91.00** |
| AIME 2024 | 12.08 | 30.21 | **45.83** | 45.31 |
| AIME 2025 | 14.17 | 26.35 | **33.33** | **33.33** |
| BBH | — | 60.34 | 67.61 | **69.19** |
| IFEval | 34.57 | 63.77 | **72.64** | **72.64** |

Percentages from [`results/`](results/). AIME is mean pass@1 over 32 samples per problem. GPQA uses Main (448 questions), BBH is micro exact match, and IFEval is prompt-level loose accuracy.

Base BBH evaluation was stopped after 4 of 27 tasks to limit compute and energy use, as extrapolation suggested roughly 34 hours for the full evaluation. No aggregate Base BBH score is reported.

| Stage | Training |
|---|---|
| **Base** | OLMo 3 7B, evaluated with the [Think-SFT](https://huggingface.co/allenai/Olmo-3-7B-Think-SFT) tokenizer/template |
| **SFT** | 6,000 steps on [Dolci-Think-SFT-7B](https://huggingface.co/datasets/allenai/Dolci-Think-SFT-7B); 16,384-token packing |
| **Refined** | 16 [repetition](https://arxiv.org/abs/2602.11149) epochs on 3,200 selected examples (2,784 steps) |
| **Think** | [Dr. GRPO](https://arxiv.org/abs/2503.20783) on [math and instruction-following rewards](https://huggingface.co/datasets/allenai/Dolci-Think-RL-7B) ; stopped at step 512 |

Further benchmark and reasoning analysis in [`notebooks/`](notebooks/). Hugging Face  Model releases coming soon.

## Setup

Python 3.12, [`uv`](https://docs.astral.sh/uv/), and four CUDA GPUs (at least ~4*40 GB VRAM) for training/evaluation. Run commands from the repo root. Use an external storage directory for models, datasets, and checkpoints.

```bash
export NANOREASON_REPO="$PWD"
export NANOREASON_ROOT=/path/to/nanoreason-data
export OLMES_PROJECT="$NANOREASON_ROOT/olmes"
mkdir -p "$NANOREASON_ROOT"/{models,datasets} logs

UV_PROJECT_ENVIRONMENT=.venv uv sync --locked --no-default-groups --group sft --group dev
UV_PROJECT_ENVIRONMENT=.venv-grpo uv sync --locked --no-default-groups --group grpo --group dev
UV_PROJECT_ENVIRONMENT=.venv-eval uv sync --locked --no-default-groups
```

`NANOREASON_ROOT` and `NANOREASON_REPO` are required in each shell. The three environments are separate because SFT, RL and evaluation use different dependency stacks.

### OLMES

Evaluation uses a pinned [OLMES](https://github.com/allenai/olmes) revision and local patch. The lockfile and patch are in [`configs/eval/olmes/`](configs/eval/olmes/).

```bash
git clone https://github.com/allenai/olmes.git "$OLMES_PROJECT"
git -C "$OLMES_PROJECT" checkout --detach 5a51f502d463b8cdc4a2dcad7d7096c41ff1197e

cp -n configs/eval/olmes/uv.lock "$OLMES_PROJECT/uv.lock"
git -C "$OLMES_PROJECT" apply --unidiff-zero "$NANOREASON_REPO/configs/eval/olmes/local.patch"

env -u UV_PROJECT_ENVIRONMENT uv sync --project "$OLMES_PROJECT" \
  --frozen --no-default-groups --extra gpu --python 3.12
```

The patch adds model-hash defaults and closes the vLLM engine after evaluation. Generation and scoring are unchanged.

## Data

Download the pinned model/tokenizer and eval dataset snapshots before using offline mode. Authenticate with Hugging Face if necessary.

```bash
.venv-grpo/bin/hf download allenai/Olmo-3-1025-7B \
  --revision a81bae42db3975be1671e27b9c9a56da1a9f980f \
  --local-dir "$NANOREASON_ROOT/models/Olmo-3-7B"

.venv-grpo/bin/hf download allenai/Olmo-3-7B-Think-SFT \
  --revision 6ff857587e040d6d523a3d5f3a56e918f5401d66 \
  --include 'tokenizer*' 'special_tokens_map.json' 'chat_template.jinja' \
  --local-dir "$NANOREASON_ROOT/models/Olmo-3-7B-Think-SFT"

.venv-grpo/bin/hf download allenai/Dolci-Think-SFT-7B --repo-type dataset \
  --revision 72ec0fe32428bada1ec686a9168a0681eecd2094 \
  --local-dir "$NANOREASON_ROOT/datasets/Dolci-Think-SFT-7B"

.venv-grpo/bin/hf download allenai/Dolci-Think-RL-7B --repo-type dataset \
  --revision 0fb6466d31ef3a9dd16985ef635e6429e05a6491 \
  --local-dir "$NANOREASON_ROOT/datasets/Dolci-Think-RL-7B"

.venv-grpo/bin/python scripts/setup_if_resources.py
env -u UV_PROJECT_ENVIRONMENT uv run --project "$OLMES_PROJECT" --frozen --offline \
  python "$NANOREASON_REPO/scripts/check_benchmarks.py" --allow-download
```

GPQA requires authorized access. Prepare repetition data **after SFT** and RLVR data **after downloading datasets**:

```bash
.venv-grpo/bin/python scripts/prepare_sft_repetition.py configs/sft/repetition-3k2.yaml
.venv-grpo/bin/python scripts/prepare_rlvr_data.py \
  --source-dir "$NANOREASON_ROOT/datasets/Dolci-Think-RL-7B" \
  --output-dir "$NANOREASON_REPO/.cache/rlvr-prepared-v4" \
  --quarantine-invalid-if-references
```

Repetition excludes matches to the recorded Base evaluation prompts. RLVR filters against [`configs/eval/exclusions.json`](configs/eval/exclusions.json). Both use exact normalized matches and do not detect paraphrases. The full 100,000-example SFT source pool has been verified to contain no exact normalized prompt matches to any evaluated benchmark.

## Training

Four GPUs per run. The Slurm wrappers use cluster-specific account/partition/module settings, so make sure to adapt them outside the original environment. Submit each stage **after the previous one finishes**:


```bash
sbatch slurm/train/sft-6k.sh
sbatch slurm/train/sft-repetition-3k2.sh 4
sbatch slurm/train/sft-repetition-3k2.sh 8
sbatch slurm/train/sft-repetition-3k2.sh 12
sbatch slurm/train/sft-repetition-3k2.sh 16
sbatch slurm/train/rlvr-512.sh
```
TODO: merge repetition job steps?

Run repetition data preparation after SFT. Prepare RLVR data before the last command.

For the historical RLVR split (due to walltime limitations), first set `runtime.stop_after_step=256`, then resume to step 512. **Keep `grpo.max_steps=1024`** as it's the original scheduler horizon, not the final stop step. Seeds, optimizer settings, and other training hyperparameters are in [`configs/`](configs/).

## Evaluation

Benchmarks use BF16 and OLMES/vLLM on four GPUs, with a 32,768-token model context. AIME uses 32 samples per problem. The generation budget is clipped by OLMES to fit the remaining context.

Choose a model directory and a result name, then submit the benchmarks you want to run:

```bash
MODEL="$NANOREASON_ROOT/models/NanoReason-7B-Think"
NAME=<name>

sbatch slurm/eval/math500.sh --model "$MODEL" --tokenizer "$MODEL" --name "$NAME"
sbatch slurm/eval/gpqa.sh --model "$MODEL" --tokenizer "$MODEL" --name "$NAME"
sbatch slurm/eval/ifeval.sh --model "$MODEL" --tokenizer "$MODEL" --name "$NAME"
sbatch slurm/eval/bbh.sh --model "$MODEL" --tokenizer "$MODEL" --name "$NAME"
sbatch slurm/eval/aime.sh --year 2024 --model "$MODEL" --tokenizer "$MODEL" --name "$NAME"
sbatch slurm/eval/aime.sh --year 2025 --model "$MODEL" --tokenizer "$MODEL" --name "$NAME"
```

Use a new `--name` for each model or rerun. Results and evaluation provenance are saved under `results/`; Slurm logs go to `logs/`. Benchmark settings are in [`configs/eval/`](configs/eval/).

## Reproducibility

- Original prepared datasets, native checkpoints, and some path-sensitive manifests are external to Git. A fresh installation and full training/benchmark rerun have **not** been verified.
- RLVR resume restores training state, but regenerates rollouts and vLLM RNG. Bitwise reproducibility across restarts or changed environments is **not** established.
- The results use the recorded evaluation configuration. Historical artifacts are preserved in [`results/`](results/); newer runs should use separate names. The reported ~210 training GPU-hours cover the recorded main runs, not all project compute.
- The current SFT dependency stack passed functional tests but differs from the original experimental environment. Bitwise reproducibility across dependency versions is not guaranteed.

Checks:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 .venv-grpo/bin/python -m unittest discover -s tests

.venv-grpo/bin/ruff check src scripts tests notebooks/utils

.venv/bin/yamllint configs

.venv-grpo/bin/python scripts/setup_if_resources.py --check

env -u UV_PROJECT_ENVIRONMENT uv run --project "$OLMES_PROJECT" --frozen --offline \
  python "$NANOREASON_REPO/scripts/check_benchmarks.py"
```

TODO: release footer with hf, zenodo links...
