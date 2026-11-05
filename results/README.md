# Benchmark Results

## Model Stages

### 1. OLMo 3 7B
- Original pretrained base model by [Allen AI](https://allenai.org/)
- Baseline for evaluating improvements across training stages

### 2. NanoReason-7B-SFT
- Initial SFT on **6,000 optimizer steps**
- Introduces structured reasoning behavior through supervised training

### 3. NanoReason-7B-SFT-Refined
- Further fine-tuning of the SFT model using **16 epochs of repetition training**
- Substantially improves reasoning behavior and the model's ability to complete its reasoning process and produce final answers

### 4. NanoReason-7B-Think
- Final model trained using [Dr. GRPO](https://arxiv.org/pdf/2503.20783)
- Performs **512 optimizer steps** during RLVR
