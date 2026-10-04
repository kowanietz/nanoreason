#!/usr/bin/env python3

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    "What is 13 + 29?",
    "Solve x^2 = 25.",
    "Convert the point (0, 3) to polar coordinates.",
    "If a train travels 120 km in 1.5 hours, what is its average speed?",
]

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--tokenizer")
parser.add_argument("--max-new-tokens", type=int, default=512)
args = parser.parse_args()

tokenizer_path = args.tokenizer or args.model

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

model = AutoModelForCausalLM.from_pretrained(
    args.model,
    dtype=torch.bfloat16,
    device_map={"": 0},
)

model.eval()

for i, question in enumerate(PROMPTS, start=1):
    messages = [
        {"role": "user", "content": question},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

    generated = output[0, inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(
        generated,
        skip_special_tokens=False,
    )

    print(f"\n{'=' * 80}")
    print(f"PROMPT {i}")
    print(question)
    print("-" * 80)
    print(text.strip())