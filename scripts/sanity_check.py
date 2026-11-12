#!/usr/bin/env python3
"""Optional local generation smoke test. CPU by default; CUDA requires Slurm allocation."""

import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = ["What is 13 + 29?", "Solve x^2 = 25."]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument(
        "--device", default="cpu", help="cpu or an allocated CUDA device, e.g. cuda:1"
    )
    parser.add_argument(
        "--dtype", choices=("auto", "float32", "bfloat16", "float16"), default="auto"
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type not in {"cpu", "cuda"}:
        parser.error("--device must be cpu or cuda[:index]")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    if device.type == "cuda" and not (
        os.environ.get("SLURM_JOB_ID") and torch.cuda.is_available()
    ):
        parser.error("CUDA generation requires an active Slurm GPU allocation")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer or args.model,
        local_files_only=True,
        trust_remote_code=False,
    )
    dtype = args.dtype if args.dtype == "auto" else getattr(torch, args.dtype)
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=dtype,
            local_files_only=True,
            trust_remote_code=False,
        )
        .to(device)
        .eval()
    )
    for question in PROMPTS:
        # Tokenize the rendered template exactly once; do not add extra BOS/EOS.
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            # Enable the KV cache for this call without editing the saved config.
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        text = tokenizer.decode(
            output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=False
        )
        if not text.strip():
            raise RuntimeError(f"Empty generation for {question!r}")
        print(f"PROMPT: {question}\n{text.strip()}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
