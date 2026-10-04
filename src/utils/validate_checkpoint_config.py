#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

CRITICAL_KEYS = [
    "rope_theta",
    "rope_scaling",
]

parser = argparse.ArgumentParser()
parser.add_argument("--base", required=True, type=Path)
parser.add_argument("--checkpoint", required=True, type=Path)
args = parser.parse_args()

base_path = args.base / "config.json"
checkpoint_path = args.checkpoint / "config.json"

if not base_path.exists():
    raise FileNotFoundError(base_path)

if not checkpoint_path.exists():
    raise FileNotFoundError(checkpoint_path)

with base_path.open() as f:
    base = json.load(f)

with checkpoint_path.open() as f:
    checkpoint = json.load(f)

errors = []

for key in CRITICAL_KEYS:
    if key not in base:
        errors.append(f"{key}: missing from base config")
    elif key not in checkpoint:
        errors.append(f"{key}: missing from checkpoint config")
    elif checkpoint[key] != base[key]:
        errors.append(
            f"{key} mismatch\n"
            f"  base:       {json.dumps(base[key], indent=2)}\n"
            f"  checkpoint: {json.dumps(checkpoint[key], indent=2)}"
        )

if checkpoint.get("use_cache") is not True:
    errors.append(
        f"use_cache should be true, got {checkpoint.get('use_cache')!r}"
    )

if errors:
    print("CHECKPOINT CONFIG VALIDATION FAILED\n", file=sys.stderr)

    for error in errors:
        print(error, file=sys.stderr)
        print(file=sys.stderr)

    sys.exit(1)

print("Checkpoint config validation PASSED")