#!/usr/bin/env python3

import argparse
import json
import shutil
from pathlib import Path

# TODO create issue on hf?

PRESERVE_FROM_BASE = [
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

backup_path = args.checkpoint / "config.backup.json"

if not backup_path.exists():
    shutil.copy2(checkpoint_path, backup_path)
    print(f"Backup created: {backup_path}")

changed = []

for key in PRESERVE_FROM_BASE:
    if key not in base:
        raise KeyError(f"Missing from base config: {key}")

    if checkpoint.get(key) != base[key]:
        print(f"Repairing {key}")
        checkpoint[key] = base[key]
        changed.append(key)

if checkpoint.get("use_cache") is not True:
    print("Setting use_cache=True")
    checkpoint["use_cache"] = True
    changed.append("use_cache")

if changed:
    checkpoint_path.write_text(
        json.dumps(checkpoint, indent=2, sort_keys=True) + "\n"
    )
    print(f"Repaired: {', '.join(changed)}")
else:
    print("No repair necessary")