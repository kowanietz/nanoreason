#!/usr/bin/env python3
"""Read-only CPU validation of a standalone FP32 or BF16 NanoReason model."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model_artifacts import validate_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--expected-sha256", help="Expected SHA-256 of model.safetensors"
    )
    args = parser.parse_args()
    try:
        report = validate_model(
            args.model,
            expected_sha256=args.expected_sha256,
        )
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print(f"Validation FAILED: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
