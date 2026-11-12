#!/usr/bin/env python3
"""Package a merged NanoReason export without changing its weights or configuration."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model_artifacts import export_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument(
        "--tokenizer", type=Path, help="Explicit evaluated tokenizer donor, if separate"
    )
    args = parser.parse_args()
    try:
        report = export_model(args.source, args.destination, tokenizer=args.tokenizer)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print(f"Export FAILED: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
