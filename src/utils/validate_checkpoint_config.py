#!/usr/bin/env python3
"""Validate an OLMo3 export or inference directory without changing files.

By default, require legacy RoPE fields and caching for vLLM 0.11.0. Use
--source-export to check that a training export can be prepared by the repair
utility, even if it still lacks those inference settings.
"""

import argparse
import json
import sys
from pathlib import Path

if __package__:
    from .repair_checkpoint_config import (
        ROOT,
        VLLM_PACKAGE,
        inspect_checkpoint,
        read_json,
    )
else:
    from repair_checkpoint_config import (
        ROOT,
        VLLM_PACKAGE,
        inspect_checkpoint,
        read_json,
    )


def validate(
    base, checkpoint, tokenizer, vllm_package=VLLM_PACKAGE, source_export=False
):
    """Run the repair utility's read-only checks and report inference readiness."""
    compatible, _, report = inspect_checkpoint(
        base, checkpoint, tokenizer, vllm_package
    )
    current = read_json(Path(checkpoint) / "config.json")
    required_changes = {
        key: {"current": current.get(key), "required": compatible[key]}
        for key in ("rope_theta", "rope_scaling", "use_cache")
        if current.get(key) != compatible[key]
    }
    if required_changes and not source_export:
        fields = ", ".join(required_changes)
        raise ValueError(
            f"Checkpoint needs inference preparation: {fields}. "
            "Run repair_checkpoint_config.py with a separate --output, or use "
            "--source-export to validate the original export without requiring legacy fields."
        )
    return {
        "checkpoint": str(Path(checkpoint).resolve()),
        "validation_mode": "source-export" if source_export else "inference",
        "inference_ready": not required_changes,
        "required_changes": required_changes,
        "validation": report,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--tokenizer", type=Path, default=ROOT / "models/Olmo-3-7B-Think-SFT"
    )
    parser.add_argument("--vllm-package", type=Path, default=VLLM_PACKAGE)
    parser.add_argument(
        "--source-export",
        action="store_true",
        help="Allow missing legacy RoPE fields and disabled caching in a source export",
    )
    args = parser.parse_args()
    try:
        result = validate(
            args.base,
            args.checkpoint,
            args.tokenizer,
            args.vllm_package,
            args.source_export,
        )
    except (OSError, ValueError, KeyError) as error:
        print(f"Checkpoint validation FAILED: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    print("Checkpoint validation PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
