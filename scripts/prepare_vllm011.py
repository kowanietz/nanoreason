#!/usr/bin/env python3
"""Historical vLLM 0.11.0 compatibility only: write a separate config and source links."""

import argparse
import copy
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model_artifacts import check_destination, model_files, read_json, validate_rope


def compatible_config(exported):
    """Translate only RoPE/cache fields representable by historical vLLM."""
    if exported.get("model_type") != "olmo3":
        raise ValueError("Expected an OLMo3 export")
    parameters = validate_rope(exported)
    full, sliding = parameters["full_attention"], parameters["sliding_attention"]
    theta = full["rope_theta"]
    if sliding != {"rope_type": "default", "rope_theta": theta}:
        raise ValueError("vLLM 0.11 requires default sliding RoPE with matching theta")
    scaling = {key: value for key, value in full.items() if key != "rope_theta"}
    allowed = {
        "rope_type",
        "factor",
        "original_max_position_embeddings",
        "attention_factor",
        "beta_fast",
        "beta_slow",
    }
    if scaling.get("rope_type") != "yarn" or scaling.keys() - allowed:
        raise ValueError("Unsupported full-attention RoPE for vLLM 0.11")
    magnitude = 1 + 0.1 * math.log(scaling["factor"])
    if not math.isclose(
        scaling.get("attention_factor", magnitude), magnitude, rel_tol=1e-12
    ):
        raise ValueError(
            "vLLM 0.11 cannot represent a nondefault YaRN attention_factor"
        )
    result = copy.deepcopy(exported)
    result.update(rope_theta=theta, rope_scaling=scaling, use_cache=True)
    return result


def validate_vllm(package):
    """Check the installed vLLM source without importing it or initializing CUDA."""
    checks = {
        "_version.py": ["version = '0.11.0'"],
        "model_executor/models/registry.py": [
            '"Olmo3ForCausalLM": ("olmo2", "Olmo2ForCausalLM")'
        ],
        "model_executor/models/olmo2.py": [
            "self.rope_theta = self.config.rope_theta",
            'layer_types[layer_idx] == "sliding_attention"',
            "sliding_window = self.config.sliding_window",
            "self.config.rope_scaling",
            "if sliding_window is None else None",
        ],
        "model_executor/layers/rotary_embedding/yarn_scaling_rope.py": [
            "yarn_get_mscale(self.scaling_factor) * attn_factor"
        ],
    }
    for relative, fragments in checks.items():
        path = Path(package) / relative
        text = path.read_text()
        if not all(fragment in text for fragment in fragments):
            raise ValueError(
                f"Unrecognized vLLM implementation: {path}; recheck compatibility"
            )


def prepare(source, tokenizer, output, vllm_package):
    files = model_files(source, tokenizer)
    output = check_destination(
        output, [Path(source), *(path.resolve().parent for path in files.values())]
    )
    config = compatible_config(read_json(files["config.json"]))
    validate_vllm(vllm_package)
    output.mkdir()
    for name, original in files.items():
        if name != "config.json":
            (output / name).symlink_to(original.resolve())
    with (output / "config.json").open("x") as stream:
        stream.write(json.dumps(config, indent=2) + "\n")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "tokenizer", "output", "vllm-package"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    try:
        output = prepare(args.source, args.tokenizer, args.output, args.vllm_package)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Legacy preparation FAILED: {error}", file=sys.stderr)
        return 1
    print(f"Prepared historical vLLM 0.11 directory: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
