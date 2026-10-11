#!/usr/bin/env python3
"""Prepare an OLMo3 export for vLLM 0.11.0 in a separate directory.

Write a compatible config and validation report. Link to the existing weights,
generation config and Think-SFT tokenizer files. Source files are read-only;
--validate-only checks an existing inference directory without writing.
"""

import argparse
import copy
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from safetensors import safe_open
from transformers import AutoTokenizer

ROOT = Path("/p/scratch/rcfd/kowanietz1")
VLLM_PACKAGE = ROOT / "olmes/.venv/lib/python3.12/site-packages/vllm"
ARCHITECTURE_FIELDS = (
    "architectures",
    "model_type",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "layer_types",
    "sliding_window",
    "max_position_embeddings",
    "hidden_act",
    "rms_norm_eps",
    "attention_bias",
    "tie_word_embeddings",
    "eos_token_id",
    "pad_token_id",
    "vocab_size",
)


def read_json(path):
    return json.loads(Path(path).read_text())


def compatible_config(base, exported):
    """Add legacy RoPE fields while keeping the nested parameters unchanged."""
    if (
        exported.get("architectures") != ["Olmo3ForCausalLM"]
        or exported.get("model_type") != "olmo3"
    ):
        raise ValueError("Expected Olmo3ForCausalLM / model_type=olmo3")
    for key in ARCHITECTURE_FIELDS:
        if key not in base or exported.get(key) != base[key]:
            raise ValueError(
                f"Architecture/token configuration differs from Base: {key}"
            )

    layers = exported["layer_types"]
    if len(layers) != exported["num_hidden_layers"] or set(layers) != {
        "full_attention",
        "sliding_attention",
    }:
        raise ValueError("Invalid full/sliding attention layer layout")
    if (
        not exported["sliding_window"]
        or exported["hidden_size"] % exported["num_attention_heads"]
    ):
        raise ValueError("Invalid sliding window or head dimensions")

    result = copy.deepcopy(exported)
    nested = exported.get("rope_parameters")
    if nested is not None:
        if set(nested) != {"full_attention", "sliding_attention"}:
            raise ValueError("Unsupported nested RoPE attention types")
        full, sliding = nested["full_attention"], nested["sliding_attention"]
        if (
            set(sliding) != {"rope_type", "rope_theta"}
            or sliding["rope_type"] != "default"
        ):
            raise ValueError("vLLM 0.11.0 requires default sliding-attention RoPE")
        theta = full.get("rope_theta")
        if theta is None or theta <= 0 or theta != sliding["rope_theta"]:
            raise ValueError(
                "vLLM 0.11.0 cannot represent different full/sliding RoPE theta"
            )
        scaling = {key: value for key, value in full.items() if key != "rope_theta"}
    else:
        theta, scaling = exported.get("rope_theta"), exported.get("rope_scaling")

    allowed = {
        "rope_type",
        "factor",
        "original_max_position_embeddings",
        "attention_factor",
        "beta_fast",
        "beta_slow",
    }
    if (
        not isinstance(scaling, dict)
        or set(scaling) - allowed
        or scaling.get("rope_type") != "yarn"
    ):
        raise ValueError("Expected supported full-attention YaRN parameters")
    factor = scaling.get("factor", 0)
    if factor <= 1 or scaling.get("original_max_position_embeddings", 0) <= 0:
        raise ValueError("Invalid YaRN factor or original context length")
    # vLLM computes the default YaRN magnitude instead of reading attention_factor.
    if not math.isclose(
        scaling.get("attention_factor", 1 + 0.1 * math.log(factor)),
        1 + 0.1 * math.log(factor),
        rel_tol=1e-12,
    ):
        raise ValueError(
            "Nondefault YaRN attention_factor is not represented by this conversion"
        )

    for key, value in {"rope_theta": theta, "rope_scaling": scaling}.items():
        if key in exported and exported[key] != value:
            raise ValueError(f"Inconsistent nested and legacy {key}")
        if base.get(key) != value:
            raise ValueError(f"Derived {key} differs from original Base")
        result[key] = value
    result["use_cache"] = True
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
    hashes = {}
    for relative, fragments in checks.items():
        path = Path(package) / relative
        text = path.read_text()
        if not all(fragment in text for fragment in fragments):
            raise ValueError(
                f"Unrecognized vLLM implementation: {path}; recheck compatibility"
            )
        hashes[relative] = hashlib.sha256(text.encode()).hexdigest()
    return hashes


def inspect_checkpoint(base, checkpoint, tokenizer, vllm_package=VLLM_PACKAGE):
    """Check config, tokenizer and weight metadata without writing files.

    Return the compatible config, source links and validation report used by
    both command-line utilities. No tensor data is loaded.
    """
    base, checkpoint, tokenizer = (
        Path(p).resolve() for p in (base, checkpoint, tokenizer)
    )

    config = compatible_config(
        read_json(base / "config.json"), read_json(checkpoint / "config.json")
    )
    vllm_hashes = validate_vllm(vllm_package)

    canonical = AutoTokenizer.from_pretrained(
        tokenizer, local_files_only=True, use_fast=True
    )
    exported = AutoTokenizer.from_pretrained(
        checkpoint, local_files_only=True, use_fast=True
    )
    for name, value in (("eos_token_id", 100257), ("pad_token_id", 100277)):
        if (
            config[name] != value
            or getattr(canonical, name) != value
            or getattr(exported, name) != value
        ):
            raise ValueError(f"Incompatible model/Think-SFT/exported tokenizer {name}")
    if len(canonical) != config["vocab_size"] or config["vocab_size"] != 100278:
        raise ValueError("Tokenizer/model vocabulary size mismatch")
    if (
        canonical.get_vocab() != exported.get_vocab()
        or canonical.chat_template != exported.chat_template
    ):
        raise ValueError(
            "Exported tokenizer vocabulary or chat template differs from Think-SFT"
        )
    if (
        canonical.convert_tokens_to_ids("<|im_end|>") != 100265
        or canonical.eos_token != "<|endoftext|>"
    ):
        raise ValueError("EOS and message-separator token conventions differ")

    generation = read_json(checkpoint / "generation_config.json")
    for key in ("eos_token_id", "pad_token_id"):
        if key in generation and generation[key] != config[key]:
            raise ValueError(f"Generation config contradicts model {key}")

    weights = checkpoint / "model.safetensors"
    dtypes, shapes = Counter(), {}
    with safe_open(weights, framework="pt", device="cpu") as tensors:
        for name in tensors.keys():  # noqa: SIM118 - safe_open exposes keys(), not dict iteration.
            view = tensors.get_slice(
                name
            )  # Read dtype and shape from the header without loading tensor data.
            dtypes[view.get_dtype()] += 1
            shapes[name] = view.get_shape()
    if set(dtypes) != {"F32"}:
        raise ValueError(
            f"Expected FP32 export; unsupported actual tensor dtypes: {dict(dtypes)}"
        )
    for name in ("model.embed_tokens.weight", "lm_head.weight"):
        if shapes.get(name) != [config["vocab_size"], config["hidden_size"]]:
            raise ValueError(f"Weight architecture mismatch: {name}")
    for layer in range(config["num_hidden_layers"]):
        if (
            shapes.get(f"model.layers.{layer}.self_attn.q_proj.weight")
            != [config["hidden_size"]] * 2
        ):
            raise ValueError(f"Missing/incompatible attention weights in layer {layer}")
    if (
        config.get("dtype", "float32") != "float32"
        or config.get("torch_dtype", "float32") != "float32"
    ):
        raise ValueError("Config dtype misrepresents actual FP32 weights")
    config["dtype"] = config["torch_dtype"] = (
        "float32"  # Both Transformers formats must describe the actual storage dtype.
    )

    links = {
        "model.safetensors": weights,
        "generation_config.json": checkpoint / "generation_config.json",
    }
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "special_tokens_map.json",
    ):
        if (tokenizer / name).is_file():
            links[name] = tokenizer / name

    report = dict(  # noqa: C408
        source=str(checkpoint),
        base=str(base),
        tokenizer=str(tokenizer),
        source_config_sha256=hashlib.sha256(
            (checkpoint / "config.json").read_bytes()
        ).hexdigest(),
        source_weight_size=weights.stat().st_size,
        source_weight_mtime_ns=weights.stat().st_mtime_ns,
        actual_tensor_dtypes=dict(dtypes),
        tensor_count=sum(dtypes.values()),
        inference_dtype="bfloat16",
        eos_token_id=canonical.eos_token_id,
        pad_token_id=canonical.pad_token_id,
        vocab_size=len(canonical),
        message_separator_id=100265,
        chat_template_sha256=hashlib.sha256(
            canonical.chat_template.encode()
        ).hexdigest(),
        original_rope_parameters=config.get("rope_parameters"),
        legacy_rope_theta=config["rope_theta"],
        legacy_rope_scaling=config["rope_scaling"],
        vllm_source_sha256=vllm_hashes,
    )

    return config, links, report


def prepare(
    base, checkpoint, tokenizer, output, vllm_package=VLLM_PACKAGE, validate_only=False
):
    base, checkpoint, tokenizer, output = (
        Path(p).resolve() for p in (base, checkpoint, tokenizer, output)
    )
    for protected in (base, checkpoint, tokenizer):
        if output.is_relative_to(protected) or protected.is_relative_to(output):
            raise ValueError(
                f"Inference output must be separate from protected source: {protected}"
            )

    config, links, report = inspect_checkpoint(
        base, checkpoint, tokenizer, vllm_package
    )

    documents = {"config.json": config, "repair_report.json": report}
    expected = set(links) | set(documents)
    if output.exists():
        unexpected = {p.name for p in output.iterdir()} - expected
        if unexpected:
            raise ValueError(
                f"Refusing incompatible output directory; unexpected files: {sorted(unexpected)}"
            )

    for name, target in links.items():
        path = output / name
        if path.exists() or path.is_symlink():
            if not path.is_symlink() or path.resolve() != target.resolve():
                raise ValueError(f"Existing inference link differs: {path}")
        elif validate_only:
            raise FileNotFoundError(path)

    for name, value in documents.items():
        path = output / name
        if path.exists() or path.is_symlink():
            if path.is_symlink() or read_json(path) != value:
                raise ValueError(f"Existing inference metadata differs: {path}")
        elif validate_only:
            raise FileNotFoundError(path)

    if not validate_only:
        output.mkdir(parents=True, exist_ok=True)
        for name, target in links.items():
            path = output / name
            if not path.is_symlink():
                path.symlink_to(target)
        for name, value in documents.items():
            path = output / name
            if not path.exists():
                with path.open("x") as stream:
                    stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")

    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Separate inference directory; in-place repair is forbidden",
    )
    parser.add_argument(
        "--tokenizer", type=Path, default=ROOT / "models/Olmo-3-7B-Think-SFT"
    )
    parser.add_argument("--vllm-package", type=Path, default=VLLM_PACKAGE)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    prepare(
        args.base,
        args.checkpoint,
        args.tokenizer,
        args.output,
        args.vllm_package,
        args.validate_only,
    )


if __name__ == "__main__":
    main()
