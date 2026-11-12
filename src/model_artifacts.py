"""Export and statically validate FP32 or BF16 NanoReason (OLMo3) models."""

import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import torch
from huggingface_hub.errors import StrictDataclassError
from jinja2 import TemplateError
from safetensors import SafetensorError, safe_open
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)

TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
)


def read_json(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def fingerprint(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def model_files(source, tokenizer=None):
    """Choose model files and, when needed, an explicit evaluated tokenizer."""
    source = Path(source)
    donor = Path(tokenizer) if tokenizer is not None else source
    files = {
        name: source / name
        for name in ("config.json", "generation_config.json", "model.safetensors")
    }
    files.update({name: donor / name for name in TOKENIZER_FILES})
    if not files["chat_template.jinja"].exists():
        files["chat_template.jinja"] = source / "chat_template.jinja"
    for name in ("special_tokens_map.json", "chat_template.jinja"):
        if not os.path.lexists(files[name]):
            del files[name]
    files.update(
        {
            name: source / name
            for name in ("README.md", "LICENSE", "NOTICE")
            if os.path.lexists(source / name)
        }
    )
    for path in files.values():
        if not path.is_file():
            raise ValueError(f"Missing file or broken link: {path}")
    if (source / "model.safetensors.index.json").exists() or any(
        path.name != "model.safetensors" for path in source.glob("*.safetensors")
    ):
        raise ValueError("Only a single merged model.safetensors is supported")
    return files


def check_destination(destination, protected):
    """Reject existing destinations and overlap with sources or resolved link targets."""
    destination = Path(destination)
    if os.path.lexists(destination):
        raise FileExistsError(f"Destination already exists: {destination}")
    destination = destination.resolve()
    for path in protected:
        path = Path(path).resolve()
        if destination.is_relative_to(path) or path.is_relative_to(destination):
            raise ValueError(f"Destination overlaps source: {path}")
    return destination


def validate_rope(raw):
    """Check NanoReason RoPE values and agreement between old and nested fields.

    Transformers supplies architecture validation, but some invalid RoPE values
    only produce warnings. Check these here without repairing the saved config.
    """
    parameters = raw.get("rope_parameters")
    if parameters is None:
        parameters = {
            "full_attention": {
                **raw.get("rope_scaling", {}),
                "rope_theta": raw.get("rope_theta"),
            },
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": raw.get("rope_theta"),
            },
        }
    if not isinstance(parameters, dict) or set(parameters) != set(raw["layer_types"]):
        raise ValueError("RoPE attention types differ from layer_types")
    for attention, values in parameters.items():
        if not isinstance(values, dict):
            raise TypeError(f"Invalid RoPE parameters for {attention}")
        if values.get("rope_type") not in {"default", "yarn"}:
            raise ValueError(f"Unsupported RoPE type for {attention}")
        for name in (
            "rope_theta",
            "factor",
            "original_max_position_embeddings",
            "attention_factor",
            "beta_fast",
            "beta_slow",
        ):
            value = values.get(name)
            if (name == "rope_theta" or value is not None) and (
                not isinstance(value, (float, int))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"Invalid RoPE {attention}.{name}")
        if values["rope_type"] == "yarn":
            if (
                attention != "full_attention"
                or values.get("factor", 0) <= 1
                or not 0
                < values.get("original_max_position_embeddings", 0)
                <= raw["max_position_embeddings"]
            ):
                raise ValueError("Invalid YaRN factor or original context length")
            if values.get("beta_fast", 32) < values.get("beta_slow", 1):
                raise ValueError("YaRN beta_fast must be at least beta_slow")
    full = parameters["full_attention"]
    if "rope_theta" in raw and raw["rope_theta"] != full["rope_theta"]:
        raise ValueError("Nested and legacy rope_theta differ")
    if "rope_scaling" in raw and raw["rope_scaling"] != {
        k: v for k, v in full.items() if k != "rope_theta"
    }:
        raise ValueError("Nested and legacy rope_scaling differ")
    return parameters


def validate_model(directory, *, expected_sha256=None):
    """Read-only checks using safetensors headers and an OLMo3 model on meta."""
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"Not a model directory: {directory}")
    for path in (directory, *directory.rglob("*")):
        if path.is_symlink() or (
            not path.is_dir() and (not path.is_file() or path.stat().st_nlink != 1)
        ):
            raise ValueError(
                f"Standalone model contains a link or special file: {path}"
            )
    files = model_files(directory)
    documents = {
        name: read_json(path) for name, path in files.items() if name.endswith(".json")
    }
    raw, generation = documents["config.json"], documents["generation_config.json"]
    if raw.get("model_type") != "olmo3" or raw.get("architectures") != [
        "Olmo3ForCausalLM"
    ]:
        raise ValueError("Expected Olmo3ForCausalLM / olmo3")
    validate_rope(raw)
    try:
        config = AutoConfig.from_pretrained(
            directory, local_files_only=True, trust_remote_code=False
        )
    except StrictDataclassError as error:
        raise ValueError(f"Invalid model config: {error}") from error
    if (
        config.hidden_size % config.num_attention_heads
        or config.num_attention_heads % config.num_key_value_heads
    ):
        raise ValueError("Inconsistent attention head dimensions")
    if "sliding_attention" in config.layer_types and config.sliding_window <= 0:
        raise ValueError("Invalid sliding_window")
    GenerationConfig.from_dict(generation)
    tokenizer = AutoTokenizer.from_pretrained(
        directory, local_files_only=True, trust_remote_code=False
    )
    if len(tokenizer) != config.vocab_size:
        raise ValueError("Tokenizer/model vocabulary size mismatch")
    for name in ("eos_token_id", "pad_token_id", "bos_token_id"):
        value = raw.get(name)
        if value is not None and (
            not 0 <= value < len(tokenizer) or value != getattr(tokenizer, name)
        ):
            raise ValueError(f"Tokenizer/model {name} mismatch")
        if name in generation and generation[name] != value:
            raise ValueError(f"Generation config contradicts model {name}")
    template = tokenizer.chat_template
    if not isinstance(template, str) or not template.strip():
        raise ValueError("Missing chat template")
    embedded = documents["tokenizer_config.json"].get("chat_template")
    if embedded is not None and embedded != template:
        raise ValueError("Embedded and loaded chat templates differ")
    if (
        "chat_template.jinja" in files
        and files["chat_template.jinja"].read_text() != template
    ):
        raise ValueError("chat_template.jinja differs from loaded template")
    try:
        tokenizer.apply_chat_template(
            [{"role": "user", "content": "Hello"}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )
    except TemplateError as error:
        raise ValueError(f"Invalid chat template: {error}") from error

    try:
        with safe_open(
            files["model.safetensors"], framework="pt", device="cpu"
        ) as tensors:
            names = tensors.keys()
            shapes = {
                name: tuple(tensors.get_slice(name).get_shape()) for name in names
            }
            dtypes = {tensors.get_slice(name).get_dtype() for name in names}
    except SafetensorError as error:
        raise ValueError(f"Invalid safetensors: {error}") from error
    if dtypes not in ({"F32"}, {"BF16"}):
        raise ValueError(f"Expected uniform FP32 or BF16 tensors, got {dtypes}")
    dtype = {"F32": "float32", "BF16": "bfloat16"}[next(iter(dtypes))]
    declarations = [raw[name] for name in ("dtype", "torch_dtype") if name in raw]
    if not declarations or any(value != dtype for value in declarations):
        raise ValueError(f"Config dtype must match {dtype} weights")
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=False)
    expected = {name: tuple(value.shape) for name, value in model.state_dict().items()}
    if shapes != expected:
        raise ValueError("Safetensors tensor names/shapes do not match model config")
    report = {
        "model": str(directory.resolve()),
        "tensors": len(shapes),
        "dtype": "FP32" if dtype == "float32" else "BF16",
        "weight_bytes": files["model.safetensors"].stat().st_size,
        "use_cache": config.use_cache,
    }
    if expected_sha256 is not None:
        digest = fingerprint(files["model.safetensors"])
        if digest != expected_sha256.lower():
            raise ValueError(
                f"Weight SHA-256 mismatch: expected {expected_sha256}, got {digest}"
            )
        report["sha256"] = digest
    return report


def export_model(source, destination, *, tokenizer=None):
    """Copy exact files, validate staging once, and promote without overwriting.

    On failure, the staging directory remains available for inspection.
    Promotion uses GNU mv -nT because os.rename can overwrite an existing directory.
    """
    source = Path(source).resolve(strict=True)
    files = model_files(source, tokenizer)
    protected = [source, *(path.resolve().parent for path in files.values())]
    destination = check_destination(destination, protected)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.", suffix=".staging", dir=destination.parent
        )
    )
    try:
        for name, original in files.items():
            with original.open("rb") as reader, (staging / name).open("xb") as writer:
                shutil.copyfileobj(reader, writer)
        report = validate_model(staging)
        for name, original in files.items():
            if fingerprint(original) != fingerprint(staging / name):
                raise ValueError(f"Copied file differs from source: {name}")
        subprocess.run(["mv", "-nT", "--", str(staging), str(destination)], check=True)
        if staging.exists():
            raise FileExistsError(
                f"Destination appeared during promotion: {destination}"
            )
    except Exception as error:
        raise ValueError(
            f"Export failed; staging retained at {staging}: {error}"
        ) from error
    report["model"] = str(destination)
    return report
