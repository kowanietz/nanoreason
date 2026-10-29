"""CPU preflight checks for the supported GRPO configuration and SFT export."""

import hashlib
import json
from pathlib import Path

from safetensors import safe_open
from transformers import AutoConfig

from training_common import make_tokenizer

from .checkpoints import fingerprint


def validate_config(config: dict) -> dict:
    args, runtime = config["grpo"], config["runtime"]
    world = runtime["expected_world_size"]
    batch = args["per_device_train_batch_size"] * world
    generation = args["generation_batch_size"]
    if (
        world != 4
        or batch <= 0
        or args["num_generations"] < 2
        or args["gradient_accumulation_steps"] < 1
        or args["max_completion_length"] < 1
        or args["vllm_tensor_parallel_size"] not in (1, 2, 4)
        or generation <= 0
        or generation % batch
        or generation % args["num_generations"]
    ):
        raise ValueError("Invalid four-rank TP/generation batch/group sizes")
    if (
        generation // batch != args["gradient_accumulation_steps"]
        or args["num_iterations"] != 1
    ):
        raise ValueError(
            "Smoke checkpoints require one fresh generation batch per optimizer step"
        )
    if (
        args["vllm_max_model_length"]
        < config["data"]["max_prompt_tokens"] + args["max_completion_length"]
        or not 0 < args["vllm_gpu_memory_utilization"] < 1
    ):
        raise ValueError("Invalid vLLM context length or memory utilization")
    if (
        args["loss_type"] != "dr_grpo"
        or args["scale_rewards"] != "none"
        or args["beta"] != 0
        or not args["bf16"]
        or not args["gradient_checkpointing"]
        or not args["fsdp"]
        or args["fsdp_config"]["version"] != 2
        or not args["fsdp_config"]["reshard_after_forward"]
        or args["fsdp_config"]["cpu_offload"]
        or args["fsdp_config"]["activation_checkpointing"]
        or not args["use_vllm"]
        or args["vllm_mode"] != "colocate"
        or not args["vllm_enable_sleep_mode"]
        or args["remove_unused_columns"]
        or args["save_only_model"]
        or args["ignore_data_skip"]
        or args["use_liger_kernel"]
        or args["torch_compile"]
    ):
        raise ValueError("Unsupported Phase 3A training settings")
    stop = runtime["stop_after_step"]
    if not 1 <= args["max_steps"] <= 10 or (
        stop is not None and not 1 <= stop <= args["max_steps"]
    ):
        raise ValueError(
            "Smoke test and intermediate stop must be within 1–10 optimizer steps"
        )
    source, destination = (
        Path(config["model"]["name_or_path"]).resolve(),
        Path(args["output_dir"]).resolve(),
    )
    if (
        source == destination
        or source in destination.parents
        or destination in source.parents
    ):
        raise ValueError("RLVR output must be isolated from the SFT source")
    if (
        config["data"]["max_examples"] < args["max_steps"]
        or config["data"]["max_prompt_tokens"] < 1
    ):
        raise ValueError("Invalid deterministic smoke subset or prompt budget")
    source_id = config["data"].get("source_id")
    if source_id is not None and (
        not isinstance(source_id, str) or not source_id.strip()
    ):
        raise ValueError("Smoke source_id must be a nonempty string")
    return {
        "world_size": world,
        "generation_batch_size": generation,
        "steps_per_generation": generation // batch,
        "unique_prompts_per_generation": generation // args["num_generations"],
    }


def validate_model(config: dict, tokenizer) -> dict:
    source = Path(config["model"]["name_or_path"])
    for name in (
        "config.json",
        "model.safetensors",
        "training_complete.json",
        "tokenizer.json",
        "chat_template.jinja",
    ):
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
    if (
        json.loads((source / "training_complete.json").read_text())["global_step"]
        != 2784
    ):
        raise ValueError("Expected the completed E16 export")
    model_config = AutoConfig.from_pretrained(
        source, local_files_only=True, trust_remote_code=False
    )
    exported_tokenizer = make_tokenizer(
        {**config["model"], "tokenizer_name_or_path": str(source)}
    )
    if (
        model_config.model_type != "olmo3"
        or model_config.architectures != ["Olmo3ForCausalLM"]
        or len(tokenizer) != model_config.vocab_size
        or tokenizer.eos_token_id != model_config.eos_token_id
        or tokenizer.pad_token_id != model_config.pad_token_id
        or exported_tokenizer.backend_tokenizer.to_str()
        != tokenizer.backend_tokenizer.to_str()
        or (source / "chat_template.jinja").read_text() != tokenizer.chat_template
    ):
        raise ValueError("E16 architecture/tokenizer/vocabulary mismatch")
    with safe_open(
        source / "model.safetensors", framework="pt", device="cpu"
    ) as weights:
        for name in ("model.embed_tokens.weight", "lm_head.weight"):
            if weights.get_slice(name).get_shape() != [
                model_config.vocab_size,
                model_config.hidden_size,
            ]:
                raise ValueError("E16 embedding/output weight shape mismatch")
    return {
        "path": str(source.resolve()),
        "config": model_config.to_dict(),
        "config_sha256": fingerprint(source / "config.json"),
        "training_complete_sha256": fingerprint(source / "training_complete.json"),
        "source_run_manifest_sha256": fingerprint(source / "run_manifest.json"),
        "weights_size": (source / "model.safetensors").stat().st_size,
        "weights_mtime_ns": (source / "model.safetensors").stat().st_mtime_ns,
        "load_dtype": "bfloat16",
        "tokenizer_sha256": fingerprint(source / "tokenizer.json"),
        "chat_template_sha256": hashlib.sha256(
            tokenizer.chat_template.encode()
        ).hexdigest(),
    }
