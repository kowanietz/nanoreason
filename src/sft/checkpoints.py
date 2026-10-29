"""SFT artifact provenance, complete-checkpoint selection, and model exports."""

import hashlib
import json
import os
import re
from pathlib import Path

import torch.distributed as dist


def resolve_checkpoint(output_dir, setting, required_files=(), reject_older=False):
    checkpoints = sorted(
        (
            path
            for path in Path(output_dir).glob("checkpoint-*")
            if path.is_dir() and re.fullmatch(r"checkpoint-\d+", path.name)
        ),
        key=lambda path: int(path.name.split("-")[-1]),
        reverse=True,
    )

    def complete(path):
        return (
            path.is_dir()
            and (path / "trainer_state.json").is_file()
            and not (path / "checkpoint-is-incomplete.txt").exists()
            and all((path / name).is_file() for name in required_files)
        )

    if setting in (None, False):
        if checkpoints:
            raise RuntimeError("Existing checkpoints found; enable resumption")
        return None

    if setting in (True, "auto"):
        for path in checkpoints:
            if complete(path):
                return str(path)
        if checkpoints or setting is True:
            raise RuntimeError("No complete checkpoint available")
        return None

    path = Path(setting)
    if not complete(path):
        raise RuntimeError(f"Invalid checkpoint: {path}")
    if reject_older and any(
        complete(other)
        and int(other.name.split("-")[-1]) > int(path.name.split("-")[-1])
        for other in checkpoints
    ):
        raise RuntimeError(
            "Refusing to rewind and overwrite completed repetition checkpoints"
        )
    return str(path)


def verify_manifest(config, files, resuming, rank):
    output_dir = Path(config["training"]["output_dir"])
    manifest_path = output_dir / "run_manifest.json"
    settings = {
        "model": {
            key: value
            for key, value in config["model"].items()
            if key != "load_dtype" or value != "bf16"
        },
        "data": config["data"],
        "training": {
            key: value
            for key, value in config["training"].items()
            if key
            not in {"resume_from_checkpoint", "stop_after_minutes", "stop_after_epoch"}
        },
        "distributed": config["distributed"],
        "files": [
            {
                "path": str(Path(path).resolve()),
                "size": os.stat(path).st_size,
                "mtime_ns": os.stat(path).st_mtime_ns,
            }
            for path in files
        ],
        "preprocessing_version": 1,
    }
    if config["data"].get("repetition_manifest"):
        settings["preprocessing_version"] = "repetition-whole-examples-v1"
        settings["repetition_manifest_sha256"] = hashlib.sha256(
            Path(config["data"]["repetition_manifest"]).read_bytes()
        ).hexdigest()
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()

    if rank == 0 and not manifest_path.exists():
        if resuming:
            raise RuntimeError("Missing manifest for existing checkpoint")
        output_dir.mkdir(parents=True, exist_ok=True)
        temp = manifest_path.with_suffix(".tmp")
        temp.write_text(json.dumps({"sha256": digest, "settings": settings}, indent=2))
        os.replace(temp, manifest_path)

    if dist.is_initialized():
        dist.barrier()

    if json.loads(manifest_path.read_text())["sha256"] != digest:
        raise RuntimeError(
            "Dataset or training configuration changed since the first job"
        )


def configure_repetition(config, world_size):
    """Derive update counts from the prepared manifest before model loading."""
    manifest = json.loads(Path(config["data"]["repetition_manifest"]).read_text())
    training = config["training"]
    if world_size != training["expected_world_size"]:
        raise ValueError(
            f"Repetition expects {training['expected_world_size']} ranks, got {world_size}"
        )
    expected = {
        "seed": training["seed"],
        "epochs": training["num_train_epochs"],
        "max_length": config["data"]["max_length"],
        "global_batch_size": world_size
        * training["per_device_batch_size"]
        * training["gradient_accumulation_steps"],
        "source_checkpoint": training["initialization_checkpoint"],
        "unique_examples": config["data"]["unique_examples"],
        "selection_seed": config["data"]["selection_seed"],
        "holdout_examples": config["data"]["holdout_examples"],
    }
    if any(manifest[key] != value for key, value in expected.items()):
        raise ValueError(
            "Prepared repetition manifest does not match training configuration"
        )
    if training["initialization_checkpoint"] != config["model"]["name_or_path"]:
        raise ValueError("Initialization checkpoint must match model weights path")
    if (
        Path(training["output_dir"]).resolve()
        == Path(training["initialization_checkpoint"]).resolve()
    ):
        raise ValueError("Source checkpoint must never be the training destination")
    if not config["data"]["packing"] or config["data"].get("shuffle", False):
        raise ValueError(
            "Repetition uses its own deterministic epoch shuffle and lossless packing"
        )
    if training["stop_after_epoch"] not in (4, 8, 12, 16):
        raise ValueError("Choose an explicit repetition milestone: 4, 8, 12, or 16")
    steps = manifest["packs_per_epoch"] // expected["global_batch_size"]
    if manifest["packs_per_epoch"] % expected["global_batch_size"]:
        raise ValueError("Pack count must be divisible by global batch size")
    training["max_steps"] = steps * manifest["epochs"]
    training["steps_per_epoch"] = steps
    return manifest


def guard_repetition_destination(output_dir):
    directory = Path(output_dir)
    if directory.exists() and not (directory / "run_manifest.json").exists():
        unexpected = [
            path.name for path in directory.iterdir() if path.name != "prepared"
        ]
        if unexpected:
            raise RuntimeError(
                f"Unmanifested artifacts in repetition destination: {unexpected}"
            )


def export_model(trainer, tokenizer, output_dir):
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model(str(output_dir))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(output_dir)
