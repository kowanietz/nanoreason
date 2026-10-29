#!/usr/bin/env python3

import glob
import hashlib
import json
import os
import time
from pathlib import Path

import hydra
import torch
import yaml
from omegaconf import DictConfig, OmegaConf
from transformers import (
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from sft.checkpoints import (
    configure_repetition,
    export_model,
    guard_repetition_destination,
    resolve_checkpoint,
    verify_manifest,
)
from sft.data import RepetitionCollator, load_training_data
from sft.trainer import RepetitionTrainer, SaveRepetitionEpochs
from training_common import StopAtDeadline, make_tokenizer


@hydra.main(version_base="1.3", config_path="../configs", config_name=None)
def main(config: DictConfig):
    start = time.monotonic()
    config = OmegaConf.to_container(config, resolve=True)
    model_config = config["model"]
    load_dtype = {"auto": "auto", "bf16": torch.bfloat16, "fp32": torch.float32}[
        model_config["load_dtype"]
    ]
    train_config = config["training"]
    log_config = config["logging"]
    repetition = bool(config["data"].get("repetition_manifest"))
    if repetition:
        guard_repetition_destination(train_config["output_dir"])
        repetition_manifest = configure_repetition(
            config, int(os.environ.get("WORLD_SIZE", "1"))
        )

    if config["data"].get("overlength_strategy", "drop") != "drop":
        raise ValueError("Only overlength_strategy: drop is supported")
    if config["distributed"]["fsdp_sharding"] != "full_shard":
        raise ValueError("Only full_shard FSDP is supported")

    training_args = TrainingArguments(
        output_dir=train_config["output_dir"],
        per_device_train_batch_size=train_config["per_device_batch_size"],
        gradient_accumulation_steps=train_config["gradient_accumulation_steps"],
        accelerator_config={
            "gradient_accumulation_kwargs": {
                "sync_each_batch": train_config.get("sync_each_batch", False)
            }
        },
        max_steps=train_config["max_steps"],
        num_train_epochs=train_config.get("num_train_epochs", 3),
        learning_rate=train_config["learning_rate"],
        lr_scheduler_type="cosine",
        warmup_steps=train_config.get("warmup_steps", train_config["warmup_ratio"]),
        weight_decay=train_config["weight_decay"],
        max_grad_norm=train_config["max_grad_norm"],
        bf16=train_config["precision"] == "bf16",
        tf32=True,
        fsdp=True,
        fsdp_config={
            "version": 2,
            "reshard_after_forward": True,
            "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
            "transformer_layer_cls_to_wrap": ["Olmo3DecoderLayer"],
            "activation_checkpointing": train_config["gradient_checkpointing"],
            "state_dict_type": "SHARDED_STATE_DICT",
        },
        use_liger_kernel=True,
        liger_kernel_config={
            "rope": False,
            "rms_norm": False,
            "swiglu": False,
            "cross_entropy": False,
            "fused_linear_cross_entropy": True,
        },
        logging_steps=log_config["logging_steps"],
        save_strategy="no" if repetition else "steps",
        save_steps=log_config["save_steps"],
        save_total_limit=log_config.get("save_total_limit", 2),
        save_only_model=False,
        ignore_data_skip=False,
        dataloader_drop_last=True,
        remove_unused_columns=not repetition,
        enable_jit_checkpoint=train_config.get("enable_jit_checkpoint", True),
        report_to=log_config["report_to"],
        seed=train_config["seed"],
    )

    files = sorted(glob.glob(config["data"]["path"]))
    if not files:
        raise FileNotFoundError(config["data"]["path"])
    if repetition:
        if (
            len(files) != 1
            or hashlib.sha256(Path(files[0]).read_bytes()).hexdigest()
            != repetition_manifest["parquet_sha256"]
        ):
            raise ValueError("Prepared dataset checksum mismatch")
        source_hashes = repetition_manifest["source_checkpoint_metadata_sha256"]
        for filename, expected in source_hashes.items():
            if (
                hashlib.sha256(
                    (Path(model_config["name_or_path"]) / filename).read_bytes()
                ).hexdigest()
                != expected
            ):
                raise ValueError(
                    "Source checkpoint metadata changed since dataset preparation"
                )
    checkpoint = resolve_checkpoint(
        train_config["output_dir"],
        train_config.get("resume_from_checkpoint"),
        required_files=(
            [
                "scheduler.pt",
                "optimizer_0/.metadata",
                "pytorch_model_fsdp_0/.metadata",
                "repetition_metrics.json",
                "repetition_sampler.json",
            ]
            + [f"rng_state_{rank}.pth" for rank in range(training_args.world_size)]
        )
        if repetition
        else (),
        reject_older=repetition,
    )
    if repetition and checkpoint:
        if (
            Path(checkpoint).resolve().parent
            != Path(train_config["output_dir"]).resolve()
        ):
            raise ValueError("Resume only checkpoints belonging to this repetition run")
        sampler = json.loads((Path(checkpoint) / "repetition_sampler.json").read_text())
        saved = json.loads((Path(checkpoint) / "trainer_state.json").read_text())
        epoch, offset = divmod(saved["global_step"], train_config["steps_per_epoch"])
        if sampler != {
            "sampler": "SequentialSampler",
            "next_epoch": epoch,
            "next_pack_offset": offset * repetition_manifest["global_batch_size"],
            "epoch_plan_sha256": repetition_manifest["epoch_plan_sha256"],
        }:
            raise ValueError("Saved repetition sampler cursor/plan mismatch")
    verify_manifest(config, files, checkpoint, training_args.process_index)
    if (
        repetition
        and checkpoint
        and saved["global_step"]
        >= train_config["steps_per_epoch"] * train_config["stop_after_epoch"]
    ):
        print(
            "Requested repetition milestone already reached; evaluate before advancing",
            flush=True,
        )
        return
    if repetition and training_args.process_index == 0:
        (Path(train_config["output_dir"]) / "config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False)
        )
        if checkpoint and (Path(checkpoint) / "epoch_metrics.json").exists():
            (Path(train_config["output_dir"]) / "epoch_metrics.json").write_bytes(
                (Path(checkpoint) / "epoch_metrics.json").read_bytes()
            )

    output_dir = Path(train_config["output_dir"])
    complete_marker = output_dir / "training_complete.json"
    if complete_marker.exists():
        state = json.loads(complete_marker.read_text())
        if state.get("global_step") == train_config["max_steps"]:
            if training_args.process_index == 0:
                print("Training complete", flush=True)
            return

    tokenizer = make_tokenizer(model_config)
    dataset = load_training_data(config, tokenizer, files, training_args)
    model = AutoModelForCausalLM.from_pretrained(
        model_config["name_or_path"],
        dtype=load_dtype,
        trust_remote_code=model_config.get("trust_remote_code", False),
        attn_implementation=model_config.get("attention_implementation", "sdpa"),
        local_files_only=True,
    )
    model.config.use_cache = train_config["use_cache"]

    callbacks = []
    if repetition:
        epoch_callback = SaveRepetitionEpochs(
            train_config["steps_per_epoch"],
            log_config["save_epochs"],
            train_config["stop_after_epoch"],
            config["data"]["repetition_manifest"],
        )
        callbacks.append(epoch_callback)
    if train_config.get("stop_after_minutes") is not None:
        callbacks.append(StopAtDeadline(start, train_config["stop_after_minutes"]))

    trainer_class = RepetitionTrainer if repetition else Trainer
    extra = {"metrics_checkpoint": checkpoint} if repetition else {}
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=dataset,
        callbacks=callbacks,
        data_collator=RepetitionCollator(tokenizer)
        if repetition
        else DataCollatorForSeq2Seq(
            tokenizer=tokenizer, label_pad_token_id=-100, pad_to_multiple_of=8
        ),
        **extra,
    )
    if repetition:
        epoch_callback.trainer = trainer
    torch.cuda.reset_peak_memory_stats()
    trainer.train(resume_from_checkpoint=checkpoint)
    torch.cuda.synchronize()
    print(
        f"rank {training_args.process_index}: "
        f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB allocated, "
        f"{torch.cuda.max_memory_reserved() / 2**30:.1f} GiB reserved",
        flush=True,
    )

    if repetition and trainer.state.global_step in {
        train_config["steps_per_epoch"] * epoch for epoch in (4, 8, 12)
    }:
        epoch = trainer.state.global_step // train_config["steps_per_epoch"]
        export_path = output_dir / f"epoch-{epoch}-model"
        if export_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite completed milestone model: {export_path}"
            )
        export_model(trainer, tokenizer, export_path)

    if trainer.state.global_step < train_config["max_steps"]:
        return

    if train_config.get("export_final_model", True):
        export_model(trainer, tokenizer, train_config["output_dir"])

    if trainer.is_world_process_zero():
        temp = complete_marker.with_suffix(".tmp")
        temp.write_text(json.dumps({"global_step": trainer.state.global_step}))
        os.replace(temp, complete_marker)


if __name__ == "__main__":
    main()
