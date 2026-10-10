#!/usr/bin/env python3

import argparse
import glob
import hashlib
import json
import os
import re
import time
from itertools import chain
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


class StopAtDeadline(TrainerCallback):
    def __init__(self, start_time, minutes):
        self.deadline = start_time + minutes * 60

    def on_step_end(self, args, state, control, **kwargs):
        expired = time.monotonic() >= self.deadline
        if dist.is_initialized():
            flag = torch.tensor(int(expired), device=f"cuda:{torch.cuda.current_device()}")
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            expired = bool(flag.item())

        if expired:
            control.should_save = True
            control.should_training_stop = True
        return control


def resolve_checkpoint(output_dir, setting):
    checkpoints = sorted(
        (
            path for path in Path(output_dir).glob("checkpoint-*")
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
    return str(path)


def verify_manifest(config, files, resuming, rank):
    output_dir = Path(config["training"]["output_dir"])
    manifest_path = output_dir / "run_manifest.json"
    settings = {
        "model": config["model"],
        "data": config["data"],
        "training": {
            key: value for key, value in config["training"].items()
            if key not in {"resume_from_checkpoint", "stop_after_minutes"}
        },
        "distributed": config["distributed"],
        "files": [
            {"path": str(Path(path).resolve()), "size": os.stat(path).st_size,
             "mtime_ns": os.stat(path).st_mtime_ns}
            for path in files
        ],
        "preprocessing_version": 1,
    }
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
        raise RuntimeError("Dataset or training configuration changed since the first job")


def make_tokenizer(model_config):
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["tokenizer_name_or_path"],
        trust_remote_code=model_config.get("trust_remote_code", False),
        use_fast=True,
        local_files_only=True,
    )
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer is required")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_training_data(config, tokenizer, files, training_args):
    data_config = config["data"]
    max_length = data_config["max_length"]
    num_proc = data_config.get("num_proc", 8)

    def assistant_span(messages, index, rendered):
        content = messages[index].get("content") or ""
        probe = [dict(message) for message in messages[:index + 1]]
        sentinel = f"\ue000ASSISTANT_CONTENT_{index}\ue001"
        while sentinel in rendered:
            sentinel += "_"
        probe[-1]["content"] = sentinel
        probe_text = tokenizer.apply_chat_template(
            probe, tokenize=False, add_generation_prompt=False
        )
        position = probe_text.find(sentinel)
        if position < 0 or not rendered.startswith(probe_text[:position]):
            raise ValueError("Assistant span does not match chat template")
        end = position + len(content)
        if rendered[position:end] != content:
            raise ValueError("Assistant content does not match chat template")
        return position, end

    def tokenize(example):
        messages = example["messages"]
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        encoded = tokenizer(
            rendered,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=True,
            max_length=max_length + 1,
        )
        input_ids = encoded["input_ids"]
        if len(input_ids) > max_length:
            return {"input_ids": [], "attention_mask": [], "labels": [], "keep": False}

        labels = [-100] * len(input_ids)
        for index, message in enumerate(messages):
            if message["role"] != "assistant":
                continue
            start, end = assistant_span(messages, index, rendered)
            if rendered.startswith("<|im_end|>", end):
                end += len("<|im_end|>")
            elif tokenizer.eos_token and rendered.startswith(tokenizer.eos_token, end):
                end += len(tokenizer.eos_token)
            for token_index, (left, right) in enumerate(encoded["offset_mapping"]):
                if left < end and start < right:
                    labels[token_index] = input_ids[token_index]

        if all(label == -100 for label in labels):
            raise ValueError("No assistant tokens found")
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
            "keep": True,
        }

    def pack(batch):
        input_ids = list(chain.from_iterable(batch["input_ids"]))
        labels = list(chain.from_iterable(batch["labels"]))
        count = len(input_ids) // max_length
        return {
            "input_ids": [input_ids[i * max_length:(i + 1) * max_length] for i in range(count)],
            "labels": [labels[i * max_length:(i + 1) * max_length] for i in range(count)],
            "attention_mask": [[1] * max_length for _ in range(count)],
        }

    with training_args.main_process_first():
        dataset = load_dataset("parquet", data_files=files, split="train")
        if data_config.get("shuffle", True):
            dataset = dataset.shuffle(seed=config["training"]["seed"])
        if "max_examples" in data_config:
            dataset = dataset.select(range(min(data_config["max_examples"], len(dataset))))

        original = len(dataset)
        dataset = dataset.map(
            tokenize, remove_columns=dataset.column_names, num_proc=num_proc
        )
        dataset = dataset.filter(lambda example: example["keep"], num_proc=num_proc)
        kept = len(dataset)
        dataset = dataset.remove_columns(["keep"])

        if data_config.get("packing", False):
            dataset = dataset.map(
                pack,
                batched=True,
                batch_size=data_config.get("packing_batch_size", 256),
                remove_columns=dataset.column_names,
                num_proc=num_proc,
            )

    if len(dataset) == 0:
        raise RuntimeError("Empty training dataset")

    needed = (
        config["training"]["max_steps"]
        * config["training"]["per_device_batch_size"]
        * config["training"]["gradient_accumulation_steps"]
        * training_args.world_size
    )
    if data_config.get("require_single_pass", False) and len(dataset) < needed:
        raise RuntimeError(f"Need {needed:,} sequences; only {len(dataset):,} available")

    if training_args.process_index == 0:
        print(f"Examples: {original:,} -> {kept:,}; sequences: {len(dataset):,}", flush=True)
        example = dataset[0]
        tokens = [token for token, label in zip(example["input_ids"], example["labels"])
                  if label != -100]
        print(tokenizer.decode(tokens[:1000], skip_special_tokens=False), flush=True)

    return dataset


def main():
    start = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    model_config = config["model"]
    train_config = config["training"]
    log_config = config["logging"]

    if config["data"].get("overlength_strategy", "drop") != "drop":
        raise ValueError("Only overlength_strategy: drop is supported")
    if config["distributed"]["fsdp_sharding"] != "full_shard":
        raise ValueError("Only full_shard FSDP is supported")

    training_args = TrainingArguments(
        output_dir=train_config["output_dir"],
        per_device_train_batch_size=train_config["per_device_batch_size"],
        gradient_accumulation_steps=train_config["gradient_accumulation_steps"],
        accelerator_config={"gradient_accumulation_kwargs": {
            "sync_each_batch": train_config.get("sync_each_batch", False)
        }},
        max_steps=train_config["max_steps"],
        learning_rate=train_config["learning_rate"],
        lr_scheduler_type="cosine",
        warmup_steps=train_config["warmup_ratio"],
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
        save_strategy="steps",
        save_steps=log_config["save_steps"],
        save_total_limit=log_config.get("save_total_limit", 2),
        save_only_model=False,
        ignore_data_skip=False,
        dataloader_drop_last=True,
        enable_jit_checkpoint=train_config.get("enable_jit_checkpoint", True),
        report_to=log_config["report_to"],
        seed=train_config["seed"],
    )

    files = sorted(glob.glob(config["data"]["path"]))
    if not files:
        raise FileNotFoundError(config["data"]["path"])
    checkpoint = resolve_checkpoint(
        train_config["output_dir"], train_config.get("resume_from_checkpoint")
    )
    verify_manifest(config, files, checkpoint, training_args.process_index)

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
        dtype=torch.bfloat16,
        trust_remote_code=model_config.get("trust_remote_code", False),
        attn_implementation=model_config.get("attention_implementation", "sdpa"),
        local_files_only=True,
    )
    model.config.use_cache = train_config["use_cache"]

    callbacks = []
    if train_config.get("stop_after_minutes") is not None:
        callbacks.append(StopAtDeadline(start, train_config["stop_after_minutes"]))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        callbacks=callbacks,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer, label_pad_token_id=-100, pad_to_multiple_of=8
        ),
    )
    torch.cuda.reset_peak_memory_stats()
    trainer.train(resume_from_checkpoint=checkpoint)
    torch.cuda.synchronize()
    print(
        f"rank {training_args.process_index}: "
        f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB allocated, "
        f"{torch.cuda.max_memory_reserved() / 2**30:.1f} GiB reserved",
        flush=True,
    )

    if trainer.state.global_step < train_config["max_steps"]:
        return

    if train_config.get("export_final_model", True):
        if trainer.is_fsdp_enabled:
            trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
        trainer.save_model(train_config["output_dir"])
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(train_config["output_dir"])

    if trainer.is_world_process_zero():
        temp = complete_marker.with_suffix(".tmp")
        temp.write_text(json.dumps({"global_step": trainer.state.global_step}))
        os.replace(temp, complete_marker)


if __name__ == "__main__":
    main()
