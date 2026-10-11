#!/usr/bin/env python3

import argparse
import glob
import hashlib
import json
import math
import os
import random
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


def resolve_checkpoint(output_dir, setting, required_files=(), reject_older=False):
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
    if reject_older and any(complete(other) and int(other.name.split("-")[-1]) > int(path.name.split("-")[-1])
                            for other in checkpoints):
        raise RuntimeError("Refusing to rewind and overwrite completed repetition checkpoints")
    return str(path)


def verify_manifest(config, files, resuming, rank):
    output_dir = Path(config["training"]["output_dir"])
    manifest_path = output_dir / "run_manifest.json"
    settings = {
        "model": config["model"],
        "data": config["data"],
        "training": {
            key: value for key, value in config["training"].items()
            if key not in {"resume_from_checkpoint", "stop_after_minutes", "stop_after_epoch"}
        },
        "distributed": config["distributed"],
        "files": [
            {"path": str(Path(path).resolve()), "size": os.stat(path).st_size,
             "mtime_ns": os.stat(path).st_mtime_ns}
            for path in files
        ],
        "preprocessing_version": 1,
    }
    if config["data"].get("repetition_manifest"):
        settings["preprocessing_version"] = "repetition-whole-examples-v1"
        settings["repetition_manifest_sha256"] = hashlib.sha256(
            Path(config["data"]["repetition_manifest"]).read_bytes()).hexdigest()
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


def tokenized_digest(rows):
    digest = hashlib.sha256()
    for row in rows:
        clean = {key: row[key] for key in ("input_ids", "attention_mask", "labels")}
        digest.update(json.dumps(clean, sort_keys=True).encode() + b"\n")
    return digest.hexdigest()


def repetition_plan(lengths, max_length, epochs, seed, global_batch_size):
    """Shuffle each epoch; concatenate whole examples without dropping tails.

    Equalize pack counts by splitting packs, never by repeating examples. Making
    the count divisible by the global batch prevents distributed sampler padding.
    """
    if not lengths or any(length <= 0 or length > max_length for length in lengths):
        raise ValueError("Repetition examples must fit completely within one sequence")
    plans = []
    for epoch in range(epochs):
        order = list(range(len(lengths)))
        random.Random(seed + epoch).shuffle(order)
        packs, current, size = [], [], 0
        for index in order:
            if current and size + lengths[index] > max_length:
                packs.append(current)
                current, size = [], 0
            current.append(index)
            size += lengths[index]
        packs.append(current)
        plans.append(packs)
    count = math.ceil(max(map(len, plans)) / global_batch_size) * global_batch_size
    if count > len(lengths):
        raise ValueError("Cannot align packs to global batch without duplicating examples")
    for packs in plans:
        while len(packs) < count:
            index = max(range(len(packs)), key=lambda i: len(packs[i]))
            if len(packs[index]) < 2:
                raise ValueError("No splittable pack remains")
            pack = packs[index]
            packs[index:index + 1] = [pack[:-1], pack[-1:]]
        if sorted(i for pack in packs for i in pack) != list(range(len(lengths))):
            raise ValueError("Epoch must present each source example exactly once")
    return plans


class RepetitionDataset(torch.utils.data.Dataset):
    def __init__(self, examples, plans, max_length, pad_token_id):
        self.examples, self.plans = examples, plans
        self.max_length, self.pad_token_id = max_length, pad_token_id
        self.epoch = 0

    def set_epoch(self, epoch):
        if not 0 <= epoch < len(self.plans):
            raise ValueError(f"Unplanned repetition epoch: {epoch}")
        self.epoch = epoch

    def __len__(self):
        return len(self.plans[0])

    def __getitem__(self, index):
        members = self.plans[self.epoch][index]
        examples = [self.examples[i] for i in members]
        ids = list(chain.from_iterable(row["input_ids"] for row in examples))
        labels = list(chain.from_iterable(row["labels"] for row in examples))
        padding = self.max_length - len(ids)
        if padding < 0:
            raise ValueError("Repetition pack exceeds sequence limit")
        return {"input_ids": ids + [self.pad_token_id] * padding,
                "labels": labels + [-100] * padding,
                "attention_mask": [1] * len(ids) + [0] * padding,
                "example_count": len(members)}


def configure_repetition(config, world_size):
    """Derive update counts from the prepared manifest before model loading."""
    manifest = json.loads(Path(config["data"]["repetition_manifest"]).read_text())
    training = config["training"]
    if world_size != training["expected_world_size"]:
        raise ValueError(f"Repetition expects {training['expected_world_size']} ranks, got {world_size}")
    expected = {"seed": training["seed"], "epochs": training["num_train_epochs"],
                "max_length": config["data"]["max_length"],
                "global_batch_size": world_size * training["per_device_batch_size"] * training["gradient_accumulation_steps"],
                "source_checkpoint": training["initialization_checkpoint"],
                "unique_examples": config["data"]["unique_examples"],
                "selection_seed": config["data"]["selection_seed"],
                "holdout_examples": config["data"]["holdout_examples"]}
    if any(manifest[key] != value for key, value in expected.items()):
        raise ValueError("Prepared repetition manifest does not match training configuration")
    if training["initialization_checkpoint"] != config["model"]["name_or_path"]:
        raise ValueError("Initialization checkpoint must match model weights path")
    if Path(training["output_dir"]).resolve() == Path(training["initialization_checkpoint"]).resolve():
        raise ValueError("Source checkpoint must never be the training destination")
    if not config["data"]["packing"] or config["data"].get("shuffle", False):
        raise ValueError("Repetition uses its own deterministic epoch shuffle and lossless packing")
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
        unexpected = [path.name for path in directory.iterdir() if path.name != "prepared"]
        if unexpected:
            raise RuntimeError(f"Unmanifested artifacts in repetition destination: {unexpected}")


def export_model(trainer, tokenizer, output_dir):
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model(str(output_dir))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(output_dir)


class RepetitionCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, label_pad_token_id=-100, pad_to_multiple_of=8)

    def __call__(self, rows):
        counts = [row["example_count"] for row in rows]
        batch = self.collator([{key: value for key, value in row.items() if key != "example_count"} for row in rows])
        batch["example_count"] = torch.tensor(counts)
        return batch


class RepetitionTrainer(Trainer):
    """Use Trainer's training loop; add ordering and Liger's masked accuracy."""
    def __init__(self, *args, metrics_checkpoint=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.repetition_totals = None
        self.repetition_base = torch.zeros(5, dtype=torch.float64)
        if metrics_checkpoint:
            stats = json.loads((Path(metrics_checkpoint) / "repetition_metrics.json").read_text())
            self.repetition_base = torch.tensor([stats[key] for key in (
                "example_presentations", "supervised_tokens", "correct_assistant_tokens", "optimization_loss_sum", "microbatches")], dtype=torch.float64)

    def _get_train_sampler(self, train_dataset=None):
        return torch.utils.data.SequentialSampler(self.train_dataset if train_dataset is None else train_dataset)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = dict(inputs)
        examples = inputs.pop("example_count").sum()
        targets = (inputs["labels"][:, 1:] != -100).sum()
        inputs["return_token_accuracy"] = True
        loss, outputs = super().compute_loss(model, inputs, True, num_items_in_batch)
        accuracy = getattr(outputs, "token_accuracy", None)
        if accuracy is None:
            raise RuntimeError("Liger did not return masked token accuracy")
        counts = torch.stack([examples.double(), targets.double(), accuracy.detach().double() * targets,
                              loss.detach().double(), torch.ones((), device=targets.device, dtype=torch.float64)])
        if self.repetition_totals is None:
            self.repetition_totals = torch.zeros_like(counts)
        self.repetition_totals += counts.detach()
        return (loss, outputs) if return_outputs else loss

    def repetition_metrics(self):
        totals = self.repetition_totals.clone() if self.repetition_totals is not None else torch.zeros(
            5, dtype=torch.float64, device=self.args.device)
        if dist.is_initialized():
            dist.all_reduce(totals)
        totals += self.repetition_base.to(totals.device)
        examples, tokens, correct, loss_sum, microbatches = totals.tolist()
        return {"example_presentations": int(examples), "supervised_tokens": int(tokens),
                "correct_assistant_tokens": correct,
                "masked_assistant_token_accuracy": correct / tokens if tokens else None,
                "optimization_loss_sum": loss_sum, "microbatches": int(microbatches),
                "optimizer_steps": self.state.global_step}

    def log(self, logs, start_time=None):
        metrics = self.repetition_metrics()
        if "train_loss" in logs and metrics["microbatches"]:
            # Trainer's final train_loss uses a session loss sum divided by the
            # cumulative step counter on resume. Our persisted sum covers all sessions.
            logs = {**logs, "train_loss": metrics["optimization_loss_sum"] / metrics["microbatches"]}
        super().log({**logs, **metrics}, start_time)


class SaveRepetitionEpochs(TrainerCallback):
    def __init__(self, steps_per_epoch, epochs, stop_after_epoch, manifest_path=None):
        self.steps_per_epoch, self.epochs, self.stop_after_epoch = steps_per_epoch, epochs, stop_after_epoch
        self.manifest_path = manifest_path
        self.trainer = None

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.steps_per_epoch == 0:
            control.should_log = True
        if state.global_step in {self.steps_per_epoch * epoch for epoch in self.epochs}:
            control.should_save = True
        if state.global_step >= self.steps_per_epoch * self.stop_after_epoch:
            control.should_save = True
            control.should_training_stop = True
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "loss" not in logs or not state.global_step or state.global_step % self.steps_per_epoch:
            return control
        path = Path(args.output_dir) / "epoch_metrics.json"
        if state.is_world_process_zero:
            history = json.loads(path.read_text()) if path.exists() else []
            epoch = state.global_step // self.steps_per_epoch
            # Use cumulative counters to retain correct totals across resumption.
            previous = next((row for row in reversed(history) if row["data_epoch"] == epoch - 1), None)
            row = {**logs, "data_epoch": epoch, "effective_optimizer_steps": self.steps_per_epoch,
                   "learning_rate": self.trainer.lr_scheduler.get_last_lr()[0]}
            for key in ("example_presentations", "supervised_tokens", "correct_assistant_tokens", "optimization_loss_sum", "microbatches"):
                row[f"epoch_{key}"] = logs[key] - (previous[key] if previous else 0)
            row["epoch_masked_assistant_token_accuracy"] = row["epoch_correct_assistant_tokens"] / row["epoch_supervised_tokens"]
            row["epoch_loss"] = row["epoch_optimization_loss_sum"] / row["epoch_microbatches"]
            if row["epoch_example_presentations"] != len(self.trainer.train_dataset.examples):
                raise RuntimeError("Actual epoch example presentations differ from selected count")
            expected = sum(sum(label != -100 for label in example["labels"]) for example in self.trainer.train_dataset.examples)
            if row["epoch_supervised_tokens"] != expected:
                raise RuntimeError("Actual epoch supervised-token count differs from prepared count")
            history = [record for record in history if record["data_epoch"] != epoch] + [row]
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(history, indent=2))
            os.replace(temporary, path)
        return control

    def on_save(self, args, state, control, **kwargs):
        metrics = self.trainer.repetition_metrics()
        metrics["data_epoch"] = state.global_step / self.steps_per_epoch
        metrics["latest_logged_loss"] = next((row["loss"] for row in reversed(state.log_history) if "loss" in row), None)
        if state.is_world_process_zero:
            path = Path(args.output_dir) / f"checkpoint-{state.global_step}" / "repetition_metrics.json"
            path.write_text(json.dumps(metrics, indent=2))
            # Sequential sampling needs no random sampler generator. Persist the
            # exact next cursor; Trainer restores RNG and skips consumed batches.
            manifest_path = Path(self.manifest_path) if self.manifest_path else Path(args.output_dir) / "prepared/manifest.json"
            manifest = json.loads(manifest_path.read_text())
            epoch, offset = divmod(state.global_step, self.steps_per_epoch)
            sampler = {"sampler": "SequentialSampler", "next_epoch": epoch,
                       "next_pack_offset": offset * manifest["global_batch_size"],
                       "epoch_plan_sha256": manifest["epoch_plan_sha256"]}
            (path.parent / "repetition_sampler.json").write_text(json.dumps(sampler, indent=2))
            epoch_history = Path(args.output_dir) / "epoch_metrics.json"
            if epoch_history.exists():
                (path.parent / "epoch_metrics.json").write_bytes(epoch_history.read_bytes())
        return control


def tokenize_sft_example(example, tokenizer, max_length):
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


def load_training_data(config, tokenizer, files, training_args):
    data_config = config["data"]
    max_length = data_config["max_length"]
    num_proc = data_config.get("num_proc", 8)

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
            lambda example: tokenize_sft_example(example, tokenizer, max_length),
            remove_columns=dataset.column_names, num_proc=num_proc
        )
        dataset = dataset.filter(lambda example: example["keep"], num_proc=num_proc)
        kept = len(dataset)
        dataset = dataset.remove_columns(["keep"])

        if data_config.get("repetition_manifest"):
            manifest = json.loads(Path(data_config["repetition_manifest"]).read_text())
            if original != manifest["unique_examples"] or kept != original:
                raise ValueError("Repetition must retain every selected example")
            if tokenized_digest(dataset) != manifest["tokenized_sha256"]:
                raise ValueError("Prepared token IDs/labels differ from production preprocessing")
            lengths = [len(row["input_ids"]) for row in dataset]
            supervised = sum(sum(label != -100 for label in row["labels"]) for row in dataset)
            if sum(lengths) != manifest["total_tokens"] or supervised != manifest["supervised_tokens_per_epoch"]:
                raise ValueError("Prepared token counts differ from production preprocessing")
            for row in dataset:
                if row["labels"][0] != -100:
                    raise ValueError("Repetition packs must begin with a masked prompt token")
                if row["input_ids"][-1] != tokenizer.eos_token_id or row["labels"][-1] != tokenizer.eos_token_id:
                    raise ValueError("Repetition example lacks supervised terminal EOS")
            plans = repetition_plan(lengths, max_length, manifest["epochs"], manifest["seed"], manifest["global_batch_size"])
            plan_digest = hashlib.sha256(json.dumps(plans).encode()).hexdigest()
            if plan_digest != manifest["epoch_plan_sha256"]:
                raise ValueError("Repetition packing plan changed")
            return RepetitionDataset(dataset, plans, max_length, tokenizer.pad_token_id)

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
    parser.add_argument("--stop-after-epoch", type=int, choices=(4, 8, 12, 16),
                        help="Advance repetition only after evaluating the preceding milestone")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    model_config = config["model"]
    train_config = config["training"]
    log_config = config["logging"]
    repetition = bool(config["data"].get("repetition_manifest"))
    if repetition:
        if args.stop_after_epoch is not None:
            train_config["stop_after_epoch"] = args.stop_after_epoch
        guard_repetition_destination(train_config["output_dir"])
        repetition_manifest = configure_repetition(config, int(os.environ.get("WORLD_SIZE", "1")))

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
        if len(files) != 1 or hashlib.sha256(Path(files[0]).read_bytes()).hexdigest() != repetition_manifest["parquet_sha256"]:
            raise ValueError("Prepared dataset checksum mismatch")
        source_hashes = repetition_manifest["source_checkpoint_metadata_sha256"]
        for filename, expected in source_hashes.items():
            if hashlib.sha256((Path(model_config["name_or_path"]) / filename).read_bytes()).hexdigest() != expected:
                raise ValueError("Source checkpoint metadata changed since dataset preparation")
    checkpoint = resolve_checkpoint(
        train_config["output_dir"], train_config.get("resume_from_checkpoint"),
        required_files=(
            ["scheduler.pt", "optimizer_0/.metadata", "pytorch_model_fsdp_0/.metadata",
             "repetition_metrics.json", "repetition_sampler.json"]
            + [f"rng_state_{rank}.pth" for rank in range(training_args.world_size)]
        ) if repetition else (),
        reject_older=repetition,
    )
    if repetition and checkpoint:
        if Path(checkpoint).resolve().parent != Path(train_config["output_dir"]).resolve():
            raise ValueError("Resume only checkpoints belonging to this repetition run")
        sampler = json.loads((Path(checkpoint) / "repetition_sampler.json").read_text())
        saved = json.loads((Path(checkpoint) / "trainer_state.json").read_text())
        epoch, offset = divmod(saved["global_step"], train_config["steps_per_epoch"])
        if sampler != {"sampler": "SequentialSampler", "next_epoch": epoch,
                       "next_pack_offset": offset * repetition_manifest["global_batch_size"],
                       "epoch_plan_sha256": repetition_manifest["epoch_plan_sha256"]}:
            raise ValueError("Saved repetition sampler cursor/plan mismatch")
    verify_manifest(config, files, checkpoint, training_args.process_index)
    if repetition and checkpoint and saved["global_step"] >= train_config["steps_per_epoch"] * train_config["stop_after_epoch"]:
        print("Requested repetition milestone already reached; evaluate before advancing", flush=True)
        return
    if repetition and training_args.process_index == 0:
        (Path(train_config["output_dir"]) / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        if checkpoint and (Path(checkpoint) / "epoch_metrics.json").exists():
            (Path(train_config["output_dir"]) / "epoch_metrics.json").write_bytes(
                (Path(checkpoint) / "epoch_metrics.json").read_bytes())

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
    if repetition:
        epoch_callback = SaveRepetitionEpochs(train_config["steps_per_epoch"], log_config["save_epochs"], train_config["stop_after_epoch"], config["data"]["repetition_manifest"])
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
        data_collator=RepetitionCollator(tokenizer) if repetition else DataCollatorForSeq2Seq(
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

    if repetition and trainer.state.global_step in {train_config["steps_per_epoch"] * epoch for epoch in (4, 8, 12)}:
        epoch = trainer.state.global_step // train_config["steps_per_epoch"]
        export_path = output_dir / f"epoch-{epoch}-model"
        if export_path.exists():
            raise FileExistsError(f"Refusing to overwrite completed milestone model: {export_path}")
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
