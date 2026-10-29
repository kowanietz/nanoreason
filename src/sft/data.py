"""SFT tokenization, deterministic whole-example packing, and collation."""

import hashlib
import json
import math
import random
from itertools import chain
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    DataCollatorForSeq2Seq,
)


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
        raise ValueError(
            "Cannot align packs to global batch without duplicating examples"
        )
    for packs in plans:
        while len(packs) < count:
            index = max(range(len(packs)), key=lambda i: len(packs[i]))
            if len(packs[index]) < 2:
                raise ValueError("No splittable pack remains")
            pack = packs[index]
            packs[index : index + 1] = [pack[:-1], pack[-1:]]
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
        return {
            "input_ids": ids + [self.pad_token_id] * padding,
            "labels": labels + [-100] * padding,
            "attention_mask": [1] * len(ids) + [0] * padding,
            "example_count": len(members),
        }


class RepetitionCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer, label_pad_token_id=-100, pad_to_multiple_of=8
        )

    def __call__(self, rows):
        counts = [row["example_count"] for row in rows]
        batch = self.collator(
            [
                {key: value for key, value in row.items() if key != "example_count"}
                for row in rows
            ]
        )
        batch["example_count"] = torch.tensor(counts)
        return batch


def tokenize_sft_example(example, tokenizer, max_length):
    def assistant_span(messages, index, rendered):
        content = messages[index].get("content") or ""
        probe = [dict(message) for message in messages[: index + 1]]
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
            "input_ids": [
                input_ids[i * max_length : (i + 1) * max_length] for i in range(count)
            ],
            "labels": [
                labels[i * max_length : (i + 1) * max_length] for i in range(count)
            ],
            "attention_mask": [[1] * max_length for _ in range(count)],
        }

    with training_args.main_process_first():
        dataset = load_dataset("parquet", data_files=files, split="train")
        if data_config.get("shuffle", True):
            dataset = dataset.shuffle(seed=config["training"]["seed"])
        if "max_examples" in data_config:
            dataset = dataset.select(
                range(min(data_config["max_examples"], len(dataset)))
            )

        original = len(dataset)
        dataset = dataset.map(
            lambda example: tokenize_sft_example(example, tokenizer, max_length),
            remove_columns=dataset.column_names,
            num_proc=num_proc,
        )
        dataset = dataset.filter(lambda example: example["keep"], num_proc=num_proc)
        kept = len(dataset)
        dataset = dataset.remove_columns(["keep"])

        if data_config.get("repetition_manifest"):
            manifest = json.loads(Path(data_config["repetition_manifest"]).read_text())
            if original != manifest["unique_examples"] or kept != original:
                raise ValueError("Repetition must retain every selected example")
            if tokenized_digest(dataset) != manifest["tokenized_sha256"]:
                raise ValueError(
                    "Prepared token IDs/labels differ from production preprocessing"
                )
            lengths = [len(row["input_ids"]) for row in dataset]
            supervised = sum(
                sum(label != -100 for label in row["labels"]) for row in dataset
            )
            if (
                sum(lengths) != manifest["total_tokens"]
                or supervised != manifest["supervised_tokens_per_epoch"]
            ):
                raise ValueError(
                    "Prepared token counts differ from production preprocessing"
                )
            for row in dataset:
                if row["labels"][0] != -100:
                    raise ValueError(
                        "Repetition packs must begin with a masked prompt token"
                    )
                if (
                    row["input_ids"][-1] != tokenizer.eos_token_id
                    or row["labels"][-1] != tokenizer.eos_token_id
                ):
                    raise ValueError("Repetition example lacks supervised terminal EOS")
            plans = repetition_plan(
                lengths,
                max_length,
                manifest["epochs"],
                manifest["seed"],
                manifest["global_batch_size"],
            )
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
        raise RuntimeError(
            f"Need {needed:,} sequences; only {len(dataset):,} available"
        )

    if training_args.process_index == 0:
        print(
            f"Examples: {original:,} -> {kept:,}; sequences: {len(dataset):,}",
            flush=True,
        )
        example = dataset[0]
        tokens = [
            token
            for token, label in zip(example["input_ids"], example["labels"])
            if label != -100
        ]
        print(tokenizer.decode(tokens[:1000], skip_special_tokens=False), flush=True)

    return dataset
