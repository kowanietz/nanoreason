#!/usr/bin/env python3

import argparse
import glob
from itertools import chain

import torch
import yaml
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

parser = argparse.ArgumentParser()
parser.add_argument("config")
args = parser.parse_args()

with open(args.config) as f:
    config = yaml.safe_load(f)

model_config = config["model"]
data_config = config["data"]
train_config = config["training"]
distributed_config = config["distributed"]
logging_config = config["logging"]

max_length = data_config["max_length"]
packing = data_config.get("packing", False)
num_proc = data_config.get("num_proc", 8)

data_files = sorted(glob.glob(data_config["path"]))

if not data_files:
    raise ValueError(f"No dataset found at {data_config['path']}")

if data_config.get("overlength_strategy", "drop") != "drop":
    raise ValueError(
        "This training script currently only supports "
        "data.overlength_strategy: drop"
    )

tokenizer = AutoTokenizer.from_pretrained(
    model_config["tokenizer_name_or_path"],
    trust_remote_code=model_config.get("trust_remote_code", False),
    use_fast=True,
    local_files_only=True,
)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

if not tokenizer.is_fast:
    raise ValueError("A fast tokenizer is required")


def find_assistant_span(messages, message_index, rendered, example_id):
    message = messages[message_index]
    content = message.get("content") or ""

    probe_messages = [
        dict(message)
        for message in messages[: message_index + 1]
    ]

    sentinel = f"\ue000ASSISTANT_CONTENT_{message_index}\ue001"

    while sentinel in rendered:
        sentinel += "_"

    probe_messages[-1]["content"] = sentinel

    probe_rendered = tokenizer.apply_chat_template(
        probe_messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    sentinel_position = probe_rendered.find(sentinel)

    if sentinel_position == -1:
        raise ValueError(
            f"Could not locate assistant sentinel for example {example_id}"
        )

    prefix = probe_rendered[:sentinel_position]

    if not rendered.startswith(prefix):
        raise ValueError(
            f"Assistant prefix does not match rendered template "
            f"for example {example_id}"
        )

    start_char = len(prefix)
    end_char = start_char + len(content)

    if rendered[start_char:end_char] != content:
        raise ValueError(
            f"Assistant content does not match rendered template "
            f"for example {example_id}"
        )

    return start_char, end_char


def tokenize(example):
    messages = example["messages"]
    example_id = example.get("id", "<unknown>")

    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_length + 1,
    )

    input_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]

    if len(input_ids) > max_length:
        return {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "keep": False,
        }

    labels = [-100] * len(input_ids)

    for i, message in enumerate(messages):
        if message["role"] != "assistant":
            continue

        start_char, end_char = find_assistant_span(
            messages=messages,
            message_index=i,
            rendered=rendered,
            example_id=example_id,
        )

        if rendered.startswith("<|im_end|>", end_char):
            end_char += len("<|im_end|>")
        elif (
            tokenizer.eos_token
            and rendered.startswith(tokenizer.eos_token, end_char)
        ):
            end_char += len(tokenizer.eos_token)

        for token_idx, (token_start, token_end) in enumerate(offsets):
            if token_start < end_char and start_char < token_end:
                labels[token_idx] = input_ids[token_idx]

    if not any(label != -100 for label in labels):
        raise ValueError(
            f"No trainable assistant tokens for example {example_id}"
        )

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "keep": True,
    }


def pack_batch(batch):
    flat_input_ids = list(chain.from_iterable(batch["input_ids"]))
    flat_labels = list(chain.from_iterable(batch["labels"]))

    if len(flat_input_ids) != len(flat_labels):
        raise ValueError("input_ids and labels have different lengths")

    usable_length = (len(flat_input_ids) // max_length) * max_length

    packed_input_ids = []
    packed_labels = []
    packed_attention_masks = []

    for start in range(0, usable_length, max_length):
        end = start + max_length

        packed_input_ids.append(flat_input_ids[start:end])
        packed_labels.append(flat_labels[start:end])
        packed_attention_masks.append([1] * max_length)

    return {
        "input_ids": packed_input_ids,
        "attention_mask": packed_attention_masks,
        "labels": packed_labels,
    }


if distributed_config["fsdp_sharding"] != "full_shard":
    raise ValueError(
        "This training script currently expects "
        "distributed.fsdp_sharding: full_shard"
    )


training_args = TrainingArguments(
    output_dir=train_config["output_dir"],
    per_device_train_batch_size=train_config["per_device_batch_size"],
    gradient_accumulation_steps=train_config["gradient_accumulation_steps"],
    accelerator_config={
        "gradient_accumulation_kwargs": {
            "sync_each_batch": train_config.get("sync_each_batch", False),
        },
    },
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
        "transformer_layer_cls_to_wrap": [
            "Olmo3DecoderLayer",
        ],
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
    logging_steps=logging_config["logging_steps"],
    save_strategy="steps",
    save_steps=logging_config["save_steps"],
    save_total_limit=logging_config.get("save_total_limit", 5),
    report_to=logging_config["report_to"],
    seed=train_config["seed"],
)


with training_args.main_process_first():
    dataset = load_dataset(
        "parquet",
        data_files=data_files,
        split="train",
    )

    dataset = dataset.shuffle(seed=train_config["seed"])

    if "max_examples" in data_config:
        dataset = dataset.select(
            range(min(data_config["max_examples"], len(dataset)))
        )

    original_examples = len(dataset)

    dataset = dataset.map(
        tokenize,
        remove_columns=dataset.column_names,
        num_proc=num_proc,
    )

    dataset = dataset.filter(
        lambda example: example["keep"],
        num_proc=num_proc,
    )

    kept_examples = len(dataset)
    dataset = dataset.remove_columns(["keep"])

    if packing:
        dataset = dataset.map(
            pack_batch,
            batched=True,
            batch_size=data_config.get("packing_batch_size", 256),
            remove_columns=dataset.column_names,
            num_proc=num_proc,
        )


if training_args.process_index == 0:
    print(
        f"Examples: {original_examples:,} -> {kept_examples:,} "
        f"({original_examples - kept_examples:,} overlength dropped)"
    )

    if packing:
        print(
            f"Packing: enabled -> {len(dataset):,} packed sequences "
            f"of {max_length:,} tokens"
        )

        for i in range(min(10, len(dataset))):
            if len(dataset[i]["input_ids"]) != max_length:
                raise ValueError(
                    f"Packed sequence {i} has "
                    f"{len(dataset[i]['input_ids'])} tokens, "
                    f"expected {max_length}"
                )
    else:
        print("Packing: disabled")

    example = dataset[0]

    trainable_ids = [
        token
        for token, label in zip(
            example["input_ids"],
            example["labels"],
        )
        if label != -100
    ]

    print("\nFirst trainable tokens:")
    print(
        tokenizer.decode(
            trainable_ids[:1000],
            skip_special_tokens=False,
        )
    )


model = AutoModelForCausalLM.from_pretrained(
    model_config["name_or_path"],
    dtype=torch.bfloat16,
    trust_remote_code=model_config.get("trust_remote_code", False),
    attn_implementation=model_config.get(
        "attention_implementation",
        "sdpa",
    ),
    local_files_only=True,
)

model.config.use_cache = train_config["use_cache"]

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    ),
)

torch.cuda.reset_peak_memory_stats()

trainer.train(
    resume_from_checkpoint=train_config.get(
        "resume_from_checkpoint",
        None,
    )
)

torch.cuda.synchronize()

print(
    f"rank {training_args.process_index} peak memory: "
    f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB allocated, "
    f"{torch.cuda.max_memory_reserved() / 2**30:.1f} GiB reserved"
)

if trainer.is_fsdp_enabled:
    trainer.accelerator.state.fsdp_plugin.set_state_dict_type(
        "FULL_STATE_DICT"
    )

trainer.save_model(train_config["output_dir"])

if trainer.is_world_process_zero():
    tokenizer.save_pretrained(train_config["output_dir"])