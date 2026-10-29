"""Validate prepared RLVR provenance and select deterministic math prompts."""

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from datasets import Dataset

from benchmark_exclusions import load_exclusions

from .checkpoints import fingerprint
from .rewards import validate_reference_answers


def load_smoke_data(config: dict, tokenizer) -> tuple[Dataset, dict]:
    directory = Path(config["data"]["directory"])
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    parquet = directory / "examples.parquet"
    digest = fingerprint(parquet)
    _, exclusions = load_exclusions(Path(config["data"]["exclusions"]))
    if (
        digest != manifest["output"]["sha256"]
        or manifest["output"]["file"] != parquet.name
        or manifest["exclusions"]["sha256"] != exclusions["sha256"]
        or manifest["exclusions"]["normalization"] != exclusions["normalization"]
        or manifest["exclusions"]["benchmarks"] != exclusions["benchmarks"]
        or hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()
        != manifest["tokenizer"]["chat_template_sha256"]
    ):
        raise ValueError("Prepared dataset/exclusion/tokenizer provenance mismatch")
    columns = [
        "prompt",
        "source_prompt",
        "ground_truth",
        "domain",
        "reward_enabled",
        "reasoning_prefix",
        "source_id",
    ]
    rows = pq.read_table(parquet, columns=columns).to_pylist()
    if (
        len(rows) != manifest["output"]["rows"]
        or dict(Counter(row["domain"] for row in rows))
        != manifest["statistics"]["retained_by_domain"]
    ):
        raise ValueError(
            "Prepared dataset row/domain counts disagree with its manifest"
        )
    selected, excluded = [], Counter()
    random.Random(config["data"]["selection_seed"]).shuffle(rows)
    for row in rows:
        if row["domain"] != "math" or row["reward_enabled"] is not True:
            excluded["not_enabled_math"] += 1
            continue
        if (
            row["reasoning_prefix"] is not True
            or len(row["prompt"]) != 1
            or row["prompt"][0]["role"] != "user"
            or row["source_prompt"] != "user: " + row["prompt"][0]["content"]
        ):
            raise ValueError(f"Unsupported prepared prompt: {row['source_id']}")
        ids = tokenizer.apply_chat_template(
            row["prompt"], tokenize=True, add_generation_prompt=True, return_dict=False
        )
        if len(ids) > config["data"]["max_prompt_tokens"]:
            excluded["prompt_over_budget"] += 1
            continue
        rendered = tokenizer.decode(ids, skip_special_tokens=False)
        if (
            not rendered.endswith("<|im_start|>assistant\n<think>")
            or rendered.count("<think>") != 1
        ):
            raise ValueError("Unexpected Think generation prefix")
        if (
            config["data"].get("source_id") is not None
            and row["source_id"] != config["data"]["source_id"]
        ):
            excluded["not_requested_source"] += 1
            continue
        if len(selected) < config["data"]["max_examples"]:
            validate_reference_answers(row["ground_truth"], prompt=row["prompt"])
            selected.append(row)
        else:
            excluded["outside_smoke_subset"] += 1
    if len(selected) != config["data"]["max_examples"]:
        raise ValueError("Insufficient math examples within the prompt budget")
    identity = [
        {
            "source_id": row["source_id"],
            "prompt": row["prompt"],
            "ground_truth": row["ground_truth"],
        }
        for row in selected
    ]
    return Dataset.from_list(selected), {
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": fingerprint(manifest_path),
        "parquet_sha256": digest,
        "exclusions_sha256": exclusions["sha256"],
        "selected_source_ids": [row["source_id"] for row in selected],
        "excluded": dict(excluded),
        "selection_sha256": hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()
        ).hexdigest(),
        "first_prompt_ids": tokenizer.apply_chat_template(
            selected[0]["prompt"],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        ),
    }
