#!/usr/bin/env python3
"""Prepare fixed Dolci training and holdout datasets offline.

Select the first user/assistant turn with complete think tags and at most
10,000 OLMo tokens. Require nonempty reasoning and a final answer, deduplicate
IDs and prompts, exclude benchmark overlaps, and check assistant/EOS labels
through the training preprocessor. --dry-run performs the checks without writing.
"""

import argparse
import bisect
import glob
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from train_sft import (
    RepetitionCollator,
    RepetitionDataset,
    guard_repetition_destination,
    make_tokenizer,
    repetition_plan,
    tokenize_sft_example,
    tokenized_digest,
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def normalized(text):
    return re.sub(r"\s+", " ", text).strip().casefold()


def first_turn(messages):
    # Keep the leading system message and the first user/assistant exchange.
    prefix = 1 if messages and messages[0]["role"] == "system" else 0
    if len(messages) < prefix + 2 or [
        m["role"] for m in messages[prefix : prefix + 2]
    ] != ["user", "assistant"]:
        return None
    return messages[: prefix + 2]


def complete_trace(content):
    if not isinstance(content, str):
        return False
    if content.count("<think>") != 1 or content.count("</think>") != 1:
        return False
    match = re.fullmatch(r"\s*<think>(.*?)</think>(.+)", content, re.DOTALL)
    return bool(match and match[1].strip() and match[2].strip())


def benchmark_questions():
    paths = sorted((ROOT / "results/base").glob("*/task-*-requests.jsonl"))
    if not paths:
        raise RuntimeError(
            "No local benchmark questions available for contamination check"
        )
    questions = set()
    provenance = []
    for path in paths:
        data = path.read_bytes()
        provenance.append({"path": str(path), "sha256": sha(data)})
        for line in data.splitlines():
            row = json.loads(line)
            doc = row["doc"]
            text = doc.get("problem") or doc.get("question")
            if text:
                questions.add(normalized(text))
    return questions, provenance


def candidate_rows(files, global_indices, ends):
    """Read candidate row groups by shard; key results by global source row."""
    grouped = defaultdict(list)
    for global_index in global_indices:
        file_index = bisect.bisect_right(ends, int(global_index))
        start = ends[file_index - 1] if file_index else 0
        grouped[file_index].append(int(global_index) - start)
    result = {}
    for file_index, indices in sorted(grouped.items()):
        parquet = pq.ParquetFile(files[file_index])
        group_ends = np.cumsum(
            [
                parquet.metadata.row_group(i).num_rows
                for i in range(parquet.num_row_groups)
            ]
        )
        groups = defaultdict(list)
        for index in indices:
            group = bisect.bisect_right(group_ends, index)
            offset = int(group_ends[group - 1]) if group else 0
            groups[group].append((index, index - offset))
        file_start = ends[file_index - 1] if file_index else 0
        for group, positions in groups.items():
            table = parquet.read_row_group(
                group, columns=["messages", "id", "dataset_source"]
            )
            selected = table.take(
                pa.array([position for _, position in positions])
            ).to_pylist()
            for (index, _), row in zip(positions, selected):
                result[file_start + index] = (row, files[file_index], index)
        if (file_index + 1) % 20 == 0:
            print(
                f"Read candidate groups through shard {file_index + 1}/{len(files)}",
                flush=True,
            )
    return result


def prepare(config, dry_run=False):
    started = time.monotonic()
    data, training = config["data"], config["training"]
    output = Path(training["output_dir"])
    source = Path(training["initialization_checkpoint"])
    if output.resolve().is_relative_to(
        source.resolve()
    ) or source.resolve().is_relative_to(output.resolve()):
        raise ValueError("Output and source checkpoint must be separate directories")
    guard_repetition_destination(output)
    destination = Path(data["repetition_manifest"]).parent
    if Path(data["path"]).resolve() != (destination / "examples.parquet").resolve():
        raise ValueError(
            "Training data and manifest must use the same prepared directory"
        )
    if destination.resolve().is_relative_to(
        source.resolve()
    ) or source.resolve().is_relative_to(destination.resolve()):
        raise ValueError(
            "Prepared data and source checkpoint must be separate directories"
        )
    if destination.exists() and any(destination.iterdir()) and not dry_run:
        raise FileExistsError(
            f"Prepared data already exists: {destination}; validate it instead of overwriting"
        )
    complete = json.loads((source / "training_complete.json").read_text())
    if complete["global_step"] != 6000 or not (source / "model.safetensors").is_file():
        raise ValueError("Expected completed 6k exported model weights")

    tokenizer = make_tokenizer(config["model"])
    source_config = json.loads((source / "config.json").read_text())
    if tokenizer.eos_token_id != source_config["eos_token_id"]:
        raise ValueError("Tokenizer EOS differs from source model EOS")

    forbidden, benchmark_provenance = benchmark_questions()
    files = sorted(glob.glob(data["source_path"]))
    if not files:
        raise FileNotFoundError(data["source_path"])
    counts = [pq.ParquetFile(file).metadata.num_rows for file in files]
    ends = np.cumsum(counts).tolist()

    original_manifest = json.loads((source / "run_manifest.json").read_text())[
        "settings"
    ]
    if [file["path"] for file in original_manifest["files"]] != [
        str(Path(file).resolve()) for file in files
    ]:
        raise ValueError(
            "Cannot reconstruct SFT1 sampling pool: source file order differs"
        )
    for previous, file in zip(original_manifest["files"], files):
        stat = Path(file).stat()
        if previous["size"] != stat.st_size or previous["mtime_ns"] != stat.st_mtime_ns:
            raise ValueError(
                "Dolci source files changed since SFT1; cannot verify holdout exclusion"
            )
    original_data = original_manifest["data"]
    if not original_data.get("shuffle", True) or "max_examples" not in original_data:
        raise ValueError(
            "Cannot reconstruct SFT1 sampling pool: unsupported original selection"
        )
    original_pool = set(
        np.random.default_rng(original_manifest["training"]["seed"])
        .permutation(ends[-1])[: original_data["max_examples"]]
        .tolist()
    )

    order = np.random.default_rng(data["selection_seed"]).permutation(ends[-1])
    target = data["unique_examples"] + data["holdout_examples"]
    selected, encoded, identities = [], [], []
    seen_ids, seen_prompts, seen_contents = set(), set(), set()
    rejected, inspected = Counter(), 0
    for start in range(0, len(order), 8192):
        candidates = order[start : start + 8192]
        rows = candidate_rows(files, candidates, ends)
        for index in candidates:
            row, file, file_row = rows[int(index)]
            inspected += 1
            if len(selected) >= data["unique_examples"] and int(index) in original_pool:
                rejected["holdout_in_SFT1_sampling_pool"] += 1
                continue
            messages = first_turn(row["messages"])
            if not messages or not complete_trace(messages[-1]["content"]):
                rejected["incomplete_or_empty_trace_or_answer"] += 1
                continue
            prompt = normalized(messages[-2]["content"])
            content_hash = sha(
                json.dumps(messages, sort_keys=True, ensure_ascii=False).encode()
            )
            if (
                row["id"] in seen_ids
                or prompt in seen_prompts
                or content_hash in seen_contents
            ):
                rejected["duplicate_id_prompt_or_content"] += 1
                continue
            if any(
                question in prompt or prompt in question
                for question in forbidden
                if len(question) >= 40
            ):
                rejected["benchmark_prompt_overlap"] += 1
                continue
            tokens = tokenize_sft_example(
                {"messages": messages}, tokenizer, data["max_length"]
            )
            if (
                not tokens["keep"]
                or len(tokens["input_ids"]) > data["selection_max_tokens"]
            ):
                rejected["overlength"] += 1
                continue
            if (
                tokens["input_ids"][-1] != tokenizer.eos_token_id
                or tokens["labels"][-1] != tokenizer.eos_token_id
            ):
                raise ValueError(f"Missing supervised terminal EOS for {row['id']}")
            # Check final-answer labels against token offsets in the rendered chat.
            rendered = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            final_start = rendered.index("</think>") + len("</think>")
            offsets = tokenizer(
                rendered, add_special_tokens=False, return_offsets_mapping=True
            )["offset_mapping"]
            assert all(
                tokens["labels"][i] == tokens["input_ids"][i]
                for i, (left, right) in enumerate(offsets)
                if right > final_start
            )
            seen_ids.add(row["id"])
            seen_prompts.add(prompt)
            seen_contents.add(content_hash)
            tokens.pop("keep")
            selected.append(
                {
                    "messages": messages,
                    "id": row["id"],
                    "dataset_source": row["dataset_source"],
                }
            )
            encoded.append(tokens)
            identities.append(
                {
                    "id": row["id"],
                    "dataset_source": row["dataset_source"],
                    "source_file": file,
                    "source_row": file_row,
                    "global_source_row": int(index),
                    "content_sha256": content_hash,
                    "prompt_sha256": sha(prompt.encode()),
                    "tokens": len(tokens["input_ids"]),
                    "supervised_tokens": sum(x != -100 for x in tokens["labels"]),
                }
            )
            if len(selected) == target:
                break
        print(
            f"Inspected {inspected:,}, retained {len(selected):,}/{target:,}",
            flush=True,
        )
        if len(selected) == target:
            break
    if len(selected) != target:
        raise RuntimeError(f"Only {len(selected)} valid unique examples available")

    n = data["unique_examples"]
    assert not {row["global_source_row"] for row in identities[n:]} & original_pool
    assert not {row["id"] for row in identities[:n]} & {
        row["id"] for row in identities[n:]
    }
    assert not {row["prompt_sha256"] for row in identities[:n]} & {
        row["prompt_sha256"] for row in identities[n:]
    }
    assert all(row["labels"][0] == -100 for row in encoded)

    lengths = [row["tokens"] for row in identities[:n]]
    global_batch = (
        training["expected_world_size"]
        * training["per_device_batch_size"]
        * training["gradient_accumulation_steps"]
    )
    plans = repetition_plan(
        lengths,
        data["max_length"],
        training["num_train_epochs"],
        training["seed"],
        global_batch,
    )
    tokens = sum(lengths)
    supervised = sum(row["supervised_tokens"] for row in identities[:n])

    # Each epoch must contain every training example exactly once.
    for plan in plans:
        assert sorted(i for pack in plan for i in pack) == list(range(n))
        assert sum(sum(lengths[i] for i in pack) for pack in plan) == tokens
        assert all(sum(lengths[i] for i in pack) <= data["max_length"] for pack in plan)

    packed = RepetitionDataset(
        encoded[:n], plans, data["max_length"], tokenizer.pad_token_id
    )
    collator = RepetitionCollator(tokenizer)
    # Check the first epoch through the training collator, including padding and EOS.
    observed_tokens = observed_eos = observed_examples = 0
    for index in range(len(packed)):
        batch = collator([packed[index]])
        labels, inputs = batch["labels"], batch["input_ids"]
        observed_examples += int(batch["example_count"].sum())
        observed_tokens += int((labels != -100).sum())
        observed_eos += int(
            ((inputs == tokenizer.eos_token_id) & (labels == inputs)).sum()
        )
        assert bool(((labels == -100) | (labels == inputs)).all())
        assert bool((labels[batch["attention_mask"] == 0] == -100).all())
        expected_labels = [
            label for member in plans[0][index] for label in encoded[member]["labels"]
        ]
        assert labels[0, : len(expected_labels)].tolist() == expected_labels
    assert observed_tokens == supervised and observed_examples == n
    source_eos = sum(
        sum(
            a == tokenizer.eos_token_id and b == a
            for a, b in zip(row["input_ids"], row["labels"])
        )
        for row in encoded[:n]
    )
    assert observed_eos == source_eos

    packs = len(plans[0])
    steps = packs // global_batch
    manifest = {
        "version": 1,
        "dataset": "allenai/Dolci-Think-SFT-7B",
        "seed": training["seed"],
        "selection_seed": data["selection_seed"],
        "selection_algorithm": "uniform random global-row permutation; filter and deduplicate until train+holdout full",
        "filters": "first turn; nonempty think trace and final answer; <=10000 production tokens; supervised terminal EOS",
        "answer_correctness": "unverified; structural completion only",
        "paper": "https://arxiv.org/html/2602.11149v2",
        "inspected_examples": inspected,
        "rejections": dict(rejected),
        "unique_examples": n,
        "holdout_examples": len(selected) - n,
        "source_checkpoint": str(source),
        "epochs": training["num_train_epochs"],
        "max_length": data["max_length"],
        "global_batch_size": global_batch,
        "total_tokens": tokens,
        "supervised_tokens_per_epoch": supervised,
        "supervised_eos_per_epoch": source_eos,
        "terminal_eos_per_epoch": n,
        "packs_per_epoch": packs,
        "optimizer_steps_per_epoch": steps,
        "milestones": {
            str(epoch): {
                "optimizer_steps": epoch * steps,
                "example_presentations": epoch * n,
                "supervised_tokens": epoch * supervised,
                "checkpoint": f"checkpoint-{epoch * steps}",
            }
            for epoch in (4, 8, 12, 16)
        },
        "epoch_plan_sha256": sha(json.dumps(plans).encode()),
        "tokenized_sha256": tokenized_digest(encoded[:n]),
        "selection_sha256": sha(json.dumps(identities, sort_keys=True).encode()),
        "selected": identities[:n],
        "holdout": identities[n:],
        "tokenizer": {
            "path": config["model"]["tokenizer_name_or_path"],
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "chat_template_sha256": sha(tokenizer.chat_template.encode()),
        },
        "source_files": [
            {
                "path": file,
                "rows": count,
                "size": Path(file).stat().st_size,
                "mtime_ns": Path(file).stat().st_mtime_ns,
            }
            for file, count in zip(files, counts)
        ],
        "source_checkpoint_metadata_sha256": {
            name: sha((source / name).read_bytes())
            for name in (
                "config.json",
                "generation_config.json",
                "training_complete.json",
                "run_manifest.json",
            )
        },
        "holdout_exclusion": {
            "SFT1_source_pool_size": len(original_pool),
            "method": "exclude original shuffled max_examples pool by global source row; conservative upper bound on SFT1 exposure",
            "limitations": "Duplicate/paraphrased prompts in other source rows and base pretraining exposure remain possible",
        },
        "benchmark_contamination_check": {
            "method": "casefolded whitespace-normalized substring in user prompt; >=40 characters",
            "unique_benchmark_questions": len(forbidden),
            "sources": benchmark_provenance,
            "limitations": "Cannot rule out paraphrases, approximate matches, or prior 6k/base contamination",
        },
        "parquet_sha256": None,
    }

    if not dry_run:
        destination.mkdir(parents=True, exist_ok=True)
        if any(destination.iterdir()):
            raise FileExistsError(
                f"Prepared destination became occupied: {destination}"
            )
        pq.write_table(
            pa.Table.from_pylist(selected[:n]), destination / "examples.parquet"
        )
        pq.write_table(
            pa.Table.from_pylist(selected[n:]), destination / "validation.parquet"
        )
        manifest["parquet_sha256"] = sha(
            (destination / "examples.parquet").read_bytes()
        )
        manifest["holdout_parquet_sha256"] = sha(
            (destination / "validation.parquet").read_bytes()
        )
        (destination / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
        )
        (destination / "config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False)
        )

    summary = {
        key: manifest[key]
        for key in (
            "unique_examples",
            "holdout_examples",
            "inspected_examples",
            "rejections",
            "total_tokens",
            "supervised_tokens_per_epoch",
            "supervised_eos_per_epoch",
            "packs_per_epoch",
            "optimizer_steps_per_epoch",
            "milestones",
            "selection_sha256",
            "epoch_plan_sha256",
            "tokenizer",
            "holdout_exclusion",
        )
    }
    summary.update(
        dry_run=dry_run,
        elapsed_seconds=time.monotonic() - started,
        actual_collator_verified_packs=len(packed),
        all_epochs_exact_coverage_verified=len(plans),
    )
    print(json.dumps(summary, indent=2), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config", nargs="?", default=str(ROOT / "configs/sft-repetition-3k2.yaml")
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for variable in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[variable] = "1"
    config_path = Path(args.config).resolve()
    with initialize_config_dir(version_base="1.3", config_dir=str(config_path.parent)):
        config = compose(config_name=config_path.stem)
    prepare(OmegaConf.to_container(config, resolve=True), args.dry_run)


if __name__ == "__main__":
    main()
