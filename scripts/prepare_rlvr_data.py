#!/usr/bin/env python3

import argparse
import ast
import hashlib
import json
import re
import sys
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from benchmark_exclusions import load_exclusions, normalized, question_hash
from rlvr.rewards import validate_reference_answers

COLUMNS = [
    "prompt",
    "ground_truth",
    "dataset",
    "dataset_source",
    "original_dataset",
    "custom_id",
    "id",
    "key",
    "constraint_type",
    "constraint",
]
DOMAINS = {"math", "ifeval"}
SCHEMA = pa.schema(
    [
        (
            "prompt",
            pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())])),
        ),
        ("source_prompt", pa.string()),
        ("ground_truth", pa.list_(pa.string())),
        ("dataset", pa.list_(pa.string())),
        ("domain", pa.string()),
        *[(key, pa.string()) for key in COLUMNS[3:]],
        ("source_id", pa.string()),
        ("source_file", pa.string()),
        ("source_row", pa.int64()),
        ("constraint_spec", pa.string()),
        ("reasoning_prefix", pa.bool_()),
        ("reward_enabled", pa.bool_()),
    ]
)


def fingerprint(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def constraint_spec(ground_truth):
    if len(ground_truth) != 1:
        raise ValueError("invalid_if_constraints")
    try:
        value = ast.literal_eval(ground_truth[0])
    except (ValueError, SyntaxError) as error:
        raise ValueError("invalid_if_constraints") from error
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], dict):
        raise ValueError("invalid_if_constraints")
    spec = value[0]
    if set(spec) != {"instruction_id", "kwargs"}:
        raise ValueError("invalid_if_constraints")
    ids, arguments = spec["instruction_id"], spec["kwargs"]
    if (
        not isinstance(ids, list)
        or not ids
        or not isinstance(arguments, list)
        or len(ids) != len(arguments)
        or any(not isinstance(key, str) or not key for key in ids)
        or any(arg is not None and not isinstance(arg, dict) for arg in arguments)
    ):
        raise ValueError("invalid_if_constraints")
    try:
        return json.dumps(spec, sort_keys=True, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("invalid_if_constraints") from error


def prepare_record(row, source_file, source_row):
    """Validate a source row and retain provenance without historical rollouts."""
    for field in ("prompt", "dataset_source", "original_dataset", "custom_id"):
        if not isinstance(row.get(field), str) or not row[field].strip():
            raise ValueError("invalid_" + field)
    for field in ("id", "key", "constraint_type", "constraint"):
        if row.get(field) is not None and not isinstance(row[field], str):
            raise ValueError("invalid_" + field)
    answers = row.get("ground_truth")
    if (
        not isinstance(answers, list)
        or not answers
        or any(not isinstance(answer, str) or not answer.strip() for answer in answers)
    ):
        raise ValueError("invalid_ground_truth")
    raw = row["prompt"]
    if not raw.startswith("user: "):
        raise ValueError("unsupported_prompt_format")
    content = raw[6:]
    if (
        not content.strip()
        or re.match(r"\s*(?:user|assistant|system):", content, re.IGNORECASE)
        or re.search(r"</?think|<\|(?:im_start|im_end|endoftext)\|>", content)
    ):
        raise ValueError("unsupported_prompt_format")
    domain = row["dataset"][0]
    spec = None
    if domain == "math":
        try:
            validate_reference_answers(answers, prompt=content)
        except ValueError as error:
            raise ValueError("unsupported_math_reference") from error
    else:
        if row.get("constraint_type") != "multi" or not row.get("constraint"):
            raise ValueError("invalid_if_constraints")
        spec = constraint_spec(answers)
    return {
        **{key: row.get(key) for key in COLUMNS[1:]},
        "prompt": [{"role": "user", "content": content}],
        "source_prompt": raw,
        "domain": domain,
        "source_id": f"{source_file}:{source_row}",
        "source_file": source_file,
        "source_row": source_row,
        "constraint_spec": spec,
        "reasoning_prefix": True,
        "reward_enabled": domain == "math",
    }


def select_records(records, exclusions):
    """Exclude benchmark overlap and all copies of conflicting duplicate prompts."""
    seen, conflicts = {}, set()
    input_counts, removed = Counter(), Counter()
    removal_examples = {}
    total = 0
    for row, source_file, source_row in records:
        total += 1
        labels = row.get("dataset")
        reason = None
        if (
            not isinstance(labels, list)
            or len(labels) != 1
            or not isinstance(labels[0], str)
            or not labels[0]
        ):
            reason = "invalid_dataset"
        else:
            input_counts[labels[0]] += 1
            if labels[0] not in DOMAINS:
                reason = "domain_not_selected"
        if reason is None:
            try:
                record = prepare_record(row, source_file, source_row)
            except ValueError as error:
                reason = str(error)
        if reason is None:
            key = normalized(record["prompt"][0]["content"])
            if question_hash(record["prompt"][0]["content"]) in exclusions:
                reason = "benchmark_overlap"
            elif key in seen:
                previous = seen[key]
                if key in conflicts or (record["domain"], record["ground_truth"]) != (
                    previous["domain"],
                    previous["ground_truth"],
                ):
                    if key not in conflicts:
                        conflicts.add(key)
                        removed["conflicting_duplicate_prompt"] += 1
                    reason = "conflicting_duplicate_prompt"
                else:
                    reason = "duplicate_prompt"
            else:
                seen[key] = record
        if reason:
            removed[reason] += 1
            removal_examples.setdefault(reason, f"{source_file}:{source_row}")
    selected = [row for key, row in seen.items() if key not in conflicts]
    assert total == len(selected) + sum(removed.values())
    return selected, {
        "input_rows": total,
        "input_by_domain": dict(sorted(input_counts.items())),
        "removed": dict(sorted(removed.items())),
        "removal_examples": removal_examples,
        "retained_rows": len(selected),
        "retained_by_domain": dict(
            sorted(Counter(row["domain"] for row in selected).items())
        ),
        "retained_by_source": dict(
            sorted(Counter(row["original_dataset"] for row in selected).items())
        ),
    }


def read_records(files, source_dir):
    for path in files:
        parquet = pq.ParquetFile(path)
        for column in COLUMNS:
            expected = (
                pa.list_(pa.string())
                if column in {"dataset", "ground_truth"}
                else pa.string()
            )
            if (
                column not in parquet.schema_arrow.names
                or parquet.schema_arrow.field(column).type != expected
            ):
                raise ValueError(f"Unsupported schema for {column} in {path}")
        offset = 0
        for batch in parquet.iter_batches(columns=COLUMNS, batch_size=1024):
            for row in batch.to_pylist():
                yield row, str(path.relative_to(source_dir)), offset
                offset += 1


def check_rendering(rows, tokenizer):
    samples = {}
    for row in rows:
        rendered = tokenizer.apply_chat_template(
            row["prompt"], tokenize=False, add_generation_prompt=True
        )
        expected = (
            "<|im_start|>user\n"
            + row["prompt"][0]["content"]
            + "<|im_end|>\n<|im_start|>assistant\n<think>"
        )
        if not rendered.endswith(expected) or rendered.count("<think>") != 1:
            raise ValueError(
                f"Unsupported tokenizer prompt rendering: {row['source_id']}"
            )
        if row["domain"] not in samples:
            ids = tokenizer.apply_chat_template(
                row["prompt"],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
            )
            if tokenizer.decode(ids, skip_special_tokens=False) != rendered:
                raise ValueError("Tokenizer does not round-trip the generation prompt")
            samples[row["domain"]] = {
                "source_id": row["source_id"],
                "tokens": len(ids),
                "rendered": rendered,
            }
    return samples


def prepare(
    source_dir: Path,
    output_dir: Path,
    tokenizer,
    exclusions: Path,
    dry_run: bool = False,
):
    """Build deterministic Parquet/manifest artifacts, or validate without writing."""
    if not dry_run and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite nonempty output directory: {output_dir}"
        )
    files = sorted(source_dir.glob("data/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No local Parquet shards in {source_dir / 'data'}")
    hashes, exclusion_info = load_exclusions(exclusions)
    rows, statistics = select_records(read_records(files, source_dir), hashes)
    if not rows:
        raise ValueError("No eligible RLVR records remain")
    samples = check_rendering(rows, tokenizer)
    sink = pa.BufferOutputStream()
    pq.write_table(
        pa.Table.from_pylist(rows, schema=SCHEMA),
        sink,
        compression="zstd",
        row_group_size=1024,
        version="2.6",
    )
    prepared = sink.getvalue()
    tokenizer_dir = Path(tokenizer.name_or_path)
    tokenizer_files = [
        tokenizer_dir / name
        for name in (
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
            "vocab.json",
            "merges.txt",
            "chat_template.jinja",
        )
        if (tokenizer_dir / name).is_file()
    ]
    manifest = {
        "preprocessing_version": 3,
        "source_directory": str(source_dir.resolve()),
        "source_files": [
            {
                "path": str(path.relative_to(source_dir)),
                "size": path.stat().st_size,
                "rows": pq.ParquetFile(path).metadata.num_rows,
                "sha256": fingerprint(path),
            }
            for path in files
        ],
        "settings": {
            "domains": sorted(DOMAINS),
            "columns": COLUMNS,
            "prompt_format": "remove one leading 'user: '; preserve user content",
            "deduplication": "normalized prompt; first in sorted shard/row order; drop conflicting groups",
            "math_references": "require supported answer types; reject bare-symbol construction targets",
            "sampling": None,
        },
        "tokenizer": {
            "path": str(tokenizer_dir),
            "chat_template_sha256": hashlib.sha256(
                tokenizer.chat_template.encode()
            ).hexdigest(),
            "files": [
                {"path": path.name, "sha256": fingerprint(path)}
                for path in tokenizer_files
            ],
        },
        "versions": {
            name: version(name)
            for name in (
                "pyarrow",
                "transformers",
                "math-verify",
                "latex2sympy2_extended",
            )
        },
        "statistics": statistics,
        "exclusions": {
            **exclusion_info,
            "excluded_training_examples": statistics["removed"].get(
                "benchmark_overlap", 0
            ),
        },
        "rewards": {
            "math": "enabled",
            "ifeval": "disabled: Open Instruct IFEvalG and its language/NLTK resources are not installed or verified",
        },
        "output": {
            "file": "examples.parquet",
            "rows": len(rows),
            "sha256": hashlib.sha256(prepared).hexdigest(),
        },
    }
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "examples.parquet").open("xb") as stream:
            stream.write(prepared)
        with (output_dir / "manifest.json").open("x") as stream:
            stream.write(
                json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False)
                + "\n"
            )
    return manifest, samples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare local Dolci math and IF prompts for RLVR."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("/p/scratch/rcfd/kowanietz1/datasets/Dolci-Think-RL-7B"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--exclusions", type=Path, default=ROOT / "benchmarks/exclusions.json"
    )
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and fingerprint without writing artifacts",
    )
    args = parser.parse_args()
    from training_common import make_tokenizer

    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        model = OmegaConf.to_container(
            compose(config_name="sft/base").model, resolve=True
        )
    if args.tokenizer:
        model["tokenizer_name_or_path"] = str(args.tokenizer)
    if not Path(model["tokenizer_name_or_path"]).is_dir():
        raise FileNotFoundError("Tokenizer must be an existing local directory")
    manifest, samples = prepare(
        args.source_dir,
        args.output_dir,
        make_tokenizer(model),
        args.exclusions,
        args.dry_run,
    )
    print(json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False))
    for domain, sample in samples.items():
        print(
            f"\n{domain}: {sample['source_id']}, {sample['tokens']} prompt tokens\n{sample['rendered']}"
        )


if __name__ == "__main__":
    main()
