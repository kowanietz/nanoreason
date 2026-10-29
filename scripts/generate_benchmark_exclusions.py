#!/usr/bin/env python3

import argparse
import csv
import hashlib
import io
import json
import os
import sys
import urllib.request
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from benchmark_exclusions import NORMALIZATION, load_exclusions, question_hash

MATH_REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
AIME_REVISION = "73e1eba765ad5847cdb5d1e2e7aaf7b22b585798"
GPQA_REVISION = "83022cefff930aea54f654c0b282e74b9eeda5c6"
IFEVAL_REVISION = "ba728e288c51126eac25e316b11dfee94d68c301"
BBH_REVISION = "9ee07bd481feebf959a6b59d61ea57bdcf30964d"
BBH_TASKS = [
    "boolean_expressions",
    "causal_judgement",
    "date_understanding",
    "disambiguation_qa",
    "dyck_languages",
    "formal_fallacies",
    "geometric_shapes",
    "hyperbaton",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "logical_deduction_three_objects",
    "movie_recommendation",
    "multistep_arithmetic_two",
    "navigate",
    "object_counting",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "ruin_names",
    "salient_translation_error_detection",
    "snarks",
    "sports_understanding",
    "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
    "web_of_lies",
    "word_sorting",
]


def read_source(
    url: str, revision: str, local: Path | None = None
) -> tuple[bytes, dict]:
    if local is not None and local.is_file():
        data = local.read_bytes()
    else:
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
    return data, {
        "url": url,
        "revision": revision,
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def hf_source(
    cache: Path, dataset: str, revision: str, filename: str
) -> tuple[bytes, dict]:
    return read_source(
        f"https://huggingface.co/datasets/{dataset}/resolve/{revision}/{filename}",
        revision,
        cache
        / ("datasets--" + dataset.replace("/", "--"))
        / "snapshots"
        / revision
        / filename,
    )


def make_benchmark(identifier, variant, questions, expected, expected_unique, sources):
    if any(not isinstance(text, str) or not text.strip() for text in questions):
        raise ValueError(f"Missing question text for {identifier}")
    hashes = sorted({question_hash(text) for text in questions})
    if len(questions) != expected or len(hashes) != expected_unique:
        raise ValueError(
            f"{identifier}: expected {expected} records / {expected_unique} unique questions, "
            f"got {len(questions)} / {len(hashes)}"
        )
    return {
        "id": identifier,
        "variant": variant,
        "expected_question_count": expected,
        "expected_unique_question_count": expected_unique,
        "sources": sources,
        "question_hashes": hashes,
    }


def generate(cache: Path) -> dict:
    """Fingerprint pinned benchmark questions using the evaluation variants."""
    benchmarks = []
    data, source = hf_source(
        cache, "HuggingFaceH4/MATH-500", MATH_REVISION, "test.jsonl"
    )
    questions = [json.loads(line)["problem"] for line in data.splitlines()]
    source.update(question_count=len(questions), question_field="problem", split="test")
    benchmarks.append(
        make_benchmark(
            "math500",
            "minerva_math_500::hamish_zs_reasoning",
            questions,
            500,
            500,
            [source],
        )
    )

    data, source = hf_source(
        cache,
        "allenai/aime-2021-2025",
        AIME_REVISION,
        "data/train-00000-of-00001.parquet",
    )
    rows = pq.read_table(pa.BufferReader(data), columns=["year", "problem"]).to_pylist()
    for year in (2024, 2025):
        questions = [row["problem"] for row in rows if str(row["year"]) == str(year)]
        provenance = {
            **source,
            "question_count": len(questions),
            "question_field": "problem",
            "split": "train",
            "year": year,
        }
        benchmarks.append(
            make_benchmark(
                f"aime{year}",
                f"aime:{year}::olmo3:adapt (I and II)",
                questions,
                30,
                30,
                [provenance],
            )
        )

    data, source = hf_source(cache, "Idavidrein/gpqa", GPQA_REVISION, "gpqa_main.csv")
    questions = [
        row["Question"] for row in csv.DictReader(io.StringIO(data.decode("utf-8")))
    ]
    source.update(
        question_count=len(questions),
        question_field="Question",
        subset="gpqa_main",
        split="train",
    )
    benchmarks.append(
        make_benchmark(
            "gpqa_main", "gpqa::olmo3:adapt / gpqa_main", questions, 448, 448, [source]
        )
    )

    data, source = read_source(
        "https://raw.githubusercontent.com/google-research/google-research/"
        f"{IFEVAL_REVISION}/instruction_following_eval/data/input_data.jsonl",
        IFEVAL_REVISION,
    )
    questions = [json.loads(line)["prompt"] for line in data.splitlines()]
    source.update(question_count=len(questions), question_field="prompt", split="train")
    benchmarks.append(
        make_benchmark(
            "ifeval",
            "OLMES IFEval / HuggingFaceH4/ifeval:train",
            questions,
            541,
            541,
            [source],
        )
    )

    questions, sources = [], []
    for task in BBH_TASKS:
        data, source = read_source(
            f"https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/{BBH_REVISION}/bbh/{task}.json",
            BBH_REVISION,
        )
        texts = [row["input"] for row in json.loads(data)["examples"]]
        expected = {
            "causal_judgement": 187,
            "penguins_in_a_table": 146,
            "snarks": 178,
        }.get(task, 250)
        if len(texts) != expected:
            raise ValueError(
                f"BBH {task}: expected {expected} questions, got {len(texts)}"
            )
        questions.extend(texts)
        source.update(
            question_count=len(texts),
            question_field="input",
            subtask=task,
            split="test",
        )
        sources.append(source)
    # causal_judgement contains four repeated normalized questions.
    benchmarks.append(
        make_benchmark(
            "bbh",
            "OLMES lukaemon/bbh:test (all 27 subtasks)",
            questions,
            6511,
            6507,
            sources,
        )
    )
    return {
        "version": 1,
        "normalization": NORMALIZATION,
        "expected_benchmark_count": 6,
        "benchmarks": benchmarks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hash pinned benchmark questions; use cached HF snapshots when available."
    )
    parser.add_argument(
        "--hf-cache",
        type=Path,
        default=Path(
            os.environ.get(
                "HF_HUB_CACHE",
                Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
                / "hub",
            )
        ),
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "benchmarks/exclusions.json"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Compare regenerated bytes with the existing manifest",
    )
    args = parser.parse_args()
    manifest = generate(args.hf_cache)
    data = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    if args.check:
        if args.output.read_bytes() != data:
            raise ValueError(
                f"Exclusion manifest differs from pinned sources: {args.output}"
            )
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(data)
    load_exclusions(args.output)
    print(
        json.dumps(
            {
                b["id"]: {
                    "records": b["expected_question_count"],
                    "unique": b["expected_unique_question_count"],
                }
                for b in manifest["benchmarks"]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
