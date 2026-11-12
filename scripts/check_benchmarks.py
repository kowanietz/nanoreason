#!/usr/bin/env python3
"""Check OLMES datasets against the training exclusion manifest."""

import argparse
import copy
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# IFEval key 2785: Google says "at least 3 placeholders" but H4/OLMES says "at least one placeholder".
# This exact pair is accepted only here. Training exclusions remain unchanged.
IFEVAL_GOOGLE_2785 = "8cdd05d2afe1a51fd6123668bae435f1ef27b10e773516a90540792eda113e0d"
IFEVAL_H4_2785 = "cc65ff6173080d6f8ebda0638fb2e12399e403cada46fa1e61d7988fed25a753"


def check_questions(identifier, count, observed, expected):
    """Check counts and hashes, allowing the single known IFEval wording variant."""
    hashes = set(expected["question_hashes"])
    known_variant = (
        identifier == "ifeval"
        and observed - hashes == {IFEVAL_H4_2785}
        and hashes - observed == {IFEVAL_GOOGLE_2785}
    )
    if count != expected["expected_question_count"] or (
        observed != hashes and not known_variant
    ):
        raise ValueError(
            f"{count}/{expected['expected_question_count']} documents; "
            f"{len(observed - hashes)} prompts missing from exclusions, "
            f"{len(hashes - observed)} exclusion prompts absent from evaluation"
        )
    return known_variant


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ[key] = "0" if args.allow_download else "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["NLTK_DATA"] = str(ROOT / ".cache/ifevalg/nltk_data")
    sys.path.insert(0, str(ROOT / "src"))

    import nltk
    import yaml
    from oe_eval.configs.task_suites import TASK_SUITE_CONFIGS
    from oe_eval.configs.tasks import TASK_CONFIGS
    from oe_eval.data.bbh_tasks import BBH_TASKS
    from oe_eval.dependencies.ifeval import instructions_util
    from oe_eval.tasks.oe_eval_tasks import TASK_REGISTRY
    from oe_eval.utils import update_nested_dict

    from benchmark_exclusions import DEFAULT_EXCLUSIONS, load_exclusions, question_hash

    nltk.word_tokenize("Offline tokenizer check.")
    instructions_util.count_sentences("Offline tokenizer check.")
    path = DEFAULT_EXCLUSIONS
    load_exclusions(path)
    benchmarks = {b["id"]: b for b in json.loads(path.read_text())["benchmarks"]}
    failures = []
    for benchmark, identifier, year, field in (
        ("math500", "math500", None, "problem"),
        ("aime", "aime2024", "2024", "problem"),
        ("aime", "aime2025", "2025", "problem"),
        ("gpqa", "gpqa_main", None, "Question"),
        ("ifeval", "ifeval", None, "prompt"),
        ("bbh", "bbh", None, "input"),
    ):
        preset = yaml.safe_load((ROOT / f"configs/eval/{benchmark}.yaml").read_text())
        aliases = [preset["tasks"][year]] if year else preset["tasks"]
        aliases = [
            task
            for alias in aliases
            for task in TASK_SUITE_CONFIGS.get(alias, {"tasks": [alias]})["tasks"]
        ]
        if benchmark == "bbh" and (
            len(aliases) != 27
            or set(aliases) != {f"bbh_{name}:cot::olmo3:adapt" for name in BBH_TASKS}
        ):
            raise ValueError("Expected all 27 BBH subtasks")
        cache = (
            str(ROOT / ".cache/olmes-datasets")
            if benchmark in {"ifeval", "bbh"}
            else None
        )
        observed, count = set(), 0
        try:
            for alias in aliases:
                config = update_nested_dict(
                    copy.deepcopy(TASK_CONFIGS[alias]), preset.get("task_args", {})
                )
                task = TASK_REGISTRY[config["task_name"]](
                    task_name=config["task_name"], task_config=config
                )
                task.make_metrics()
                task.download(cache_dir=cache)
                docs = task.get_eval_docs()
                # AIME filters its split by year.
                raw = docs if year else task.dataset[task.task_config["split"]]
                if len(docs) != len(raw):
                    raise ValueError(
                        f"{alias}: evaluation document count differs from the loaded split"
                    )
                observed.update(question_hash(doc[field]) for doc in raw)
                count += len(raw)
            known_variant = check_questions(
                identifier, count, observed, benchmarks[identifier]
            )
        except (ConnectionError, FileNotFoundError, ValueError) as error:
            failures.append(f"{identifier}: {error}")
            continue
        status = (
            "known H4 wording variant at key 2785 accepted; exclusions unchanged"
            if known_variant
            else "exclusions match"
        )
        print(
            f"{identifier}: {count} documents, {len(aliases)} tasks; {status}",
            flush=True,
        )
    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    main()
