#!/usr/bin/env python3
"""Evaluate a model on MATH-500, AIME, GPQA, IFEval, or all 27 BBH tasks with OLMES."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from benchmark_exclusions import DEFAULT_EXCLUSIONS


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--config",
        choices=("math500", "aime", "gpqa", "ifeval", "bbh"),
        default="math500",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--tokenizer", help="Defaults to the Think-SFT tokenizer.")
    parser.add_argument("--year", choices=("2024", "2025"), help="Required for AIME.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve OLMES recipes and print commands without model loading",
    )

    args = parser.parse_args()

    if args.config == "aime" and args.year is None:
        parser.error("--year is required for AIME")

    if args.config != "aime" and args.year is not None:
        parser.error("--year is only supported for AIME")

    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        config = compose(config_name=f"eval/{args.config}")

    model_args = OmegaConf.to_container(config.model_args, resolve=True)
    model_args["tokenizer"] = args.tokenizer or model_args["tokenizer"]

    task_args = []

    if "task_args" in config:
        task_args = [
            "--task-args",
            json.dumps(OmegaConf.to_container(config.task_args, resolve=True)),
        ]

    tasks = config.tasks

    benchmark = args.config

    if args.config == "aime":
        tasks = [config.tasks[args.year]]
        benchmark = f"aime-{args.year}"

    output_dir = ROOT / "results" / args.name / benchmark
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    olmes_env = os.environ.copy()
    olmes_env.pop("UV_PROJECT_ENVIRONMENT", None)
    if args.config in {"ifeval", "bbh"}:
        resource_root = ROOT / ".cache/ifevalg/nltk_data"
        olmes_env["NLTK_DATA"] = str(resource_root)
        olmes_env["HF_DATASETS_CACHE"] = str(ROOT / ".cache/olmes-datasets")
    project = Path(config.olmes_project)
    revision = subprocess.check_output(
        ["git", "-C", str(project), "rev-parse", "HEAD"],
        env={**olmes_env, "GIT_CONFIG_NOSYSTEM": "1"},
        text=True,
    ).strip()
    changes = subprocess.check_output(
        ["git", "-C", str(project), "diff", "--binary", "HEAD", "--"],
        env={**olmes_env, "GIT_CONFIG_NOSYSTEM": "1"},
    )
    provenance = {
        "olmes_revision": revision,
        "olmes_diff_sha256": hashlib.sha256(changes).hexdigest(),
        "olmes_lock_sha256": hashlib.sha256(
            (project / "uv.lock").read_bytes()
        ).hexdigest(),
        "exclusions_sha256": hashlib.sha256(
            DEFAULT_EXCLUSIONS.read_bytes()
        ).hexdigest(),
    }
    if not args.dry_run:
        path = output_dir / "benchmark_provenance.json"
        if path.exists() and json.loads(path.read_text()) != provenance:
            raise ValueError(
                f"Benchmark provenance changed; choose a new --name: {path}"
            )
        path.write_text(json.dumps(provenance, indent=2) + "\n")

    subprocess.run(
        [
            "uv",
            "run",
            "--project",
            config.olmes_project,
            "--frozen",
            "--offline",
            "olmes",
            "--model",
            args.model,
            "--model-type",
            config.model_type,
            "--model-args",
            json.dumps(model_args),
            "--task",
            *tasks,
            *task_args,
            "--batch-size",
            str(config.batch_size),
            "--gpus",
            str(config.gpus),
            "--num-workers",
            str(config.num_workers),
            "--output-dir",
            str(output_dir),
            *(["--dry-run"] if args.dry_run else []),
        ],
        check=True,
        cwd=ROOT,
        env=olmes_env,
    )


if __name__ == "__main__":
    main()
