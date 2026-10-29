#!/usr/bin/env python3
"""Evaluate a model on MATH-500, AIME, or GPQA with OLMES."""

import argparse
import json
import os
import subprocess
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--config", choices=("math500", "aime", "gpqa"), default="math500"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--tokenizer", help="Defaults to the Think-SFT tokenizer.")
    parser.add_argument("--year", choices=("2024", "2025"), help="Required for AIME.")

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
    output_dir.mkdir(parents=True, exist_ok=True)

    olmes_env = os.environ.copy()
    olmes_env.pop("UV_PROJECT_ENVIRONMENT", None)

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
        ],
        check=True,
        cwd=ROOT,
        env=olmes_env,
    )


if __name__ == "__main__":
    main()
