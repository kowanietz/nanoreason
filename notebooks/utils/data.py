import json

import numpy as np
import pandas as pd

from .config import (
    BENCHMARKS,
    MODELS,
    PASS_K,
    PRIMARY_METRICS,
    STAGES,
)


def load_metrics(results_dir):
    runs = {}
    rows = []
    for stage, folder in MODELS.items():
        for benchmark in BENCHMARKS:
            path = results_dir / folder / benchmark / "metrics.json"
            if not path.is_file():
                continue
            result = json.loads(path.read_text())
            runs[stage, benchmark] = result
            task = result["tasks"][0]
            config = task["task_config"]
            generation = config.get("generation_kwargs", {})
            metrics = task["metrics"]
            assert np.isclose(
                metrics["primary_score"], metrics[PRIMARY_METRICS[benchmark]]
            )
            rows.append(
                {
                    "stage": stage,
                    "benchmark": benchmark,
                    "score": 100 * metrics["primary_score"],
                    "num_instances": task["num_instances"],
                    "metric": config.get("primary_metric"),
                    "model_path": result["model_config"].get("model"),
                    "model_max_length": result["model_config"].get("max_length"),
                    "requested_max_gen_toks": generation.get("max_gen_toks"),
                    "temperature": generation.get("temperature"),
                    "top_p": generation.get("top_p"),
                    "repeats": generation.get("repeats", 1),
                    "dataset": config.get("dataset_path"),
                    "dataset_name": config.get("dataset_name"),
                    "source": f"results/{folder}/{benchmark}/metrics.json",
                }
            )

    expected = {(stage, benchmark) for stage in STAGES for benchmark in BENCHMARKS}
    missing = expected - runs.keys()
    assert missing == {("Base", "bbh")}, f"Unexpected missing results: {missing}"
    metrics = pd.DataFrame(rows)
    assert metrics.score.between(0, 100).all()

    for (stage, benchmark), result in runs.items():
        task = result["tasks"][0]
        if benchmark.startswith("aime-"):
            assert task["num_instances"] == 30
            assert task["task_config"]["generation_kwargs"]["repeats"] == 32
            values = [task["metrics"][f"pass_at_{k}"] for k in PASS_K]
            assert np.all(np.diff(values) >= -1e-12)
        elif benchmark == "bbh":
            parts = result["tasks"][1:]
            assert len(parts) == 27
            assert sum(part["num_instances"] for part in parts) == 6511
            micro = (
                sum(
                    part["metrics"]["primary_score"] * part["num_instances"]
                    for part in parts
                )
                / task["num_instances"]
            )
            assert np.isclose(micro, task["metrics"]["primary_score"])
    return runs, metrics


def score_table(metrics):
    return metrics.pivot(index="benchmark", columns="stage", values="score").reindex(
        index=BENCHMARKS, columns=STAGES
    )


def load_tokens(results_dir, runs):
    """Average generated tokens across every response, including AIME repeats."""
    rows = []
    for benchmark in BENCHMARKS:
        for stage in STAGES:
            if (stage, benchmark) not in runs:
                continue
            suite = runs[stage, benchmark]["tasks"][0]
            parts = (
                runs[stage, benchmark]["tasks"][1:] if benchmark == "bbh" else [suite]
            )
            expected = {part["alias"].split(":")[0]: part for part in parts}
            directory = results_dir / MODELS[stage] / benchmark
            paths = sorted(directory.glob("task-*-predictions.jsonl"))
            assert len(paths) == len(parts), (stage, benchmark, "prediction files")

            total_tokens = total_responses = total_problems = 0
            score_sum = 0.0
            seen_tasks = set()
            metric = "exact_match" if benchmark == "bbh" else PRIMARY_METRICS[benchmark]
            for path in paths:
                name = path.name.split("-", 2)[2].removesuffix("-predictions.jsonl")
                assert name in expected and name not in seen_tasks
                seen_tasks.add(name)
                part = expected[name]
                repeats = (
                    part["task_config"].get("generation_kwargs", {}).get("repeats", 1)
                )
                file_tokens = file_responses = 0
                seen_ids = set()
                with path.open() as handle:
                    for line in handle:
                        prediction = json.loads(line)
                        assert prediction["doc_id"] not in seen_ids
                        seen_ids.add(prediction["doc_id"])
                        outputs = prediction["model_output"]
                        assert len(outputs) == repeats, (
                            stage,
                            benchmark,
                            name,
                            "repeats",
                        )
                        counts = [output["num_tokens"] for output in outputs]
                        assert all(type(count) is int and count > 0 for count in counts)
                        file_tokens += sum(counts)
                        file_responses += len(counts)
                        score_sum += prediction["metrics"][metric]
                        if "num_tokens" in prediction["metrics"]:
                            assert np.isclose(
                                prediction["metrics"]["num_tokens"], np.mean(counts)
                            )

                assert len(seen_ids) == part["num_instances"], (stage, benchmark, name)
                assert file_responses == part["num_instances"] * repeats
                total_tokens += file_tokens
                total_responses += file_responses
                total_problems += len(seen_ids)

            assert seen_tasks == expected.keys()
            assert total_problems == suite["num_instances"]
            assert np.isclose(
                score_sum / total_problems, suite["metrics"]["primary_score"]
            )
            mean_tokens = total_tokens / total_responses
            aggregate_key = "num_tokens_micro" if benchmark == "bbh" else "num_tokens"
            aggregate_mean = (
                suite["metrics"].get("extra_metrics", {}).get(aggregate_key)
            )
            if aggregate_mean is not None:
                assert np.isclose(mean_tokens, aggregate_mean, rtol=1e-12)
            rows.append(
                {
                    "benchmark": benchmark,
                    "stage": stage,
                    "score": 100 * suite["metrics"]["primary_score"],
                    "mean_tokens": mean_tokens,
                    "problems": total_problems,
                    "responses": total_responses,
                    "prediction_files": len(paths),
                    "token_field": "model_output[].num_tokens",
                    "aggregate_mean_tokens": aggregate_mean,
                }
            )
    return pd.DataFrame(rows)
