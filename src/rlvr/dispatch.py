"""TRL grouped-rollout reward dispatch and domain telemetry."""

from collections import Counter
from statistics import fmean, pstdev

import torch.distributed as dist

from .ifeval import ifeval_score
from .rewards import final_response, math_reward


def rollout_metrics(rows: list[dict], num_generations: int) -> dict:
    """Summarize gathered completions, preserving group boundaries across ranks."""
    metrics = {}
    for domain in sorted({row["domain"] for row in rows}):
        selected = [r for r in rows if r["domain"] == domain]
        rewards = [r["reward"] for r in selected]
        prefix = f"domain/{domain}"
        metrics.update(
            {
                f"{prefix}/count": len(selected),
                f"{prefix}/reward_mean": fmean(rewards),
                f"{prefix}/reward_std": pstdev(rewards),
                f"{prefix}/reward_min": min(rewards),
                f"{prefix}/reward_max": max(rewards),
                f"{prefix}/zero_rate": fmean(v == 0 for v in rewards),
                f"{prefix}/partial_rate": fmean(0 < v < 1 for v in rewards),
                f"{prefix}/strict_rate": fmean(r["strict"] for r in selected),
                f"{prefix}/invalid_final_rate": fmean(
                    not r["valid_final"] for r in selected
                ),
                f"{prefix}/truncated_rate": fmean(r["truncated"] for r in selected),
            }
        )
        if all(r.get("completion_tokens") is not None for r in selected):
            metrics[f"{prefix}/completion_tokens"] = sum(
                r["completion_tokens"] for r in selected
            )
        passed, counts = Counter(), Counter()
        for row in selected:
            for key, ok in zip(row["constraint_ids"], row["passed"], strict=True):
                counts[key] += 1
                passed[key] += ok
        for key in counts:
            metrics[f"constraint/{key}/count"] = counts[key]
            metrics[f"constraint/{key}/pass_rate"] = passed[key] / counts[key]
    if len(rows) % num_generations:
        raise ValueError("Gathered rollout does not contain complete generation groups")
    groups = [
        rows[i : i + num_generations] for i in range(0, len(rows), num_generations)
    ]
    for group in groups:
        if len({(r["source_id"], r["domain"]) for r in group}) != 1:
            raise ValueError("TRL rollout group metadata is misaligned")
    for domain in ["all", *sorted({r["domain"] for r in rows})]:
        selected = (
            groups
            if domain == "all"
            else [g for g in groups if g[0]["domain"] == domain]
        )
        mixed = [len({r["reward"] for r in g}) > 1 for g in selected]
        metrics[f"group/{domain}/mixed_reward_rate"] = fmean(mixed)
        metrics[f"group/{domain}/uniform_reward_rate"] = 1 - fmean(mixed)
    return metrics


def domain_rewards(
    completions,
    ground_truth,
    domain,
    reasoning_prefix=None,
    prompts=None,
    completion_ids=None,
    source_id=None,
    *,
    eos_token_ids=(),
    num_generations=4,
    log_extra=None,
    log_metric=None,
    **kwargs,
) -> list[float]:
    """Score aligned local rows; gather metadata for global group/domain metrics."""
    count = len(completions)
    prefixes = [False] * count if reasoning_prefix is None else reasoning_prefix
    questions = [None] * count if prompts is None else prompts
    sources = list(range(count)) if source_id is None else source_id
    tokens = [None] * count if completion_ids is None else completion_ids
    if any(
        len(values) != count
        for values in (ground_truth, domain, prefixes, questions, sources, tokens)
    ):
        raise ValueError("Reward metadata must align with completions")
    rows = []
    for completion, reference, name, prefix, prompt, ids, source in zip(
        completions,
        ground_truth,
        domain,
        prefixes,
        questions,
        tokens,
        sources,
        strict=True,
    ):
        truncated = bool(eos_token_ids and (not ids or ids[-1] not in eos_token_ids))
        if name == "math":
            reward = math_reward(
                completion, reference, reasoning_prefix=prefix, prompt=prompt
            )
            score = {
                "reward": reward,
                "strict": reward,
                "constraint_ids": [],
                "passed": [],
                "valid_final": final_response(completion, reasoning_prefix=prefix)
                is not None,
            }
        elif name == "ifeval":
            score = ifeval_score(completion, reference, reasoning_prefix=prefix)
        else:
            raise ValueError(f"Unsupported reward domain: {name}")
        if truncated:
            score.update(reward=0.0, strict=0.0, passed=[False] * len(score["passed"]))
        rows.append(
            {
                **score,
                "domain": name,
                "source_id": source,
                "truncated": truncated,
                "completion_tokens": len(ids) if ids is not None else None,
            }
        )
    if log_extra is not None:
        for key in (
            "domain",
            "source_id",
            "strict",
            "valid_final",
            "truncated",
            "constraint_ids",
            "passed",
            "completion_tokens",
        ):
            log_extra(key, [row[key] for row in rows])
    if log_metric is not None:
        gathered = [None] * dist.get_world_size() if dist.is_initialized() else [rows]
        if dist.is_initialized():
            dist.all_gather_object(gathered, rows)
        global_rows = [row for rank_rows in gathered for row in rank_rows]
        for key, value in rollout_metrics(global_rows, num_generations).items():
            log_metric(key, value)
    return [row["reward"] for row in rows]


def make_domain_rewards(tokenizer, num_generations):
    """Bind tokenizer EOS IDs and rollout group size."""
    eos = tokenizer.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    eos_ids.add(tokenizer.pad_token_id)

    def rewards(**kwargs):
        return domain_rewards(
            **kwargs, eos_token_ids=eos_ids, num_generations=num_generations
        )

    rewards.__name__ = "domain_rewards"
    return rewards
