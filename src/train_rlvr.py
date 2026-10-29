#!/usr/bin/env python3

import hashlib
import json
import math
import os
import random
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import hydra
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
from datasets import Dataset
from omegaconf import DictConfig, OmegaConf
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, TrainerCallback
from transformers.models.olmo3.modeling_olmo3 import Olmo3Model
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOConfig, GRPOTrainer

from benchmark_exclusions import load_exclusions
from rlvr_rewards import math_rewards, validate_reference_answers
from training_common import StopAtDeadline, make_tokenizer

ROOT = Path(__file__).resolve().parents[1]


class TrainingOlmo3Model(Olmo3Model):
    # vLLM mutates Olmo3Model.__call__; training must retain PyTorch's hooks.
    __call__ = torch.nn.Module.__call__


@torch.no_grad()
def down_proj_sample(
    parameter: torch.Tensor, tp_rank: int, tp_size: int
) -> torch.Tensor:
    """Sample the policy columns held by one vLLM row-parallel down projection."""
    full = parameter.full_tensor() if hasattr(parameter, "full_tensor") else parameter
    if full.ndim != 2 or full.shape[1] % tp_size:
        raise ValueError("Unexpected down-projection shape for TP readback")
    columns = full.shape[1] // tp_size
    indices = torch.arange(
        0,
        full.shape[0] * columns,
        max(1, full.shape[0] * columns // 1024),
        device=full.device,
    )
    return (
        full[indices // columns, indices % columns + tp_rank * columns]
        .detach()
        .float()
        .clone()
    )


def fingerprint(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_config(config: dict) -> dict:
    args, runtime = config["grpo"], config["runtime"]
    world = runtime["expected_world_size"]
    batch = args["per_device_train_batch_size"] * world
    generation = args["generation_batch_size"]
    if (
        world != 4
        or batch <= 0
        or args["num_generations"] < 2
        or args["gradient_accumulation_steps"] < 1
        or args["max_completion_length"] < 1
        or args["vllm_tensor_parallel_size"] not in (1, 2, 4)
        or generation <= 0
        or generation % batch
        or generation % args["num_generations"]
    ):
        raise ValueError("Invalid four-rank TP/generation batch/group sizes")
    if (
        generation // batch != args["gradient_accumulation_steps"]
        or args["num_iterations"] != 1
    ):
        raise ValueError(
            "Smoke checkpoints require one fresh generation batch per optimizer step"
        )
    if (
        args["vllm_max_model_length"]
        < config["data"]["max_prompt_tokens"] + args["max_completion_length"]
        or not 0 < args["vllm_gpu_memory_utilization"] < 1
    ):
        raise ValueError("Invalid vLLM context length or memory utilization")
    if (
        args["loss_type"] != "dr_grpo"
        or args["scale_rewards"] != "none"
        or args["beta"] != 0
        or not args["bf16"]
        or not args["gradient_checkpointing"]
        or not args["fsdp"]
        or args["fsdp_config"]["version"] != 2
        or not args["fsdp_config"]["reshard_after_forward"]
        or args["fsdp_config"]["cpu_offload"]
        or args["fsdp_config"]["activation_checkpointing"]
        or not args["use_vllm"]
        or args["vllm_mode"] != "colocate"
        or not args["vllm_enable_sleep_mode"]
        or args["remove_unused_columns"]
        or args["save_only_model"]
        or args["ignore_data_skip"]
        or args["use_liger_kernel"]
        or args["torch_compile"]
    ):
        raise ValueError("Unsupported Phase 3A training settings")
    stop = runtime["stop_after_step"]
    if not 1 <= args["max_steps"] <= 10 or (
        stop is not None and not 1 <= stop <= args["max_steps"]
    ):
        raise ValueError(
            "Smoke test and intermediate stop must be within 1–10 optimizer steps"
        )
    source, destination = (
        Path(config["model"]["name_or_path"]).resolve(),
        Path(args["output_dir"]).resolve(),
    )
    if (
        source == destination
        or source in destination.parents
        or destination in source.parents
    ):
        raise ValueError("RLVR output must be isolated from the SFT source")
    if (
        config["data"]["max_examples"] < args["max_steps"]
        or config["data"]["max_prompt_tokens"] < 1
    ):
        raise ValueError("Invalid deterministic smoke subset or prompt budget")
    source_id = config["data"].get("source_id")
    if source_id is not None and (
        not isinstance(source_id, str) or not source_id.strip()
    ):
        raise ValueError("Smoke source_id must be a nonempty string")
    return {
        "world_size": world,
        "generation_batch_size": generation,
        "steps_per_generation": generation // batch,
        "unique_prompts_per_generation": generation // args["num_generations"],
    }


def validate_model(config: dict, tokenizer) -> dict:
    source = Path(config["model"]["name_or_path"])
    for name in (
        "config.json",
        "model.safetensors",
        "training_complete.json",
        "tokenizer.json",
        "chat_template.jinja",
    ):
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
    if (
        json.loads((source / "training_complete.json").read_text())["global_step"]
        != 2784
    ):
        raise ValueError("Expected the completed E16 export")
    model_config = AutoConfig.from_pretrained(
        source, local_files_only=True, trust_remote_code=False
    )
    exported_tokenizer = make_tokenizer(
        {**config["model"], "tokenizer_name_or_path": str(source)}
    )
    if (
        model_config.model_type != "olmo3"
        or model_config.architectures != ["Olmo3ForCausalLM"]
        or len(tokenizer) != model_config.vocab_size
        or tokenizer.eos_token_id != model_config.eos_token_id
        or tokenizer.pad_token_id != model_config.pad_token_id
        or exported_tokenizer.backend_tokenizer.to_str()
        != tokenizer.backend_tokenizer.to_str()
        or (source / "chat_template.jinja").read_text() != tokenizer.chat_template
    ):
        raise ValueError("E16 architecture/tokenizer/vocabulary mismatch")
    with safe_open(
        source / "model.safetensors", framework="pt", device="cpu"
    ) as weights:
        for name in ("model.embed_tokens.weight", "lm_head.weight"):
            if weights.get_slice(name).get_shape() != [
                model_config.vocab_size,
                model_config.hidden_size,
            ]:
                raise ValueError("E16 embedding/output weight shape mismatch")
    return {
        "path": str(source.resolve()),
        "config": model_config.to_dict(),
        "config_sha256": fingerprint(source / "config.json"),
        "training_complete_sha256": fingerprint(source / "training_complete.json"),
        "source_run_manifest_sha256": fingerprint(source / "run_manifest.json"),
        "weights_size": (source / "model.safetensors").stat().st_size,
        "weights_mtime_ns": (source / "model.safetensors").stat().st_mtime_ns,
        "load_dtype": "bfloat16",
        "tokenizer_sha256": fingerprint(source / "tokenizer.json"),
        "chat_template_sha256": hashlib.sha256(
            tokenizer.chat_template.encode()
        ).hexdigest(),
    }


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


def resume_checkpoint(
    output: Path, setting: str | bool | None, world: int
) -> str | None:
    latest = get_last_checkpoint(str(output)) if output.is_dir() else None
    if setting in (None, False):
        if latest:
            raise ValueError("Existing checkpoints require resumption")
        return None
    checkpoint = latest if setting == "auto" else str(Path(setting).resolve())
    if checkpoint is None:
        return None
    path = Path(checkpoint)
    required = [
        "trainer_state.json",
        "scheduler.pt",
        "optimizer_0/.metadata",
        "pytorch_model_fsdp_0/.metadata",
        *[f"rng_state_{rank}.pth" for rank in range(world)],
    ]
    if (
        path.resolve().parent != output.resolve()
        or path.resolve() != Path(latest or "").resolve()
        or (path / "checkpoint-is-incomplete.txt").exists()
        or any(not (path / name).is_file() for name in required)
    ):
        raise ValueError(f"Incomplete or non-latest native FSDP checkpoint: {path}")
    return str(path)


class SmokeDiagnostics(TrainerCallback):
    def __init__(self, stop_after_step: int | None):
        self.stop_after_step = stop_after_step
        self.trainer = None
        self.before = {}
        self.updated_steps = 0
        self.updated_policy_steps = set()
        self.max_delta = 0.0
        self.max_gradient = 0.0
        self.nonzero_gradient_steps = set()
        self.sync_steps = set()
        self.sync_before = {}
        self.verified_sync_steps = set()
        self.vllm_readback_delta = 0.0
        self.mixed_rewards = False
        self.distinct_completions = False

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        parameters = [
            (name, p) for name, p in model.named_parameters() if p.requires_grad
        ]
        gradients = [
            p.grad.to_local() if hasattr(p.grad, "to_local") else p.grad
            for _, p in parameters
            if p.grad is not None
        ]
        finite = (
            torch.stack([torch.isfinite(gradient).all() for gradient in gradients])
            .all()
            .int()
        )
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not finite.item():
            raise FloatingPointError("Nonfinite gradients before optimizer step")
        gradient_max = torch.zeros((), device=self.trainer.accelerator.device)
        for gradient in gradients:
            if gradient.numel():
                gradient_max = torch.maximum(
                    gradient_max, gradient.detach().abs().max().float()
                )
        dist.all_reduce(gradient_max, op=dist.ReduceOp.MAX)
        self.max_gradient = max(self.max_gradient, gradient_max.item())
        if gradient_max.item() > 0:
            self.nonzero_gradient_steps.add(state.global_step + 1)
        self.before = {}
        for name, p in parameters:
            if name.endswith("mlp.down_proj.weight"):
                local = p.to_local() if hasattr(p, "to_local") else p
                self.before[name] = (
                    local.detach()
                    .flatten()[:: max(1, local.numel() // 1024)]
                    .float()
                    .clone()
                )
                if args.max_steps == 1:
                    self.sync_before[name] = down_proj_sample(
                        p,
                        args.process_index % args.vllm_tensor_parallel_size,
                        args.vllm_tensor_parallel_size,
                    )
                if len(self.before) == 2:
                    break

    def on_optimizer_step(self, args, state, control, model=None, **kwargs):
        delta = torch.zeros((), device=self.trainer.accelerator.device)
        for name, p in model.named_parameters():
            if name in self.before:
                local = p.to_local() if hasattr(p, "to_local") else p
                after = (
                    local.detach().flatten()[:: max(1, local.numel() // 1024)].float()
                )
                delta = torch.maximum(delta, (after - self.before[name]).abs().max())
        dist.all_reduce(delta, op=dist.ReduceOp.MAX)
        self.max_delta = max(self.max_delta, delta.item())
        self.updated_steps += int(delta.item() > 0)
        if delta.item() > 0:
            self.updated_policy_steps.add(state.global_step + 1)
        self.sync_steps.add(self.trainer._last_loaded_step)
        print(
            json.dumps(
                {
                    "event": "optimizer_update",
                    "rank": args.process_index,
                    "step": state.global_step + 1,
                    "sampled_parameter_delta_max": delta.item(),
                    "vllm_synced_policy_step": self.trainer._last_loaded_step,
                    "torch_peak_allocated_gib": torch.cuda.max_memory_allocated()
                    / 2**30,
                    "torch_peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                    "device_used_gib": torch.cuda.device_memory_used(args.device)
                    / 2**30,
                }
            ),
            flush=True,
        )

    def on_step_end(self, args, state, control, **kwargs):
        if (
            self.stop_after_step is not None
            and state.global_step >= self.stop_after_step
        ):
            control.should_save = True
            control.should_training_stop = True
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if (
            args.max_steps == 1
            and state.global_step == 1
            and self.updated_policy_steps & self.nonzero_gradient_steps
            and self.mixed_rewards
            and self.distinct_completions
        ):
            # Retain the subsequent-rollout sync check without a second optimizer step.
            prompts = [self.trainer.train_dataset[0]["prompt"]]
            ids, images, fields = self.trainer._tokenize_prompts(prompts)
            generation = self.trainer.vllm_generation
            original_sync = generation.sync_weights
            verified = False

            def checked_sync():
                nonlocal verified
                original_sync()
                # Sleep level 2 discards weights after generation: read back before it.
                self.verify_vllm_weights(args)
                verified = True

            generation.sync_weights = checked_sync
            try:
                completions, _ = self.trainer._generate_single_turn(
                    ids, images, fields, args.num_generations
                )
            finally:
                generation.sync_weights = original_sync
            if not verified or self.trainer._last_loaded_step != state.global_step:
                raise RuntimeError(
                    "Post-update rollout did not verify freshly synchronized vLLM weights"
                )
            if len(completions) != 1 or not completions[0]:
                raise RuntimeError(
                    "Post-update validation rollout returned no completion"
                )
            self.sync_steps.add(self.trainer._last_loaded_step)
            print(
                json.dumps(
                    {
                        "event": "post_update_rollout",
                        "step": state.global_step,
                        "vllm_synced_policy_step": self.trainer._last_loaded_step,
                        "updated_weights_readback_verified": self.vllm_readback_delta
                        > 0,
                        "local_completion_tokens": len(completions[0]),
                    }
                ),
                flush=True,
            )
            if self.vllm_readback_delta == 0:
                raise RuntimeError(
                    "vLLM matches the BF16 policy and the post-update rollout completed, but no changed "
                    "sample survives BF16 rounding; an updated inference policy was not proven"
                )
            self.verified_sync_steps.add(state.global_step)

    def verify_vllm_weights(self, args):
        generation = self.trainer.vllm_generation
        inference = (
            generation.llm.llm_engine.model_executor.driver_worker.model_runner.model
        )
        matches = torch.ones(
            (), dtype=torch.int, device=self.trainer.accelerator.device
        )
        changed = torch.zeros((), device=self.trainer.accelerator.device)
        raw_changed = torch.zeros((), device=self.trainer.accelerator.device)
        details = []
        parameters = dict(self.trainer.model.named_parameters())
        for name, before in self.sync_before.items():
            parameter = parameters[name]
            expected = down_proj_sample(
                parameter,
                args.process_index % args.vllm_tensor_parallel_size,
                args.vllm_tensor_parallel_size,
            )
            loaded = inference.get_parameter(generation._fix_param_name_to_vllm(name))
            shape = (
                parameter.shape[0],
                parameter.shape[1] // args.vllm_tensor_parallel_size,
            )
            if tuple(loaded.shape) != shape:
                raise ValueError("Unexpected vLLM row-parallel down-projection shape")
            if loaded.dtype != torch.bfloat16:
                raise ValueError("Expected BF16 vLLM down-projection weights")
            # Accelerate maintains FP32 FSDP2 master weights; vLLM's copy_ casts them.
            master = expected
            expected = master.to(loaded.dtype).float()
            previous = before.to(loaded.dtype).float()
            actual = (
                loaded.detach().flatten()[:: max(1, loaded.numel() // 1024)].float()
            )
            matches *= torch.eq(actual, expected).all().int()
            changed = torch.maximum(changed, (expected - previous).abs().max())
            raw_changed = torch.maximum(raw_changed, (master - before).abs().max())
            details.append(
                {
                    "parameter": name,
                    "policy_dtype": str(parameter.dtype),
                    "vllm_dtype": str(loaded.dtype),
                    "tp_rank": args.process_index % args.vllm_tensor_parallel_size,
                    "policy_shape": list(parameter.shape),
                    "vllm_shape": list(loaded.shape),
                    "samples": actual.numel(),
                    "mismatched_samples": (actual != expected).sum().item(),
                    "representably_changed_samples": (expected != previous)
                    .sum()
                    .item(),
                    "fp32_change_max": (master - before).abs().max().item(),
                    "bf16_change_max": (expected - previous).abs().max().item(),
                    "readback_error_max": (actual - expected).abs().max().item(),
                }
            )
        dist.all_reduce(matches, op=dist.ReduceOp.MIN)
        dist.all_reduce(changed, op=dist.ReduceOp.MAX)
        dist.all_reduce(raw_changed, op=dist.ReduceOp.MAX)
        print(
            json.dumps(
                {
                    "event": "vllm_weight_readback",
                    "rank": args.process_index,
                    "step": self.trainer.state.global_step,
                    "all_samples_match_policy": bool(matches.item()),
                    "fp32_change_max": raw_changed.item(),
                    "bf16_change_max": changed.item(),
                    "parameters": details,
                }
            ),
            flush=True,
        )
        if not matches.item():
            raise RuntimeError(
                "vLLM weight readback failed: stale/mismatched weights against the BF16-cast updated policy"
            )
        self.vllm_readback_delta = changed.item()

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        if "frac_reward_zero_std" in logs:
            self.mixed_rewards |= logs["frac_reward_zero_std"] < 1
        completions = list(self.trainer._logs["completion"])
        size = args.num_generations
        distinct = [
            len({json.dumps(c, sort_keys=True) for c in completions[i : i + size]})
            for i in range(0, len(completions) - size + 1, size)
        ]
        self.distinct_completions |= any(count > 1 for count in distinct)
        if args.process_index == 0 and distinct:
            print(
                json.dumps(
                    {
                        "event": "completion_diversity",
                        "step": state.global_step,
                        "distinct_completions_per_group": distinct,
                    }
                ),
                flush=True,
            )


def write_run_manifest(
    output: Path, checkpoint: str | None, settings: dict, config: dict, rank: int
) -> None:
    """Validate resume provenance before any checkpoint or output is written."""
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    manifest_path = output / "run_manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text())["sha256"] != digest:
            raise ValueError(
                "Training configuration or inputs changed since the first job"
            )
    elif checkpoint or (output.exists() and any(output.iterdir())):
        raise ValueError("Refusing an unmanifested output directory")
    dist.barrier()
    if not manifest_path.exists() and rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(
                {"sha256": digest, "settings": settings, "resolved_config": config},
                indent=2,
            )
        )
    dist.barrier()


def finish_smoke_test(
    trainer: GRPOTrainer,
    diagnostics: SmokeDiagnostics,
    checkpoint: str | None,
    started: float,
) -> None:
    """Verify the saved checkpoint and persist the learning/synchronization evidence."""
    args = trainer.args
    output = Path(args.output_dir)
    trainer.save_state()
    dist.barrier()
    saved = resume_checkpoint(output, "auto", args.world_size)
    if (
        saved is None
        or json.loads((Path(saved) / "trainer_state.json").read_text())["global_step"]
        != trainer.state.global_step
    ):
        raise RuntimeError("Final native checkpoint was not saved at the stopping step")
    losses = [entry["loss"] for entry in trainer.state.log_history if "loss" in entry]
    finite_losses = bool(losses) and all(math.isfinite(loss) for loss in losses)
    result = {
        "global_step": trainer.state.global_step,
        "resumed_from": checkpoint,
        "updated_steps_this_job": diagnostics.updated_steps,
        "parameter_delta_max": diagnostics.max_delta,
        "gradient_abs_max": diagnostics.max_gradient,
        "nonzero_gradient_steps_this_job": sorted(diagnostics.nonzero_gradient_steps),
        "vllm_synced_policy_steps_this_job": sorted(diagnostics.sync_steps),
        "finite_logged_losses": finite_losses,
        "updated_policy_resynchronized": bool(
            diagnostics.updated_policy_steps & diagnostics.sync_steps
        ),
        "updated_policy_weights_verified": bool(
            diagnostics.updated_policy_steps & diagnostics.verified_sync_steps
        ),
        "vllm_weight_readback_delta_max": diagnostics.vllm_readback_delta,
        "mixed_reward_group_observed": diagnostics.mixed_rewards,
        "distinct_completion_group_observed": diagnostics.distinct_completions,
        "checkpoint": saved,
        "trajectory_reproducibility": "not established; native resume regenerates rollouts and vLLM RNG",
        "seconds": time.monotonic() - started,
    }
    result["checks_passed"] = (
        finite_losses
        and bool(diagnostics.nonzero_gradient_steps & diagnostics.updated_policy_steps)
        and diagnostics.updated_steps > 0
        and diagnostics.mixed_rewards
        and diagnostics.distinct_completions
        and result["updated_policy_resynchronized"]
        and (args.max_steps != 1 or result["updated_policy_weights_verified"])
    )
    if trainer.is_world_process_zero():
        (output / f"smoke-result-{trainer.state.global_step}.json").write_text(
            json.dumps(result, indent=2)
        )
    print(json.dumps(result), flush=True)
    if not finite_losses:
        raise FloatingPointError("Nonfinite training loss")
    if not diagnostics.mixed_rewards:
        raise RuntimeError(
            "Uniform-reward sampling produced zero group advantages; learning was not validated"
        )
    if not diagnostics.nonzero_gradient_steps & diagnostics.updated_policy_steps:
        raise RuntimeError(
            "No optimizer step had both a nonzero gradient and an observed parameter update"
        )
    if not diagnostics.distinct_completions:
        raise RuntimeError(
            "Smoke test needs a distinct-completion group with mixed correctness rewards"
        )
    if not result["updated_policy_resynchronized"]:
        raise RuntimeError(
            "No subsequent rollout synchronized an observed policy update; more smoke steps are needed"
        )
    if args.max_steps == 1 and not result["updated_policy_weights_verified"]:
        raise RuntimeError(
            "Updated-policy vLLM weights were not verified by readback before a separate rollout"
        )


@hydra.main(version_base="1.3", config_path="../configs", config_name="rlvr-smoke")
def main(config: DictConfig) -> None:
    started = time.monotonic()
    config = OmegaConf.to_container(config, resolve=True)
    batch = validate_config(config)
    tokenizer = make_tokenizer(config["model"])
    tokenizer.padding_side = "left"
    model_info = validate_model(config, tokenizer)
    dataset, data_info = load_smoke_data(config, tokenizer)
    versions = {
        name: version(name)
        for name in ("trl", "vllm", "torch", "transformers", "accelerate", "datasets")
    }
    if not versions["trl"].startswith("1.16.") or versions["vllm"] != "0.30.0":
        raise ValueError(
            "Phase 3A requires the locked TRL 1.16.x / vLLM 0.30.0 environment"
        )
    print(
        json.dumps(
            {
                "event": "preflight",
                "batch": batch,
                "model": model_info,
                "data": data_info,
                "versions": versions,
                "config": config,
            },
            indent=2,
        ),
        flush=True,
    )
    if config["runtime"]["validate_only"]:
        cpu_args = {**config["grpo"], "use_cpu": True}
        GRPOConfig(**cpu_args)
        print(
            "CPU validation passed; no model weights loaded or rollouts generated",
            flush=True,
        )
        return
    if (
        not os.environ.get("SLURM_JOB_ID")
        or not torch.cuda.is_available()
        or int(os.environ.get("WORLD_SIZE", "1")) != batch["world_size"]
        or int(os.environ.get("LOCAL_WORLD_SIZE", "1")) != batch["world_size"]
        or int(os.environ.get("SLURM_JOB_NUM_NODES", "1")) != 1
        or torch.cuda.device_count() != 4
    ):
        raise RuntimeError(
            "Training requires one allocated Slurm GPU node and four torchrun ranks"
        )
    args = GRPOConfig(**config["grpo"])
    output = Path(args.output_dir)
    checkpoint = resume_checkpoint(
        output, config["runtime"]["resume_from_checkpoint"], args.world_size
    )
    settings = {
        "model": model_info,
        "data": data_info,
        "grpo": config["grpo"],
        "model_config": config["model"],
        "data_config": config["data"],
        "versions": versions,
        "uv_lock_sha256": fingerprint(ROOT / "uv.lock"),
        "world_size": args.world_size,
    }
    write_run_manifest(output, checkpoint, settings, config, args.process_index)
    if checkpoint:
        step = json.loads((Path(checkpoint) / "trainer_state.json").read_text())[
            "global_step"
        ]
        if step >= (config["runtime"]["stop_after_step"] or args.max_steps):
            result_path = output / f"smoke-result-{step}.json"
            if not result_path.is_file() or not json.loads(result_path.read_text()).get(
                "checks_passed"
            ):
                raise RuntimeError(
                    "Checkpoint reached this milestone without verified smoke diagnostics; inspect its logs"
                )
            print(
                f"Requested smoke milestone already reached at step {step}", flush=True
            )
            return
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["name_or_path"],
        dtype=torch.bfloat16,
        attn_implementation=config["model"]["attention_implementation"],
        trust_remote_code=False,
        local_files_only=True,
    )
    model.config.use_cache = False
    model.model.__class__ = TrainingOlmo3Model
    if model.dtype != torch.bfloat16:
        raise ValueError("Policy was not initialized in BF16")
    diagnostics = SmokeDiagnostics(config["runtime"]["stop_after_step"])
    trainer = GRPOTrainer(
        model=model,
        args=args,
        reward_funcs=math_rewards,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[
            diagnostics,
            StopAtDeadline(started, config["runtime"]["stop_after_minutes"]),
        ],
    )
    diagnostics.trainer = trainer
    if (
        trainer.ref_model is not None
        or trainer.vllm_generation.llm.llm_engine.model_config.dtype != torch.bfloat16
    ):
        raise ValueError("Unexpected reference model or non-BF16 vLLM engine")
    print(
        json.dumps(
            {
                "event": "gpu_initialized",
                "rank": args.process_index,
                "gpu": torch.cuda.get_device_name(args.device),
                "world_size": args.world_size,
                "device_total_gib": torch.cuda.get_device_properties(
                    args.device
                ).total_memory
                / 2**30,
                "torch_initialization_peak_gib": torch.cuda.max_memory_allocated()
                / 2**30,
                "device_used_gib": torch.cuda.device_memory_used(args.device) / 2**30,
            }
        ),
        flush=True,
    )
    torch.cuda.reset_peak_memory_stats()
    trainer.train(resume_from_checkpoint=checkpoint)
    finish_smoke_test(trainer, diagnostics, checkpoint, started)


if __name__ == "__main__":
    main()
