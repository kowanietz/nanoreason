"""Smoke-test evidence for learning, BF16 weight synchronization, and rollouts."""

import json
import math
import time
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import TrainerCallback
from trl import GRPOTrainer

from .checkpoints import resume_checkpoint


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
