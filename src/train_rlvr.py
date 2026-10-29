#!/usr/bin/env python3

import json
import os
import time
from importlib.metadata import version
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM
from transformers.models.olmo3.modeling_olmo3 import Olmo3Model
from trl import GRPOConfig, GRPOTrainer

from rlvr.checkpoints import fingerprint, resume_checkpoint, write_run_manifest
from rlvr.data import load_smoke_data
from rlvr.diagnostics import SmokeDiagnostics, finish_smoke_test
from rlvr.rewards import math_rewards
from rlvr.validation import validate_config, validate_model
from training_common import StopAtDeadline, make_tokenizer

ROOT = Path(__file__).resolve().parents[1]


class TrainingOlmo3Model(Olmo3Model):
    # vLLM mutates Olmo3Model.__call__; training must retain PyTorch's hooks.
    __call__ = torch.nn.Module.__call__


@hydra.main(version_base="1.3", config_path="../configs", config_name="rlvr/smoke")
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
