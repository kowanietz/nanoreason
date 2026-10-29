import time

import torch
import torch.distributed as dist
from transformers import AutoTokenizer, TrainerCallback


class StopAtDeadline(TrainerCallback):
    def __init__(self, start_time: float, minutes: float):
        self.deadline = start_time + minutes * 60

    def on_step_end(self, args, state, control, **kwargs):
        expired = time.monotonic() >= self.deadline
        if dist.is_initialized():
            flag = torch.tensor(
                int(expired), device=f"cuda:{torch.cuda.current_device()}"
            )
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            expired = bool(flag.item())

        if expired:
            control.should_save = True
            control.should_training_stop = True
        return control


def make_tokenizer(model_config: dict):
    """Load the local tokenizer shared by preparation, SFT, and RLVR."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["tokenizer_name_or_path"],
        trust_remote_code=model_config.get("trust_remote_code", False),
        use_fast=True,
        local_files_only=True,
    )
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer is required")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
