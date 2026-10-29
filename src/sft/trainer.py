"""Repetition ordering, cumulative training metrics, and epoch checkpoints."""

import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import (
    Trainer,
    TrainerCallback,
)


class RepetitionTrainer(Trainer):
    """Use Trainer's training loop; add ordering and Liger's masked accuracy."""

    def __init__(self, *args, metrics_checkpoint=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.repetition_totals = None
        self.repetition_base = torch.zeros(5, dtype=torch.float64)
        if metrics_checkpoint:
            stats = json.loads(
                (Path(metrics_checkpoint) / "repetition_metrics.json").read_text()
            )
            self.repetition_base = torch.tensor(
                [
                    stats[key]
                    for key in (
                        "example_presentations",
                        "supervised_tokens",
                        "correct_assistant_tokens",
                        "optimization_loss_sum",
                        "microbatches",
                    )
                ],
                dtype=torch.float64,
            )

    def _get_train_sampler(self, train_dataset=None):
        return torch.utils.data.SequentialSampler(
            self.train_dataset if train_dataset is None else train_dataset
        )

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        inputs = dict(inputs)
        examples = inputs.pop("example_count").sum()
        targets = (inputs["labels"][:, 1:] != -100).sum()
        inputs["return_token_accuracy"] = True
        loss, outputs = super().compute_loss(model, inputs, True, num_items_in_batch)
        accuracy = getattr(outputs, "token_accuracy", None)
        if accuracy is None:
            raise RuntimeError("Liger did not return masked token accuracy")
        counts = torch.stack(
            [
                examples.double(),
                targets.double(),
                accuracy.detach().double() * targets,
                loss.detach().double(),
                torch.ones((), device=targets.device, dtype=torch.float64),
            ]
        )
        if self.repetition_totals is None:
            self.repetition_totals = torch.zeros_like(counts)
        self.repetition_totals += counts.detach()
        return (loss, outputs) if return_outputs else loss

    def repetition_metrics(self):
        totals = (
            self.repetition_totals.clone()
            if self.repetition_totals is not None
            else torch.zeros(5, dtype=torch.float64, device=self.args.device)
        )
        if dist.is_initialized():
            dist.all_reduce(totals)
        totals += self.repetition_base.to(totals.device)
        examples, tokens, correct, loss_sum, microbatches = totals.tolist()
        return {
            "example_presentations": int(examples),
            "supervised_tokens": int(tokens),
            "correct_assistant_tokens": correct,
            "masked_assistant_token_accuracy": correct / tokens if tokens else None,
            "optimization_loss_sum": loss_sum,
            "microbatches": int(microbatches),
            "optimizer_steps": self.state.global_step,
        }

    def log(self, logs, start_time=None):
        metrics = self.repetition_metrics()
        if "train_loss" in logs and metrics["microbatches"]:
            # Trainer's final train_loss uses a session loss sum divided by the
            # cumulative step counter on resume. Our persisted sum covers all sessions.
            logs = {
                **logs,
                "train_loss": metrics["optimization_loss_sum"]
                / metrics["microbatches"],
            }
        super().log({**logs, **metrics}, start_time)


class SaveRepetitionEpochs(TrainerCallback):
    def __init__(self, steps_per_epoch, epochs, stop_after_epoch, manifest_path=None):
        self.steps_per_epoch, self.epochs, self.stop_after_epoch = (
            steps_per_epoch,
            epochs,
            stop_after_epoch,
        )
        self.manifest_path = manifest_path
        self.trainer = None

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.steps_per_epoch == 0:
            control.should_log = True
        if state.global_step in {self.steps_per_epoch * epoch for epoch in self.epochs}:
            control.should_save = True
        if state.global_step >= self.steps_per_epoch * self.stop_after_epoch:
            control.should_save = True
            control.should_training_stop = True
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if (
            not logs
            or "loss" not in logs
            or not state.global_step
            or state.global_step % self.steps_per_epoch
        ):
            return control
        path = Path(args.output_dir) / "epoch_metrics.json"
        if state.is_world_process_zero:
            history = json.loads(path.read_text()) if path.exists() else []
            epoch = state.global_step // self.steps_per_epoch
            # Use cumulative counters to retain correct totals across resumption.
            previous = next(
                (row for row in reversed(history) if row["data_epoch"] == epoch - 1),
                None,
            )
            row = {
                **logs,
                "data_epoch": epoch,
                "effective_optimizer_steps": self.steps_per_epoch,
                "learning_rate": self.trainer.lr_scheduler.get_last_lr()[0],
            }
            for key in (
                "example_presentations",
                "supervised_tokens",
                "correct_assistant_tokens",
                "optimization_loss_sum",
                "microbatches",
            ):
                row[f"epoch_{key}"] = logs[key] - (previous[key] if previous else 0)
            row["epoch_masked_assistant_token_accuracy"] = (
                row["epoch_correct_assistant_tokens"] / row["epoch_supervised_tokens"]
            )
            row["epoch_loss"] = (
                row["epoch_optimization_loss_sum"] / row["epoch_microbatches"]
            )
            if row["epoch_example_presentations"] != len(
                self.trainer.train_dataset.examples
            ):
                raise RuntimeError(
                    "Actual epoch example presentations differ from selected count"
                )
            expected = sum(
                sum(label != -100 for label in example["labels"])
                for example in self.trainer.train_dataset.examples
            )
            if row["epoch_supervised_tokens"] != expected:
                raise RuntimeError(
                    "Actual epoch supervised-token count differs from prepared count"
                )
            history = [
                record for record in history if record["data_epoch"] != epoch
            ] + [row]
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(history, indent=2))
            os.replace(temporary, path)
        return control

    def on_save(self, args, state, control, **kwargs):
        metrics = self.trainer.repetition_metrics()
        metrics["data_epoch"] = state.global_step / self.steps_per_epoch
        metrics["latest_logged_loss"] = next(
            (row["loss"] for row in reversed(state.log_history) if "loss" in row), None
        )
        if state.is_world_process_zero:
            path = (
                Path(args.output_dir)
                / f"checkpoint-{state.global_step}"
                / "repetition_metrics.json"
            )
            path.write_text(json.dumps(metrics, indent=2))
            # Sequential sampling needs no random sampler generator. Persist the
            # exact next cursor; Trainer restores RNG and skips consumed batches.
            manifest_path = (
                Path(self.manifest_path)
                if self.manifest_path
                else Path(args.output_dir) / "prepared/manifest.json"
            )
            manifest = json.loads(manifest_path.read_text())
            epoch, offset = divmod(state.global_step, self.steps_per_epoch)
            sampler = {
                "sampler": "SequentialSampler",
                "next_epoch": epoch,
                "next_pack_offset": offset * manifest["global_batch_size"],
                "epoch_plan_sha256": manifest["epoch_plan_sha256"],
            }
            (path.parent / "repetition_sampler.json").write_text(
                json.dumps(sampler, indent=2)
            )
            epoch_history = Path(args.output_dir) / "epoch_metrics.json"
            if epoch_history.exists():
                (path.parent / "epoch_metrics.json").write_bytes(
                    epoch_history.read_bytes()
                )
        return control
