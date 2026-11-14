"""CPU regressions for lossless repetition data and SFT checkpoint contracts."""

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from transformers import TrainerControl, TrainerState

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_sft_repetition import benchmark_questions

from sft.checkpoints import (
    configure_repetition,
    guard_repetition_destination,
    resolve_checkpoint,
    verify_manifest,
)
from sft.data import (
    RepetitionCollator,
    RepetitionDataset,
    repetition_plan,
    tokenize_sft_example,
)
from sft.trainer import SaveRepetitionEpochs
from training_common import make_tokenizer


def repetition_config():
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        return OmegaConf.to_container(
            compose(config_name="sft/repetition-3k2"), resolve=True
        )


class BenchmarkQuestionsTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(temporary.cleanup)
        directory = Path(temporary.name)
        self.checkout = directory / "checkout"
        self.private = directory / "private"
        self.private.mkdir()
        self.gpqa = self.private / "task-000-gpqa-requests.jsonl"
        self.gpqa.write_text(
            json.dumps(
                {"doc": {"question": "  Synthetic QUESTION\nwith extra whitespace.  "}}
            )
            + "\n"
        )
        self.digest = hashlib.sha256(self.gpqa.read_bytes()).hexdigest()
        for benchmark in ("math500", "aime-2024", "aime-2025"):
            path = self.checkout / "results/Olmo-3-7B" / benchmark
            path.mkdir(parents=True)
            (path / "task-000-synthetic-requests.jsonl").write_text(
                json.dumps({"doc": {"problem": f"Synthetic {benchmark} problem."}})
                + "\n"
            )
        for name, value in (
            ("ROOT", self.checkout),
            ("GPQA_REQUESTS_SHA256", self.digest),
        ):
            patched = patch(f"prepare_sft_repetition.{name}", value)
            patched.start()
            self.addCleanup(patched.stop)

    def test_private_questions_preserve_text_normalization_and_provenance(self):
        questions, provenance = benchmark_questions(self.private)
        self.assertEqual(
            questions,
            {
                "synthetic question with extra whitespace.",
                "synthetic math500 problem.",
                "synthetic aime-2024 problem.",
                "synthetic aime-2025 problem.",
            },
        )
        self.assertEqual(len(provenance), 4)
        self.assertIn({"path": str(self.gpqa), "sha256": self.digest}, provenance)
        for source in provenance:
            self.assertEqual(
                source["sha256"],
                hashlib.sha256(Path(source["path"]).read_bytes()).hexdigest(),
            )

    def test_missing_or_changed_private_inputs_fail_closed(self):
        self.gpqa.unlink()
        with self.assertRaises(FileNotFoundError):
            benchmark_questions(self.private)
        self.gpqa.write_text('{"doc": {"question": "Different synthetic question"}}\n')
        with self.assertRaisesRegex(ValueError, "original baseline SHA-256"):
            benchmark_questions(self.private)

    def test_private_directory_must_be_external(self):
        with self.assertRaisesRegex(ValueError, "outside the repository"):
            benchmark_questions(self.checkout / "private")

    def test_missing_public_benchmark_cannot_silently_reduce_exclusions(self):
        path = self.checkout / "results/Olmo-3-7B/math500"
        next(path.iterdir()).unlink()
        with self.assertRaisesRegex(FileNotFoundError, "math500"):
            benchmark_questions(self.private)


class RepetitionPackingTest(unittest.TestCase):
    def test_epochs_preserve_examples_without_sampler_padding(self):
        lengths = [3, 4, 2, 5, 2, 6, 1, 7]
        plans = repetition_plan(lengths, 10, 4, 73, 2)
        self.assertEqual(plans, repetition_plan(lengths, 10, 4, 73, 2))
        self.assertEqual(len({len(plan) for plan in plans}), 1)
        for plan in plans:
            self.assertEqual(len(plan) % 2, 0)
            self.assertEqual(
                sorted(index for pack in plan for index in pack),
                list(range(len(lengths))),
            )
            self.assertTrue(
                all(sum(lengths[index] for index in pack) <= 10 for pack in plan)
            )
        for invalid in ([], [0], [11]):
            with self.subTest(lengths=invalid), self.assertRaises(ValueError):
                repetition_plan(invalid, 10, 4, 73, 2)

    def test_manifest_derives_steps_and_rejects_changed_inputs(self):
        config = repetition_config()
        training, data = config["training"], config["data"]
        batch = (
            4
            * training["per_device_batch_size"]
            * training["gradient_accumulation_steps"]
        )
        manifest = {
            "seed": training["seed"],
            "epochs": training["num_train_epochs"],
            "max_length": data["max_length"],
            "global_batch_size": batch,
            "source_checkpoint": training["initialization_checkpoint"],
            "unique_examples": data["unique_examples"],
            "selection_seed": data["selection_seed"],
            "holdout_examples": data["holdout_examples"],
            "packs_per_epoch": 3 * batch,
        }
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(manifest))
            config["data"]["repetition_manifest"] = str(path)
            configure_repetition(config, 4)
            self.assertEqual(config["training"]["steps_per_epoch"], 3)
            self.assertEqual(config["training"]["max_steps"], 3 * manifest["epochs"])
            for changed in (
                {**manifest, "seed": -1},
                {**manifest, "packs_per_epoch": batch + 1},
            ):
                path.write_text(json.dumps(changed))
                with self.subTest(manifest=changed), self.assertRaises(ValueError):
                    configure_repetition(copy.deepcopy(config), 4)


class SFTCheckpointTest(unittest.TestCase):
    def test_resume_falls_back_to_complete_checkpoint_but_cannot_rewind(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            output = Path(directory)
            self.assertIsNone(resolve_checkpoint(output, "auto"))
            for step in (1, 2):
                checkpoint = output / f"checkpoint-{step}"
                checkpoint.mkdir()
                (checkpoint / "trainer_state.json").write_text(
                    json.dumps({"global_step": step})
                )
            incomplete = output / "checkpoint-2/checkpoint-is-incomplete.txt"
            incomplete.touch()
            self.assertEqual(
                resolve_checkpoint(output, "auto"), str(output / "checkpoint-1")
            )
            incomplete.unlink()
            self.assertEqual(
                resolve_checkpoint(output, "auto"), str(output / "checkpoint-2")
            )
            with self.assertRaisesRegex(RuntimeError, "rewind"):
                resolve_checkpoint(
                    output, str(output / "checkpoint-1"), reject_older=True
                )
            with self.assertRaisesRegex(RuntimeError, "enable resumption"):
                resolve_checkpoint(output, False)

    def test_manifest_rejects_changed_config_and_unmanifested_destination(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            output = Path(directory)
            config = repetition_config()
            config["training"]["output_dir"] = str(output)
            source = output / "examples.parquet"
            source.write_bytes(b"manifest fixture")
            with self.assertRaisesRegex(RuntimeError, "Unmanifested"):
                guard_repetition_destination(output)
            # Omit the repetition artifact to test the generic SFT provenance contract.
            config["data"].pop("repetition_manifest")
            verify_manifest(config, [str(source)], False, 0)
            verify_manifest(config, [str(source)], True, 0)
            config["training"]["seed"] += 1
            with self.assertRaisesRegex(RuntimeError, "configuration changed"):
                verify_manifest(config, [str(source)], True, 0)

    def test_epoch_checkpoint_persists_exact_sampler_cursor(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            output = Path(directory)
            checkpoint = output / "checkpoint-3"
            checkpoint.mkdir()
            manifest = output / "manifest.json"
            manifest.write_text(
                json.dumps({"global_batch_size": 4, "epoch_plan_sha256": "fixed-plan"})
            )
            callback = SaveRepetitionEpochs(2, [2, 4], 4, str(manifest))
            callback.trainer = SimpleNamespace(
                repetition_metrics=lambda: {"supervised_tokens": 12}
            )
            state = TrainerState(
                global_step=3, is_world_process_zero=True, log_history=[{"loss": 0.25}]
            )
            callback.on_save(
                SimpleNamespace(output_dir=str(output)), state, TrainerControl()
            )
            self.assertEqual(
                json.loads((checkpoint / "repetition_sampler.json").read_text()),
                {
                    "sampler": "SequentialSampler",
                    "next_epoch": 1,
                    "next_pack_offset": 4,
                    "epoch_plan_sha256": "fixed-plan",
                },
            )
            control = callback.on_step_end(
                None, TrainerState(global_step=8), TrainerControl()
            )
            self.assertTrue(control.should_save and control.should_training_stop)


class SFTTokenizationTest(unittest.TestCase):
    def test_tokenization_and_collation_preserve_assistant_eos_and_padding(self):
        config = repetition_config()
        if not Path(config["model"]["tokenizer_name_or_path"]).is_dir():
            self.skipTest("Local OLMo tokenizer is unavailable")
        tokenizer = make_tokenizer(config["model"])
        encoded = []
        for answer in ("42", "43", "44"):
            example = {
                "messages": [
                    {"role": "user", "content": "Compute the answer."},
                    {"role": "assistant", "content": f"<think>work</think>{answer}"},
                ]
            }
            row = tokenize_sft_example(example, tokenizer, 256)
            self.assertTrue(row.pop("keep"))
            self.assertEqual(row["labels"][0], -100)
            self.assertEqual(row["input_ids"][-1], tokenizer.eos_token_id)
            self.assertEqual(row["labels"][-1], tokenizer.eos_token_id)
            self.assertFalse(tokenize_sft_example(example, tokenizer, 4)["keep"])
            encoded.append(row)
        lengths = [len(row["input_ids"]) for row in encoded]
        max_length = sum(lengths) + 8
        plans = repetition_plan(lengths, max_length, 2, 73, 1)
        dataset = RepetitionDataset(encoded, plans, max_length, tokenizer.pad_token_id)
        collator = RepetitionCollator(tokenizer)
        for epoch in (0, 1):
            dataset.set_epoch(epoch)
            row = dataset[0]
            expected = [
                label for index in plans[epoch][0] for label in encoded[index]["labels"]
            ]
            self.assertEqual(row["labels"][: len(expected)], expected)
            batch = collator([row])
            self.assertEqual(batch["example_count"].item(), len(encoded))
            self.assertTrue(
                (batch["labels"][batch["attention_mask"] == 0] == -100).all()
            )
            self.assertEqual(
                int((batch["labels"] != -100).sum()),
                sum(label != -100 for label in expected),
            )


if __name__ == "__main__":
    unittest.main()
