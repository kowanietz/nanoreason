"""Small offline fixtures for model export safety and static validation."""

import copy
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import Olmo3Config, Olmo3ForCausalLM, PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import export_model as export_cli
import validate_model as validate_cli
from prepare_vllm011 import compatible_config, prepare

from model_artifacts import export_model, fingerprint, validate_model


def tiny_export(path, dtype=torch.float32):
    """Real OLMo3 tensors and a usable local chat tokenizer, all only a few KiB."""
    backend = Tokenizer(
        WordLevel({"<unk>": 0, "<eos>": 1, "<pad>": 2, "hello": 3}, unk_token="<unk>")
    )
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        eos_token="<eos>",
        pad_token="<pad>",
    )
    tokenizer.chat_template = "{% for m in messages %}{{ m['content'] + ' ' }}{% endfor %}{% if add_generation_prompt %}hello{% endif %}"
    config = Olmo3Config(
        vocab_size=4,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
        sliding_window=8,
        layer_types=["sliding_attention", "full_attention"],
        eos_token_id=1,
        pad_token_id=2,
        bos_token_id=None,
        use_cache=False,
        rope_parameters={
            "full_attention": {
                "rope_type": "yarn",
                "rope_theta": 500000,
                "factor": 2.0,
                "original_max_position_embeddings": 16,
            },
            "sliding_attention": {"rope_type": "default", "rope_theta": 500000},
        },
    )
    Olmo3ForCausalLM(config).to(dtype=dtype).save_pretrained(path)
    tokenizer.save_pretrained(path)
    (path / "LICENSE").write_text("fixture license\n")
    return path


class ModelArtifactsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = tiny_export(self.root / "source")
        self.destination = self.root / "release"

    def test_export_preserves_exact_files_and_sources(self):
        for precision, dtype in (("FP32", torch.float32), ("BF16", torch.bfloat16)):
            with self.subTest(precision=precision):
                source = tiny_export(self.root / precision, dtype=dtype)
                destination = self.root / f"{precision}-export"
                before = {p.name: fingerprint(p) for p in source.iterdir()}
                report = export_model(source, destination)
                self.assertEqual(report["dtype"], precision)
                self.assertFalse(report["use_cache"])
                for name, digest in before.items():
                    original, copied = source / name, destination / name
                    self.assertEqual(fingerprint(original), digest)
                    self.assertEqual(fingerprint(copied), digest)
                    self.assertNotEqual(original.stat().st_ino, copied.stat().st_ino)
                    self.assertEqual(copied.stat().st_nlink, 1)
                    self.assertFalse(copied.is_symlink())
                validate_model(destination, expected_sha256=before["model.safetensors"])
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    validate_model(destination, expected_sha256="0" * 64)

    def test_inference_source_links_and_tokenizer_donor(self):
        inference = self.root / "inference"
        inference.mkdir()
        for name in ("config.json", "generation_config.json", "model.safetensors"):
            (inference / name).symlink_to(self.source / name)
        export_model(inference, self.destination, tokenizer=self.source)
        self.assertFalse(any(p.is_symlink() for p in self.destination.iterdir()))
        for p in self.source.iterdir():
            if p.name != "LICENSE":  # The inference source only supplies model files.
                self.assertEqual(
                    p.read_bytes(), (self.destination / p.name).read_bytes()
                )
        with self.assertRaisesRegex(ValueError, "overlaps"):
            export_model(inference, self.source / "nested", tokenizer=self.source)

    def test_destinations_and_resolved_source_targets_are_protected(self):
        for destination in (self.source, self.source / "nested", self.root):
            with (
                self.subTest(destination=destination),
                self.assertRaises((ValueError, FileExistsError)),
            ):
                export_model(self.source, destination)
        self.destination.mkdir()
        marker = self.destination / "marker"
        marker.write_text("keep")
        with self.assertRaises(FileExistsError):
            export_model(self.source, self.destination)
        self.assertEqual(marker.read_text(), "keep")
        alias = self.root / "alias"
        alias.symlink_to(self.source, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "overlaps"):
            export_model(self.source, alias / "nested")
        broken = self.root / "broken"
        broken.symlink_to(self.root / "absent")
        with self.assertRaises(FileExistsError):
            export_model(self.source, broken)
        external = self.root / "external"
        external.mkdir()
        weight = self.source / "model.safetensors"
        weight.rename(external / weight.name)
        weight.symlink_to(external / weight.name)
        with self.assertRaisesRegex(ValueError, "overlaps"):
            export_model(self.source, external / "nested")

    def test_validation_failure_retains_staging_without_finalizing(self):
        digest = fingerprint(self.source / "model.safetensors")
        with (
            patch(
                "model_artifacts.validate_model", side_effect=ValueError("bad model")
            ),
            self.assertRaisesRegex(ValueError, "staging retained"),
        ):
            export_model(self.source, self.destination)
        self.assertFalse(self.destination.exists())
        staging = list(self.root.glob(".release.*.staging"))
        self.assertEqual(len(staging), 1)
        self.assertEqual(fingerprint(staging[0] / "model.safetensors"), digest)
        self.assertEqual(fingerprint(self.source / "model.safetensors"), digest)

    def test_promotion_never_replaces_a_racing_destination(self):
        run = subprocess.run

        def race(command, **kwargs):
            self.destination.mkdir()
            return run(command, **kwargs)

        with (
            patch("model_artifacts.subprocess.run", side_effect=race),
            self.assertRaisesRegex(ValueError, "appeared during promotion"),
        ):
            export_model(self.source, self.destination)
        self.assertEqual(list(self.destination.iterdir()), [])
        self.assertEqual(len(list(self.root.glob(".release.*.staging"))), 1)

    def test_config_rope_and_special_token_failures(self):
        path = self.source / "config.json"
        original = json.loads(path.read_text())
        changes = [
            {"dtype": "bfloat16"},
            {"rope_theta": 1},
            {"num_hidden_layers": 3},
            {"num_attention_heads": 3},
            {"sliding_window": 0},
            {"eos_token_id": 0},
            {"rope_scaling": {"rope_type": "default"}},
        ]
        invalid_rope = copy.deepcopy(original["rope_parameters"])
        invalid_rope["full_attention"]["factor"] = -1
        changes.append({"rope_parameters": invalid_rope})
        for change in changes:
            path.write_text(json.dumps({**original, **change}))
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_model(self.source)

    def test_missing_files_json_and_chat_template(self):
        path = self.source / "generation_config.json"
        original = path.read_bytes()
        path.unlink()
        with self.assertRaisesRegex(ValueError, "Missing file"):
            validate_model(self.source)
        path.write_text("{")
        with self.assertRaises(json.JSONDecodeError):
            validate_model(self.source)
        path.write_text('{"eos_token_id": 0}')
        with self.assertRaisesRegex(ValueError, "Generation config"):
            validate_model(self.source)
        path.write_bytes(original)
        config = self.source / "tokenizer_config.json"
        doc = json.loads(config.read_text())
        doc["chat_template"] = (self.source / "chat_template.jinja").read_text()
        config.write_text(json.dumps(doc))
        (self.source / "chat_template.jinja").unlink()
        validate_model(self.source)  # Embedded templates are sufficient.
        doc["chat_template"] = "{{"
        config.write_text(json.dumps(doc))
        with self.assertRaisesRegex(ValueError, "Invalid chat template"):
            validate_model(self.source)

    def test_tensor_integrity_precision_and_architecture(self):
        path = self.source / "model.safetensors"
        original = path.read_bytes()
        path.write_bytes(original[:-16])
        with self.assertRaisesRegex(ValueError, "Invalid safetensors"):
            validate_model(self.source)
        path.write_bytes(original)
        tensors = load_file(path)
        key = next(iter(tensors))
        variants = [
            dict(tensors, unexpected=torch.zeros(1)),
            {k: v for k, v in tensors.items() if k != key},
            {**tensors, key: torch.zeros(1)},
            {**tensors, key: tensors[key].bfloat16()},
        ]
        for changed in variants:
            save_file(changed, path)
            with self.assertRaisesRegex(ValueError, "FP32|names/shapes"):
                validate_model(self.source)

    def test_standalone_links_and_shards_are_rejected(self):
        link = self.source / "unexpected"
        link.symlink_to(self.root / "missing")
        with self.assertRaisesRegex(ValueError, "link"):
            validate_model(self.source)
        link.unlink()
        os.link(self.source / "LICENSE", link)
        with self.assertRaisesRegex(ValueError, "link"):
            validate_model(self.source)
        link.unlink()
        (self.source / "model.safetensors.index.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "single merged"):
            validate_model(self.source)

    def test_legacy_preparation_is_separate_and_non_mutating(self):
        raw = json.loads((self.source / "config.json").read_text())
        original = copy.deepcopy(raw)
        converted = compatible_config(raw)
        self.assertEqual(raw, original)
        self.assertTrue(converted["use_cache"])
        self.assertNotIn("rope_theta", raw)
        output = self.root / "legacy"
        with patch("prepare_vllm011.validate_vllm"):
            prepare(self.source, self.source, output, self.root / "vllm")
            with self.assertRaises(FileExistsError):
                prepare(self.source, self.source, output, self.root / "vllm")
        self.assertTrue((output / "model.safetensors").is_symlink())
        self.assertEqual(json.loads((output / "config.json").read_text()), converted)
        self.assertFalse(validate_model(self.source)["use_cache"])

    def test_cli_export_validation_and_clear_failure(self):
        for module, arguments in (
            (
                export_cli,
                ["--source", str(self.source), "--destination", str(self.destination)],
            ),
            (validate_cli, ["--model", str(self.destination)]),
        ):
            with (
                patch.object(sys, "argv", ["tool", *arguments]),
                redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(module.main(), 0)
        error = io.StringIO()
        with (
            patch.object(sys, "argv", ["tool", "--model", str(self.root / "absent")]),
            redirect_stderr(error),
        ):
            self.assertEqual(validate_cli.main(), 1)
        self.assertIn("Validation FAILED", error.getvalue())

    def test_bf16_dtype_declarations_must_match_weights(self):
        source = tiny_export(self.root / "bf16", dtype=torch.bfloat16)
        config_path = source / "config.json"
        config = json.loads(config_path.read_text())
        config["dtype"] = "float32"
        config_path.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "Config dtype"):
            validate_model(source)
        config.update(dtype="bfloat16", torch_dtype="float32")
        config_path.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "Config dtype"):
            validate_model(source)
        config["torch_dtype"] = "bfloat16"
        config_path.write_text(json.dumps(config))
        self.assertEqual(validate_model(source)["dtype"], "BF16")

    def test_mixed_or_unsupported_weight_dtypes_are_rejected(self):
        path = self.source / "model.safetensors"
        original = load_file(path)
        mixed = dict(original)
        name = next(iter(mixed))
        mixed[name] = mixed[name].bfloat16()
        for tensors in (
            mixed,
            {name: tensor.half() for name, tensor in original.items()},
        ):
            save_file(tensors, path)
            with self.assertRaisesRegex(ValueError, "uniform FP32 or BF16"):
                validate_model(self.source)


if __name__ == "__main__":
    unittest.main()
