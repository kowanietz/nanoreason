"""Hydra composition contracts shared by entrypoints and preparation scripts."""

import sys
import tempfile
import unittest
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_sft_repetition import load_config

CONFIGS = ROOT / "configs"


class ConfigCompositionTest(unittest.TestCase):
    def test_training_presets_stay_global_and_share_hydra_policy(self):
        for name in (
            "sft/base",
            "sft/packed-6k",
            "sft/repetition-3k2",
            "rlvr/smoke",
            "rlvr/smoke-16k",
        ):
            with (
                self.subTest(name=name),
                initialize_config_dir(version_base="1.3", config_dir=str(CONFIGS)),
            ):
                config = compose(config_name=name, return_hydra_config=True)
                self.assertIn("model", config)
                self.assertIn("data", config)
                self.assertNotIn("sft", config)
                self.assertNotIn("rlvr", config)
                self.assertEqual(config.hydra.run.dir, ".")
                self.assertIsNone(config.hydra.output_subdir)
                self.assertIs(config.hydra.job.chdir, False)
                self.assertEqual(
                    config.hydra.runtime.choices["hydra/job_logging"], "none"
                )
                self.assertEqual(
                    config.hydra.runtime.choices["hydra/hydra_logging"], "none"
                )

    def test_leaf_and_cli_overrides_take_precedence(self):
        with initialize_config_dir(version_base="1.3", config_dir=str(CONFIGS)):
            config = compose(
                config_name="rlvr/smoke-16k",
                overrides=[
                    "grpo.learning_rate=0.000003",
                    "grpo.output_dir=/tmp/config-test-only",
                    "runtime.stop_after_step=1",
                ],
            )
        self.assertEqual(config.grpo.learning_rate, 3e-6)
        self.assertEqual(config.grpo.output_dir, "/tmp/config-test-only")
        self.assertEqual(config.grpo.max_steps, 1)
        self.assertEqual(config.grpo.max_completion_length, 16384)
        self.assertEqual(config.grpo.vllm_max_model_length, 16896)
        self.assertEqual(config.data.max_examples, 1)
        self.assertIs(config.runtime.resume_from_checkpoint, False)

    def test_evaluation_presets_preserve_task_shapes(self):
        expected = {
            "aime": {
                "2024": "aime:2024::olmo3:adapt",
                "2025": "aime:2025::olmo3:adapt",
            },
            "gpqa": ["gpqa::olmo3:adapt"],
            "math500": ["minerva_math_500::hamish_zs_reasoning"],
        }
        for name, tasks in expected.items():
            with (
                self.subTest(name=name),
                initialize_config_dir(version_base="1.3", config_dir=str(CONFIGS)),
            ):
                config = compose(config_name=f"eval/{name}")
                self.assertNotIn("eval", config)
                self.assertEqual(
                    OmegaConf.to_container(config.tasks, resolve=True), tasks
                )
                self.assertEqual(config.model_args.dtype, "bfloat16")
                self.assertEqual(config.model_args.max_length, 32768)
                self.assertEqual(config.gpus, 4)
                self.assertEqual("task_args" in config, name != "aime")

    def test_preparation_loads_nested_and_standalone_configs(self):
        with initialize_config_dir(version_base="1.3", config_dir=str(CONFIGS)):
            expected = OmegaConf.to_container(
                compose(config_name="sft/repetition-3k2"), resolve=True
            )
        self.assertEqual(load_config(CONFIGS / "sft/repetition-3k2.yaml"), expected)
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            # Standalone custom config files retain their own local search root.
            path = Path(directory)
            (path / "base.yaml").write_text(
                "model:\n  name: custom\ntraining:\n  seed: 73\n"
            )
            preset = path / "custom.yaml"
            preset.write_text(
                "defaults:\n  - base\n  - _self_\ntraining:\n  seed: 74\n"
            )
            self.assertEqual(
                load_config(preset),
                {"model": {"name": "custom"}, "training": {"seed": 74}},
            )


if __name__ == "__main__":
    unittest.main()
