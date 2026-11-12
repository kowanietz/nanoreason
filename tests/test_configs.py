"""Hydra composition contracts shared by entrypoints and preparation scripts."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_sft_repetition import load_config

from benchmark_exclusions import DEFAULT_EXCLUSIONS

CONFIGS = ROOT / "configs"


class ConfigCompositionTest(unittest.TestCase):
    def test_storage_repository_and_olmes_paths_are_configurable(self):
        with (
            patch.dict(
                os.environ,
                {
                    "NANOREASON_ROOT": "/tmp/nanoreason-storage",
                    "NANOREASON_REPO": str(ROOT),
                    "OLMES_PROJECT": "/tmp/olmes-checkout",
                },
            ),
            initialize_config_dir(version_base="1.3", config_dir=str(CONFIGS)),
        ):
            sft = compose(config_name="sft/repetition-3k2")
            rlvr = compose(config_name="rlvr/mixed-512-16k")
            evaluation = compose(config_name="eval/math500")
            self.assertTrue(
                sft.data.source_path.startswith("/tmp/nanoreason-storage/datasets/")
            )
            self.assertTrue(
                sft.training.output_dir.startswith(
                    "/tmp/nanoreason-storage/checkpoints/"
                )
            )
            self.assertEqual(rlvr.data.directory, str(ROOT / ".cache/rlvr-prepared-v4"))
            self.assertEqual(Path(rlvr.data.exclusions), DEFAULT_EXCLUSIONS)
            self.assertEqual(evaluation.olmes_project, "/tmp/olmes-checkout")

    def test_training_presets_stay_global_and_share_hydra_policy(self):
        presets = sorted(CONFIGS.glob("sft/*.yaml")) + sorted(
            CONFIGS.glob("rlvr/*.yaml")
        )
        for path in presets:
            name = path.relative_to(CONFIGS).with_suffix("").as_posix()
            with (
                self.subTest(name=name),
                initialize_config_dir(version_base="1.3", config_dir=str(CONFIGS)),
            ):
                config = compose(config_name=name, return_hydra_config=True)
                self.assertIn("model", config)
                self.assertIn("data", config)
                self.assertEqual(config.hydra.run.dir, ".")
                self.assertIsNone(config.hydra.output_subdir)
                self.assertIs(config.hydra.job.chdir, False)
                if name.startswith("rlvr/"):
                    self.assertEqual(Path(config.data.exclusions), DEFAULT_EXCLUSIONS)

    def test_evaluation_presets_preserve_task_shapes(self):
        expected = {
            "aime": {
                "2024": "aime:2024::olmo3:adapt",
                "2025": "aime:2025::olmo3:adapt",
            },
            "gpqa": ["gpqa::olmo3:adapt"],
            "math500": ["minerva_math_500::hamish_zs_reasoning"],
            "ifeval": ["ifeval::olmo3:adapt"],
            "bbh": ["bbh:cot::olmo3:adapt"],
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
