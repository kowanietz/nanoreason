import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, PropertyMock, patch

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from hydra import compose, initialize_config_dir
from hydra._internal.utils import get_args_parser
from omegaconf import OmegaConf
from transformers import Olmo3Config, Olmo3ForCausalLM, TrainerControl, TrainerState
from transformers.models.olmo3.modeling_olmo3 import Olmo3Model
from trl import GRPOConfig, GRPOTrainer
from trl.trainer.utils import add_fused_lm_head

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from benchmark_exclusions import load_exclusions
from rlvr.checkpoints import resume_checkpoint
from rlvr.data import load_smoke_data
from rlvr.diagnostics import SmokeDiagnostics, down_proj_sample
from rlvr.rewards import math_rewards
from rlvr.validation import validate_config
from train_rlvr import TrainingOlmo3Model
from training_common import make_tokenizer


def smoke_config(name="rlvr/smoke"):
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        return OmegaConf.to_container(compose(config_name=name), resolve=True)


class TrainingDecoderTest(unittest.TestCase):
    def test_training_hooks_and_gradients_survive_vllm_class_mutation(self):
        config = Olmo3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=32,
            pad_token_id=0,
            eos_token_id=2,
            use_cache=False,
            layer_types=["full_attention"],
        )
        model = Olmo3ForCausalLM(config)
        model.model.__class__ = TrainingOlmo3Model
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        add_fused_lm_head(model)
        tokens = torch.tensor([[1, 3, 4, 5]])
        events = []
        model.model.register_forward_pre_hook(lambda *args: events.append("pre"))
        model.model.register_forward_hook(lambda *args: events.append("post"))
        # vLLM patches the shared decoder class; the training subclass must retain PyTorch hooks.
        with patch.object(
            Olmo3Model, "__call__", side_effect=AssertionError("vLLM-only call")
        ):
            result = model(input_ids=tokens, labels=tokens, fused_lm_head=True)
            self.assertTrue(torch.isfinite(result.log_probs).all())
            result.loss.backward()
        self.assertEqual(events, ["pre", "post"])
        self.assertTrue(
            all(
                p.grad is not None and torch.isfinite(p.grad).all()
                for p in model.parameters()
            )
        )
        self.assertGreater(model.model.embed_tokens.weight.grad.abs().max().item(), 0)


class SmokeConfigTest(unittest.TestCase):
    def test_invalid_settings_fail(self):
        for key, value in (
            ("num_generations", 0),
            ("num_generations", 3),
            ("generation_batch_size", 6),
            ("gradient_accumulation_steps", 2),
            ("vllm_tensor_parallel_size", 3),
            ("vllm_max_model_length", 1024),
            ("vllm_gpu_memory_utilization", 0),
            ("max_completion_length", 0),
            ("max_steps", 100),
            ("beta", 0.01),
            ("bf16", False),
            ("remove_unused_columns", True),
            ("save_only_model", True),
        ):
            config = smoke_config()
            config["grpo"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_config(config)

    def test_output_isolated_from_source(self):
        config = smoke_config()
        config["grpo"]["output_dir"] = config["model"]["name_or_path"]
        with self.assertRaisesRegex(ValueError, "isolated"):
            validate_config(config)

    def test_native_checkpoint_completeness_and_no_rewind(self):
        callback = SmokeDiagnostics(4)
        for step, stopping in ((3, False), (4, True)):
            control = callback.on_step_end(
                None, TrainerState(global_step=step), TrainerControl()
            )
            self.assertEqual(control.should_save, stopping)
            self.assertEqual(control.should_training_stop, stopping)
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            output = Path(directory)
            self.assertIsNone(resume_checkpoint(output, "auto", 4))
            checkpoint = output / "checkpoint-4"
            checkpoint.mkdir()
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                resume_checkpoint(output, "auto", 4)
            files = [
                "trainer_state.json",
                "scheduler.pt",
                "optimizer_0/.metadata",
                "pytorch_model_fsdp_0/.metadata",
                *[f"rng_state_{rank}.pth" for rank in range(4)],
            ]
            for name in files:
                path = checkpoint / name
                path.parent.mkdir(exist_ok=True)
                path.touch()
            self.assertEqual(resume_checkpoint(output, "auto", 4), str(checkpoint))
            with self.assertRaisesRegex(ValueError, "require resumption"):
                resume_checkpoint(output, False, 4)
            (output / "checkpoint-8").mkdir()
            with self.assertRaises(ValueError):
                resume_checkpoint(output, str(checkpoint), 4)


class TokenizerAndRewardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = smoke_config()
        if not Path(cls.config["model"]["tokenizer_name_or_path"]).is_dir():
            raise unittest.SkipTest("Local OLMo tokenizer is unavailable")
        cls.tokenizer = make_tokenizer(cls.config["model"])

    def bare_trainer(self):
        trainer = object.__new__(GRPOTrainer)
        trainer.processing_class = self.tokenizer
        trainer._tokenizer = self.tokenizer
        trainer._is_vlm = False
        trainer.environment_factories = None
        trainer.tools = None
        trainer.chat_template = None
        trainer.chat_template_kwargs = {}
        trainer.state = TrainerState()
        trainer.args = SimpleNamespace(report_to=[])
        trainer.accelerator = SimpleNamespace(
            device=torch.device("cpu"), is_main_process=True
        )
        trainer.model = SimpleNamespace(training=True)
        trainer.reward_funcs = [math_rewards]
        trainer.reward_processing_classes = [None]
        trainer.reward_func_names = ["math_rewards"]
        trainer.environments = None
        trainer._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        trainer._pending_metrics = defaultdict(list)
        trainer._pending_extra_logs = {}
        return trainer

    def test_trl_prompt_ids_and_completion_prefix(self):
        prompt = [{"role": "user", "content": "What is six times seven?"}]
        trainer = self.bare_trainer()
        ids, images, fields = GRPOTrainer._tokenize_prompts(trainer, [prompt] * 4)
        expected = self.tokenizer.apply_chat_template(
            prompt, tokenize=True, add_generation_prompt=True, return_dict=False
        )
        self.assertEqual(ids, [expected] * 4)
        rendered = self.tokenizer.decode(ids[0], skip_special_tokens=False)
        self.assertTrue(rendered.endswith("<|im_start|>assistant\n<think>"))
        self.assertEqual(rendered.count("<think>"), 1)
        decoded = self.tokenizer.decode(
            self.tokenizer.encode("work</think>42", add_special_tokens=False),
            skip_special_tokens=True,
        )
        self.assertEqual(decoded, "work</think>42")
        self.assertIsNone(images)
        self.assertEqual(fields, {})

    def test_actual_trl_reward_dispatch_and_metadata_alignment(self):
        trainer = self.bare_trainer()
        prompt = [{"role": "user", "content": "What is six times seven?"}]
        inputs = [
            {
                "prompt": prompt,
                "ground_truth": ["42"],
                "reasoning_prefix": True,
                "domain": "math",
            }
        ] * 4
        completions = [
            [{"role": "assistant", "content": content}]
            for content in (
                r"work</think>Thus \boxed{42}.",
                r"work</think>Thus \boxed{43}.",
                "unfinished",
                "42</think>42",
            )
        ]
        rewards = GRPOTrainer._calculate_rewards(
            trainer, inputs, [prompt] * 4, completions, [[1]] * 4
        )
        self.assertEqual(rewards[:, 0].tolist(), [1.0, 0.0, 0.0, 1.0])
        inputs[1] = {**inputs[1], "ground_truth": ["43"]}
        aligned = GRPOTrainer._calculate_rewards(
            trainer, inputs, [prompt] * 4, completions, [[1]] * 4
        )
        self.assertEqual(aligned[:, 0].tolist(), [1.0, 1.0, 0.0, 1.0])
        invalid = copy.deepcopy(inputs)
        invalid[0]["ground_truth"] = ["X"]
        invalid[0]["prompt"] = [
            {"role": "user", "content": "Construct a point X on a circle."}
        ]
        with self.assertRaises(ValueError):
            GRPOTrainer._calculate_rewards(
                trainer,
                invalid,
                [row["prompt"] for row in invalid],
                completions,
                [[1]] * 4,
            )

    def test_production_importance_sampling_modes_at_16k(self):
        length = 16384
        prompt = [{"role": "user", "content": "What is six times seven?"}]
        inputs = [
            {
                "prompt": prompt,
                "ground_truth": ["42"],
                "reasoning_prefix": True,
                "domain": "math",
            }
        ] * 4
        completions = [
            [{"role": "assistant", "content": content}]
            for content in (
                "work</think>42",
                "work</think>43",
                "unfinished",
                "work</think>43",
            )
        ]
        for mode, difference in (
            ("sequence_mask", -0.0006),
            ("token_truncate", -0.0006),
            ("token_truncate", 2.0),
        ):
            with self.subTest(mode=mode, difference=difference):
                trainer = self.bare_trainer()
                trainer.args = SimpleNamespace(
                    per_device_train_batch_size=4,
                    steps_per_generation=1,
                    gradient_accumulation_steps=1,
                    gradient_checkpointing_kwargs={},
                    report_to=[],
                )
                trainer.accelerator.gather = lambda x: x
                trainer.accelerator.process_index = 0
                trainer.model.is_gradient_checkpointing = False
                for key, value in {
                    "pad_to_multiple_of": None,
                    "mask_truncated_completions": False,
                    "use_vllm": True,
                    "vllm_importance_sampling_correction": True,
                    "vllm_importance_sampling_mode": mode,
                    "vllm_importance_sampling_clip_min": None,
                    "vllm_importance_sampling_clip_max": 3.0,
                    "num_iterations": 1,
                    "num_generations": 4,
                    "beta": 0.0,
                    "multi_objective_aggregation": "sum_then_normalize",
                    "scale_rewards": "none",
                    "reward_weights": torch.ones(1),
                    "log_multimodal": False,
                }.items():
                    setattr(trainer, key, value)
                trainer._logs = {
                    "prompt": [],
                    "completion": [],
                    "rewards": {"math_rewards": []},
                    "advantages": [],
                    "extra": defaultdict(list),
                }
                trainer._generate = Mock(
                    return_value=(
                        [[1]] * 4,
                        [[1] * length] * 4,
                        None,
                        completions,
                        [[-1.0] * length] * 4,
                        {},
                        None,
                        None,
                    )
                )
                trainer._get_per_token_logps_and_entropies = Mock(
                    return_value=(
                        torch.full((4, length), -1.0 + difference),
                        None,
                        None,
                    )
                )
                result = GRPOTrainer._generate_and_score_completions(trainer, inputs)
                ratios = result["importance_sampling_ratio"]
                torch.testing.assert_close(
                    result["advantages"], torch.tensor([0.75, -0.25, -0.25, -0.25])
                )
                if mode == "sequence_mask":
                    self.assertEqual(tuple(ratios.shape), (4, 1))
                    self.assertLess(ratios.max().item(), 6e-5)
                    self.assertGreater(ratios.min().item(), 5e-5)
                else:
                    self.assertEqual(tuple(ratios.shape), (4, length))
                    expected = float(torch.exp(torch.tensor(difference)).clamp(max=3.0))
                    torch.testing.assert_close(
                        ratios, torch.full_like(ratios, expected)
                    )

    def test_dataset_checksum_filtering_lengths_and_reproducibility(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            config = copy.deepcopy(self.config)
            config["data"].update(
                directory=directory, max_examples=2, max_prompt_tokens=128
            )
            rows = []
            for i, content in enumerate(
                (
                    "What is six times seven?",
                    "What is 41+1?",
                    "What is 40+2?",
                    "word " * 1000,
                    "Write a sentence.",
                )
            ):
                rows.append(
                    {
                        "prompt": [{"role": "user", "content": content}],
                        "source_prompt": "user: " + content,
                        "ground_truth": ["42"],
                        "domain": "math" if i < 4 else "ifeval",
                        "reward_enabled": i < 4,
                        "reasoning_prefix": True,
                        "source_id": str(i),
                        "outputs": ["historical rollout must not be used"],
                    }
                )
            parquet = Path(directory) / "examples.parquet"
            pq.write_table(pa.Table.from_pylist(rows), parquet)
            _, exclusions = load_exclusions(Path(config["data"]["exclusions"]))
            manifest = {
                "output": {
                    "file": "examples.parquet",
                    "rows": 5,
                    "sha256": hashlib.sha256(parquet.read_bytes()).hexdigest(),
                },
                "statistics": {
                    "retained_by_domain": dict(Counter(r["domain"] for r in rows))
                },
                "tokenizer": {
                    "chat_template_sha256": hashlib.sha256(
                        self.tokenizer.chat_template.encode()
                    ).hexdigest()
                },
                "exclusions": exclusions,
            }
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(manifest))
            first, info = load_smoke_data(config, self.tokenizer)
            second, repeated_info = load_smoke_data(config, self.tokenizer)
            self.assertEqual(first.to_list(), second.to_list())
            self.assertEqual(info, repeated_info)
            self.assertNotIn("outputs", first.column_names)
            config["data"].update(max_examples=1, source_id=first[1]["source_id"])
            targeted, targeted_info = load_smoke_data(config, self.tokenizer)
            self.assertEqual(targeted.to_list(), [first[1]])
            self.assertEqual(
                targeted_info["selected_source_ids"], [first[1]["source_id"]]
            )
            config["data"]["source_id"] = "missing"
            with self.assertRaisesRegex(ValueError, "Insufficient"):
                load_smoke_data(config, self.tokenizer)
            self.assertEqual(info["excluded"]["prompt_over_budget"], 1)
            self.assertEqual(info["excluded"]["not_enabled_math"], 1)
            for section, key in (("output", "sha256"), ("exclusions", "sha256")):
                invalid = copy.deepcopy(manifest)
                invalid[section][key] = "0" * 64
                path.write_text(json.dumps(invalid))
                with (
                    self.subTest(section=section),
                    self.assertRaisesRegex(ValueError, "provenance"),
                ):
                    load_smoke_data(config, self.tokenizer)


class SingleStepSmokeTest(unittest.TestCase):
    def test_16k_config_preserves_training_setup(self):
        base, config = smoke_config(), smoke_config("rlvr/smoke-16k")
        validate_config(config)
        self.assertEqual(config["grpo"]["max_steps"], 1)
        self.assertEqual(config["grpo"]["max_completion_length"], 16384)
        self.assertEqual(config["grpo"]["vllm_max_model_length"], 16896)
        self.assertTrue(
            config["grpo"]["output_dir"].endswith("/dr-grpo-smoke-16k-token-is")
        )
        self.assertNotEqual(config["grpo"]["output_dir"], base["grpo"]["output_dir"])
        self.assertEqual(
            config["data"]["source_id"], "data/train-00002-of-00009.parquet:5184"
        )
        self.assertEqual(config["data"]["max_examples"], 1)
        self.assertIs(config["runtime"]["resume_from_checkpoint"], False)
        self.assertEqual(config["runtime"]["stop_after_step"], 1)
        self.assertTrue(config["grpo"]["vllm_importance_sampling_correction"])
        self.assertEqual(
            config["grpo"]["vllm_importance_sampling_mode"], "token_truncate"
        )
        self.assertEqual(config["grpo"]["vllm_importance_sampling_clip_max"], 3.0)
        with patch.object(
            GRPOConfig, "world_size", new_callable=PropertyMock, return_value=4
        ):
            args = GRPOConfig(**{**copy.deepcopy(config["grpo"]), "use_cpu": True})
        self.assertEqual(args.vllm_importance_sampling_mode, "token_truncate")
        self.assertIsNone(args.vllm_importance_sampling_clip_min)
        self.assertEqual(args.steps_per_generation, 1)
        self.assertEqual(args.loss_type, "dr_grpo")
        self.assertEqual(args.fsdp_plugin_args["fsdp_version"], 2)
        self.assertTrue(args.fsdp_plugin_args["reshard_after_forward"])
        self.assertEqual(args.fsdp_plugin_args["state_dict_type"], "SHARDED_STATE_DICT")
        for key in base["grpo"]:
            if key not in {
                "output_dir",
                "max_steps",
                "max_completion_length",
                "vllm_max_model_length",
                "save_steps",
            }:
                self.assertEqual(config["grpo"][key], base["grpo"][key], key)
        for source_id in ("", 42):
            config["data"]["source_id"] = source_id
            with self.assertRaisesRegex(ValueError, "source_id"):
                validate_config(config)

    def test_mixed_rewards_have_nonzero_gradient_and_real_update(self):
        parameter = torch.nn.Parameter(torch.zeros(4, 2))
        model = SimpleNamespace(
            named_parameters=lambda: [("mlp.down_proj.weight", parameter)],
            training=True,
        )
        callback = SmokeDiagnostics(1)
        callback.trainer = SimpleNamespace(
            accelerator=SimpleNamespace(device="cpu"), _last_loaded_step=0
        )
        rewards = torch.tensor(
            math_rewards(
                [r"Thus \boxed{42}.", "wrong", "unfinished", "42"], [["42"]] * 4
            )
        )
        self.assertEqual(rewards.tolist(), [1.0, 0.0, 0.0, 1.0])
        loss_trainer = object.__new__(GRPOTrainer)
        loss_trainer.model = model
        loss_trainer.args = SimpleNamespace(delta=None)
        loss_trainer.accelerator = SimpleNamespace(
            reduce=lambda x, **kwargs: x, gather=lambda x: x
        )
        loss_trainer._metrics = {"train": defaultdict(list)}
        loss_trainer._get_per_token_logps_and_entropies = lambda *args, **kwargs: (
            parameter.log_softmax(-1)[:, :1],
            torch.zeros(4, 1),
            None,
        )
        for key, value in {
            "aux_loss_enabled": False,
            "top_entropy_quantile": 1.0,
            "off_policy_mask_threshold": None,
            "importance_sampling_level": "token",
            "beta": 0.0,
            "epsilon_low": 0.2,
            "epsilon_high": 0.2,
            "loss_type": "dr_grpo",
            "use_vllm": True,
            "vllm_importance_sampling_correction": True,
            "max_completion_length": 16384,
            "current_gradient_accumulation_steps": 1,
            "_entropy_bonus_enabled": False,
        }.items():
            setattr(loss_trainer, key, value)
        inputs = {
            "prompt_ids": torch.ones(4, 1, dtype=torch.long),
            "prompt_mask": torch.ones(4, 1),
            "completion_ids": torch.ones(4, 1, dtype=torch.long),
            "completion_mask": torch.ones(4, 1),
            "advantages": rewards - rewards.mean(),
            "importance_sampling_ratio": torch.ones(4, 1),
        }
        # Run the production DR-GRPO loss with differentiable CPU log-probabilities.
        loss = GRPOTrainer._compute_loss(loss_trainer, model, inputs)
        loss.backward()
        unsuppressed_gradient = parameter.grad.clone()
        parameter.grad = None
        inputs["importance_sampling_ratio"].fill_(5.811e-5)
        GRPOTrainer._compute_loss(loss_trainer, model, inputs).backward()
        self.assertGreater(parameter.grad.abs().max().item(), 0)
        torch.testing.assert_close(
            parameter.grad, unsuppressed_gradient * 5.811e-5, rtol=1e-5, atol=0
        )
        parameter.grad = unsuppressed_gradient
        inputs["importance_sampling_ratio"].fill_(1.0)
        optimizer = torch.optim.AdamW([parameter], lr=1e-6, weight_decay=0)
        args = SimpleNamespace(
            process_index=0, device="cpu", max_steps=1, vllm_tensor_parallel_size=1
        )
        state = TrainerState()
        with (
            patch("rlvr.diagnostics.dist.all_reduce"),
            patch("rlvr.diagnostics.torch.cuda.max_memory_allocated", return_value=0),
            patch("rlvr.diagnostics.torch.cuda.max_memory_reserved", return_value=0),
            patch("rlvr.diagnostics.torch.cuda.device_memory_used", return_value=0),
        ):
            callback.on_pre_optimizer_step(args, state, None, model=model)
            optimizer.step()
            callback.on_optimizer_step(args, state, None, model=model)
            self.assertGreater(callback.max_gradient, 0)
            self.assertGreater(callback.max_delta, 0)
            self.assertEqual(
                callback.nonzero_gradient_steps & callback.updated_policy_steps, {1}
            )
            parameter.grad = None
            inputs["advantages"] = torch.zeros(4)
            uniform_loss = GRPOTrainer._compute_loss(loss_trainer, model, inputs)
            uniform_loss.backward()
            self.assertEqual(parameter.grad.abs().max().item(), 0)
            callback.on_pre_optimizer_step(
                args, TrainerState(global_step=1), None, model=model
            )
            self.assertNotIn(2, callback.nonzero_gradient_steps)
            parameter.grad.fill_(float("nan"))
            with self.assertRaises(FloatingPointError):
                callback.on_pre_optimizer_step(args, state, None, model=model)

    def test_one_step_keeps_subsequent_rollout_sync_validation(self):
        callback = SmokeDiagnostics(1)
        callback.updated_policy_steps = {1}
        callback.nonzero_gradient_steps = {1}
        callback.mixed_rewards = callback.distinct_completions = True
        name = "model.layers.0.mlp.down_proj.weight"
        parameter = torch.nn.Parameter(torch.zeros(4, 8))
        callback.sync_before[name] = down_proj_sample(parameter, 1, 4)
        with torch.no_grad():
            parameter.copy_(torch.arange(32).view(4, 8) / 128 + 0.25 + 1e-5)
        loaded = torch.nn.Parameter(
            torch.zeros(4, 2, dtype=torch.bfloat16), requires_grad=False
        )
        inference = SimpleNamespace(get_parameter=Mock(return_value=loaded))
        trainer = object.__new__(GRPOTrainer)
        trainer.model = SimpleNamespace(named_parameters=lambda: [(name, parameter)])
        trainer.accelerator = SimpleNamespace(device="cpu", is_main_process=True)
        trainer.args = SimpleNamespace(report_to=[])
        trainer.state = TrainerState(global_step=1)
        trainer.use_vllm = True
        trainer.train_dataset = [{"prompt": [{"role": "user", "content": "question"}]}]
        trainer._tokenize_prompts = Mock(return_value=([[1, 2]], None, {}))
        trainer._last_loaded_step = 0
        callback.trainer = trainer
        events = []

        def sync():
            events.append("sync")
            loaded.data.copy_(parameter[:, 2:4])

        def generate(**kwargs):
            events.append("rollout")
            self.assertEqual(trainer._last_loaded_step, 1)
            torch.testing.assert_close(loaded, parameter[:, 2:4].to(torch.bfloat16))
            return [[1, 2]], [[3, 4]], [[[0.0], [0.0]]], None

        # Exercise the real TRL sync/marker/generate path with a CPU inference stand-in.
        generation = SimpleNamespace(
            sync_weights=Mock(side_effect=sync),
            generate=Mock(side_effect=generate),
            _fix_param_name_to_vllm=lambda name: name,
            llm=SimpleNamespace(
                llm_engine=SimpleNamespace(
                    model_executor=SimpleNamespace(
                        driver_worker=SimpleNamespace(
                            model_runner=SimpleNamespace(model=inference)
                        )
                    )
                )
            ),
        )
        trainer.vllm_generation = generation
        original_sync = generation.sync_weights
        args = SimpleNamespace(
            max_steps=1, num_generations=4, process_index=1, vllm_tensor_parallel_size=4
        )
        with patch("rlvr.diagnostics.dist.all_reduce"):
            callback.on_train_end(args, trainer.state, None)
        self.assertEqual(events, ["sync", "rollout"])
        self.assertEqual(callback.verified_sync_steps, {1})
        self.assertGreater(callback.vllm_readback_delta, 0)
        self.assertIs(generation.sync_weights, original_sync)
        inference.get_parameter.assert_called_once_with(name)
        self.assertEqual(callback.updated_policy_steps & callback.sync_steps, {1})

        generation.generate.reset_mock()
        callback.mixed_rewards = False
        callback.on_train_end(args, TrainerState(global_step=1), None)
        generation.generate.assert_not_called()
        callback.mixed_rewards = True
        callback.on_train_end(
            SimpleNamespace(max_steps=8), TrainerState(global_step=1), None
        )
        generation.generate.assert_not_called()

        # A step marker alone must fail even when the tensors happen to match.
        callback.verified_sync_steps.clear()
        with self.assertRaisesRegex(RuntimeError, "did not verify"):
            callback.on_train_end(args, trainer.state, None)
        self.assertEqual(callback.verified_sync_steps, set())

        # A successful sync call that leaves stale weights must fail before generation.
        trainer._last_loaded_step = 0
        loaded.data.zero_()
        original_sync.side_effect = lambda: None
        generation.generate.reset_mock()
        with (
            patch("rlvr.diagnostics.dist.all_reduce"),
            self.assertRaisesRegex(RuntimeError, "stale/mismatched"),
        ):
            callback.on_train_end(args, trainer.state, None)
        generation.generate.assert_not_called()
        self.assertIs(generation.sync_weights, original_sync)
        self.assertEqual(callback.verified_sync_steps, set())

        original_sync.side_effect = sync
        callback.sync_before[name] = down_proj_sample(parameter, 1, 4)
        with (
            patch("rlvr.diagnostics.dist.all_reduce"),
            self.assertRaisesRegex(RuntimeError, "no changed sample.*BF16 rounding"),
        ):
            callback.on_train_end(args, trainer.state, None)
        generation.generate.assert_called_once()

        # Sub-ULP master updates match a correct BF16 load but cannot prove a changed inference policy.
        trainer._last_loaded_step = 0
        generation.generate.reset_mock()
        with torch.no_grad():
            parameter.add_(1e-5)
        self.assertGreater(
            (down_proj_sample(parameter, 1, 4) - callback.sync_before[name])
            .abs()
            .max()
            .item(),
            0,
        )
        with (
            patch("rlvr.diagnostics.dist.all_reduce"),
            self.assertRaisesRegex(RuntimeError, "survives BF16 rounding"),
        ):
            callback.on_train_end(args, trainer.state, None)
        generation.generate.assert_called_once()
        self.assertEqual(callback.verified_sync_steps, set())

        # A wrong TP column shard must not pass the dtype-aware comparison.
        trainer._last_loaded_step = 0
        generation.generate.reset_mock()
        loaded.data.copy_(parameter[:, :2])
        original_sync.side_effect = lambda: None
        with (
            patch("rlvr.diagnostics.dist.all_reduce"),
            self.assertRaisesRegex(RuntimeError, "stale/mismatched"),
        ):
            callback.on_train_end(args, trainer.state, None)
        generation.generate.assert_not_called()

        original_sync.side_effect = sync
        callback.sync_before[name].zero_()
        generation.generate.side_effect = lambda **kwargs: ([[1, 2]], [[]], [[]], None)
        with (
            patch("rlvr.diagnostics.dist.all_reduce"),
            self.assertRaisesRegex(RuntimeError, "no completion"),
        ):
            callback.on_train_end(args, trainer.state, None)
        self.assertEqual(callback.verified_sync_steps, set())

    def test_launcher_forwards_hydra_options_before_stop_override(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            directory = Path(temporary)
            for name in ("module", "nvidia-smi", "srun"):
                path = directory / name
                path.write_text(
                    "#!/bin/bash\n"
                    + (
                        'printf "%s\\n" "$@" > "$AUDIT_ARGV"\n'
                        if name == "srun"
                        else "exit 0\n"
                    )
                )
                path.chmod(0o700)
            capture = directory / "argv"
            env = {
                **os.environ,
                "PATH": str(directory) + os.pathsep + os.environ["PATH"],
                "SLURM_JOB_ID": "cpu-audit",
                "SLURM_CPUS_PER_TASK": "16",
                "AUDIT_ARGV": str(capture),
            }
            # All external launcher commands are stubs; no CUDA or Slurm calls run.
            subprocess.run(
                [
                    "bash",
                    str(ROOT / "slurm/train/rlvr-smoke.sh"),
                    "1",
                    "--config-name=rlvr/smoke-16k",
                    "runtime.validate_only=true",
                ],
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            argv = capture.read_text().splitlines()
            self.assertEqual(
                argv[:8],
                [
                    "uv",
                    "run",
                    "--group",
                    "grpo",
                    "--no-default-groups",
                    "--locked",
                    "--offline",
                    "torchrun",
                ],
            )
            self.assertIn("--nproc_per_node=4", argv)
            overrides = argv[argv.index(str(ROOT / "src/train_rlvr.py")) + 1 :]
            parsed = get_args_parser().parse_args(overrides)
            self.assertEqual(parsed.config_name, "rlvr/smoke-16k")
            self.assertEqual(
                parsed.overrides,
                ["runtime.validate_only=true", "runtime.stop_after_step=1"],
            )


if __name__ == "__main__":
    unittest.main()
