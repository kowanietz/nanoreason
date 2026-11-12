"""Focused CPU contracts using the real upstream constraint checks."""

import copy
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rlvr.dispatch import domain_rewards, rollout_metrics
from rlvr.ifeval import IFValidationError, ifeval_score, verifier_info


def reference(ids=None, args=None):
    return [
        repr(
            [
                {
                    "instruction_id": ids
                    or ["change_case:english_lowercase", "punctuation:no_comma"],
                    "kwargs": [None, None] if args is None else args,
                }
            ]
        )
    ]


class IFRewardTest(unittest.TestCase):
    def test_fractional_and_strict_scores_exclude_reasoning(self):
        for final, fraction, strict in [
            ("hello world", 1, 1),
            ("hello, world", 0.5, 0),
            ("HELLO, WORLD", 0, 0),
        ]:
            score = ifeval_score(
                "UPPERCASE, THINKING</think>" + final, reference(), True
            )
            self.assertEqual((score["reward"], score["strict"]), (fraction, strict))
        constrained = reference(["keywords:existence"], [{"keywords": ["secret"]}])
        self.assertEqual(
            ifeval_score("secret</think>answer", constrained, True)["reward"], 0
        )

    def test_empty_invalid_and_unclosed_thinking_receive_zero(self):
        for text in [
            None,
            "",
            "hello world",
            "work</think>",
            "<think>work</think>hello",
            "work</think><think>more</think>hello",
            [{"role": "user", "content": "hello"}],
        ]:
            self.assertEqual(ifeval_score(text, reference(), True)["reward"], 0)
        self.assertEqual(
            ifeval_score(
                [{"role": "assistant", "content": "work</think>hello world"}],
                reference(),
                True,
            )["reward"],
            1,
        )

    def test_bad_references_raise_even_for_empty_responses(self):
        bad = [
            [],
            ["[]"],
            ["[{}]"],
            ["invalid"],
            reference(["unknown:constraint"], [None]),
            reference(["keywords:existence"], [None]),
            reference(["keywords:frequency"], [{"keyword": "cat", "frequency": 2}]),
            reference(["punctuation:no_comma"], [{"extra": True}]),
            reference(["keywords:existence"], [{"keywords": 42}]),
            reference(
                ["length_constraints:nth_paragraph_first_word"],
                [{"num_paragraphs": 2, "nth_paragraph": 3, "first_word": "cat"}],
            ),
            reference(
                ["new:copy_span_idx"],
                [{"n_start": 0, "n_end": 3, "prompt_to_repeat": "abcdef"}],
            ),
        ]
        before = random.getstate()
        for value in bad:
            with self.subTest(value=value), self.assertRaises(IFValidationError):
                ifeval_score("", value)
        self.assertEqual(before, random.getstate())

    def test_upstream_sentence_and_counting_resources_work_offline(self):
        info = verifier_info()
        self.assertEqual(len(info["registry_ids"]), 54)
        self.assertEqual(
            ifeval_score(
                "hello world.",
                reference(
                    ["length_constraints:number_sentences", "count:count_unique"],
                    [{"num_sentences": 1, "relation": "at least"}, None],
                ),
            )["reward"],
            1,
        )
        self.assertEqual(
            ifeval_score("hello hello.", reference(["count:count_unique"], [None]))[
                "reward"
            ],
            0,
        )

    def test_missing_resources_raise_an_explicit_validation_error(self):
        verifier_info.cache_clear()
        try:
            with (
                tempfile.TemporaryDirectory(dir=ROOT) as directory,
                patch("rlvr.ifeval.RESOURCE_ROOT", Path(directory)),
                self.assertRaisesRegex(IFValidationError, "resources unavailable"),
            ):
                verifier_info()
        finally:
            verifier_info.cache_clear()

    def test_mixed_dispatch_fractional_rewards_and_truncated_finals(self):
        extra, metrics = {}, {}
        rewards = domain_rewards(
            completions=[
                "work</think>42",
                "work</think>43",
                "unfinished",
                "work</think>42",
                "work</think>hello world",
                "work</think>hello, world",
                "unfinished",
                "work</think>hello world",
            ],
            ground_truth=[["42"]] * 4 + [reference()] * 4,
            domain=["math"] * 4 + ["ifeval"] * 4,
            reasoning_prefix=[True] * 8,
            source_id=["math-1"] * 4 + ["if-1"] * 4,
            completion_ids=[[1, 2]] * 7 + [[1, 1]],
            eos_token_ids={2},
            log_extra=extra.__setitem__,
            log_metric=metrics.__setitem__,
        )
        self.assertEqual(rewards, [1, 0, 0, 1, 1, 0.5, 0, 0])
        self.assertEqual(metrics["domain/ifeval/strict_rate"], 0.25)
        self.assertEqual(metrics["domain/ifeval/truncated_rate"], 0.25)
        self.assertEqual(metrics["domain/ifeval/completion_tokens"], 8)
        self.assertEqual(extra["completion_tokens"], [2] * 8)
        self.assertEqual(metrics["group/all/mixed_reward_rate"], 1)
        self.assertEqual(extra["strict"], [1, 0, 0, 1, 1, 0, 0, 0])
        uniform = domain_rewards(
            ["work</think>hello world"] * 4,
            [reference()] * 4,
            ["ifeval"] * 4,
            [True] * 4,
            source_id=["if-1"] * 4,
            log_metric=metrics.__setitem__,
        )
        self.assertEqual(uniform, [1] * 4)
        self.assertEqual(metrics["group/all/uniform_reward_rate"], 1)
        for arguments in [
            {"ground_truth": [reference()], "domain": ["ifeval", "ifeval"]},
            {"ground_truth": [reference()], "domain": ["unsupported"]},
        ]:
            with self.assertRaises(ValueError):
                domain_rewards([""], **arguments)

    def test_group_metadata_validation(self):
        row = {
            "domain": "math",
            "source_id": "one",
            "reward": 1,
            "strict": 1,
            "valid_final": True,
            "truncated": False,
            "constraint_ids": [],
            "passed": [],
        }
        rows = [copy.deepcopy(row) for _ in range(4)]
        self.assertEqual(rollout_metrics(rows, 4)["group/math/uniform_reward_rate"], 1)
        rows[1]["source_id"] = "other"
        with self.assertRaisesRegex(ValueError, "misaligned"):
            rollout_metrics(rows, 4)
        with self.assertRaisesRegex(ValueError, "complete"):
            rollout_metrics(rows[:3], 4)


if __name__ == "__main__":
    unittest.main()
