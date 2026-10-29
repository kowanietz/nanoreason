import copy
import hashlib
import json
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from generate_benchmark_exclusions import make_benchmark
from prepare_rlvr_data import (
    COLUMNS,
    check_rendering,
    prepare,
    select_records,
)

from benchmark_exclusions import NORMALIZATION, load_exclusions, question_hash
from rlvr_rewards import math_reward, math_rewards, validate_reference_answers


def record(prompt="user: What is six times seven?", domain="math", answers=None):
    row = dict.fromkeys(COLUMNS)
    row.update(
        prompt=prompt,
        dataset=[domain],
        ground_truth=["42"] if answers is None else answers,
        dataset_source="source/mix",
        original_dataset="source/original",
        custom_id="example-1",
    )
    if domain == "ifeval":
        row.update(
            constraint_type="multi",
            constraint="Use lowercase and no commas.",
            ground_truth=[
                repr(
                    [
                        {
                            "instruction_id": [
                                "change_case:english_lowercase",
                                "punctuation:no_comma",
                            ],
                            "kwargs": [None, None],
                        }
                    ]
                )
            ],
        )
    return row


def select(rows, questions=None):
    return select_records(
        ((row, "data/a.parquet", i) for i, row in enumerate(rows)), questions or set()
    )


def exclusion_manifest():
    return {
        "version": 1,
        "normalization": copy.deepcopy(NORMALIZATION),
        "expected_benchmark_count": 1,
        "benchmarks": [
            {
                "id": "example",
                "expected_question_count": 1,
                "expected_unique_question_count": 1,
                "question_hashes": [question_hash("Example question")],
                "sources": [
                    {
                        "url": "https://example.test/questions",
                        "revision": "fixture-v1",
                        "sha256": hashlib.sha256(b"fixture").hexdigest(),
                        "question_count": 1,
                    }
                ],
            }
        ],
    }


class ExclusionManifestTest(unittest.TestCase):
    def test_manifest_provenance_and_coverage(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / "exclusions.json"
            path.write_text(json.dumps(exclusion_manifest()))
            hashes, info = load_exclusions(path)
            self.assertEqual(hashes, {question_hash(" EXAMPLE\nquestion ")})
            self.assertEqual(
                info["sha256"], hashlib.sha256(path.read_bytes()).hexdigest()
            )
        hashes, info = load_exclusions(ROOT / "benchmarks/exclusions.json")
        self.assertEqual(
            {b["id"]: b["expected_unique_question_count"] for b in info["benchmarks"]},
            {
                "math500": 500,
                "aime2024": 30,
                "aime2025": 30,
                "gpqa_main": 448,
                "ifeval": 541,
                "bbh": 6507,
            },
        )
        self.assertEqual(len(hashes), 8056)

    def test_rejects_invalid_manifests(self):
        mutations = [
            ("version", True),
            ("version", 2),
            ("normalization", {}),
            ("expected_benchmark_count", 2),
            ("benchmarks", []),
        ]
        benchmark_mutations = [
            ("id", ""),
            ("expected_question_count", 0),
            ("expected_question_count", 2),
            ("expected_unique_question_count", True),
            ("expected_unique_question_count", 2),
            ("question_hashes", ["g" * 64]),
            ("question_hashes", ["F" * 64]),
            ("question_hashes", ["f" * 63]),
            ("question_hashes", []),
            ("sources", []),
            ("sources", [{"url": "https://example.test", "sha256": "invalid"}]),
        ]
        manifests = []
        for key, value in mutations:
            manifests.append({**exclusion_manifest(), key: value})
        for key, value in benchmark_mutations:
            manifest = exclusion_manifest()
            manifest["benchmarks"][0][key] = value
            manifests.append(manifest)
        duplicate = exclusion_manifest()
        duplicate["benchmarks"] *= 2
        duplicate["expected_benchmark_count"] = 2
        manifests.append(duplicate)
        for hashes in (["0" * 64] * 2, ["f" * 64, "0" * 64]):
            manifest = exclusion_manifest()
            manifest["benchmarks"][0].update(
                expected_question_count=2,
                expected_unique_question_count=2,
                question_hashes=hashes,
            )
            manifest["benchmarks"][0]["sources"][0]["question_count"] = 2
            manifests.append(manifest)
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / "exclusions.json"
            for text in [
                "",
                "{",
                "[]",
                "null",
                '{"version":1,"version":1}',
                *map(json.dumps, manifests),
            ]:
                with self.subTest(manifest=text), self.assertRaises(ValueError):
                    path.write_text(text)
                    load_exclusions(path)

    def test_generator_requires_expected_counts(self):
        for expected, unique in ((2, 1), (1, 2)):
            with (
                self.subTest(expected=expected, unique=unique),
                self.assertRaisesRegex(ValueError, "expected"),
            ):
                make_benchmark("fixture", "test", ["question"], expected, unique, [])


class MathRewardsTest(unittest.TestCase):
    def test_answer_formats_and_equivalence(self):
        cases = [
            ("42", "42"),
            (r"\boxed{42}", "42"),
            (r"The final answer is $\boxed{42}$.", "42"),
            (r"Answer: \(42\)", "42"),
            ("Final answer: 42", "42"),
            ("x=42", "42"),
            (r"\sqrt{8}", r"2\sqrt{2}"),
            ("0.5", r"\frac{1}{2}"),
            ("1/2", "0.5"),
            ("50%", "0.5"),
            (r"50\%", "0.5"),
            ("1e-3", "0.001"),
            ("1e3", "1000"),
            ("0.5", "x=1/2"),
            ("x=50%", "0.5"),
            ("x=1e3", "1000"),
            (r"x=\pi", r"\pi"),
            (r"Therefore the result is \[\boxed{217}\]", "217"),
            (r"The result is \boxed{\frac{1}{2}}. Thank you.", "0.5"),
            (r"The solution set is \boxed{\{1,2\}}.", r"\{2,1\}"),
            ("Explanation. " * 250 + r"\boxed{42}", "42"),
        ]
        polynomial = "x^4 - 3x^3 - 5x^2 + 11x + 2"
        cases += [
            ("Thus, \\boxed{" + expression + "}.", polynomial)
            for expression in (
                polynomial,
                "2+11x-5x^2-3x^3+x^4",
                "x*(x*(x*(x-3)-5)+11)+2",
            )
        ]
        for completion, reference in cases:
            with self.subTest(completion=completion, reference=reference):
                self.assertEqual(math_reward(completion, [reference]), 1.0)

    def test_reasoning_boundary(self):
        for completion, prefix, reward in [
            ("<think>42</think>43", False, 0.0),
            ("<think>43</think>42", False, 1.0),
            ("work</think>Thus \\boxed{42}.", True, 1.0),
            ("work</think>43", True, 0.0),
            (r"Still thinking: \boxed{42}", True, 0.0),
            ("<think>42</think>42", True, 0.0),
            ("<think>42", False, 0.0),
            ("<think>42</think>", False, 0.0),
            ("<think>42</think><think>42</think>42", False, 0.0),
            ("</think>42", False, 0.0),
        ]:
            with self.subTest(completion=completion, prefix=prefix):
                self.assertEqual(
                    math_reward(completion, ["42"], reasoning_prefix=prefix), reward
                )

    def test_wrong_malformed_and_ambiguous_answers(self):
        for completion in [
            None,
            "",
            "43",
            "No answer.",
            "I saw 42 birds.",
            "41 or 42",
            "41,42",
            "41\n42",
            "x=0=42",
            "0=42",
            "x=1,y=42",
            r"\frac{1}{}",
            "42 +",
            r"Thus \boxed{42",
            r"Thus \boxed{42}}",
            r"Thus \boxed42",
            r"Thus \boxed{}",
            r"Thus \boxed{\frac{1}{}}",
            r"Thus \boxed{42 +}",
            r"\boxed{41} or \boxed{42}",
            r"\boxed{42} and \boxed{42}",
            r"\boxed{\boxed{42}}",
            r"\boxed{42 or 43}",
            "\\boxed{" + "1+" * 1100 + "41}",
        ]:
            with self.subTest(completion=completion):
                self.assertEqual(math_reward(completion, ["42"]), 0.0)
        self.assertEqual(
            math_reward(r"\boxed{x^3-3x^2+6x+2}", ["x^4-3x^3-5x^2+11x+2"]), 0.0
        )

    def test_equations_and_structured_types(self):
        for completion, reference, expected in [
            ("y=x", "x=y", 1.0),
            ("x=y+0", "x=y", 1.0),
            ("x+y", "y+x", 1.0),
            ("x=2+2", "x=4", 1.0),
            ("y", "x=y", 0.0),
            ("z=y", "x=y", 0.0),
            ("x=2y", "x=y", 0.0),
            ("x=x", "x=y", 0.0),
            ("x+y", "x-y", 0.0),
            ("y=42", "x=42", 0.0),
            ("x=43", "42", 0.0),
            ("x=y", "y", 0.0),
            (r"\{1,2\}", r"\{2,1\}", 1.0),
            (r"\{1,2\}", "1", 0.0),
            ("{2}", "2", 0.0),
            ("2", "{2}", 0.0),
            ("{}", r"\emptyset", 1.0),
            ("(2;3)", "(2;3)", 1.0),
            ("(3;2)", "(2;3)", 0.0),
            ("(2;3)", "{2,3}", 0.0),
            ("(1,2,3)", "(1,2,3)", 1.0),
            ("(1,2,4)", "(1,2,3)", 0.0),
            ("(1;(2;3))", "(1,2,3)", 0.0),
        ]:
            with self.subTest(completion=completion, reference=reference):
                self.assertEqual(math_reward(completion, [reference]), expected)
        intervals = ("(2,3)", "[2,3]", "[2,3)", "(2,3]")
        for reference in intervals:
            for completion in (*intervals, "{2,3}", "(2;3)", "2"):
                with self.subTest(completion=completion, reference=reference):
                    self.assertEqual(
                        math_reward(completion, [reference]),
                        float(reference == completion),
                    )

    def test_reference_validation(self):
        for reference in [
            None,
            [],
            [None],
            [42],
            [""],
            {"answer": "42"},
            ["Friday"],
            [r"\frac{1}{}"],
            ["1/0"],
            ["05:54"],
            ["42", "not a mathematical answer"],
            [r"Therefore \boxed{42}."],
            [r"x\in(-1;2)"],
            ["(1;(2;3))"],
            ["{(1;2),(3;4)}"],
        ]:
            with self.subTest(reference=reference), self.assertRaises(ValueError):
                math_reward("", reference)
        with self.assertRaisesRegex(ValueError, "verifiable construction"):
            math_reward(
                "X", ["X"], prompt="Construct a point X on a circle such that XA = XB."
            )
        self.assertEqual(math_reward("x", ["x"], prompt="Simplify 2x-x."), 1.0)
        self.assertEqual(math_reward("y", ["x"], prompt="Simplify 2x-x."), 0.0)
        self.assertEqual(math_reward("0.5", ["1/2", "2/4"]), 1.0)
        self.assertEqual(math_reward("42", "42"), 1.0)
        with (
            ThreadPoolExecutor(max_workers=1) as executor,
            self.assertRaisesRegex(RuntimeError, "main thread"),
        ):
            executor.submit(math_reward, "42", ["42"]).result()

    def test_trl_reward_metadata(self):
        completions = [
            [{"role": "assistant", "content": content}]
            for content in ("work</think>42", "work</think>43")
        ]
        self.assertEqual(
            math_rewards(
                completions,
                [["42"], ["42"]],
                reasoning_prefix=[True, True],
                domain=["math", "math"],
                prompts=[[], []],
                trainer_state=None,
            ),
            [1.0, 0.0],
        )
        for arguments in [
            {"ground_truth": [["42"]]},
            {"ground_truth": [["42"], ["42"]], "domain": ["math", "ifeval"]},
            {"ground_truth": [["42"], ["42"]], "reasoning_prefix": [True]},
            {"ground_truth": [["42"], ["42"]], "prompts": [[]]},
        ]:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                math_rewards(completions, **arguments)


class PreparationTest(unittest.TestCase):
    def test_shared_reference_validation(self):
        for reference in (
            "x=y",
            "x=1/2",
            "(2,3)",
            "[2,3]",
            "{2,3}",
            "(2;3)",
            "(1,2,3)",
            "x",
        ):
            row = record(
                prompt="user: Give the requested mathematical value.",
                answers=[reference],
            )
            selected, _ = select([row])
            self.assertEqual(len(selected), 1)
            self.assertEqual(
                math_reward(
                    reference, selected[0]["ground_truth"], prompt=selected[0]["prompt"]
                ),
                1.0,
            )
        for prompt, reference in (
            ("user: Construct a point X on a circle such that XA = XB.", "X"),
            ("user: Give a value.", r"x\in(-1;2)"),
            ("user: Give a value.", "{(1;2),(3;4)}"),
        ):
            selected, stats = select([record(prompt=prompt, answers=[reference])])
            self.assertEqual(selected, [])
            self.assertEqual(stats["removed"], {"unsupported_math_reference": 1})
            with self.assertRaises(ValueError):
                validate_reference_answers([reference], prompt=prompt[6:])

    def test_list_fields_and_domain_filtering(self):
        rows, stats = select(
            [
                record(),
                record("user: Write a sentence.", "ifeval"),
                record(domain="code"),
            ]
        )
        self.assertEqual(stats["retained_by_domain"], {"ifeval": 1, "math": 1})
        self.assertEqual(stats["removed"], {"domain_not_selected": 1})
        self.assertEqual(rows[0]["ground_truth"], ["42"])
        self.assertEqual(rows[0]["dataset"], ["math"])
        self.assertFalse(rows[1]["reward_enabled"])
        self.assertEqual(json.loads(rows[1]["constraint_spec"])["kwargs"], [None, None])

    def test_deterministic_deduplication(self):
        inputs = [record(), record("user:  WHAT is six\n times seven? ")]
        first, stats = select(inputs)
        second, repeated_stats = select(copy.deepcopy(inputs))
        self.assertEqual(first, second)
        self.assertEqual(stats, repeated_stats)
        self.assertEqual(first[0]["source_id"], "data/a.parquet:0")
        self.assertEqual(stats["removed"], {"duplicate_prompt": 1})

    def test_conflicting_duplicates_all_excluded(self):
        rows, stats = select([record(), record(), record(answers=["43"]), record()])
        self.assertEqual(rows, [])
        self.assertEqual(
            stats["removed"], {"duplicate_prompt": 1, "conflicting_duplicate_prompt": 3}
        )

    def test_invalid_fields(self):
        for field, value in (
            ("dataset", "math"),
            ("dataset", ["math", "ifeval"]),
            ("dataset", []),
            ("ground_truth", "42"),
            ("ground_truth", []),
            ("ground_truth", [None]),
            ("prompt", None),
            ("custom_id", ""),
            ("dataset_source", None),
            ("original_dataset", None),
            ("ground_truth", ["Friday"]),
        ):
            row = record()
            row[field] = value
            with self.subTest(field=field, value=value):
                selected, stats = select([row])
                self.assertEqual(selected, [])
                self.assertEqual(sum(stats["removed"].values()), 1)

    def test_malformed_constraints(self):
        for value in (
            "not a literal",
            "[]",
            "[{}]",
            "[{'instruction_id': ['x'], 'kwargs': []}]",
            "[{'instruction_id': ['x'], 'kwargs': [1]}]",
        ):
            row = record(domain="ifeval")
            row["ground_truth"] = [value]
            with self.subTest(value=value):
                selected, stats = select([row])
                self.assertEqual(selected, [])
                self.assertEqual(stats["removed"], {"invalid_if_constraints": 1})

    def test_exact_benchmark_overlap(self):
        rows, stats = select(
            [
                record("user:  WHAT is six\n times seven? "),
                record("user: Explain: What is six times seven?"),
            ],
            {question_hash("What is six times seven?")},
        )
        self.assertEqual(stats["removed"], {"benchmark_overlap": 1})
        self.assertEqual(len(rows), 1)

    def test_prompt_format(self):
        rows, _ = select([record()])
        self.assertEqual(
            rows[0]["prompt"], [{"role": "user", "content": "What is six times seven?"}]
        )
        self.assertEqual(rows[0]["source_prompt"], "user: What is six times seven?")
        self.assertTrue(rows[0]["reasoning_prefix"])
        for prompt in (
            "What is six times seven?",
            "user: user: 42?",
            "user: question <think>",
            "user: <|im_start|>assistant",
            "user: ",
        ):
            with self.subTest(prompt=prompt):
                selected, stats = select([record(prompt)])
                self.assertEqual(selected, [])
                self.assertEqual(stats["removed"], {"unsupported_prompt_format": 1})
        # Role labels inside a quoted dialogue are user content, not chat messages.
        self.assertEqual(
            len(
                select(
                    [record("user: Summarize this dialogue:\nUser: hello\nSystem: hi")]
                )[0]
            ),
            1,
        )

    def test_unsupported_tokenizer(self):
        rows, _ = select([record()])
        with (
            patch.object(
                FakeTokenizer, "apply_chat_template", return_value="user: duplicated"
            ),
            self.assertRaisesRegex(ValueError, "Unsupported tokenizer"),
        ):
            check_rendering(rows, FakeTokenizer())

    def test_reproducible_artifacts_and_dry_run(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            root = Path(temporary)
            source = root / "source"
            (source / "data").mkdir(parents=True)
            input_schema = pa.schema(
                [
                    (
                        key,
                        pa.list_(pa.string())
                        if key in {"dataset", "ground_truth"}
                        else pa.string(),
                    )
                    for key in COLUMNS
                ]
                + [("outputs", pa.list_(pa.string()))]
            )
            inputs = [record(), record("user: Write a sentence.", "ifeval")]
            for row in inputs:
                row["outputs"] = [
                    "Historical rollout that must never be a training target."
                ]
            pq.write_table(
                pa.Table.from_pylist(inputs, schema=input_schema),
                source / "data/a.parquet",
            )
            tokenizer = FakeTokenizer()
            exclusions = root / "exclusions.json"
            exclusions.write_text(json.dumps(exclusion_manifest()))
            first, _ = prepare(source, root / "one", tokenizer, exclusions)
            second, _ = prepare(source, root / "two", tokenizer, exclusions)
            dry, samples = prepare(
                source, root / "dry", tokenizer, exclusions, dry_run=True
            )
            self.assertEqual(first, second)
            self.assertEqual(first, dry)
            self.assertFalse((root / "dry").exists())
            data = (root / "one/examples.parquet").read_bytes()
            self.assertEqual(data, (root / "two/examples.parquet").read_bytes())
            self.assertEqual(
                first["output"]["sha256"], hashlib.sha256(data).hexdigest()
            )
            self.assertEqual(
                first["exclusions"]["sha256"],
                hashlib.sha256(exclusions.read_bytes()).hexdigest(),
            )
            self.assertEqual(first["exclusions"]["excluded_training_examples"], 0)
            self.assertEqual(
                (root / "one/manifest.json").read_bytes(),
                (root / "two/manifest.json").read_bytes(),
            )
            self.assertNotIn(
                "outputs", pq.read_table(root / "one/examples.parquet").column_names
            )
            self.assertEqual(samples["math"]["rendered"].count("<think>"), 1)
            with self.assertRaises(FileExistsError):
                prepare(source, root / "one", tokenizer, exclusions)


class FakeTokenizer:
    name_or_path = "synthetic-test-tokenizer"
    chat_template = "synthetic-test-template"

    def apply_chat_template(
        self, prompt, tokenize, add_generation_prompt, return_dict=False
    ):
        assert add_generation_prompt
        text = (
            "<|im_start|>user\n"
            + prompt[0]["content"]
            + "<|im_end|>\n<|im_start|>assistant\n<think>"
        )
        return list(text.encode()) if tokenize else text

    def decode(self, ids, skip_special_tokens):
        return bytes(ids).decode()


if __name__ == "__main__":
    unittest.main()
