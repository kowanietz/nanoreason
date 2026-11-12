"""Offline checks for benchmark exclusion compatibility."""

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from check_benchmarks import IFEVAL_GOOGLE_2785, IFEVAL_H4_2785, check_questions


class BenchmarkQuestionsTest(unittest.TestCase):
    def setUp(self):
        manifest = json.loads((ROOT / "configs/eval/exclusions.json").read_text())
        self.expected = next(b for b in manifest["benchmarks"] if b["id"] == "ifeval")
        self.count = self.expected["expected_question_count"]
        self.hashes = set(self.expected["question_hashes"])
        self.assertIn(IFEVAL_GOOGLE_2785, self.hashes)
        self.variant = self.hashes - {IFEVAL_GOOGLE_2785} | {IFEVAL_H4_2785}

    def test_exact_questions_and_known_ifeval_variant(self):
        self.assertFalse(
            check_questions("ifeval", self.count, self.hashes, self.expected)
        )
        self.assertTrue(
            check_questions("ifeval", self.count, self.variant, self.expected)
        )

    def test_other_mismatches_still_fail(self):
        for identifier, count, observed in (
            ("ifeval", self.count - 1, self.variant),
            ("math500", self.count, self.variant),
            ("ifeval", self.count, self.variant | {"0" * 64}),
            (
                "ifeval",
                self.count,
                self.variant - {min(self.hashes - {IFEVAL_GOOGLE_2785})},
            ),
        ):
            with (
                self.subTest(
                    identifier=identifier, count=count, questions=len(observed)
                ),
                self.assertRaises(ValueError),
            ):
                check_questions(identifier, count, observed, self.expected)


if __name__ == "__main__":
    unittest.main()
