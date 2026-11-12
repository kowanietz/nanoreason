"""Offline checks for benchmark exclusion compatibility."""

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import eval as evaluation
from check_benchmarks import IFEVAL_GOOGLE_2785, IFEVAL_H4_2785, check_questions


class EvaluationProvenanceTest(unittest.TestCase):
    def test_local_changes_are_recorded_and_changed_provenance_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            directory = Path(temporary)
            project = directory / "olmes"
            project.mkdir()
            (project / "uv.lock").write_text("frozen dependencies")
            argv = [
                "eval.py",
                "--model",
                "/unused/model",
                "--name",
                str(directory / "run"),
                "--config",
                "math500",
            ]
            with (
                patch.dict(os.environ, {"OLMES_PROJECT": str(project)}),
                patch.object(sys, "argv", argv),
                patch.object(
                    evaluation.subprocess,
                    "check_output",
                    side_effect=["revision\n", b"local edits"],
                ) as git,
                patch.object(evaluation.subprocess, "run") as run,
            ):
                evaluation.main()
                path = directory / "run/math500/benchmark_provenance.json"
                original = path.read_bytes()
                provenance = json.loads(original)
                self.assertEqual(
                    provenance["olmes_diff_sha256"],
                    hashlib.sha256(b"local edits").hexdigest(),
                )
                run.assert_called_once()
                run.reset_mock()
                git.side_effect = ["revision\n", b"changed edits"]
                with self.assertRaisesRegex(ValueError, "provenance changed"):
                    evaluation.main()
                run.assert_not_called()
                self.assertEqual(path.read_bytes(), original)
                git.side_effect = ["revision\n", b"changed edits"]
                with patch.object(sys, "argv", [*argv, "--dry-run"]):
                    evaluation.main()
                self.assertEqual(run.call_args.args[0][-1], "--dry-run")
                self.assertEqual(path.read_bytes(), original)


class BenchmarkQuestionsTest(unittest.TestCase):
    def test_olmes_lock_snapshot_matches_recorded_benchmark_provenance(self):
        lock = ROOT / "configs/eval/olmes/uv.lock"
        digest = hashlib.sha256(lock.read_bytes()).hexdigest()
        paths = list((ROOT / "results").glob("*/*/benchmark_provenance.json"))
        self.assertTrue(paths)
        for path in paths:
            with self.subTest(path=path):
                self.assertEqual(
                    json.loads(path.read_text())["olmes_lock_sha256"], digest
                )

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
