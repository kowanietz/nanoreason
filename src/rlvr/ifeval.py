"""Dolci references and pinned Open-Instruct IFEvalG checks."""

import ast
import hashlib
import json
import random
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path

from .rewards import final_response

VENDOR = Path(__file__).parent / "_vendor" / "ifevalg"
RESOURCE_ROOT = Path(__file__).resolve().parents[2] / ".cache/ifevalg/nltk_data"


class IFValidationError(ValueError):
    """Invalid reference, unsupported ID, or unavailable verifier resource."""


def check_resources() -> list[dict]:
    """Check the pinned NLTK files shared by setup and offline preflight."""
    resources = json.loads((VENDOR / "UPSTREAM.json").read_text())["resources"]
    for resource in resources:
        for name, digest in resource["files"].items():
            if (
                hashlib.sha256((RESOURCE_ROOT / name).read_bytes()).hexdigest()
                != digest
            ):
                raise IFValidationError(f"IFEvalG resource mismatch: {name}")
    return resources


@lru_cache(maxsize=1)
def verifier_info() -> dict:
    """Verify pinned source, dependencies, and English tokenizer data offline."""
    try:
        versions = {
            name: version(name)
            for name in ("absl-py", "immutabledict", "langdetect", "nltk")
        }
        upstream = json.loads((VENDOR / "UPSTREAM.json").read_text())
        for name, digest in upstream["vendored_sha256"].items():
            if hashlib.sha256((VENDOR / name).read_bytes()).hexdigest() != digest:
                raise IFValidationError(f"Modified IFEvalG source: {name}")
        resources = check_resources()
        del upstream["resources"]
        import nltk
        from langdetect import DetectorFactory

        # Language detection is otherwise nondeterministic. Do not seed training RNG.
        DetectorFactory.seed = 0
        nltk.data.path.insert(0, str(RESOURCE_ROOT))
        nltk.word_tokenize("An offline tokenizer check.")
        from ._vendor.ifevalg import instructions_registry

    except (ImportError, OSError, LookupError) as error:
        raise IFValidationError(
            "IFEvalG dependencies/resources unavailable; sync --group grpo and run "
            "scripts/setup_if_resources.py before offline preparation/training"
        ) from error
    return {
        "upstream": upstream,
        "dependencies": versions,
        "resources": resources,
        "registry_ids": sorted(instructions_registry.INSTRUCTION_DICT),
        "reward": "fraction of official constraints passed; strict final response only",
    }


def constraint_spec(ground_truth) -> dict:
    """Parse the original Dolci list[str] containing a Python-literal list[dict]."""
    if (
        not isinstance(ground_truth, list)
        or len(ground_truth) != 1
        or not isinstance(ground_truth[0], str)
    ):
        raise IFValidationError("IF ground_truth must be one literal reference string")
    try:
        value = ast.literal_eval(ground_truth[0])
        if (
            not isinstance(value, list)
            or len(value) != 1
            or not isinstance(value[0], dict)
        ):
            raise ValueError("expected one constraint dictionary")
        spec = value[0]
        if set(spec) != {"instruction_id", "kwargs"}:
            raise ValueError("expected instruction_id and kwargs")
        ids, arguments = spec["instruction_id"], spec["kwargs"]
        if (
            not isinstance(ids, list)
            or not ids
            or any(not isinstance(key, str) or not key for key in ids)
            or not isinstance(arguments, list)
            or len(ids) != len(arguments)
            or any(arg is not None and not isinstance(arg, dict) for arg in arguments)
            or any(not isinstance(k, str) for arg in arguments if arg for k in arg)
        ):
            raise ValueError("invalid constraint IDs/argument alignment")
        canonical = json.dumps(
            spec, sort_keys=True, ensure_ascii=False, allow_nan=False
        )
    except (ValueError, SyntaxError, TypeError) as error:
        raise IFValidationError(f"Malformed IF reference: {error}") from error
    validate_spec(canonical)
    return spec


@lru_cache(maxsize=32768)
def validate_spec(canonical: str) -> None:
    verifier_info()
    spec = json.loads(canonical)
    for key, args in zip(spec["instruction_id"], spec["kwargs"], strict=True):
        build_checker(key, args)


def build_checker(key: str, args):
    from ._vendor.ifevalg.instructions_registry import INSTRUCTION_DICT

    if key not in INSTRUCTION_DICT:
        raise IFValidationError(f"Unsupported IF constraint ID: {key}")
    # Upstream removes null kwargs before constructing the official checker.
    args = {k: v for k, v in (args or {}).items() if v is not None}
    checker = INSTRUCTION_DICT[key](key)
    state = random.getstate()
    try:
        checker.build_description(**args)
        if random.getstate() != state:
            raise ValueError("reference invokes random default constraint arguments")
        effective = checker.get_instruction_args() or {}
        if set(effective) != set(args):
            raise ValueError(
                f"expected arguments {sorted(effective)}, received {sorted(args)}"
            )
        for name, value in effective.items():
            if type(value) is not type(args[name]):
                raise ValueError(f"invalid argument type for {name}")
    except (TypeError, ValueError, AssertionError, AttributeError, IndexError) as error:
        raise IFValidationError(f"Malformed arguments for {key}: {error}") from error
    finally:
        random.setstate(state)
    return checker


def ifeval_score(completion, ground_truth, reasoning_prefix=False) -> dict:
    """Return the upstream fractional reward and the strict all-constraints metric."""
    spec = constraint_spec(ground_truth)  # Validate even when the response is empty.
    answer = final_response(completion, reasoning_prefix=reasoning_prefix)
    passed = []
    for key, args in zip(spec["instruction_id"], spec["kwargs"], strict=True):
        checker = build_checker(key, args)
        passed.append(bool(answer and checker.check_following(answer)))
    return {
        "reward": sum(passed) / len(passed),
        "strict": float(all(passed)),
        "constraint_ids": spec["instruction_id"],
        "passed": passed,
        "valid_final": answer is not None,
    }
