import hashlib
import json
import re
from pathlib import Path

DEFAULT_EXCLUSIONS = (
    Path(__file__).resolve().parents[1] / "configs/eval/exclusions.json"
)

NORMALIZATION = {
    "version": "whitespace_casefold_v1",
    "algorithm": "re.sub(r'\\s+', ' ', text).strip().casefold()",
    "encoding": "utf-8",
    "hash": "sha256",
}


def normalized(text):
    return re.sub(r"\s+", " ", text).strip().casefold()


def question_hash(text):
    return hashlib.sha256(normalized(text).encode("utf-8")).hexdigest()


def unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate exclusion manifest key: {key}")
        result[key] = value
    return result


def load_exclusions(path):
    data = path.read_bytes()
    try:
        manifest = json.loads(data, object_pairs_hook=unique_keys)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"Malformed exclusion manifest: {path}") from error
    if (
        not isinstance(manifest, dict)
        or type(manifest.get("version")) is not int
        or manifest["version"] != 1
    ):
        raise ValueError("Unsupported exclusion manifest version")
    if manifest.get("normalization") != NORMALIZATION:
        raise ValueError("Unsupported exclusion normalization version or algorithm")
    benchmarks = manifest.get("benchmarks")
    expected = manifest.get("expected_benchmark_count")
    if (
        not isinstance(benchmarks, list)
        or not benchmarks
        or type(expected) is not int
        or expected != len(benchmarks)
    ):
        raise ValueError("Empty or incomplete exclusion manifest")
    identifiers, hashes, provenance = set(), set(), []
    for benchmark in benchmarks:
        if not isinstance(benchmark, dict):
            raise TypeError("Malformed benchmark exclusion entry")
        identifier = benchmark.get("id")
        if (
            not isinstance(identifier, str)
            or not identifier.strip()
            or identifier in identifiers
        ):
            raise ValueError("Missing or duplicate benchmark identifier")
        identifiers.add(identifier)
        count = benchmark.get("expected_question_count")
        unique = benchmark.get("expected_unique_question_count")
        values = benchmark.get("question_hashes")
        if (
            type(count) is not int
            or type(unique) is not int
            or not 0 < unique <= count
            or not isinstance(values, list)
            or len(values) != unique
        ):
            raise ValueError(f"Inconsistent exclusion counts for {identifier}")
        if any(
            not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
            for value in values
        ):
            raise ValueError(f"Invalid question hash for {identifier}")
        if len(set(values)) != len(values):
            raise ValueError(f"Duplicate question hash for {identifier}")
        if values != sorted(values):
            raise ValueError(f"Question hashes must be sorted for {identifier}")
        sources = benchmark.get("sources")
        if not isinstance(sources, list) or not sources:
            raise ValueError(f"Missing exclusion provenance for {identifier}")
        for source in sources:
            if (
                not isinstance(source, dict)
                or not isinstance(source.get("url"), str)
                or not source["url"]
                or "revision" not in source
                or (
                    source["revision"] is not None
                    and (
                        not isinstance(source["revision"], str)
                        or not source["revision"].strip()
                    )
                )
                or not isinstance(source.get("sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", source["sha256"])
                or type(source.get("question_count")) is not int
                or source["question_count"] <= 0
            ):
                raise ValueError(f"Invalid exclusion provenance for {identifier}")
        if sum(source["question_count"] for source in sources) != count:
            raise ValueError(f"Inconsistent source question counts for {identifier}")
        hashes.update(values)
        provenance.append(
            {key: value for key, value in benchmark.items() if key != "question_hashes"}
        )
    return hashes, {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(data).hexdigest(),
        "version": manifest["version"],
        "normalization": manifest["normalization"],
        "method": "exact membership of SHA-256(normalized(user prompt).encode('utf-8'))",
        "unique_questions": len(hashes),
        "benchmarks": provenance,
    }
