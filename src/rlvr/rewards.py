"""Strict final-answer extraction and binary mathematical-equivalence rewards."""

import re
import threading
from functools import lru_cache

from math_verify import LatexExtractionConfig, LatexNormalizationConfig, parse, verify
from sympy import And, Basic, Eq, Expr, FiniteSet, Interval, S, Set, Symbol, Tuple
from sympy.core.relational import Relational

EXTRACTION = LatexExtractionConfig(
    try_extract_without_anchor=False,
    boxed_match_priority=0,
    normalization_config=LatexNormalizationConfig(
        basic_latex=True,
        units=False,
        malformed_operators=False,
        nits=False,
        boxed="all",
        equations=False,
    ),
)


def final_response(completion: object, reasoning_prefix: bool = False) -> str | None:
    """Return final text only after a well-formed reasoning boundary."""
    if isinstance(completion, list):
        if (
            len(completion) != 1
            or not isinstance(completion[0], dict)
            or completion[0].get("role") != "assistant"
        ):
            return None
        completion = completion[0].get("content")
    if not isinstance(completion, str):
        return None
    text = completion.strip()
    # OLMos generation prompt already contains the reasoning tag
    if reasoning_prefix:
        text = "<think>" + text
    if "<think" in text or "</think" in text:
        if (
            not text.startswith("<think>")
            or text.count("<think>") != 1
            or text.count("</think>") != 1
        ):
            return None
        text = text.split("</think>", 1)[1].strip()
        if "<think" in text or "</think" in text:
            return None
    return text or None


def answer_expression(text: object) -> str:
    """Accept one expression, optionally boxed or introduced by an answer label."""
    if not isinstance(text, str) or not text.strip() or len(text) > 2048:
        raise ValueError("Missing or oversized mathematical answer")
    text = re.sub(
        r"^(?:the\s+)?(?:final\s+)?answer\s*(?::|is\b)\s*",
        "",
        text.strip(),
        flags=re.IGNORECASE,
    ).strip()
    if text.endswith("."):
        text = text[:-1].rstrip()
    for left, right in (("$$", "$$"), ("$", "$"), (r"\[", r"\]"), (r"\(", r"\)")):
        if text.startswith(left) and text.endswith(right):
            text = text[len(left) : -len(right)].strip()
            break
    if text.startswith(r"\boxed{") and text.endswith("}"):
        text = text[7:-1].strip()
    if not text or re.search(r"[\n\r$`:]|\\(?:boxed|text|mbox)\b", text):
        raise ValueError("Expected a single mathematical expression")
    prefix = ""
    if "=" in text and not re.search(r"[<>]=", text):
        if text.count("=") != 1 or not re.fullmatch(
            r"(?:[a-zA-Z]|\\[a-zA-Z]+)\s*", text.split("=")[0]
        ):
            raise ValueError("Only a single variable assignment is supported")
        left, text = text.split("=", 1)
        prefix = left.strip() + "="
        text = text.strip()
    if text.endswith("%"):
        number = text[:-1].removesuffix("\\").strip()
        if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", number):
            raise ValueError("Unsupported percentage expression")
        text = r"\frac{" + number + "}{100}"
    scientific = re.fullmatch(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))[eE]([+-]?\d+)", text)
    if scientific:
        text = scientific[1] + r"\times 10^{" + scientific[2] + "}"
    depth = 0
    for char in re.sub(r"\\[{}]", "", text):
        depth += (char == "{") - (char == "}")
        if depth < 0:
            raise ValueError("Unbalanced answer braces")
    if depth:
        raise ValueError("Unbalanced answer braces")
    words = re.findall(r"[a-zA-Z]{2,}", re.sub(r"\\[a-zA-Z]+", "", text))
    if any(
        word not in {"sin", "cos", "tan", "log", "ln", "exp", "sqrt", "pi"}
        for word in words
    ):
        raise ValueError("Textual or multiple candidate answers are unsupported")
    if not re.fullmatch(r"[\x20-\x7eπ∞°]+", text):
        raise ValueError("Unsupported mathematical answer characters")
    return prefix + text


def answer_type(value: Basic) -> type | None:
    """Classify values so equivalence never collapses equations or structured types."""
    if isinstance(value, Eq) and (
        not isinstance(value.lhs, Symbol) or not isinstance(value.rhs, Expr)
    ):
        return None
    if isinstance(value, And):
        return (
            And
            if all(
                isinstance(arg, Relational) and not isinstance(arg, Eq)
                for arg in value.args
            )
            else None
        )
    for kind in (Eq, Relational, Interval, FiniteSet, Tuple, Set, Expr):
        if isinstance(value, kind):
            return kind
    return None


def completion_expression(final: str | None) -> str:
    """Extract one balanced box from the final response, or a strict bare answer."""
    if not isinstance(final, str) or not final.strip():
        raise ValueError("Missing final response")
    boxes = list(re.finditer(r"\\boxed\b", final))
    if not boxes:
        return answer_expression(final)
    if len(boxes) != 1:
        raise ValueError("Multiple boxed answers are unsupported")
    balance = 0
    for match in re.finditer(r"\\[{}]|[{}]", final):
        balance += (match[0] == "{") - (match[0] == "}")
        if balance < 0:
            raise ValueError("Unbalanced final response braces")
    if balance:
        raise ValueError("Unbalanced final response braces")
    opening = re.match(r"\s*\{", final[boxes[0].end() :])
    if opening is None:
        raise ValueError("Malformed answer box")
    start = boxes[0].end() + opening.end()
    depth = 1
    # Escaped braces are set delimiters, not LaTeX grouping braces.
    for match in re.finditer(r"\\[{}]|[{}]", final[start:]):
        depth += (match[0] == "{") - (match[0] == "}")
        if depth == 0:
            return answer_expression(final[start : start + match.start()])
    raise ValueError("Unbalanced answer box")


@lru_cache(maxsize=8192)
def parse_expression(expression: str) -> tuple[Basic, ...]:
    # Supplying exactly one box prevents permissive extraction from arbitrary prose.
    result = parse(
        r"\boxed{" + expression + "}",
        extraction_config=[EXTRACTION],
        fallback_mode="no_fallback",
        extraction_mode="first_match",
    )
    if len(result) != 1:
        return ()
    value = result[0]
    kind = answer_type(value)
    if kind is None or value.has(S.NaN, S.ComplexInfinity):
        return ()
    if kind in (FiniteSet, Tuple) and any(not isinstance(item, Expr) for item in value):
        return ()
    return (value,)


def validate_reference_answers(ground_truth, prompt=None) -> list[Basic]:
    """Reject unsupported references before scoring, including empty completions."""
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("math-verify timeouts require the main thread")
    answers = [ground_truth] if isinstance(ground_truth, str) else ground_truth
    if not isinstance(answers, list) or not answers:
        raise ValueError(
            "ground_truth must be a nonempty string or list of answer strings"
        )
    if isinstance(prompt, list):
        if any(
            not isinstance(message, dict) or not isinstance(message.get("content"), str)
            for message in prompt
        ):
            raise ValueError("Prompt messages must contain string content")
        prompt = "\n".join(
            message["content"] for message in prompt if message.get("role") == "user"
        )
    if prompt is not None and not isinstance(prompt, str):
        raise TypeError("prompt must be a string or conversational prompt")
    parsed = []
    for answer in answers:
        try:
            expression = answer_expression(answer)
        except ValueError as error:
            raise ValueError(
                f"Invalid or unsupported math reference: {answer!r}"
            ) from error
        result = parse_expression(expression)
        if len(result) != 1:
            raise ValueError(f"Unparseable math reference: {answer!r}")
        if (
            isinstance(result[0], Symbol)
            and prompt
            and re.search(r"\bconstruct(?:ion)?\b", prompt, re.IGNORECASE)
        ):
            raise ValueError("A bare symbol does not specify a verifiable construction")
        parsed.append(result[0])
    return parsed


def math_reward(
    completion, ground_truth, *, reasoning_prefix=False, prompt=None
) -> float:
    references = validate_reference_answers(ground_truth, prompt)
    if not isinstance(reasoning_prefix, bool):
        raise TypeError("reasoning_prefix must be boolean")
    final = final_response(completion, reasoning_prefix)
    try:
        expression = completion_expression(final)
    except ValueError:
        return 0.0
    prediction = parse_expression(expression)
    if len(prediction) != 1:
        return 0.0
    for reference in references:
        gold, target = reference, prediction[0]
        # Allow scalar shorthand for constant assignments, while preserving two equations.
        if isinstance(gold, Eq) and answer_type(target) is Expr and gold.rhs.is_number:
            gold = gold.rhs
        elif (
            isinstance(target, Eq)
            and answer_type(gold) is Expr
            and target.rhs.is_number
        ):
            target = target.rhs
        # math-verify permits cross-type comparisons, including interval endpoints as sets.
        if answer_type(gold) is answer_type(target) and verify(
            gold, target, strict=True
        ):
            return 1.0
    return 0.0


def math_rewards(
    completions,
    ground_truth,
    reasoning_prefix=None,
    domain=None,
    prompts=None,
    **kwargs,
) -> list[float]:
    """TRL reward callable; prepared rows carry the reasoning-prefix flag."""
    count = len(completions)
    if len(ground_truth) != count:
        raise ValueError("Completions and ground truths must have equal lengths")
    if domain is not None and (
        len(domain) != count or any(value != "math" for value in domain)
    ):
        raise ValueError(
            "Math rewards require math rows; instruction-following rewards are disabled"
        )
    prefixes = [False] * count if reasoning_prefix is None else reasoning_prefix
    if len(prefixes) != count:
        raise ValueError(
            "Completions and reasoning-prefix flags must have equal lengths"
        )
    questions = [None] * count if prompts is None else prompts
    if len(questions) != count:
        raise ValueError("Completions and prompts must have equal lengths")
    return [
        math_reward(completion, reference, reasoning_prefix=prefix, prompt=question)
        for completion, reference, prefix, question in zip(
            completions, ground_truth, prefixes, questions, strict=True
        )
    ]
