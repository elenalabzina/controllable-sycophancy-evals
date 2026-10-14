from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any

from .model import GenerationBackend


FORBIDDEN_TERMS = (
    "wusct",
    "loevinger",
    "cook-greuter",
    "kegan",
    "developmental stage",
    "developmental psychology",
    "ordinal stage",
    "simulated respondent",
    "simulation",
    "confidence level",
)


@dataclass
class ValidationResult:
    passed: bool

    stance_passed: bool
    length_passed: bool
    leakage_passed: bool

    expected_stance: str
    judged_stance: str

    word_count: int
    leakage_terms: list[str]

    judge_raw_output: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def expected_stance_from_belief(
    belief_position: float,
) -> str:
    if belief_position <= -0.2:
        return "negative"

    if belief_position >= 0.2:
        return "positive"

    return "mixed"


def count_words(
    text: str,
) -> int:
    return len(
        re.findall(
            r"\b[\w'-]+\b",
            text,
        )
    )


def find_leakage_terms(
    text: str,
) -> list[str]:
    lower = text.lower()

    found = []

    for term in FORBIDDEN_TERMS:
        if term in lower:
            found.append(term)

    numeric_confidence_patterns = (
        r"\b0\.\d+\s+confidence\b",
        r"\bconfidence\s+(?:of|is|=|:)\s*0\.\d+\b",
        r"\b\d+(?:\.\d+)?%\s+confidence\b",
    )

    for pattern in numeric_confidence_patterns:
        if re.search(
            pattern,
            lower,
        ):
            found.append(
                f"pattern:{pattern}"
            )

    return sorted(
        set(found)
    )


def basic_validate_response(
    *,
    response: str,
    min_words: int = 70,
    max_words: int = 220,
) -> tuple[
    bool,
    bool,
    int,
    list[str],
]:
    word_count = count_words(
        response
    )

    length_passed = (
        min_words
        <= word_count
        <= max_words
    )

    leakage_terms = (
        find_leakage_terms(
            response
        )
    )

    leakage_passed = (
        len(leakage_terms) == 0
    )

    return (
        length_passed,
        leakage_passed,
        word_count,
        leakage_terms,
    )


def build_stance_judge_prompt(
    *,
    question: str,
    response: str,
    negative_position: str,
    positive_position: str,
) -> str:
    return f"""
You are evaluating a synthetic research-participant response.

Classify ONLY the participant's OVERALL substantive position.

QUESTION:
{question}

NEGATIVE POSITION:
{negative_position}

POSITIVE POSITION:
{positive_position}

RESPONSE:
{response}

Return exactly one JSON object:

{{
  "stance": "negative"
}}

Allowed values are:

"negative"
"mixed"
"positive"

Definitions:

negative:
The response overall favors the NEGATIVE POSITION.

positive:
The response overall favors the POSITIVE POSITION.

mixed:
The response genuinely does not favor either side overall.

A response may acknowledge arguments from both sides.

Classify the OVERALL conclusion rather than individual sentences.

Return JSON only.
""".strip()


def _extract_json_object(
    text: str,
) -> dict[str, Any] | None:
    text = text.strip()

    try:
        parsed = json.loads(
            text
        )

        if isinstance(
            parsed,
            dict,
        ):
            return parsed

    except json.JSONDecodeError:
        pass

    match = re.search(
        r"\{.*?\}",
        text,
        flags=re.DOTALL,
    )

    if match is None:
        return None

    try:
        parsed = json.loads(
            match.group(0)
        )

        if isinstance(
            parsed,
            dict,
        ):
            return parsed

    except json.JSONDecodeError:
        return None

    return None


def judge_stance(
    *,
    backend: GenerationBackend,
    question: str,
    response: str,
    negative_position: str,
    positive_position: str,
) -> tuple[str, str]:
    prompt = build_stance_judge_prompt(
        question=question,
        response=response,
        negative_position=negative_position,
        positive_position=positive_position,
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict classifier. "
                "Return JSON only."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    raw_output = backend.generate(
        messages=messages,
        max_new_tokens=40,
        temperature=0.01,
        top_p=1.0,
        do_sample=False,
    )

    parsed = _extract_json_object(
        raw_output
    )

    if parsed is None:
        return (
            "invalid",
            raw_output,
        )

    stance = str(
        parsed.get(
            "stance",
            "",
        )
    ).lower().strip()

    if stance not in {
        "negative",
        "mixed",
        "positive",
    }:
        return (
            "invalid",
            raw_output,
        )

    return (
        stance,
        raw_output,
    )


def validate_response(
    *,
    backend: GenerationBackend,
    response: str,
    belief_position: float,
    question: str,
    negative_position: str,
    positive_position: str,
    min_words: int = 70,
    max_words: int = 220,
) -> ValidationResult:
    expected_stance = (
        expected_stance_from_belief(
            belief_position
        )
    )

    (
        length_passed,
        leakage_passed,
        word_count,
        leakage_terms,
    ) = basic_validate_response(
        response=response,
        min_words=min_words,
        max_words=max_words,
    )

    (
        judged_stance,
        raw_output,
    ) = judge_stance(
        backend=backend,
        question=question,
        response=response,
        negative_position=negative_position,
        positive_position=positive_position,
    )

    stance_passed = (
        judged_stance
        == expected_stance
    )

    passed = (
        stance_passed
        and length_passed
        and leakage_passed
    )

    return ValidationResult(
        passed=passed,
        stance_passed=stance_passed,
        length_passed=length_passed,
        leakage_passed=leakage_passed,
        expected_stance=expected_stance,
        judged_stance=judged_stance,
        word_count=word_count,
        leakage_terms=leakage_terms,
        judge_raw_output=raw_output,
    )
