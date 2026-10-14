from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


_CONFIDENCE_RANK = {
    "low": 0,
    "medium": 1,
    "medium_high": 2,
    "high": 3,
}


@dataclass(frozen=True)
class CalibrationAnchor:
    example_id: str
    anonymous_protocol_anchor: str
    item_number: int
    stem: str
    response: str
    stage_code: str
    stage_order: int
    stage_label: str
    confidence: str
    developmental_signature: str
    why_this_level: str
    why_not_adjacent_level: str


@dataclass(frozen=True)
class StageCalibration:
    stage_code: str
    stage_order: int
    stage_label: str
    core_organization: str
    typical_reasoning: list[str]
    avoid: list[str]
    adjacent_below: str | None
    adjacent_above: str | None
    anchors: list[CalibrationAnchor]
    lower_contrast_anchors: list[CalibrationAnchor]
    higher_contrast_anchors: list[CalibrationAnchor]


def load_stage_definitions(
    path: str | Path,
) -> dict[str, dict[str, Any]]:
    """
    Load source-aligned developmental stage definitions.

    Expected YAML structure:

    reference:
      ...

    stages:
      E2:
        order: 2
        label: Impulsive
        ...
      E3:
        ...
    """

    path = Path(path)

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        payload = yaml.safe_load(f)

    stages = payload.get(
        "stages",
        {},
    )

    if not stages:
        raise ValueError(
            f"No 'stages' mapping found in {path}"
        )

    return stages


def load_calibration_anchors(
    path: str | Path,
) -> list[CalibrationAnchor]:
    """
    Load calibration anchors from CSV.

    Public releases may use synthetic demonstration anchors; private
    research workflows may use appropriately governed human anchors.
    """

    path = Path(path)

    anchors: list[CalibrationAnchor] = []

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            anchors.append(
                CalibrationAnchor(
                    example_id=row[
                        "example_id"
                    ],
                    anonymous_protocol_anchor=row[
                        "anonymous_protocol_anchor"
                    ],
                    item_number=int(
                        row[
                            "item_number"
                        ]
                    ),
                    stem=row[
                        "stem"
                    ],
                    response=row[
                        "response_verbatim"
                    ],
                    stage_code=row[
                        "provisional_item_level"
                    ],
                    stage_order=int(
                        row[
                            "ego_stage_number"
                        ]
                    ),
                    stage_label=row[
                        "item_stage_label"
                    ],
                    confidence=row[
                        "item_level_confidence"
                    ],
                    developmental_signature=row[
                        "developmental_signature"
                    ],
                    why_this_level=row[
                        "why_this_level"
                    ],
                    why_not_adjacent_level=row[
                        "why_not_adjacent_level"
                    ],
                )
            )

    return anchors


def _confidence_rank(
    confidence: str,
) -> int:
    """
    Convert confidence labels to an ordinal rank.
    """

    return _CONFIDENCE_RANK.get(
        confidence,
        -1,
    )


def _ranked(
    anchors: list[CalibrationAnchor],
) -> list[CalibrationAnchor]:
    """
    Rank anchors primarily by confidence.

    Within equal confidence, keep ordering deterministic.
    """

    return sorted(
        anchors,
        key=lambda anchor: (
            -_confidence_rank(
                anchor.confidence
            ),
            anchor.anonymous_protocol_anchor,
            anchor.item_number,
            anchor.example_id,
        ),
    )


def _take_diverse(
    anchors: list[CalibrationAnchor],
    n: int,
) -> list[CalibrationAnchor]:
    """
    Select up to n anchors.

    Prefer examples from different human protocols before selecting
    multiple responses from the same protocol.

    This reduces the risk that the prompt learns one participant's
    biography, idiosyncratic vocabulary, or surface writing style.
    """

    if n <= 0:
        return []

    ranked = _ranked(
        anchors
    )

    chosen: list[
        CalibrationAnchor
    ] = []

    used_protocols: set[
        str
    ] = set()

    # First pass:
    # maximize protocol diversity.
    for anchor in ranked:

        if (
            anchor.anonymous_protocol_anchor
            in used_protocols
        ):
            continue

        chosen.append(
            anchor
        )

        used_protocols.add(
            anchor.anonymous_protocol_anchor
        )

        if len(chosen) >= n:
            return chosen

    # Second pass:
    # if we still need more examples,
    # allow repeated protocols.
    for anchor in ranked:

        if anchor in chosen:
            continue

        chosen.append(
            anchor
        )

        if len(chosen) >= n:
            break

    return chosen


def build_stage_calibration(
    *,
    stage_code: str,
    definitions: dict[
        str,
        dict[str, Any],
    ],
    anchors: list[
        CalibrationAnchor
    ],
    examples_per_stage: int = 4,
    contrast_examples_per_adjacent_stage: int = 1,
    min_confidence: str = "medium",
) -> StageCalibration:
    """
    Construct the calibration context for one requested stage.

    The context contains:

    1. source-aligned stage definition;
    2. positive human examples for the target stage;
    3. contrastive examples from the immediately lower stage;
    4. contrastive examples from the immediately higher stage.

    Exact stage identity is preserved throughout calibration.
    """

    if stage_code not in definitions:
        raise KeyError(
            f"No calibration definition "
            f"for stage {stage_code}"
        )

    if (
        min_confidence
        not in _CONFIDENCE_RANK
    ):
        raise ValueError(
            "Unknown confidence level: "
            f"{min_confidence}"
        )

    target_def = definitions[
        stage_code
    ]

    target_order = int(
        target_def[
            "order"
        ]
    )

    min_rank = (
        _CONFIDENCE_RANK[
            min_confidence
        ]
    )

    eligible = [
        anchor
        for anchor
        in anchors
        if (
            _confidence_rank(
                anchor.confidence
            )
            >= min_rank
        )
    ]

    target_anchors = (
        _take_diverse(
            [
                anchor
                for anchor
                in eligible
                if (
                    anchor.stage_code
                    == stage_code
                )
            ],
            examples_per_stage,
        )
    )

    lower_anchors = (
        _take_diverse(
            [
                anchor
                for anchor
                in eligible
                if (
                    anchor.stage_order
                    == target_order - 1
                )
            ],
            contrast_examples_per_adjacent_stage,
        )
    )

    higher_anchors = (
        _take_diverse(
            [
                anchor
                for anchor
                in eligible
                if (
                    anchor.stage_order
                    == target_order + 1
                )
            ],
            contrast_examples_per_adjacent_stage,
        )
    )

    adjacent = (
        target_def.get(
            "adjacent_contrast",
            {},
        )
        or {}
    )

    return StageCalibration(
        stage_code=stage_code,
        stage_order=target_order,
        stage_label=str(
            target_def[
                "label"
            ]
        ),
        core_organization=str(
            target_def[
                "core_organization"
            ]
        ).strip(),
        typical_reasoning=[
            str(item).strip()
            for item
            in target_def.get(
                "typical_reasoning",
                [],
            )
        ],
        avoid=[
            str(item).strip()
            for item
            in target_def.get(
                "avoid",
                [],
            )
        ],
        adjacent_below=(
            adjacent.get(
                "below"
            )
        ),
        adjacent_above=(
            adjacent.get(
                "above"
            )
        ),
        anchors=target_anchors,
        lower_contrast_anchors=(
            lower_anchors
        ),
        higher_contrast_anchors=(
            higher_anchors
        ),
    )


def _format_anchor(
    *,
    anchor: CalibrationAnchor,
    label: str,
) -> list[str]:
    """
    Format one calibration example.

    Kept as a helper so positive and contrastive examples
    have a consistent representation.
    """

    return [
        label,
        f'Stem: "{anchor.stem}"',
        f'Response: "{anchor.response}"',
    ]


def format_stage_calibration_for_prompt(
    calibration: StageCalibration,
) -> str:
    """
    Convert StageCalibration into a prompt-ready text block.

    Important design principle:

    Calibration examples should teach the model developmental structure,
    not participant identity or writing style.

    Contrast examples are described dynamically relative to the
    current target stage. We deliberately do NOT reuse the static
    CSV field `why_not_adjacent_level` as the boundary explanation,
    because that explanation may have been written relative to a
    different comparison.
    """

    lines: list[str] = []

    # ------------------------------------------------------------
    # TARGET DEFINITION
    # ------------------------------------------------------------

    lines.append(
        "SOURCE-ALIGNED DEVELOPMENTAL CALIBRATION"
    )

    lines.append("")

    lines.append(
        f"Target: "
        f"{calibration.stage_code} "
        f"— {calibration.stage_label}"
    )

    lines.append("")

    lines.append(
        "Core organization:"
    )

    lines.append(
        calibration.core_organization
    )

    # ------------------------------------------------------------
    # TYPICAL REASONING
    # ------------------------------------------------------------

    if (
        calibration.typical_reasoning
    ):

        lines.append("")

        lines.append(
            "Typical reasoning structure:"
        )

        for item in (
            calibration.typical_reasoning
        ):
            lines.append(
                f"- {item}"
            )

    # ------------------------------------------------------------
    # AVOID / FAILURE MODES
    # ------------------------------------------------------------

    if calibration.avoid:

        lines.append("")

        lines.append(
            "Do NOT collapse the target into:"
        )

        for item in (
            calibration.avoid
        ):
            lines.append(
                f"- {item}"
            )

    # ------------------------------------------------------------
    # ABSTRACT ADJACENT-STAGE BOUNDARY
    # ------------------------------------------------------------

    if (
        calibration.adjacent_below
        or calibration.adjacent_above
    ):

        lines.append("")

        lines.append(
            "Adjacent-stage boundary:"
        )

        if (
            calibration.adjacent_below
        ):
            lines.append(
                "- Lower neighbor: "
                f"{calibration.adjacent_below}"
            )

        if (
            calibration.adjacent_above
        ):
            lines.append(
                "- Higher neighbor: "
                f"{calibration.adjacent_above}"
            )

    # ------------------------------------------------------------
    # POSITIVE HUMAN EXAMPLES
    # ------------------------------------------------------------

    if calibration.anchors:

        lines.append("")

        lines.append(
            "CALIBRATION EXAMPLES"
        )

        lines.append("")

        lines.append(
            "Use these only to learn the organization "
            "of meaning-making."
        )

        lines.append(
            "Do not copy topic content, biography, "
            "grammar, spelling, verbosity, or wording."
        )

        for i, anchor in enumerate(
            calibration.anchors,
            start=1,
        ):

            lines.append("")

            lines.extend(
                _format_anchor(
                    anchor=anchor,
                    label=(
                        f"Target example {i}:"
                    ),
                )
            )

            lines.append(
                "Why it fits: "
                f"{anchor.why_this_level}"
            )

            lines.append(
                "Calibration confidence: "
                f"{anchor.confidence}"
            )

    # ------------------------------------------------------------
    # LOWER-STAGE CONTRAST
    # ------------------------------------------------------------

    if (
        calibration.lower_contrast_anchors
    ):

        lines.append("")

        lines.append(
            "LOWER-STAGE CONTRAST"
        )

        lines.append("")

        lines.append(
            "These examples come from the immediately "
            "lower developmental stage."
        )

        lines.append(
            "They are NOT examples of the target stage."
        )

        lines.append(
            "Use them to understand what is structurally "
            "LESS differentiated than the target."
        )

        for i, anchor in enumerate(
            calibration.lower_contrast_anchors,
            start=1,
        ):

            lines.append("")

            lines.extend(
                _format_anchor(
                    anchor=anchor,
                    label=(
                        f"Lower-neighbor example {i} "
                        f"({anchor.stage_code} "
                        f"— {anchor.stage_label}):"
                    ),
                )
            )

            lines.append(
                "Developmental signature of this "
                "lower-stage response: "
                f"{anchor.developmental_signature}"
            )

    # ------------------------------------------------------------
    # HIGHER-STAGE CONTRAST
    # ------------------------------------------------------------

    if (
        calibration.higher_contrast_anchors
    ):

        lines.append("")

        lines.append(
            "HIGHER-STAGE CONTRAST"
        )

        lines.append("")

        lines.append(
            "These examples come from the immediately "
            "higher developmental stage."
        )

        lines.append(
            "They are NOT examples of the target stage."
        )

        lines.append(
            "Use them to understand what is structurally "
            "MORE differentiated than the target."
        )

        for i, anchor in enumerate(
            calibration.higher_contrast_anchors,
            start=1,
        ):

            lines.append("")

            lines.extend(
                _format_anchor(
                    anchor=anchor,
                    label=(
                        f"Higher-neighbor example {i} "
                        f"({anchor.stage_code} "
                        f"— {anchor.stage_label}):"
                    ),
                )
            )

            lines.append(
                "Developmental signature of this "
                "higher-stage response: "
                f"{anchor.developmental_signature}"
            )

    # ------------------------------------------------------------
    # GLOBAL PROMPT SAFETY / CONSTRUCT RULE
    # ------------------------------------------------------------

    lines.append("")

    lines.append(
        "CRITICAL CONSTRUCT RULE"
    )

    lines.append("")

    lines.append(
        "Developmental stage concerns the organization "
        "of meaning-making."
    )

    lines.append(
        "It does NOT imply intelligence, education, "
        "language proficiency, demographic background, "
        "political orientation, or factual knowledge."
    )

    lines.append(
        "Do not make lower-stage respondents less articulate "
        "or grammatically weaker merely because of stage."
    )

    lines.append(
        "Do not make higher-stage respondents artificially "
        "verbose, philosophical, or abstract."
    )

    lines.append(
        "Preserve natural human variation while keeping "
        "the target developmental organization."
    )

    return "\n".join(
        lines
    ).strip()


def calibration_example_ids(
    calibration: StageCalibration,
) -> dict[str, list[str]]:
    """
    Return the exact calibration examples used for a stage.

    This should later be stored in the generated JSONL record so
    every synthetic sample is reproducible and auditable.
    """

    return {
        "target": [
            anchor.example_id
            for anchor
            in calibration.anchors
        ],
        "lower_contrast": [
            anchor.example_id
            for anchor
            in calibration.lower_contrast_anchors
        ],
        "higher_contrast": [
            anchor.example_id
            for anchor
            in calibration.higher_contrast_anchors
        ],
    }
