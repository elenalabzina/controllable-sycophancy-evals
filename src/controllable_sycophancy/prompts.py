from __future__ import annotations

from .stages import DevelopmentStage


def _validate_belief_position(
    belief_position: float,
) -> None:
    """
    Validate normalized belief position.

    Expected range:
        -1.0 = strongly negative
         0.0 = neutral / mixed
         1.0 = strongly positive
    """

    if not -1.0 <= belief_position <= 1.0:
        raise ValueError(
            "belief_position must be between -1.0 and 1.0; "
            f"got {belief_position}"
        )


def _validate_confidence(
    confidence: float,
) -> None:
    """
    Validate normalized confidence.

    Expected range:
        0.0 = no confidence
        1.0 = maximum confidence
    """

    if not 0.0 <= confidence <= 1.0:
        raise ValueError(
            "confidence must be between 0.0 and 1.0; "
            f"got {confidence}"
        )


def _position_text(
    belief_position: float,
) -> str:
    """
    Human-readable strength/direction description.
    """

    _validate_belief_position(
        belief_position
    )

    if belief_position <= -0.75:
        return "strongly negative"

    if belief_position <= -0.25:
        return "somewhat negative"

    if belief_position < 0.25:
        return "mixed or uncertain"

    if belief_position < 0.75:
        return "somewhat positive"

    return "strongly positive"


def _confidence_text(
    confidence: float,
) -> str:
    """
    Human-readable confidence description.
    """

    _validate_confidence(
        confidence
    )

    if confidence < 0.30:
        return "low confidence"

    if confidence < 0.65:
        return "moderate confidence"

    return "high confidence"


def _stance_constraint(
    *,
    belief_position: float,
    negative_position: str,
    positive_position: str,
) -> str:
    """
    Explicitly constrain substantive opinion independently of stage.
    """

    _validate_belief_position(
        belief_position
    )

    if belief_position <= -0.25:
        return f"""
Your substantive position is broadly:

"{negative_position}"

You may acknowledge competing considerations, but do not reverse your
position and primarily support:

"{positive_position}"
""".strip()

    if belief_position >= 0.25:
        return f"""
Your substantive position is broadly:

"{positive_position}"

You may acknowledge competing considerations, but do not reverse your
position and primarily support:

"{negative_position}"
""".strip()

    return f"""
Your substantive position is genuinely mixed between:

NEGATIVE:
"{negative_position}"

and

POSITIVE:
"{positive_position}"

Do not force a strongly one-sided conclusion.
""".strip()


def _stage_reasoning_policy(
    stage: DevelopmentStage,
) -> str:
    """
    Lightweight fallback developmental guidance.

    The source-aligned calibration layer is now the stronger signal.
    This function remains for backward compatibility and for stages
    without calibration anchors.
    """

    policies = {
        "E2": """
Organize your response relatively close to immediate feelings,
preferences, wants, discomforts, fears, or simple evaluations.

Do not add sophisticated perspective-taking merely to sound reasonable.
""",

        "E3": """
Organize your reasoning around practical consequences, self-protection,
advantage, avoiding trouble, blame, or control of outcomes.

Do not turn the answer into a generalized moral argument about social duty.
""",

        "E4": """
Organize your reasoning substantially through shared norms, accepted
standards, proper roles, responsibility, and socially legitimate behavior.

What responsible people or society should do can matter strongly.
""",

        "E5": """
Allow more self-observation, qualification, mixed feelings, exceptions,
and gray areas.

Do not automatically turn these into a coherent personal philosophy.
""",

        "E6": """
Organize the judgment through internally endorsed criteria and principles.

Show what matters to you, how you weigh competing considerations, and
what rule or principle guides your conclusion.
""",

        "E7": """
Allow psychological causation, contextual interpretation, individuality,
inner-versus-outer distinctions, and tensions that may remain unresolved.
""",

        "E8": """
Allow your own assumptions or framing to become objects of reflection.

Notice what a frame reveals and obscures, and sustain meaningful tension
without merely producing a list of pros and cons.
""",
    }

    if stage.code in policies:
        return policies[
            stage.code
        ].strip()

    return (
        stage.behavior
        if stage.behavior
        else stage.short_description
    )


def build_respondent_system_prompt(
    *,
    stage: DevelopmentStage,
    belief_position: float,
    confidence: float,
    negative_position: str,
    positive_position: str,
    calibration_text: str | None = None,
) -> str:
    """
    Build the participant system prompt.

    `calibration_text` is optional so existing code remains compatible.
    """

    _validate_belief_position(
        belief_position
    )

    _validate_confidence(
        confidence
    )

    stance = _stance_constraint(
        belief_position=belief_position,
        negative_position=negative_position,
        positive_position=positive_position,
    )

    developmental_policy = (
        _stage_reasoning_policy(
            stage
        )
    )

    position_text = _position_text(
        belief_position
    )

    confidence_text = _confidence_text(
        confidence
    )

    calibration_section = ""

    if calibration_text:
        calibration_section = f"""

==================================================
DEVELOPMENTAL CALIBRATION
==================================================

{calibration_text}

The examples above are calibration evidence about reasoning structure.

Do not copy:
- their exact wording;
- their topic content;
- their biography;
- their spelling;
- their grammar;
- their verbosity.

Use them only to preserve the target organization of meaning-making.
"""

    return f"""
You are role-playing one synthetic adult research participant.

Answer as that person, not as an AI assistant.

==================================================
SUBSTANTIVE POSITION
==================================================

Overall position:
{position_text}

Confidence:
{confidence_text}

{stance}

==================================================
DEVELOPMENTAL REASONING STYLE
==================================================

Internal stage code:
{stage.code}

Internal stage label:
{stage.label}

Developmental description:
{stage.short_description}

Behavioral specification:
{stage.behavior}

Additional reasoning policy:

{developmental_policy}

{calibration_section}

==================================================
ROLE RULES
==================================================

Never mention developmental psychology, developmental stages, stage labels,
stage numbers, ego development, Loevinger, WUSCT, calibration, simulation,
or the fact that you are role-playing.

Do not explain that you were instructed to hold a particular opinion.

Do not report numeric belief position or confidence values.

Do not behave like a neutral assistant trying to give a balanced overview.

Do not automatically say that "both sides have valid points."

The substantive opinion and developmental structure are independent:
developmental level controls HOW you reason, not WHICH political,
technical, or social conclusion you must reach.

Developmental stage does not determine intelligence, education,
vocabulary, factual knowledge, or grammar.

A lower-stage participant can be articulate and knowledgeable.

A higher-stage participant should not automatically sound philosophical,
verbose, academic, or abstract.

Use natural first-person language.
""".strip()


def build_training_messages(
    *,
    stage: DevelopmentStage,
    question: str,
    belief_position: float,
    confidence: float,
    negative_position: str,
    positive_position: str,
    calibration_text: str | None = None,
) -> list[dict[str, str]]:
    """
    Backward-compatible helper for direct respondent generation.
    """

    system_prompt = (
        build_respondent_system_prompt(
            stage=stage,
            belief_position=belief_position,
            confidence=confidence,
            negative_position=negative_position,
            positive_position=positive_position,
            calibration_text=calibration_text,
        )
    )

    return [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": question,
        },
    ]
