from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .model import GenerationBackend
from .stages import DevelopmentStage


@dataclass
class ReasoningPlanResult:
    valid: bool
    plan: dict[str, str] | None
    raw_output: str
    error: str | None = None


# ---------------------------------------------------------------------
# STAGE-SPECIFIC REASONING SCHEMAS
# ---------------------------------------------------------------------

STAGE_REASONING_SCHEMAS: dict[
    str,
    dict[str, str],
] = {
    "E2": {
        "immediate_reaction": (
            "The respondent's immediate emotional, bodily, or intuitive "
            "reaction to the issue."
        ),
        "main_want_or_feeling": (
            "What the respondent most immediately wants, likes, dislikes, "
            "fears, or feels."
        ),
        "simple_evaluation": (
            "A simple good/bad, helpful/harmful, fair/unfair, or "
            "comfortable/uncomfortable evaluation."
        ),
        "conclusion": (
            "The respondent's final position, consistent with the assigned "
            "belief direction."
        ),
    },

    "E3": {
        "self_interest_or_threat": (
            "What the respondent sees as personally advantageous, risky, "
            "threatening, unfair, blame-producing, or protective of their "
            "position."
        ),
        "consequence_or_advantage": (
            "The practical consequence, payoff, loss, advantage, or "
            "disadvantage that matters most."
        ),
        "instrumental_strategy": (
            "The practical rule, strategy, control mechanism, or arrangement "
            "that would help the respondent obtain the desired outcome or "
            "avoid the unwanted one."
        ),
        "other_people_or_authority": (
            "How other people, managers, rules, or authority affect the "
            "respondent mainly through consequences, accountability, "
            "approval, punishment, blame, or control."
        ),
        "conclusion": (
            "The respondent's final position, consistent with the assigned "
            "belief direction."
        ),
    },

    "E4": {
        "shared_norm": (
            "The socially recognizable norm, shared value, duty, or "
            "expectation that organizes the respondent's view."
        ),
        "trusted_standard_or_expectation": (
            "The accepted social, institutional, professional, moral, or "
            "community standard that provides guidance."
        ),
        "responsible_behavior": (
            "What a responsible, reasonable, good, or proper person, "
            "organization, or society is expected to do."
        ),
        "other_view": (
            "A competing view that the respondent can recognize without "
            "fully entering its independent logic."
        ),
        "how_disagreement_is_resolved": (
            "How disagreement is resolved through accepted norms, shared "
            "expectations, trusted standards, institutional legitimacy, or "
            "socially recognized responsibility."
        ),
        "conclusion": (
            "The respondent's final position, consistent with the assigned "
            "belief direction."
        ),
    },

    "E5": {
        "initial_reaction": (
            "The respondent's initial position or reaction."
        ),
        "second_reaction_or_qualification": (
            "A second feeling, qualification, complication, or recognition "
            "that the first reaction is incomplete."
        ),
        "context_or_exception": (
            "A context, exception, individual difference, or circumstance "
            "that changes how the issue looks."
        ),
        "self_observation": (
            "Something the respondent notices about their own feelings, "
            "preferences, reactions, or uncertainty, without requiring a "
            "fully developed self-authored framework."
        ),
        "remaining_ambiguity": (
            "A genuine uncertainty, mixed feeling, or unresolved tension "
            "that remains."
        ),
        "conclusion": (
            "The respondent's qualified final position, still consistent "
            "with the assigned belief direction."
        ),
    },

    "E6": {
        "primary_criterion": (
            "The respondent's main personally endorsed criterion or principle "
            "for evaluating the issue."
        ),
        "secondary_criteria": (
            "Other personally meaningful criteria that also matter but are "
            "not necessarily decisive."
        ),
        "competing_consideration": (
            "A legitimate consideration that points in a different direction "
            "from the respondent's preferred conclusion."
        ),
        "tradeoff": (
            "How the respondent weighs the competing considerations against "
            "their own criteria."
        ),
        "decision_rule": (
            "The respondent's internally endorsed way of deciding which "
            "consideration should receive greater weight."
        ),
        "responsibility": (
            "How the respondent understands their own responsibility for "
            "making and owning the judgment."
        ),
        "conclusion": (
            "The respondent's final position, derived from their own "
            "criteria and consistent with the assigned belief direction."
        ),
    },

    "E7": {
        "personal_position": (
            "The respondent's own position on the issue. It should be "
            "personally owned rather than merely borrowed from convention, "
            "authority, or social expectation."
        ),
        "personally_owned_criterion": (
            "State the respondent's own evaluative basis for judgment. "
            "It must be internally endorsed and independently organized, "
            "rather than merely a practical preference, an instrumental "
            "goal such as productivity or convenience, or a socially "
            "supplied rule. It may be contextual and need not be expressed "
            "as a rigid principle or hierarchy."
        ),
        "psychological_or_contextual_cause": (
            "A psychological, interpersonal, historical, situational, or "
            "contextual process that helps explain why different people may "
            "experience or interpret the issue differently."
        ),
        "inner_outer_distinction": (
            "A distinction between external behavior or circumstances and "
            "internal motives, identity, experience, meaning, or needs."
        ),
        "alternative_perspective": (
            "A genuinely different perspective represented through its own "
            "logic, motives, or values rather than as a simple objection."
        ),
        "individuality_or_authenticity": (
            "How individuality, authenticity, personal history, identity, "
            "or different legitimate ways of functioning matter to the "
            "respondent's understanding."
        ),
        "unresolved_conflict": (
            "A meaningful tension between legitimate considerations that "
            "does not disappear simply because the respondent reaches a "
            "position."
        ),
        "conclusion": (
            "The respondent's final position. It should remain grounded in "
            "the personally owned criterion while acknowledging contextual "
            "complexity and unresolved tension."
        ),
    },

    "E8": {
        "personally_owned_criterion": (
            "State the respondent's independently owned evaluative basis "
            "for judgment before examining its limits. The criterion must "
            "express a broader personally endorsed value or way of judging, "
            "not merely an organizational, instrumental, or performance "
            "criterion. Productivity, efficiency, accountability, cohesion, "
            "shared purpose, structure, flexibility, safety, convenience, "
            "or collective momentum do NOT by themselves qualify. If such "
            "considerations matter, explain the deeper personally owned "
            "value through which they matter—for example autonomy, dignity, "
            "meaningful participation, integrity, reciprocity, human "
            "connection, responsibility, or another independently endorsed "
            "basis. The criterion may be contextual, provisional, and "
            "revisable."
        ),
        "initial_frame": (
            "The respondent's initial way of framing or organizing the "
            "problem."
        ),
        "assumption_inside_initial_frame": (
            "An assumption embedded in the respondent's initial frame that "
            "could shape what they notice, value, or conclude."
        ),
        "alternative_frame": (
            "A genuinely different way of framing the issue, not merely the "
            "opposite conclusion."
        ),
        "what_initial_frame_reveals": (
            "What becomes visible, important, or compelling from the "
            "respondent's initial frame."
        ),
        "what_initial_frame_hides": (
            "What the initial frame may obscure, neglect, or make difficult "
            "to notice."
        ),
        "what_alternative_frame_reveals": (
            "What becomes visible or important under the alternative frame."
        ),
        "reciprocal_or_complex_causation": (
            "A description of reciprocal, circular, systemic, or "
            "multi-directional causation rather than a single simple causal "
            "story."
        ),
        "unresolved_tension": (
            "A meaningful tension that remains legitimate and unresolved "
            "even after the respondent forms a position."
        ),
        "relation_to_own_frame": (
            "How the respondent relates to their own frame and criterion as "
            "partial, contextual, assumption-dependent, or revisable, "
            "without abandoning their capacity to judge."
        ),
        "conclusion": (
            "The respondent's final position. It should remain consistent "
            "with the assigned belief direction while preserving both an "
            "independently owned evaluative standpoint and reflection on "
            "the limitations of that standpoint."
        ),
    },
}


# ---------------------------------------------------------------------
# JSON UTILITIES
# ---------------------------------------------------------------------

def _extract_json_object(
    text: str,
) -> dict[str, Any] | None:
    """
    Extract the first plausible JSON object from model output.
    """

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
        r"\{.*\}",
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


# ---------------------------------------------------------------------
# STANCE HELPERS
# ---------------------------------------------------------------------

def _stance_description(
    *,
    belief_position: float,
    negative_position: str,
    positive_position: str,
) -> str:
    """
    Convert the numeric belief direction into a planning constraint.
    """

    if belief_position < 0:
        return (
            "The respondent's required position is NEGATIVE / CRITICAL.\n\n"
            "The reasoning plan and final conclusion must remain broadly "
            "consistent with this position:\n\n"
            f'"{negative_position}"'
        )

    if belief_position > 0:
        return (
            "The respondent's required position is POSITIVE / SUPPORTIVE.\n\n"
            "The reasoning plan and final conclusion must remain broadly "
            "consistent with this position:\n\n"
            f'"{positive_position}"'
        )

    return (
        "The respondent's position is genuinely MIXED / UNCERTAIN.\n\n"
        "The reasoning plan should preserve meaningful uncertainty rather "
        "than forcing either supplied position."
    )


# ---------------------------------------------------------------------
# SCHEMA FORMATTING
# ---------------------------------------------------------------------

def _format_schema_for_prompt(
    stage_code: str,
) -> str:
    """
    Render the required reasoning-plan fields for one stage.
    """

    if stage_code not in STAGE_REASONING_SCHEMAS:
        raise KeyError(
            f"No reasoning-plan schema for stage {stage_code}"
        )

    schema = STAGE_REASONING_SCHEMAS[
        stage_code
    ]

    lines = []

    for field, description in schema.items():
        lines.append(
            f'- "{field}": {description}'
        )

    return "\n".join(
        lines
    )


def _json_template_for_stage(
    stage_code: str,
) -> str:
    """
    Build an explicit JSON template for the requested stage.
    """

    schema = STAGE_REASONING_SCHEMAS[
        stage_code
    ]

    template = {
        field: "..."
        for field in schema
    }

    return json.dumps(
        template,
        indent=2,
        ensure_ascii=False,
    )


# ---------------------------------------------------------------------
# PLANNING PROMPT
# ---------------------------------------------------------------------

def build_reasoning_plan_prompt(
    *,
    stage: DevelopmentStage,
    question: str,
    belief_position: float,
    negative_position: str,
    positive_position: str,
    calibration_text: str | None = None,
) -> str:
    """
    Build the hidden reasoning-plan prompt.

    The plan operationalizes developmental structure before natural-language
    realization. It is never shown to the synthetic participant.
    """

    schema_text = _format_schema_for_prompt(
        stage.code
    )

    json_template = _json_template_for_stage(
        stage.code
    )

    stance_text = _stance_description(
        belief_position=belief_position,
        negative_position=negative_position,
        positive_position=positive_position,
    )

    calibration_section = ""

    if calibration_text:
        calibration_section = f"""
==================================================
DEVELOPMENTAL CALIBRATION
==================================================

{calibration_text}

Use the calibration examples to understand the ORGANIZATION OF MEANING.

Do not copy:
- wording;
- biography;
- grammar;
- topic content;
- idiosyncratic phrases.

The current topic is different. Transfer only the developmental structure.
"""

    stage_specific_addition = ""

    if stage.code == "E7":
        stage_specific_addition = """
==================================================
CUMULATIVE DEVELOPMENTAL REQUIREMENT
==================================================

E7 must PRESERVE self-authorship while adding contextual and psychological
complexity.

Do not replace an independently owned evaluative standpoint with generic
balance, relativism, or a list of instrumental considerations.

The field "personally_owned_criterion" is REQUIRED.

It should express a basis for judgment that belongs to the respondent and is
not merely:

- productivity;
- efficiency;
- convenience;
- avoiding blame;
- maintaining order;
- following expectations;
- maximizing a simple outcome.

Those considerations may appear, but the plan must show how they are
evaluated from a personally owned standpoint.

The criterion does NOT need to be rigid, universal, or expressed as a slogan.

E7 can retain unresolved conflict and recognize contextual variation while
still possessing a standpoint of its own.
"""

    elif stage.code == "E8":
        stage_specific_addition = """
==================================================
CUMULATIVE DEVELOPMENTAL REQUIREMENT
==================================================

E8 must PRESERVE self-authorship while making the respondent's own frame
available for reflection.

Do not generate frame-reflection by deleting the respondent's independently
owned evaluative standpoint.

The field "personally_owned_criterion" is REQUIRED and should be established
before the plan examines its assumptions or limits.

CRITICAL:
The personally owned criterion must NOT merely restate an organizational,
instrumental, or performance objective.

The following do NOT by themselves count as the owned evaluative basis:

- productivity;
- efficiency;
- accountability;
- organizational cohesion;
- shared purpose;
- collective momentum;
- structure;
- flexibility;
- convenience;
- safety as a bare outcome;
- maintaining order;
- institutional expectations;
- achieving organizational goals.

These may appear as consequences or considerations, but the plan must identify
the deeper personally endorsed value through which they matter.

For example:

BAD:
"I value accountability and productivity."

BETTER:
"I care about whether people can participate in work in a way that preserves
both meaningful responsibility to others and genuine autonomy."

BAD:
"I value organizational cohesion and shared purpose."

BETTER:
"I care about whether work lets people experience themselves as genuine
participants in something shared without requiring them to surrender their
individual ways of functioning."

BAD:
"I value structure because it improves performance."

BETTER:
"I care about whether people can rely on one another while still being able
to exercise judgment for themselves."

The criterion does NOT need to sound philosophical or abstract.

It should simply reveal a personally owned basis for deciding what ultimately
matters, rather than treating an instrumental outcome as the final value.

After establishing this standpoint, the later E8 fields should examine how
that standpoint may itself be:

- partial;
- contextual;
- assumption-dependent;
- limited;
- revisable;
- shaped by the respondent's own experience.

E8 should therefore combine:

1. capacity to judge from an independently owned standpoint;
2. capacity to examine the assumptions and limitations of that standpoint.

Reflection must not collapse into indecision or generic "both sides" language.
"""

    return f"""
You are constructing a HIDDEN reasoning plan for a synthetic human research
participant.

The plan will later be converted into natural first-person prose.

Do NOT write the final response yet.

==================================================
TASK
==================================================

Question:

{question}

{stance_text}

==================================================
DEVELOPMENTAL CONDITION
==================================================

Internal stage code: {stage.code}
Internal stage label: {stage.label}

The developmental stage should affect the STRUCTURE of reasoning, not factual
knowledge, vocabulary, political orientation, intelligence, or verbosity.

Do not caricature the stage.

Do not make later stages automatically:
- more moderate;
- more agreeable;
- more uncertain;
- more politically liberal;
- more factually knowledgeable;
- less confident.

{calibration_section}

==================================================
REQUIRED REASONING-PLAN FIELDS
==================================================

Produce every field below.

{schema_text}

{stage_specific_addition}

==================================================
GENERAL RULES
==================================================

- Preserve the required belief direction.
- Acknowledge counter-considerations only in ways compatible with the
  developmental structure.
- Do not reverse the assigned position.
- Do not mention developmental psychology.
- Do not mention stage labels or stage numbers in field values.
- Do not mention WUSCT, Loevinger, Cook-Greuter, Kegan, simulation, prompts,
  hidden variables, or research instructions.
- Do not make the plan sound like an AI policy analysis.
- Use ordinary substantive reasoning that could plausibly underlie a human
  response.
- Each field must contain meaningful content.
- Do not put multiple alternative plans inside one field.
- Do not omit required fields.
- Do not add fields.

For E7 and E8 specifically:
- preserve an independently owned evaluative standpoint;
- do not substitute mere complexity for self-authorship;
- do not substitute instrumental criteria for an owned evaluative basis;
- later reflection may qualify the criterion but should not erase it.

==================================================
OUTPUT FORMAT
==================================================

Return valid JSON only.

Use exactly this object structure:

{json_template}
""".strip()


# ---------------------------------------------------------------------
# PLAN VALIDATION
# ---------------------------------------------------------------------

def _validate_reasoning_plan(
    *,
    stage_code: str,
    payload: dict[str, Any],
) -> tuple[
    dict[str, str] | None,
    str | None,
]:
    """
    Validate that all and only the required fields are present and usable.
    """

    if stage_code not in STAGE_REASONING_SCHEMAS:
        return (
            None,
            f"Unknown stage code: {stage_code}",
        )

    expected_fields = list(
        STAGE_REASONING_SCHEMAS[
            stage_code
        ].keys()
    )

    missing = [
        field
        for field in expected_fields
        if field not in payload
    ]

    if missing:
        return (
            None,
            (
                "Missing required reasoning-plan "
                f"fields: {missing}"
            ),
        )

    extra = [
        field
        for field in payload
        if field not in expected_fields
    ]

    if extra:
        return (
            None,
            (
                "Unexpected reasoning-plan "
                f"fields: {extra}"
            ),
        )

    normalized: dict[
        str,
        str,
    ] = {}

    for field in expected_fields:
        value = payload.get(
            field
        )

        if not isinstance(
            value,
            str,
        ):
            return (
                None,
                (
                    f"Reasoning-plan field "
                    f"'{field}' is not a string."
                ),
            )

        value = value.strip()

        if not value:
            return (
                None,
                (
                    f"Reasoning-plan field "
                    f"'{field}' is empty."
                ),
            )

        normalized[
            field
        ] = value

    return (
        normalized,
        None,
    )


# ---------------------------------------------------------------------
# PLAN GENERATION
# ---------------------------------------------------------------------

def generate_reasoning_plan(
    *,
    backend: GenerationBackend,
    stage: DevelopmentStage,
    question: str,
    belief_position: float,
    negative_position: str,
    positive_position: str,
    calibration_text: str | None = None,
) -> ReasoningPlanResult:
    """
    Generate and validate one hidden developmental reasoning plan.
    """

    prompt = build_reasoning_plan_prompt(
        stage=stage,
        question=question,
        belief_position=belief_position,
        negative_position=negative_position,
        positive_position=positive_position,
        calibration_text=calibration_text,
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You generate structured hidden reasoning plans for a "
                "synthetic research-participant simulator. "
                "Follow the requested developmental structure exactly. "
                "Return valid JSON only."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    raw_output = backend.generate(
        messages=messages,
        max_new_tokens=700,
        temperature=0.45,
        top_p=0.85,
        do_sample=True,
    )

    payload = _extract_json_object(
        raw_output
    )

    if payload is None:
        return ReasoningPlanResult(
            valid=False,
            plan=None,
            raw_output=raw_output,
            error=(
                "Could not parse reasoning plan "
                "as a JSON object."
            ),
        )

    plan, error = _validate_reasoning_plan(
        stage_code=stage.code,
        payload=payload,
    )

    if plan is None:
        return ReasoningPlanResult(
            valid=False,
            plan=None,
            raw_output=raw_output,
            error=error,
        )

    return ReasoningPlanResult(
        valid=True,
        plan=plan,
        raw_output=raw_output,
        error=None,
    )


# ---------------------------------------------------------------------
# NATURAL-LANGUAGE REALIZATION
# ---------------------------------------------------------------------

def build_realization_messages(
    *,
    participant_system_prompt: str,
    question: str,
    reasoning_plan: dict[str, str],
    calibration_text: str | None = None,
) -> list[
    dict[str, str]
]:
    """
    Build messages that convert a hidden structured plan into natural
    first-person participant prose.

    The plan controls reasoning structure but must not be exposed.
    """

    reasoning_plan_text = json.dumps(
        reasoning_plan,
        indent=2,
        ensure_ascii=False,
    )

    calibration_section = ""

    if calibration_text:
        calibration_section = f"""
==================================================
CALIBRATION REMINDER
==================================================

The developmental calibration below is provided only to preserve the
organization of meaning during realization.

Do not copy its wording, biography, grammar, or topic content.

{calibration_text}
"""

    cumulative_instruction = ""

    if "personally_owned_criterion" in reasoning_plan:
        cumulative_instruction = """
==================================================
IMPORTANT CUMULATIVE-STRUCTURE RULE
==================================================

The hidden plan contains a "personally_owned_criterion".

Preserve this element in the natural response.

Do not turn it into:
- a generic preference;
- a simple practical goal;
- a list of pros and cons;
- generic balance;
- a socially supplied rule.

The final prose does not need to explicitly say "my criterion is..." or use
formal philosophical language.

It should nevertheless make visible that the respondent has a personally
owned basis for judgment.

If the plan also reflects on limitations, assumptions, context, or alternative
frames, preserve BOTH:

1. the respondent's capacity to judge from their own standpoint;
2. the respondent's capacity to qualify or examine that standpoint.

Do not let reflection erase authorship.

If the hidden criterion contains organizational or instrumental terms such as
productivity, accountability, cohesion, structure, flexibility, shared
purpose, or efficiency, do not present those terms as the ultimate basis of
judgment by themselves.

Preserve the deeper personally owned value that explains WHY those outcomes
matter to the respondent.
"""

    e8_reflection_instruction = ""

    if (
        "initial_frame" in reasoning_plan
        and "assumption_inside_initial_frame" in reasoning_plan
        and "what_initial_frame_hides" in reasoning_plan
        and "relation_to_own_frame" in reasoning_plan
    ):
        e8_reflection_instruction = """
==================================================
IMPORTANT FRAME-REFLECTION RULE
==================================================

This plan contains explicit frame-reflective structure.

The final response MUST preserve that structure in natural prose.

At some point in the response, make visible ALL THREE of the following:

1. the respondent recognizes something about how they are initially framing
   the issue;

2. the respondent identifies or questions at least one assumption inside
   that frame;

3. the respondent recognizes something that the assumption or frame may
   cause them to overlook, underweight, or misunderstand.

Do not merely write:

- "I know other people disagree";
- "my view may be influenced by my experience";
- "my perspective is not the only valid one";
- "I could be wrong";
- "there are different ways to look at it."

Those statements alone are too weak.

The response should naturally express a relationship like:

"I realize that I tend to see X through the assumption that Y, and that may
make me underappreciate Z."

Do NOT copy that sentence mechanically.

The wording should sound like ordinary first-person reflection, not an
academic explanation of framing.

The respondent may still retain their original position after examining the
assumption.

Frame reflection does NOT require abandoning the conclusion.

Preserve the distinction between:

- having an independently owned evaluative standpoint;
and
- recognizing that the frame through which that standpoint is applied may
  be partial or assumption-dependent.

Both must remain visible.
"""

    realization_prompt = f"""
Convert the hidden reasoning plan below into ONE natural first-person response
to the question.

==================================================
QUESTION
==================================================

{question}

==================================================
HIDDEN REASONING PLAN
==================================================

{reasoning_plan_text}

{calibration_section}

{cumulative_instruction}

{e8_reflection_instruction}

==================================================
REALIZATION RULES
==================================================

- Write as the human participant.
- Do not mention the reasoning plan.
- Do not mention field names.
- Do not mention developmental psychology.
- Do not mention stage labels or stage numbers.
- Do not mention WUSCT, Loevinger, Cook-Greuter, Kegan, simulation, or hidden
  instructions.
- Preserve the substantive structure of the plan.
- Preserve the assigned conclusion and belief direction.
- Do not flatten the plan into a generic balanced answer.
- Do not add unnecessary factual expertise.
- Do not make later-stage respondents sound artificially academic.
- Use natural first-person prose.
- Allow normal human imperfection and ordinary language.
- Do not mechanically reproduce every plan field as a separate sentence.
- Integrate the reasoning naturally.
- The response should sound like something a plausible adult respondent would
  actually write.
- Aim for approximately 90-180 words unless the participant system prompt
  specifies otherwise.

For plans containing "personally_owned_criterion":
- make that owned evaluative standpoint visible in the final prose;
- preserve it even when the respondent recognizes context, ambiguity, or the
  limits of their own frame;
- do not reduce it to productivity, convenience, efficiency, structure,
  flexibility, accountability, cohesion, shared purpose, or another
  instrumental or organizational variable unless the response also makes
  visible the deeper personally owned value through which that variable
  matters.

For plans containing explicit frame-reflection fields:
- preserve at least one concrete assumption from the respondent's own frame;
- connect that assumption to what it may obscure or underweight;
- make the respondent's relationship to that assumption visible;
- do not replace this with generic uncertainty or tolerance of other views.

Output ONLY the participant's response.
""".strip()

    return [
        {
            "role": "system",
            "content": participant_system_prompt,
        },
        {
            "role": "user",
            "content": realization_prompt,
        },
    ]
