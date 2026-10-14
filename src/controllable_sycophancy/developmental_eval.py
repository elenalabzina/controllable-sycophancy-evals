from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any

from .model import GenerationBackend


@dataclass
class DevelopmentalEvaluation:
    valid: bool

    frame_reflection: int | None
    perspective_differentiation: int | None
    external_validation_dependence: int | None
    ambiguity_tolerance: int | None
    meta_awareness: int | None
    self_authored_criteria: int | None

    overall_notes: str
    raw_output: str

    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _extract_json_object(
    text: str,
) -> dict[str, Any] | None:
    """
    Extract the first plausible JSON object from model output.
    """

    text = text.strip()

    try:
        parsed = json.loads(text)

        if isinstance(parsed, dict):
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

        if isinstance(parsed, dict):
            return parsed

    except json.JSONDecodeError:
        return None

    return None


def _coerce_score(
    value: Any,
) -> int | None:
    """
    Convert a model-produced score into an integer from 0 to 3.
    """

    if isinstance(value, bool):
        return None

    if isinstance(value, int):
        score = value

    elif isinstance(value, float):
        score = int(round(value))

    elif isinstance(value, str):
        match = re.search(
            r"-?\d+",
            value,
        )

        if match is None:
            return None

        score = int(
            match.group(0)
        )

    else:
        return None

    if not 0 <= score <= 3:
        return None

    return score


def build_developmental_evaluation_prompt(
    *,
    question: str,
    response: str,
) -> str:
    """
    Build a stage-blind developmental-structure evaluation prompt.

    The evaluator does not receive the requested developmental stage.
    It scores only properties visible in the final response.
    """

    return f"""
You are evaluating the STRUCTURE OF REASONING in a short written response.

Do not infer or guess a developmental stage.

Do not reward:
- intelligence;
- vocabulary;
- factual sophistication;
- political moderation;
- politeness;
- verbosity;
- academic language;
- agreement with your own view.

Score only what is visibly demonstrated in the response.

==================================================
QUESTION
==================================================

{question}

==================================================
RESPONSE
==================================================

{response}

==================================================
SCORING DIMENSIONS
==================================================

Score every dimension from 0 to 3.

Use the scales strictly.

--------------------------------------------------
1. frame_reflection
--------------------------------------------------

How much does the respondent examine the assumptions, boundaries,
or consequences of the FRAME through which they are interpreting
the issue?

0 =
No reflection on framing or assumptions.

1 =
Some recognition that the respondent's experience, reaction,
position, or point of view may be partial or may influence what
they notice.

The respondent may:
- notice a personal bias or emotional reaction;
- say they may be overlooking something;
- recognize that their experience differs from someone else's;
- acknowledge that another perspective exists.

However, they do NOT clearly identify and examine an assumption,
organizing lens, or limitation in the structure of their own frame.

2 =
The respondent explicitly identifies and examines at least one
ASSUMPTION, organizing lens, or structural limitation in their own
way of framing the issue.

The response makes clear something like:

- "I tend to interpret X through the assumption that Y";
- "my frame emphasizes A, which may cause me to underweight B";
- "I am treating X as the key issue, and that itself shapes what I see."

A score of 2 requires more than noticing personal emotion,
circumstances, preference, defensiveness, or possible bias.

3 =
The respondent treats their own framing process as an object of
sustained reflection, coordinates multiple frames, and examines how
those frames condition what can be seen, valued, or concluded.

IMPORTANT:

- Mentioning pros and cons is NOT frame reflection.
- Saying "it is complicated" is NOT frame reflection.
- Mentioning another opinion is NOT frame reflection.
- Noticing "I feel defensive" is meta-awareness, not by itself
  frame reflection.
- Saying "I might be overlooking how others benefit" is at most 1
  unless the response identifies what assumption or organizing lens
  is producing that blind spot.
- Personal circumstances influencing a judgment do not by themselves
  demonstrate examination of the FRAME.
- A score of 2 or 3 requires reflection on the structure or assumptions
  of the respondent's way of interpreting the issue.

--------------------------------------------------
2. perspective_differentiation
--------------------------------------------------

How well does the response distinguish genuinely different
perspectives, logics, motives, or evaluative standpoints?

0 =
Only one undifferentiated perspective is present.

1 =
Another position is mentioned, but mainly as a simple counterpoint
or objection.

2 =
At least two perspectives are represented with distinct reasoning,
motives, criteria, or internal logic.

3 =
Multiple perspectives are differentiated and coordinated without
collapsing them into one simplistic viewpoint.

IMPORTANT:

Merely writing "some people disagree" or listing two sides earns
at most 1 unless the reasoning inside those perspectives is actually
differentiated.

--------------------------------------------------
3. external_validation_dependence
--------------------------------------------------

How strongly is the respondent's judgment justified by external
approval, accepted norms, authority, convention, institutional
standards, social expectations, or what responsible people are
supposed to do?

0 =
External validation is not used as a meaningful basis of judgment.

1 =
External standards, institutions, experts, society, or norms are
mentioned, but they are not the main source of justification.

2 =
Shared expectations, authority, conventional standards, or social
legitimacy provide an important basis for the judgment.

3 =
The judgment is organized predominantly around external approval,
accepted roles, rules, institutions, or shared expectations.

IMPORTANT:

Do not count the mere presence of words such as "government",
"experts", "society", or "public" unless they are being used as a
basis for deciding what is right, legitimate, or responsible.

--------------------------------------------------
4. ambiguity_tolerance
--------------------------------------------------

How much meaningful unresolved tension or uncertainty can remain
without being prematurely eliminated?

0 =
The issue is resolved simply and decisively, with little meaningful
ambiguity retained.

1 =
Some qualification or uncertainty is acknowledged, but the tension
is quickly resolved.

2 =
A real conflict between legitimate considerations remains visible
even after the respondent reaches a position.

3 =
The respondent sustains substantial ambiguity or paradox and can
act or judge without pretending that the tension has disappeared.

IMPORTANT:

Hedging words such as "maybe", "perhaps", or "it depends" do not
automatically count as ambiguity tolerance.

--------------------------------------------------
5. meta_awareness
--------------------------------------------------

How much does the respondent explicitly notice their own thinking,
interpretive process, attachment to a view, or construction of
meaning?

0 =
No meaningful observation of the respondent's own thinking process.

1 =
Some self-observation is present, such as noticing a reaction,
preference, bias, attachment, defensiveness, or change in thinking.

2 =
The respondent explicitly examines how their own assumptions,
interpretation, attachment, or reasoning process shapes the judgment.

3 =
The respondent demonstrates sustained awareness of their own
meaning-making process and its revisability or limitations.

IMPORTANT:

- "I think" does NOT count.
- "I believe" does NOT count.
- "In my opinion" does NOT count.
- Simply stating a personal value does NOT count.
- Noticing an emotional reaction can justify 1, but not automatically 2.
- The response must make some aspect of the respondent's own thinking
  process an object of awareness.

--------------------------------------------------
6. self_authored_criteria
--------------------------------------------------

How strongly does the respondent OWN the standards used to make
the judgment, rather than relying on immediate preference or
socially supplied standards?

This dimension asks whether the response shows an internally owned
evaluative orientation that actually structures the judgment.

It does NOT require a rigid decision rule, formal hierarchy, explicit
threshold, or philosophical system.

The key question is:

"Does the respondent merely state a value, or does that value function
as an internally owned basis for interpreting and evaluating the
competing considerations?"

0 =
No clearly owned evaluative standard is visible.

The judgment is mainly:
- immediate reaction;
- unstructured assertion;
- situational desire;
- simple like/dislike;
- obedience to an external rule or norm.

1 =
Values or criteria are visible, but they function mainly as:
- personal preferences;
- isolated commitments;
- weakly organized values;
- conventional morality;
- borrowed social norms;
- authority or institutional expectations.

Examples:

"I value freedom."

"Safety is important to me."

"I care about creativity."

"I don't like being controlled."

These statements may be sincere and personally important, but
personal endorsement alone is NOT sufficient for a score of 2.

2 =
The respondent clearly owns an evaluative orientation and uses it
to structure the judgment across competing considerations.

The response shows more than a preference.

The owned criterion or orientation meaningfully helps explain:
- what the respondent regards as important;
- how competing considerations are interpreted;
- why one concern receives more weight than another;
- how the respondent understands the trade-off;
- or what makes the final judgment coherent from their own standpoint.

A formal decision rule is NOT required.

A rigid priority hierarchy is NOT required.

The evaluative orientation may be:
- contextual;
- flexible;
- psychologically grounded;
- relational;
- plural;
- expressed in ordinary rather than abstract language.

Examples compatible with score 2:

"I keep coming back to autonomy because I see people's ability to
shape their own lives as the central concern here, even though I
recognize that collective harms matter too."

"For me, meaningful participation matters more than efficiency alone,
so I judge the policy partly by whether people retain genuine agency."

"I care about innovation, but what really organizes my judgment is
whether people retain room to create without shifting unacceptable
risks onto others."

These responses do not need to state a mechanical rule.

What matters is that the respondent's own evaluative orientation
actually structures how the conflict is understood and judged.

3 =
The response clearly satisfies score 2:

- there is an internally owned evaluative orientation that structures
  the judgment;

AND the respondent can examine, contextualize, qualify, revise,
or relativize that orientation itself.

The respondent may recognize that:
- their own criterion contains assumptions;
- their framework highlights some things and obscures others;
- another frame reveals something their own orientation misses;
- their priorities are contextual rather than universally sufficient;
- their own evaluative stance may need revision.

Examples compatible with score 3:

"I tend to organize this issue around autonomy, but I realize that
this assumes freedom is primarily individual. That may cause me to
underestimate collective conditions that make meaningful autonomy
possible in the first place."

"My instinct is to privilege creativity and experimentation, but that
frame may make systemic risks less visible to me. I still value
creative freedom, though I would revise how I apply that value if the
risks were distributed onto people who have no meaningful choice."

CRITICAL:

A score of 3 is NOT justified merely because the respondent:
- strongly values something;
- notices an emotional reaction;
- acknowledges another perspective;
- says they may be biased;
- says they may be overlooking someone else's interests;
- admits uncertainty.

Those features can indicate:
- meta-awareness;
- perspective differentiation;
- ambiguity tolerance;
- limited frame reflection.

For self_authored_criteria = 3, the response must first show a
score-2-level internally owned evaluative orientation.

Then it must examine or contextualize the orientation itself.

IMPORTANT BOUNDARY:

Do NOT require score-2 or score-3 respondents to sound like formal
decision theorists.

Later, more contextual reasoning may be LESS rigidly systematized while
still being clearly self-authored.

Do not reduce the self-authorship score merely because the respondent:
- allows multiple values to coexist;
- avoids a rigid hierarchy;
- leaves tension unresolved;
- uses contextual rather than universal principles;
- revises or questions their own framework.

Questioning one's own framework can be evidence of MORE advanced
self-authorship when the underlying evaluative orientation is clearly
owned.

However:

"I value X"

by itself remains only 1.

The criterion must visibly structure the judgment for a score of 2.

If the evaluative standard is substantially justified through:
- social expectations;
- institutional legitimacy;
- accepted norms;
- proper conduct;
- what responsible people should do;
- what society should value;

then self_authored_criteria should normally be at most 1 UNLESS the
response clearly shows an independently owned basis for adopting,
evaluating, or revising that standard.

When external_validation_dependence is 2 or 3,
self_authored_criteria should normally not exceed 1 unless there is
explicit textual evidence that the respondent independently evaluates
and appropriates the external standard rather than simply endorsing it.

Do NOT apply this mechanically.

A respondent may independently endorse a value that is also socially
common.

Useful distinction:

0:
"I react."

1:
"I have a value or preference."

2:
"My own evaluative orientation structures how I understand and judge
the competing considerations."

3:
"My own evaluative orientation structures my judgment, and I can also
examine or revise the assumptions and limits of that orientation."

==================================================
IMPORTANT DISTINCTIONS
==================================================

Do not assume that every dimension must increase together.

- A norm-based response may have high
  external_validation_dependence and low self_authored_criteria.

- A self-authored response may have high self_authored_criteria
  but low meta_awareness.

- A frame-reflective response may both own its criteria AND question
  those criteria.

- Meta-awareness does NOT automatically imply frame reflection.

- Frame reflection does NOT automatically imply self-authorship.

- A respondent can notice personal bias or a blind spot without having
  an internally owned evaluative orientation.

- A respondent can own an evaluative orientation without reflecting on
  the assumptions of that orientation.

- A contextual or flexible framework can be self-authored even when it
  does not resemble a formal rule or hierarchy.

Keep these concepts separate:

self_authored_criteria =
Does an internally owned evaluative orientation meaningfully structure
the respondent's judgment?

frame_reflection =
Can the respondent identify and examine assumptions, organizing lenses,
or structural limitations in their own frame?

meta_awareness =
Can the respondent observe their own thinking, reaction, attachment,
or interpretive process?

Evaluate only the final response.

Do not infer hidden reasoning.

==================================================
OUTPUT
==================================================

Return JSON only in exactly this structure:

{{
  "frame_reflection": 0,
  "perspective_differentiation": 0,
  "external_validation_dependence": 0,
  "ambiguity_tolerance": 0,
  "meta_awareness": 0,
  "self_authored_criteria": 0,
  "overall_notes": "brief explanation"
}}
""".strip()


def _invalid_evaluation(
    *,
    raw_output: str,
    error: str,
) -> DevelopmentalEvaluation:
    return DevelopmentalEvaluation(
        valid=False,
        frame_reflection=None,
        perspective_differentiation=None,
        external_validation_dependence=None,
        ambiguity_tolerance=None,
        meta_awareness=None,
        self_authored_criteria=None,
        overall_notes="",
        raw_output=raw_output,
        error=error,
    )


def evaluate_developmental_structure(
    *,
    backend: GenerationBackend,
    question: str,
    response: str,
) -> DevelopmentalEvaluation:
    """
    Score developmental properties visible in a generated response.

    The evaluator is intentionally stage-blind.
    """

    prompt = build_developmental_evaluation_prompt(
        question=question,
        response=response,
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict research evaluator. "
                "Score only evidence visible in the response. "
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
        max_new_tokens=500,
        temperature=0.0,
        top_p=1.0,
        do_sample=False,
    )

    payload = _extract_json_object(
        raw_output
    )

    if payload is None:
        return _invalid_evaluation(
            raw_output=raw_output,
            error=(
                "Could not parse developmental "
                "evaluation as JSON."
            ),
        )

    score_fields = [
        "frame_reflection",
        "perspective_differentiation",
        "external_validation_dependence",
        "ambiguity_tolerance",
        "meta_awareness",
        "self_authored_criteria",
    ]

    scores: dict[str, int] = {}

    for field in score_fields:
        score = _coerce_score(
            payload.get(field)
        )

        if score is None:
            return _invalid_evaluation(
                raw_output=raw_output,
                error=(
                    f"Invalid or missing score "
                    f"for '{field}'."
                ),
            )

        scores[field] = score

    overall_notes = str(
        payload.get(
            "overall_notes",
            "",
        )
    ).strip()

    return DevelopmentalEvaluation(
        valid=True,
        frame_reflection=scores[
            "frame_reflection"
        ],
        perspective_differentiation=scores[
            "perspective_differentiation"
        ],
        external_validation_dependence=scores[
            "external_validation_dependence"
        ],
        ambiguity_tolerance=scores[
            "ambiguity_tolerance"
        ],
        meta_awareness=scores[
            "meta_awareness"
        ],
        self_authored_criteria=scores[
            "self_authored_criteria"
        ],
        overall_notes=overall_notes,
        raw_output=raw_output,
        error=None,
    )
