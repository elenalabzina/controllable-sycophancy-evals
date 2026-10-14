#!/usr/bin/env python

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
import time
from pathlib import Path

from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(
        ROOT
        / "src"
    ),
)


from controllable_sycophancy.calibration import (  # noqa: E402
    StageCalibration,
    build_stage_calibration,
    calibration_example_ids,
    format_stage_calibration_for_prompt,
    load_calibration_anchors,
    load_stage_definitions,
)
from controllable_sycophancy.config import load_yaml  # noqa: E402
from controllable_sycophancy.developmental_eval import (  # noqa: E402
    DevelopmentalEvaluation,
    evaluate_developmental_structure,
)
from controllable_sycophancy.model import (  # noqa: E402
    GenerationBackend,
    build_generation_backend,
)
from controllable_sycophancy.prompts import (  # noqa: E402
    build_respondent_system_prompt,
)
from controllable_sycophancy.reasoning_plan import (  # noqa: E402
    build_realization_messages,
    generate_reasoning_plan,
)
from controllable_sycophancy.stages import (  # noqa: E402
    get_stage,
    load_stages,
)
from controllable_sycophancy.validators import (  # noqa: E402
    validate_response,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic respondents using "
            "developmentally calibrated plan -> response generation."
        )
    )

    parser.add_argument(
        "--config",
        default="configs/generate_validation.example.yaml",
    )

    parser.add_argument(
        "--replicas",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--stage-orders",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional stage-order override. "
            "Example: --stage-orders 7 runs E8 only."
        ),
    )

    parser.add_argument(
        "--output-path",
        default=None,
        help=(
            "Optional output JSONL path override. "
            "Useful for smoke tests without editing the YAML config."
        ),
    )

    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--keep-failed",
        action="store_true",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from an existing output JSONL file. "
            "Completed respondent IDs are skipped."
        ),
    )

    parser.add_argument(
        "--evaluator-retries",
        type=int,
        default=5,
        help=(
            "Maximum number of attempts for each developmental "
            "evaluation, including the first attempt."
        ),
    )

    return parser.parse_args()


def generate_final_response(
    *,
    backend: GenerationBackend,
    messages: list[
        dict[str, str]
    ],
    generation_config: dict,
) -> str:
    """
    Convert a hidden reasoning plan into natural participant prose.
    """

    return backend.generate(
        messages=messages,
        max_new_tokens=int(
            generation_config.get(
                "max_new_tokens",
                260,
            )
        ),
        temperature=float(
            generation_config.get(
                "temperature",
                0.7,
            )
        ),
        top_p=float(
            generation_config.get(
                "top_p",
                0.8,
            )
        ),
        do_sample=bool(
            generation_config.get(
                "do_sample",
                True,
            )
        ),
    )


def evaluate_with_retries(
    *,
    backend: GenerationBackend,
    question: str,
    response: str,
    max_attempts: int,
) -> tuple[DevelopmentalEvaluation, int]:
    """
    Run the developmental evaluator with retry/backoff.

    Retries both transient API exceptions and invalid evaluator outputs.
    If all attempts fail, return an invalid DevelopmentalEvaluation rather
    than crashing the entire synthetic-generation run.
    """

    max_attempts = max(
        1,
        int(max_attempts),
    )

    last_error = ""
    last_raw_output = ""

    for attempt in range(
        1,
        max_attempts + 1,
    ):
        try:
            result = (
                evaluate_developmental_structure(
                    backend=backend,
                    question=question,
                    response=response,
                )
            )

            if result.valid:
                return (
                    result,
                    attempt,
                )

            last_error = (
                result.error
                or "Invalid developmental evaluation."
            )
            last_raw_output = (
                result.raw_output
                or ""
            )

        except Exception as exc:
            last_error = (
                f"{type(exc).__name__}: {exc}"
            )
            last_raw_output = ""

        if attempt < max_attempts:
            delay_seconds = min(
                2 ** attempt,
                30,
            )

            tqdm.write(
                (
                    "Developmental evaluator attempt "
                    f"{attempt}/{max_attempts} failed; "
                    f"retrying in {delay_seconds}s. "
                    f"Reason: {last_error}"
                )
            )

            time.sleep(
                delay_seconds
            )

    return (
        DevelopmentalEvaluation(
            valid=False,
            frame_reflection=None,
            perspective_differentiation=None,
            external_validation_dependence=None,
            ambiguity_tolerance=None,
            meta_awareness=None,
            self_authored_criteria=None,
            overall_notes="",
            raw_output=last_raw_output,
            error=(
                "Developmental evaluation failed after "
                f"{max_attempts} attempts. "
                f"Last error: {last_error}"
            ),
        ),
        max_attempts,
    )


def format_developmental_scores(
    evaluation: DevelopmentalEvaluation,
) -> str:
    """
    Format developmental-evaluation scores for compact checkpoint logging.
    """

    if not evaluation.valid:
        return (
            "developmental_evaluation=INVALID"
            + (
                f" | error={evaluation.error}"
                if evaluation.error
                else ""
            )
        )

    return (
        f"frame={evaluation.frame_reflection} | "
        f"perspective={evaluation.perspective_differentiation} | "
        f"external={evaluation.external_validation_dependence} | "
        f"ambiguity={evaluation.ambiguity_tolerance} | "
        f"meta={evaluation.meta_awareness} | "
        f"self_authored={evaluation.self_authored_criteria}"
    )


def select_topics(
    *,
    all_topics: list[dict],
    selected_topic_ids: list[str] | None,
    max_topics: int | None,
) -> list[dict]:

    if selected_topic_ids:

        topics_by_id = {
            str(
                topic["id"]
            ): topic
            for topic
            in all_topics
        }

        missing = [
            topic_id
            for topic_id
            in selected_topic_ids
            if topic_id
            not in topics_by_id
        ]

        if missing:
            raise KeyError(
                f"Unknown topic IDs: "
                f"{missing}"
            )

        return [
            topics_by_id[
                topic_id
            ]
            for topic_id
            in selected_topic_ids
        ]

    if max_topics is not None:
        return all_topics[
            :max_topics
        ]

    return all_topics


def build_balanced_cells(
    *,
    stage_orders: list[int],
    topics: list[dict],
    belief_positions: list[float],
    confidence_values: list[float],
    replicas: int,
) -> list[tuple]:
    """
    Interleave stages within matched content cells.

    Example:

    E2 replica 0
    E4 replica 0
    E6 replica 0
    E8 replica 0

    E2 replica 1
    E4 replica 1
    ...
    """

    cells = []

    for (
        topic,
        belief_position,
        confidence,
        replica,
    ) in itertools.product(
        topics,
        belief_positions,
        confidence_values,
        range(
            replicas
        ),
    ):

        for stage_order in (
            stage_orders
        ):
            cells.append(
                (
                    stage_order,
                    topic,
                    belief_position,
                    confidence,
                    replica,
                )
            )

    return cells


def load_calibration_resources(
    *,
    config: dict,
) -> tuple[
    dict | None,
    list | None,
    dict,
]:
    """
    Load calibration resources once.

    Returns:

    definitions
    anchors
    calibration config
    """

    calibration_config = dict(
        config.get(
            "calibration",
            {},
        )
    )

    if not calibration_config.get(
        "enabled",
        False,
    ):
        return (
            None,
            None,
            calibration_config,
        )

    definitions_path = (
        calibration_config.get(
            "definitions_path"
        )
    )

    anchors_path = (
        calibration_config.get(
            "anchors_path"
        )
    )

    if not definitions_path:
        raise ValueError(
            "Calibration is enabled but "
            "'definitions_path' is missing."
        )

    if not anchors_path:
        raise ValueError(
            "Calibration is enabled but "
            "'anchors_path' is missing."
        )

    definitions = (
        load_stage_definitions(
            ROOT
            / definitions_path
        )
    )

    anchors = (
        load_calibration_anchors(
            ROOT
            / anchors_path
        )
    )

    return (
        definitions,
        anchors,
        calibration_config,
    )


def build_calibration_cache(
    *,
    stages,
    stage_orders: list[int],
    definitions: dict | None,
    anchors: list | None,
    calibration_config: dict,
) -> dict[
    str,
    tuple[
        StageCalibration,
        str,
        dict[str, list[str]],
    ],
]:
    """
    Build calibration context once per stage.

    The exact same calibration-anchor selection is then used across all
    matched replicas of that stage.
    """

    cache = {}

    if (
        definitions is None
        or anchors is None
    ):
        return cache

    examples_per_stage = int(
        calibration_config.get(
            "examples_per_stage",
            4,
        )
    )

    contrast_examples = int(
        calibration_config.get(
            "contrast_examples_per_adjacent_stage",
            1,
        )
    )

    min_confidence = (
        calibration_config.get(
            "min_confidence",
            "medium",
        )
    )

    for stage_order in stage_orders:

        stage = get_stage(
            stages,
            stage_order,
        )

        if (
            stage.code
            not in definitions
        ):
            tqdm.write(
                (
                    "No calibration "
                    f"definition for {stage.code}; "
                    "using the stage description."
                )
            )

            continue

        calibration = (
            build_stage_calibration(
                stage_code=stage.code,
                definitions=definitions,
                anchors=anchors,
                examples_per_stage=(
                    examples_per_stage
                ),
                contrast_examples_per_adjacent_stage=(
                    contrast_examples
                ),
                min_confidence=(
                    min_confidence
                ),
            )
        )

        calibration_text = (
            format_stage_calibration_for_prompt(
                calibration
            )
        )

        example_ids = (
            calibration_example_ids(
                calibration
            )
        )

        cache[
            stage.code
        ] = (
            calibration,
            calibration_text,
            example_ids,
        )

    return cache


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------
    # LOAD CONFIG
    # ------------------------------------------------------------

    config = load_yaml(
        ROOT
        / args.config
    )

    model_config = load_yaml(
        ROOT
        / config[
            "model_config"
        ]
    )

    evaluation_model_config_path = config.get(
        "evaluation_model_config"
    )

    if evaluation_model_config_path:
        evaluation_model_config = load_yaml(
            ROOT
            / evaluation_model_config_path
        )
    else:
        evaluation_model_config = model_config

    stages = load_stages(
        ROOT
        / config[
            "stages_path"
        ]
    )

    all_topics = load_yaml(
        ROOT
        / config[
            "topics_path"
        ]
    )[
        "topics"
    ]

    selected_topic_ids = (
        config.get(
            "selected_topic_ids"
        )
    )

    if (
        selected_topic_ids
        is not None
    ):
        selected_topic_ids = [
            str(
                value
            )
            for value
            in selected_topic_ids
        ]

    max_topics = config.get(
        "max_topics"
    )

    if (
        max_topics
        is not None
    ):
        max_topics = int(
            max_topics
        )

    topics = select_topics(
        all_topics=all_topics,
        selected_topic_ids=(
            selected_topic_ids
        ),
        max_topics=max_topics,
    )

    stage_orders = (
        [
            int(
                value
            )
            for value
            in args.stage_orders
        ]
        if args.stage_orders
        is not None
        else [
            int(
                value
            )
            for value
            in config[
                "anchor_stage_orders"
            ]
        ]
    )

    belief_positions = [
        float(
            value
        )
        for value
        in config[
            "belief_positions"
        ]
    ]

    confidence_values = [
        float(
            value
        )
        for value
        in config[
            "confidence_values"
        ]
    ]

    seed = int(
        config.get(
            "seed",
            42,
        )
    )

    random.seed(
        seed
    )

    replicas = (
        args.replicas
        if args.replicas
        is not None
        else int(
            config.get(
                "replicas_per_cell",
                1,
            )
        )
    )

    max_attempts = (
        args.max_attempts
        if args.max_attempts
        is not None
        else int(
            config.get(
                "max_attempts_per_cell",
                4,
            )
        )
    )

    validation_config = dict(
        config.get(
            "validation",
            {},
        )
    )

    min_words = int(
        validation_config.get(
            "min_words",
            70,
        )
    )

    max_words = int(
        validation_config.get(
            "max_words",
            220,
        )
    )

    generation_config = dict(
        model_config.get(
            "generation",
            {},
        )
    )

    # ------------------------------------------------------------
    # BUILD MODEL BACKENDS
    # ------------------------------------------------------------

    print(
        "Building inference backend..."
    )

    backend = (
        build_generation_backend(
            model_config
        )
    )

    print(
        f"Using generation model: "
        f"{backend.model_name}"
    )

    developmental_evaluation_backend = (
        build_generation_backend(
            evaluation_model_config
        )
    )

    print(
        f"Using developmental evaluator model: "
        f"{developmental_evaluation_backend.model_name}"
    )

    # ------------------------------------------------------------
    # LOAD DEVELOPMENTAL CALIBRATION
    # ------------------------------------------------------------

    (
        calibration_definitions,
        calibration_anchors,
        calibration_config,
    ) = load_calibration_resources(
        config=config
    )

    calibration_enabled = (
        calibration_definitions
        is not None
        and calibration_anchors
        is not None
    )

    if calibration_enabled:

        print(
            "Developmental calibration: ENABLED"
        )

        print(
            f"Calibration anchors loaded: "
            f"{len(calibration_anchors)}"
        )

    else:

        print(
            "Developmental calibration: DISABLED"
        )

    calibration_cache = (
        build_calibration_cache(
            stages=stages,
            stage_orders=stage_orders,
            definitions=(
                calibration_definitions
            ),
            anchors=(
                calibration_anchors
            ),
            calibration_config=(
                calibration_config
            ),
        )
    )

    if calibration_cache:

        print(
            "Calibrated stages: "
            + ", ".join(
                calibration_cache.keys()
            )
        )

    # ------------------------------------------------------------
    # BUILD MATCHED CELLS
    # ------------------------------------------------------------

    cells = build_balanced_cells(
        stage_orders=stage_orders,
        topics=topics,
        belief_positions=belief_positions,
        confidence_values=confidence_values,
        replicas=replicas,
    )

    if (
        args.max_examples
        is not None
    ):
        cells = cells[
            :args.max_examples
        ]

    print(
        f"Respondent cells scheduled: "
        f"{len(cells)}"
    )

    print(
        "Generation mode: "
        "calibrated reasoning plan "
        "-> natural response"
    )

    # ------------------------------------------------------------
    # OUTPUT / CHECKPOINT SETUP
    # ------------------------------------------------------------

    output_path = (
        ROOT
        / (
            args.output_path
            if args.output_path
            is not None
            else config[
                "output_path"
            ]
        )
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = []
    completed_respondent_ids: set[str] = set()

    if (
        args.resume
        and output_path.exists()
    ):
        with output_path.open(
            "r",
            encoding="utf-8",
        ) as f:
            for line_number, line in enumerate(
                f,
                start=1,
            ):
                line = line.strip()

                if not line:
                    continue

                try:
                    existing_record = json.loads(
                        line
                    )
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        "Could not resume because "
                        f"{output_path} contains invalid JSON "
                        f"on line {line_number}."
                    ) from exc

                records.append(
                    existing_record
                )

                respondent_id = existing_record.get(
                    "respondent_id"
                )

                if respondent_id:
                    completed_respondent_ids.add(
                        str(
                            respondent_id
                        )
                    )

        print(
            f"Resume mode: loaded "
            f"{len(records)} existing records "
            f"from {output_path}"
        )

    else:
        output_path.write_text(
            "",
            encoding="utf-8",
        )

        if args.resume:
            print(
                "Resume mode requested, but no existing "
                "output file was found. Starting fresh."
            )
        else:
            print(
                f"Starting fresh output file: "
                f"{output_path}"
            )

    # ------------------------------------------------------------
    # GENERATION
    # ------------------------------------------------------------

    rejected_attempts = 0
    failed_cells = 0
    planning_failures = 0
    developmental_eval_failures = 0

    for index, (
        stage_order,
        topic,
        belief_position,
        confidence,
        replica,
    ) in enumerate(
        tqdm(
            cells,
            desc=(
                "Generating synthetic respondents"
            ),
        ),
        start=1,
    ):

        stage = get_stage(
            stages,
            stage_order,
        )

        respondent_id = (
            f"syn_{index:05d}"
        )

        if (
            respondent_id
            in completed_respondent_ids
        ):
            tqdm.write(
                f"Skipping completed {respondent_id}"
            )
            continue

        # --------------------------------------------------------
        # CALIBRATION FOR THIS EXACT STAGE
        # --------------------------------------------------------

        calibration_text = None

        calibration_metadata = {
            "enabled": False,
            "stage": (
                stage.code
            ),
            "target_example_ids": [],
            "lower_contrast_ids": [],
            "higher_contrast_ids": [],
        }

        if (
            stage.code
            in calibration_cache
        ):

            (
                stage_calibration,
                calibration_text,
                example_ids,
            ) = calibration_cache[
                stage.code
            ]

            calibration_metadata = {
                "enabled": True,
                "stage": (
                    stage.code
                ),
                "stage_label": (
                    stage_calibration.stage_label
                ),
                "target_example_ids": (
                    example_ids[
                        "target"
                    ]
                ),
                "lower_contrast_ids": (
                    example_ids[
                        "lower_contrast"
                    ]
                ),
                "higher_contrast_ids": (
                    example_ids[
                        "higher_contrast"
                    ]
                ),
            }

        # --------------------------------------------------------
        # RESPONDENT ROLE PROMPT
        # --------------------------------------------------------

        participant_system_prompt = (
            build_respondent_system_prompt(
                stage=stage,
                belief_position=(
                    belief_position
                ),
                confidence=confidence,
                negative_position=topic[
                    "negative_position"
                ],
                positive_position=topic[
                    "positive_position"
                ],
                calibration_text=(
                    calibration_text
                ),
            )
        )

        accepted_response = None
        accepted_validation = None
        accepted_plan = None

        attempts = []

        # --------------------------------------------------------
        # RETRY LOOP
        # --------------------------------------------------------

        for attempt_number in range(
            1,
            max_attempts + 1,
        ):

            # ----------------------------------------------------
            # STAGE 1:
            # DEVELOPMENTALLY CALIBRATED HIDDEN PLAN
            # ----------------------------------------------------

            plan_result = (
                generate_reasoning_plan(
                    backend=backend,
                    stage=stage,
                    question=topic[
                        "question"
                    ],
                    belief_position=(
                        belief_position
                    ),
                    negative_position=topic[
                        "negative_position"
                    ],
                    positive_position=topic[
                        "positive_position"
                    ],
                    calibration_text=(
                        calibration_text
                    ),
                )
            )

            if not plan_result.valid:

                planning_failures += 1
                rejected_attempts += 1

                attempts.append(
                    {
                        "attempt": (
                            attempt_number
                        ),
                        "plan_valid": False,
                        "plan": None,
                        "plan_raw_output": (
                            plan_result.raw_output
                        ),
                        "plan_error": (
                            plan_result.error
                        ),
                        "response": None,
                        "validation": None,
                    }
                )

                tqdm.write(
                    (
                        f"Rejected "
                        f"{respondent_id} "
                        f"({stage.code}) "
                        f"attempt "
                        f"{attempt_number}: "
                        f"invalid reasoning plan: "
                        f"{plan_result.error}"
                    )
                )

                continue

            reasoning_plan = (
                plan_result.plan
            )

            assert (
                reasoning_plan
                is not None
            )

            # ----------------------------------------------------
            # STAGE 2:
            # NATURAL RESPONSE REALIZATION
            # ----------------------------------------------------

            realization_messages = (
                build_realization_messages(
                    participant_system_prompt=(
                        participant_system_prompt
                    ),
                    question=topic[
                        "question"
                    ],
                    reasoning_plan=(
                        reasoning_plan
                    ),
                    calibration_text=(
                        calibration_text
                    ),
                )
            )

            response = (
                generate_final_response(
                    backend=backend,
                    messages=(
                        realization_messages
                    ),
                    generation_config=(
                        generation_config
                    ),
                )
            )

            # ----------------------------------------------------
            # BASIC VALIDATION
            # ----------------------------------------------------

            validation = (
                validate_response(
                    backend=backend,
                    response=response,
                    belief_position=(
                        belief_position
                    ),
                    question=topic[
                        "question"
                    ],
                    negative_position=topic[
                        "negative_position"
                    ],
                    positive_position=topic[
                        "positive_position"
                    ],
                    min_words=min_words,
                    max_words=max_words,
                )
            )

            attempts.append(
                {
                    "attempt": (
                        attempt_number
                    ),
                    "plan_valid": True,
                    "plan": (
                        reasoning_plan
                    ),
                    "plan_raw_output": (
                        plan_result.raw_output
                    ),
                    "plan_error": None,
                    "response": (
                        response
                    ),
                    "validation": (
                        validation.to_dict()
                    ),
                }
            )

            if validation.passed:

                accepted_plan = (
                    reasoning_plan
                )

                accepted_response = (
                    response
                )

                accepted_validation = (
                    validation
                )

                break

            rejected_attempts += 1

            tqdm.write(
                (
                    f"Rejected "
                    f"{respondent_id} "
                    f"({stage.code}) "
                    f"attempt "
                    f"{attempt_number}: "
                    f"stance="
                    f"{validation.judged_stance}/"
                    f"{validation.expected_stance}; "
                    f"words="
                    f"{validation.word_count}; "
                    f"leakage="
                    f"{validation.leakage_terms}"
                )
            )

        # --------------------------------------------------------
        # FAILED CELL
        # --------------------------------------------------------

        if (
            accepted_response
            is None
        ):

            failed_cells += 1

            tqdm.write(
                (
                    f"FAILED "
                    f"{respondent_id} "
                    f"({stage.code}): "
                    f"no valid plan+response "
                    f"after {max_attempts} attempts"
                )
            )

            if not args.keep_failed:
                continue

            usable_attempt = None

            for attempt in reversed(
                attempts
            ):
                if (
                    attempt[
                        "response"
                    ]
                    is not None
                ):
                    usable_attempt = (
                        attempt
                    )
                    break

            if usable_attempt is None:
                continue

            accepted_plan = (
                usable_attempt[
                    "plan"
                ]
            )

            accepted_response = str(
                usable_attempt[
                    "response"
                ]
            )

            accepted_validation = None

        # --------------------------------------------------------
        # DEVELOPMENTAL EVALUATION
        # --------------------------------------------------------

        (
            developmental_evaluation,
            developmental_evaluation_attempts,
        ) = evaluate_with_retries(
            backend=developmental_evaluation_backend,
            question=topic[
                "question"
            ],
            response=(
                accepted_response
            ),
            max_attempts=(
                args.evaluator_retries
            ),
        )

        if (
            not developmental_evaluation.valid
        ):

            developmental_eval_failures += 1

            tqdm.write(
                (
                    "Developmental evaluator "
                    f"failed for "
                    f"{respondent_id} "
                    f"after "
                    f"{developmental_evaluation_attempts} "
                    "attempts; saving invalid "
                    "evaluation and continuing."
                )
            )

        # --------------------------------------------------------
        # FINAL VALIDATION RECORD
        # --------------------------------------------------------

        if (
            accepted_validation
            is not None
        ):

            validation_passed = (
                accepted_validation.passed
            )

            final_validation = (
                accepted_validation.to_dict()
            )

        else:

            validation_passed = False

            final_validation = None

            for attempt in reversed(
                attempts
            ):
                if (
                    attempt[
                        "validation"
                    ]
                    is not None
                ):
                    final_validation = (
                        attempt[
                            "validation"
                        ]
                    )
                    break

        # --------------------------------------------------------
        # SAVE RECORD
        # --------------------------------------------------------

        record = {
            "respondent_id": (
                respondent_id
            ),

            "generation_model": (
                backend.model_name
            ),

            "generation_mode": (
                "calibrated_reasoning_plan_then_realization"
            ),

            "backend": (
                model_config.get(
                    "backend",
                    "transformers",
                )
            ),

            "seed": seed,

            "replica": replica,

            "stage_order": (
                stage.order
            ),

            "stage_code": (
                stage.code
            ),

            "stage_label": (
                stage.label
            ),

            "topic_id": (
                topic[
                    "id"
                ]
            ),

            "question": (
                topic[
                    "question"
                ]
            ),

            "negative_position": (
                topic[
                    "negative_position"
                ]
            ),

            "positive_position": (
                topic[
                    "positive_position"
                ]
            ),

            "belief_position": (
                belief_position
            ),

            "confidence": (
                confidence
            ),

            "calibration": (
                calibration_metadata
            ),

            "reasoning_plan": (
                accepted_plan
            ),

            "system_prompt": (
                participant_system_prompt
            ),

            "response": (
                accepted_response
            ),

            "generation_attempts": (
                len(
                    attempts
                )
            ),

            "validation_passed": (
                validation_passed
            ),

            "validation": (
                final_validation
            ),

            "developmental_evaluation_model": (
                developmental_evaluation_backend.model_name
            ),

            "developmental_evaluation_backend": (
                evaluation_model_config.get(
                    "backend",
                    "transformers",
                )
            ),

            "developmental_evaluation_attempts": (
                developmental_evaluation_attempts
            ),

            "developmental_evaluation": (
                developmental_evaluation.to_dict()
            ),
        }

        records.append(
            record
        )

        completed_respondent_ids.add(
            respondent_id
        )

        with output_path.open(
            "a",
            encoding="utf-8",
        ) as f:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )
            f.flush()

        tqdm.write(
            (
                f"Checkpointed {respondent_id} "
                f"({stage.code}) | "
                f"{format_developmental_scores(developmental_evaluation)}"
            )
        )

    # ------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------

    print()

    print(
        f"Output contains "
        f"{len(records)} examples "
        f"at {output_path}"
    )

    print(
        f"Rejected generation attempts: "
        f"{rejected_attempts}"
    )

    print(
        f"Reasoning-plan failures: "
        f"{planning_failures}"
    )

    print(
        f"Failed respondent cells: "
        f"{failed_cells}"
    )

    print(
        "Developmental evaluator failures: "
        f"{developmental_eval_failures}"
    )

    print(
        f"Accepted respondent cells: "
        f"{len(records)}/"
        f"{len(cells)}"
    )


if __name__ == "__main__":
    main()
