# Controllable Sycophancy Evals

**User-side infrastructure for behavioral evaluation of LLM sycophancy across developmentally structured reasoning profiles.**

Most sycophancy evaluations treat the user as fixed. This project asks a different engineering question:

> Can we build controlled synthetic users whose **organization of reasoning** varies systematically while topic, substantive belief, and confidence remain fixed?

The current release focuses on the **controlled-user layer** needed for heterogeneous sycophancy experiments. It separates latent reasoning structure from natural-language realization, applies constraint checks, and evaluates generated behavior with a stage-blind judge. The goal is to make downstream model-behavior evaluation more controlled, auditable, and reproducible.

This repository is a **sanitized research-engineering release**. It contains the simulator, evaluator, demo calibration assets, and held-out validation summaries. It intentionally excludes the downstream human-study design, private participant anchors, and unpublished causal predictions.

## Why this is different

Synthetic users are often implemented as surface personas: age, occupation, tone, ideology, or writing style. That makes it hard to know whether an observed effect comes from the intended latent construct or from superficial linguistic differences.

Here, the controlled variable is the **structure of meaning-making**. The current implementation uses ordered reasoning profiles informed by constructive-developmental theory. Profiles vary in dimensions such as:

- reliance on external norms and validation;
- self-authored evaluative criteria;
- perspective differentiation;
- ambiguity tolerance;
- reflection on one's own framing;
- meta-awareness of the reasoning process.

The simulator is explicitly instructed **not** to encode later profiles as greater intelligence, education, factual knowledge, verbosity, or linguistic sophistication.

## Architecture

```text
reasoning-profile specification
        +
calibration anchors
        +
topic / stance / confidence controls
        ↓
structured latent reasoning plan
        ↓
natural-language realization
        ↓
constraint validation
        ↓
stage-blind behavioral evaluator
```

Generation and evaluation are separated so that the judge does not receive the intended profile label.

## Human calibration and the digital-twin connection

The private research workflow uses **anonymized human responses collected through Prolific as calibration anchors**. Those responses are not distributed in this repository.

The public release includes synthetic demonstration anchors with the same schema so that the pipeline can be inspected and run without exposing participant material.

This is related to digital-twin research, but the goal is narrower than a full person-level digital twin. A full digital twin attempts to reproduce a particular individual across many behaviors. This project instead builds **construct-calibrated synthetic participants**: controlled behavioral models designed to vary a targeted reasoning structure while other experimental variables are held fixed.

## Controlled variables

A generated participant is conditioned separately on:

```text
reasoning profile
topic
substantive belief direction
confidence
```

This supports matched comparisons in which the semantic position is deliberately held constant while reasoning structure changes.

## Evaluator

The stage-blind evaluator scores six dimensions from 0 to 3:

1. `frame_reflection`
2. `perspective_differentiation`
3. `external_validation_dependence`
4. `ambiguity_tolerance`
5. `meta_awareness`
6. `self_authored_criteria`

The rubric contains guardrails against common false positives. For example, hedging is not automatically ambiguity tolerance; mentioning two sides is not automatically perspective differentiation; and a simple first-person preference is not automatically a self-authored criterion.

## Validation strategy

The evaluator and generator were iteratively debugged on development topics and then frozen before held-out validation.

Two held-out topics were used with stance and confidence matched across profiles:

- regulation of advanced AI systems;
- private-car access in city centers.

The frozen system produced the intended qualitative signatures across both topics. In particular, externally validated reasoning peaked in the conformist profile, self-authored criteria emerged in the self-authoring profiles, and frame reflection / meta-awareness were strongest in the autonomous profile.

See [docs/validation.md](docs/validation.md) for the validation summary and [docs/methodology.md](docs/methodology.md) for design principles and limitations.

## Quick start

Python 3.11+ is supported.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e '.[dev]'
python -m pytest -q
```

Inspect the public calibration prompt:

```bash
python scripts/inspect_calibration_prompt.py --stage E6
```

Run the public demo configuration after pointing `configs/model.example.yaml` and `configs/evaluator.example.yaml` at models available in your environment:

```bash
python scripts/generate_synthetic_data.py \\
  --config configs/generate_validation.example.yaml
```

## What this repository does **not** claim

A successful synthetic validation means that the simulator can express the supplied reasoning-profile specification in a repeatable way.

It does **not** establish that:

- the profiles are psychometrically validated measurements of real people;
- the simulator reproduces the distribution of human developmental stages;
- synthetic results establish causal effects on humans;
- later reasoning profiles are "better," smarter, or more correct;
- the public demo anchors substitute for human validation.

The downstream human-study design and substantive causal hypotheses are intentionally outside the scope of this repository.

## Status

Research-engineering release. Ongoing research continues in a separate private workspace.

## License

Apache-2.0 for repository code. External models and datasets remain governed by their respective licenses.
