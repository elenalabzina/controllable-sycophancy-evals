# Methodology

## Goal

The system is designed to vary **reasoning organization** while controlling variables that commonly become entangled in persona simulation.

The intended comparison is not "simple language versus sophisticated language." Instead, the target manipulation concerns how judgments are organized: through immediate reaction, social validation, internally endorsed criteria, contextual differentiation, or reflection on the framing process itself.

## Design principles

### Reasoning structure is not intelligence

Profile assignment must not determine intelligence, education, grammar, vocabulary, political orientation, or factual knowledge.

### Structure is separated from realization

Generation occurs in two stages:

1. create a structured latent reasoning plan;
2. realize that plan as a natural first-person response.

This reduces the tendency of direct persona prompting to encode the construct mainly through style.

### Substantive position is independently controlled

Topic, belief direction, confidence, and reasoning profile are supplied separately. The same substantive stance can therefore be expressed through different reasoning organizations.

### Generator and evaluator are separated

The evaluator receives the question and response, but not the intended profile label. It scores behavioral dimensions rather than simply predicting a stage name.

### Failure analysis precedes held-out validation

During development, evaluator failure modes were inspected directly. Examples included:

- treating a personal preference as evidence of self-authorship;
- treating "I think" or "I feel" as meta-awareness;
- rewarding generic balanced prose as perspective differentiation;
- over-penalizing later profiles unless they used explicit formal criteria.

The rubric was refined on development data and then frozen.

### Held-out validation is a stopping rule

After freezing, the system was evaluated on new topics. The goal was not perfect profile classification. The goal was to test whether the expected *dimension-level signatures* survive topic changes.

### Synthetic evidence is infrastructure evidence

Synthetic validation supports claims about controllability, construct separation, logging, and evaluation infrastructure. It does not establish a human causal effect.

## Human-informed calibration

The private research workflow uses anonymized sentence-completion responses collected through Prolific as calibration anchors. These ground distinctions in observed human reasoning rather than only hand-written profile descriptions.

The human responses are deliberately not included in this repository. The public demo provides synthetic anchors so that the engineering pipeline remains inspectable without redistributing participant material.

## Relation to digital twins

This project is adjacent to digital-twin research but intentionally narrower.

A full person-level digital twin aims to reproduce a specific person's responses across a broad range of situations. Here, the target is a **construct-calibrated synthetic participant**: a model whose behavior is constrained along a particular reasoning dimension while other experimental inputs remain controllable.

That narrower target is useful for evals because it trades person-level fidelity for experimental identifiability.

## Limitations

- The reasoning profiles are operational simulation constructs, not clinical or psychometric diagnoses.
- Model-generated language can still contain residual style differences that correlate with profile.
- A model judge is an imperfect measurement instrument.
- Human calibration anchors do not make the synthetic system equivalent to human participants.
- The current validation set is intentionally small and diagnostic rather than a population benchmark.
