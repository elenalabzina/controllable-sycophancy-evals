# Held-out validation

## Purpose

The validation asks whether the frozen simulator and stage-blind evaluator preserve the intended **dimension-level reasoning signatures** on topics that were not used for final rubric tuning.

This is a construct-separation test, not a test of a downstream causal hypothesis.

## Design

The results below come from the human-informed research workflow. The public repository ships synthetic demonstration anchors with the same interface, so the pipeline is inspectable without releasing participant material; those demo anchors are not intended to numerically reproduce this table.

Two held-out topics were used:

- **AI regulation**
- **Private-car access in city centers**

For each topic:

- seven profiles (E2-E8) were generated;
- five replicas were generated per profile;
- substantive stance was fixed;
- confidence was fixed;
- the evaluator did not receive the target profile.

Total: 35 responses per topic.

## Self-authored criteria across profiles

| Profile | AI regulation | Urban cars |
|---|---:|---:|
| E2 | 1.4 | 1.0 |
| E3 | 1.0 | 1.0 |
| E4 | 1.2 | 1.2 |
| E5 | 1.8 | 1.4 |
| E6 | 2.0 | 2.0 |
| E7 | 2.0 | 2.0 |
| E8 | 3.0 | 2.8 |

## Diagnostic signatures

| Signature | AI regulation | Urban cars |
|---|---:|---:|
| E4 external-validation dependence | 1.6 | 2.2 |
| E5 perspective differentiation | 2.0 | 2.0 |
| E5 ambiguity tolerance | 2.6 | 2.0 |
| E6 self-authored criteria | 2.0 | 2.0 |
| E7 self-authored criteria | 2.0 | 2.0 |
| E7 ambiguity tolerance | 2.2 | 2.2 |
| E8 frame reflection | 2.0 | 2.0 |
| E8 meta-awareness | 2.6 | 2.2 |
| E8 self-authored criteria | 3.0 | 2.8 |

## Interpretation

The important result is not monotonic movement on every dimension. Different profiles are intended to have different signatures.

- E4 is characterized by stronger dependence on socially validated standards.
- E5 increases differentiation and tolerance for mixed considerations.
- E6 introduces a stable internally endorsed evaluative orientation.
- E7 preserves self-authorship while increasing contextual and psychological differentiation.
- E8 adds explicit reflection on assumptions, frames, and the reasoning process itself.

The held-out topics reproduced these qualitative distinctions after the evaluator was frozen.

## Why the evaluator was frozen

Earlier rubric versions had two opposite failure modes:

1. **Too permissive:** ordinary preferences were sometimes counted as self-authored criteria.
2. **Too strict:** later profiles were required to state an overly formal decision rule, collapsing E7/E8 scores.

The final rubric distinguishes an owned evaluative orientation from both a simple preference and an explicitly formal rule. Once this distinction behaved sensibly on development data, the evaluator was frozen before the held-out runs.

## Scope

This validation supports the simulator as **experimental infrastructure** for controlled behavioral research.

It does not show that real humans with corresponding developmental profiles will respond in the same way, and it does not establish any causal effect of sycophantic model behavior.
