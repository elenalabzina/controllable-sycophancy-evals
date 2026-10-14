#!/usr/bin/env python

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from controllable_sycophancy.calibration import (  # noqa: E402
    build_stage_calibration,
    format_stage_calibration_for_prompt,
    load_calibration_anchors,
    load_stage_definitions,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="E4")
    parser.add_argument("--examples", type=int, default=1)
    parser.add_argument("--contrasts", type=int, default=1)
    args = parser.parse_args()

    definitions = load_stage_definitions(
        ROOT / "synthetic/calibration/stage_definitions.yaml"
    )
    anchors = load_calibration_anchors(
        ROOT / "synthetic/calibration/example_anchors.csv"
    )

    calibration = build_stage_calibration(
        stage_code=args.stage,
        definitions=definitions,
        anchors=anchors,
        examples_per_stage=args.examples,
        contrast_examples_per_adjacent_stage=args.contrasts,
        min_confidence="medium",
    )

    print(format_stage_calibration_for_prompt(calibration))


if __name__ == "__main__":
    main()
