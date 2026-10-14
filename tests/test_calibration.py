from controllable_sycophancy.calibration import (
    build_stage_calibration,
    load_calibration_anchors,
    load_stage_definitions,
)


def test_demo_calibration_loads_without_private_participant_data():
    definitions = load_stage_definitions(
        "synthetic/calibration/stage_definitions.yaml"
    )
    anchors = load_calibration_anchors(
        "synthetic/calibration/example_anchors.csv"
    )
    calibration = build_stage_calibration(
        stage_code="E6",
        definitions=definitions,
        anchors=anchors,
        examples_per_stage=1,
        contrast_examples_per_adjacent_stage=1,
    )
    assert calibration.stage_code == "E6"
    assert calibration.anchors
    assert calibration.anchors[0].anonymous_protocol_anchor.startswith("synthetic_demo_")
