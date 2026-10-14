from pathlib import Path
import csv

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_public_example_config_paths_exist():
    config_path = ROOT / "configs/generate_validation.example.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    required_paths = [
        config["model_config"],
        config["evaluation_model_config"],
        config["stages_path"],
        config["topics_path"],
        config["calibration"]["definitions_path"],
        config["calibration"]["anchors_path"],
    ]

    for relative_path in required_paths:
        assert (ROOT / relative_path).exists(), relative_path


def test_public_demo_anchor_schema_has_no_binary_group():
    path = ROOT / "synthetic/calibration/example_anchors.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert "binary_group" not in (reader.fieldnames or [])
        rows = list(reader)

    assert rows
    assert all(row["anonymous_protocol_anchor"].startswith("synthetic_demo_") for row in rows)


def test_public_stage_range_is_e2_through_e8():
    path = ROOT / "synthetic/stages.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    codes = [stage["code"] for stage in payload["stages"]]
    assert codes == ["E2", "E3", "E4", "E5", "E6", "E7", "E8"]
