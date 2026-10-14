from controllable_sycophancy.stages import get_stage, load_stages


def test_public_stage_file_has_seven_ordered_profiles():
    stages = load_stages("synthetic/stages.yaml")
    assert len(stages) == 7
    assert [stage.order for stage in stages] == list(range(1, 8))


def test_anchor_stage_labels():
    stages = load_stages("synthetic/stages.yaml")
    assert get_stage(stages, 3).code == "E4"
    assert get_stage(stages, 5).code == "E6"
    assert get_stage(stages, 7).code == "E8"
