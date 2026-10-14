import pytest

from controllable_sycophancy.prompts import build_respondent_system_prompt
from controllable_sycophancy.stages import get_stage, load_stages


def test_prompt_contains_stage_conditioning_but_role_rules_hide_it_from_output():
    stage = get_stage(load_stages("synthetic/stages.yaml"), 5)
    prompt = build_respondent_system_prompt(
        stage=stage,
        belief_position=0.8,
        confidence=0.7,
        negative_position="Position A",
        positive_position="Position B",
    )
    assert "E6" in prompt
    assert "Never mention developmental psychology" in prompt
    assert "Position A" in prompt
    assert "Position B" in prompt


def test_prompt_rejects_invalid_belief_position():
    stage = get_stage(load_stages("synthetic/stages.yaml"), 5)
    with pytest.raises(ValueError):
        build_respondent_system_prompt(
            stage=stage,
            belief_position=1.5,
            confidence=0.7,
            negative_position="A",
            positive_position="B",
        )
