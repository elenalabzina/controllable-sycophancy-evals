from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import load_yaml


@dataclass(frozen=True)
class DevelopmentStage:
    order: int
    code: str
    label: str
    short_description: str
    features: dict[str, str]
    behavior: str

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "DevelopmentStage":
        required = {"order", "code", "label", "short_description", "features", "behavior"}
        missing = required - set(raw)
        if missing:
            raise ValueError(f"Stage is missing required keys: {sorted(missing)}")
        return cls(
            order=int(raw["order"]),
            code=str(raw["code"]),
            label=str(raw["label"]),
            short_description=str(raw["short_description"]).strip(),
            features={str(k): str(v) for k, v in dict(raw["features"]).items()},
            behavior=str(raw["behavior"]).strip(),
        )


def load_stages(path: str | Path) -> list[DevelopmentStage]:
    payload = load_yaml(path)
    raw_stages = payload.get("stages")
    if not isinstance(raw_stages, list) or not raw_stages:
        raise ValueError("stages.yaml must contain a non-empty 'stages' list")

    stages = [DevelopmentStage.from_dict(item) for item in raw_stages]
    orders = [stage.order for stage in stages]
    if len(set(orders)) != len(orders):
        raise ValueError("Stage orders must be unique")
    if sorted(orders) != list(range(min(orders), max(orders) + 1)):
        raise ValueError("Stage orders must form a contiguous ordinal sequence")
    return sorted(stages, key=lambda stage: stage.order)


def get_stage(stages: list[DevelopmentStage], order: int) -> DevelopmentStage:
    for stage in stages:
        if stage.order == order:
            return stage
    valid = [stage.order for stage in stages]
    raise KeyError(f"Unknown stage order {order}; valid values: {valid}")
