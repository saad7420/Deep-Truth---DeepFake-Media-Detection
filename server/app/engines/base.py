from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class EngineInput:
    media_key: str
    modality: str
    artifact_path: str
    task_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class EngineResult:
    modality: str
    fake_prob: float
    real_prob: float
    confidence: float
    evidence: dict[str, Any] = field(default_factory=dict)
    model_version: str = "unknown"
    processed_at: str = field(default_factory=now_iso)
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class Engine(ABC):

    modality: str = "unknown"

    def supports(self, modality: str) -> bool:
        return modality == self.modality

    @abstractmethod
    def analyze(self, inp: EngineInput) -> EngineResult:
        raise NotImplementedError


# confidence 0 means 'ignore this result'
def neutral_result(modality: str, reason: str) -> EngineResult:
    return EngineResult(
        modality=modality,
        fake_prob=0.5,
        real_prob=0.5,
        confidence=0.0,
        evidence={"note": reason},
        model_version="stub-0",
    )
