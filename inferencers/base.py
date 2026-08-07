from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class InferenceResult:
    media_key: str
    modality: str
    trust_score: float                           # P(fake) in [0, 1] (or NaN)
    verdict: str                                 # "FAKE" | "REAL" | "UNKNOWN"
    confidence: float                            # 0..1, models-agreement
    per_model: dict[str, float] = field(default_factory=dict)
    rationale: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


class Inferencer(ABC):
    modality: str = "base"

    @abstractmethod
    def supports(self, media_kind: str) -> bool: ...

    @abstractmethod
    def predict(self, media_key: str, preprocessed: dict[str, Any],
                **opts) -> InferenceResult: ...
