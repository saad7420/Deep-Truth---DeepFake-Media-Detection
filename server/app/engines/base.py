"""
The one contract every AI engine (M6 AudioFakeNet, M7 Visual Forensics,
M8 SRM Noise Analysis) implements.

M9 (fusion) only ever depends on this file's shapes. It never imports
anything from engines/audio/, engines/visual/, or engines/srm/ directly
by concrete class — it goes through the Registry (app/registry.py).

This means: whoever builds M6 or M8 for real just has to subclass
`Engine`, return `EngineResult`, and register it. Nothing else in the
codebase (M3, M4, M5, M9) needs to change.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class EngineInput:
    """What M4/M5 hand to an engine. `artifact_path` is modality-specific:
    a WAV path for audio, a directory of extracted frames for visual/SRM."""
    media_key: str
    modality: str                      # "audio" | "visual" | "srm"
    artifact_path: str
    task_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class EngineResult:
    """What every engine hands back to M9. `fake_prob`/`confidence` are
    the only fields M9's fusion math touches — everything else
    (`evidence`) is passed straight through to storage/dashboard."""
    modality: str
    fake_prob: float                   # 0..1 — required
    real_prob: float                   # 1 - fake_prob, kept explicit for DB rows
    confidence: float                  # 0..1 — engine's own certainty signal.
                                        # 0.0 must mean "ignore me" (used by stubs
                                        # so M9's confidence-weighted fusion degrades
                                        # gracefully before a real engine exists).
    evidence: dict[str, Any] = field(default_factory=dict)
    model_version: str = "unknown"
    processed_at: str = field(default_factory=now_iso)
    error: str | None = None           # set instead of raising, when the engine
                                        # can produce a partial/neutral result

    def to_dict(self) -> dict:
        return asdict(self)


class Engine(ABC):
    """Base class for M6/M7/M8. Subclasses must set `modality` and
    implement `analyze()`. Keep `analyze()` synchronous — M5's worker
    calls it inside a thread/process pool, not the event loop directly."""

    modality: str = "unknown"

    def supports(self, modality: str) -> bool:
        return modality == self.modality

    @abstractmethod
    def analyze(self, inp: EngineInput) -> EngineResult:
        """Must not raise for expected failure modes (bad file, model
        unavailable, etc.) — catch internally and return an EngineResult
        with confidence=0.0 and `error` set, so a single bad engine
        doesn't take down the whole fusion step for that task."""
        raise NotImplementedError


def neutral_result(modality: str, reason: str) -> EngineResult:
    """Shared helper for stub engines and error paths: a 0.5/0-confidence
    result that M9's fusion treats as 'contributes nothing'."""
    return EngineResult(
        modality=modality,
        fake_prob=0.5,
        real_prob=0.5,
        confidence=0.0,
        evidence={"note": reason},
        model_version="stub-0",
    )
