from __future__ import annotations

from ..base import Engine, EngineInput, EngineResult, neutral_result


class AudioFakeNetStub(Engine):
    modality = "audio"

    def analyze(self, inp: EngineInput) -> EngineResult:
        return neutral_result("audio", "AudioFakeNet (M6) not yet wired in — stub result")
