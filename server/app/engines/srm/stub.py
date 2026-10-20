from __future__ import annotations

from ..base import Engine, EngineInput, EngineResult, neutral_result


class SRMNoiseStub(Engine):
    modality = "srm"

    def analyze(self, inp: EngineInput) -> EngineResult:
        return neutral_result("srm", "SRM Noise Analysis (M8) not yet built — stub result")
