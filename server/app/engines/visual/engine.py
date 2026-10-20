from __future__ import annotations

import logging
from pathlib import Path

from app._dtp import get_pipeline
from ..artifacts import publish as publish_artifact_map
from ..base import Engine, EngineInput, EngineResult, neutral_result

log = logging.getLogger(__name__)


class VisualForensicsEngine(Engine):
    modality = "visual"

    def analyze(self, inp: EngineInput) -> EngineResult:
        try:
            pipeline = get_pipeline()
            result = pipeline.analyze(Path(inp.artifact_path),
                                      media_kind_hint="video",
                                      # audio is handled separately by the fusion step
                                      modalities=["video"],
                                      threshold=0.5)
        except Exception as e:
            log.exception("[M7] visual analysis failed")
            return neutral_result("visual", f"VisualForensicsEngine failed: {e}")

        ir = result.inferences.get("video")
        if ir is None:
            reason = "; ".join(result.warnings) or "no video inference produced"
            return neutral_result("visual", reason)

        trust = ir["trust_score"]
        if trust != trust:
            return neutral_result("visual", ir.get("rationale", "ensemble returned NaN"))

        return EngineResult(
            modality="visual",
            fake_prob=float(trust),
            real_prob=float(1.0 - trust),
            confidence=float(ir["confidence"]),
            evidence={
                "rationale":        ir.get("rationale"),
                "per_checkpoint":   ir.get("per_model"),
                "face_avg":         ir.get("face_avg"),
                "genvideo_score":   ir.get("genvideo_score"),
                "n_face_detected":  ir.get("n_face_detected"),
                "threshold":        ir.get("threshold"),
                "artifact_map":     publish_artifact_map(ir.get("artifact_map")),
            },
            model_version="vivit-lora-ensemble",
        )
