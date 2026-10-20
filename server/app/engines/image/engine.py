from __future__ import annotations

import logging
from pathlib import Path

from app._dtp import get_pipeline
from ..artifacts import publish as publish_artifact_map
from ..base import Engine, EngineInput, EngineResult, neutral_result

log = logging.getLogger(__name__)


class ImageForensicsEngine(Engine):
    modality = "image"

    def analyze(self, inp: EngineInput) -> EngineResult:
        try:
            pipeline = get_pipeline()
            result = pipeline.analyze(Path(inp.artifact_path),
                                      media_kind_hint="image",
                                      modalities=["image"],
                                      threshold=0.5)
        except Exception as e:
            log.exception("[M7b] image analysis failed")
            return neutral_result("image", f"ImageForensicsEngine failed: {e}")

        ir = result.inferences.get("image")
        if ir is None:
            reason = "; ".join(result.warnings) or "no image inference produced"
            return neutral_result("image", reason)

        trust = ir["trust_score"]
        if trust != trust:
            return neutral_result("image", ir.get("rationale",
                                                  "image ensemble returned NaN"))

        return EngineResult(
            modality="image",
            fake_prob=float(trust),
            real_prob=float(1.0 - trust),
            confidence=float(ir["confidence"]),
            evidence={
                "rationale":       ir.get("rationale"),
                "per_checkpoint":  ir.get("per_model"),
                "per_model_role":  ir.get("per_model_role"),
                "policy":          ir.get("policy"),
                "generalist_avg":  ir.get("generalist_avg"),
                "face_avg":        ir.get("face_avg"),
                "face_detected":   ir.get("face_detected"),
                "face_trusted":    ir.get("face_trusted"),
                "n_generalist":    ir.get("n_generalist"),
                "n_face":          ir.get("n_face"),
                "skipped":         ir.get("skipped"),
                "threshold":       ir.get("threshold"),
                "artifact_map":    publish_artifact_map(ir.get("artifact_map")),
            },
            model_version="vit-b16-lora-ensemble",
        )
