"""
M7 — Visual Forensics Engine (video), backed by `deeptruth_pipeline`.

This replaces server's old app/services/model_loader/detector.py +
app/services/inference/predictor.py entirely, rather than patching
them, because those had two compounding bugs:

  1. Checkpoint ensemble weights were on inconsistent scales (face
     checkpoints 0-100, genvideo 0-1), silencing genvideo in the
     weighted average.
  2. Architecturally more serious: the base model was built at
     NUM_FRAMES=32 with no temporal position-embedding resize, but
     the LoRA checkpoints were trained at NUM_FRAMES=16 *after*
     resizing those position embeddings (see
     deeptruth_train.py::_resize_temporal_pos_embed). The resize
     touches base-model weights that aren't part of the LoRA adapter
     file, so server's checkpoints were being evaluated on a model
     with different position embeddings than the ones they were
     trained against.

`deeptruth_pipeline` already gets both of these right:
  - `ensemble.py` does max(face_avg, genvideo) with a face-frame trust
    gate instead of a naively-weighted average.
  - `train_bridge.py` rebuilds the model through the *actual training
    script* (`_build_model`, which calls `_resize_temporal_pos_embed`
    before LoRA is applied), guaranteeing train/inference parity.

Note this engine does its own preprocessing internally (frame decode +
MTCNN face crop, via deeptruth_pipeline's VideoPreprocessor) rather
than consuming pre-extracted frames from M4. `inp.artifact_path` here
is the *original video file path*, not a frame directory — M4's job
for the video branch is reduced to handing this engine the raw file
(after DRM checks / format validation); M4's real preprocessing work
(audio extraction, informative frame sampling) matters for M6, not M7.

The Pipeline instance and all path/checkpoint wiring live in
`app/_dtp.py`, shared with the image engine.
"""
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
        if trust != trust:  # NaN check — ensemble_decide() returns NaN when unusable
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
                # M7 FE-3. A clip's map is a cube, not a grid — ViViT embeds
                # tubelets, so the evidence is localised in time as well as
                # space, and the record carries a per-segment profile
                # alongside the rendered contact sheet.
                "artifact_map":     publish_artifact_map(ir.get("artifact_map")),
            },
            model_version="vivit-lora-ensemble",
        )
