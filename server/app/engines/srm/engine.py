"""
M8 — Steganalysis Rich Model (SRM) Noise Analysis Stream, engine wrapper.

Goes through `deeptruth_pipeline.Pipeline` exactly like the visual and image
engines, so preprocessing caching (and here specifically: SRM's reuse of
frames the video/image engine already decoded) works the same way.

This engine is deliberately never the source of a case's primary verdict.
`analyser.py` calls it as a supplementary pass after the real modality
engine has already produced the fused risk score, and its result is written
as a `tier: "secondary"` row — a distinct tier from the `"summary"` row the
console treats as the verdict. That is a structural guarantee, not a
convention someone could forget: the client's `readAnalysis()` only ever
picks the first `tier: "summary"` row as `summary`, so a second one from SRM
is simply invisible to anything that computes the headline score. See
app/lib/analysis.ts's `secondarySignals` for where it does surface.

Two reasons for keeping it structurally separate rather than blending it
into the fused score:
  1. The classifier head has not been trained yet (see
     inferencers/srm.py) — exactly the audio situation before
     training, and the same lesson applies: an untrained model's opinion
     must never be allowed to move a verdict, silently or otherwise.
  2. Even once trained, SRM answers a different question than the video/
     image ensembles (local noise-residual anomaly vs. learned content
     features) and a genuinely principled way to combine the two needs
     calibration data neither engine has today. Keeping it as visible,
     separate evidence is honest about that; averaging it into one number
     would not be.
"""
from __future__ import annotations

import logging
from pathlib import Path

from app._dtp import get_pipeline
from ..base import Engine, EngineInput, EngineResult, neutral_result

log = logging.getLogger(__name__)


class SRMEngine(Engine):
    modality = "srm"

    def analyze(self, inp: EngineInput) -> EngineResult:
        media_kind = inp.extra.get("media_kind")
        if media_kind not in ("video", "image"):
            return neutral_result(
                "srm", f"SRMEngine needs extra['media_kind'] to be 'video' or "
                       f"'image' (got {media_kind!r})")

        src = Path(inp.artifact_path)
        if not src.exists():
            return neutral_result("srm", f"file missing: {src}")

        try:
            pipeline = get_pipeline()
            result = pipeline.analyze(src, media_kind_hint=media_kind,
                                      modalities=["srm"])
        except Exception as exc:  # noqa: BLE001
            log.exception("[M8] SRM analysis failed")
            return neutral_result("srm", f"SRMEngine failed: {exc}")

        ir = result.inferences.get("srm")
        if ir is None:
            reason = "; ".join(result.warnings) or "no SRM inference produced"
            return neutral_result("srm", reason)

        fake_prob = ir.get("trust_score")
        if fake_prob is None or fake_prob != fake_prob:  # None or NaN
            # Expected until a checkpoint is trained: real features were
            # computed (kept for evidence/debugging) but no classifier
            # exists yet to score them. Still a neutral result either way —
            # the distinction is only in what evidence.note explains.
            reason = ir.get("rationale") or "SRM: no checkpoint configured"
            result = neutral_result("srm", reason)
            if ir.get("features_computed"):
                result.evidence["features_computed"] = True
                result.evidence["n_frames_used"] = ir.get("n_frames_used")
            return result

        return EngineResult(
            modality="srm",
            fake_prob=float(fake_prob),
            real_prob=float(1.0 - fake_prob),
            confidence=float(ir.get("confidence", 1.0)),
            evidence={
                "rationale":      ir.get("rationale"),
                "per_checkpoint": ir.get("per_model"),  # per-frame scores
                "threshold":      ir.get("threshold"),
                "n_frames_used":  ir.get("n_frames_used"),
                "feature_dim":    ir.get("feature_dim"),
            },
            model_version="srm-5filter-mlp",
        )
