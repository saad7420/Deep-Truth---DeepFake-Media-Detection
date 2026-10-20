from __future__ import annotations

import logging
import os
from pathlib import Path

from app._dtp import get_pipeline
from ..base import Engine, EngineInput, EngineResult, neutral_result

log = logging.getLogger(__name__)


def _calibrate(raw: float, threshold: float) -> float:
    threshold = min(max(threshold, 1e-9), 1.0 - 1e-9)
    raw = min(max(raw, 0.0), 1.0)

    if raw <= threshold:
        return 0.5 * (raw / threshold)
    return 0.5 + 0.5 * (raw - threshold) / (1.0 - threshold)


class WavLMAudioEngine(Engine):
    modality = "audio"

    def __init__(self, threshold: float | None = None):
        if threshold is None:
            try:
                from app._dtp import bootstrap
                bootstrap()
                from deeptruth_pipeline.config import AUDIO_THRESHOLD
                threshold = float(AUDIO_THRESHOLD)
            except Exception:
                threshold = 0.5
        self._threshold = threshold

        self._experimental = os.getenv(
            "DEEPTRUTH_AUDIO_EXPERIMENTAL", "true"
        ).lower() not in ("false", "0", "no")

    def analyze(self, inp: EngineInput) -> EngineResult:
        src = Path(inp.artifact_path)
        if not src.exists():
            return neutral_result("audio", f"audio file missing: {src}")

        try:
            pipeline = get_pipeline()
            result = pipeline.analyze(src,
                                      media_kind_hint="audio",
                                      modalities=["audio"],
                                      threshold=self._threshold)
        except FileNotFoundError as exc:
            return neutral_result(
                "audio",
                f"audio checkpoint not found ({exc}). Train one with "
                f"train_pipeline/train_audio_asvspoof.py, then set "
                f"DEEPTRUTH_AUDIO_CHECKPOINT.")
        except NotImplementedError:
            return neutral_result(
                "audio",
                "no audio checkpoint configured — set DEEPTRUTH_AUDIO_CHECKPOINT "
                "and restart the server.")
        except RuntimeError as exc:
            return neutral_result(
                "audio",
                f"audio model did not load ({exc}). If this mentions missing or "
                f"unexpected keys, the trained architecture has drifted from "
                f"inferencers/audio.py.")
        except Exception as exc:
            log.exception("[M6] audio analysis failed")
            return neutral_result("audio", f"WavLMAudioEngine failed: {exc}")

        ir = result.inferences.get("audio")
        if ir is None:
            reason = "; ".join(result.warnings) or "no audio inference produced"
            return neutral_result("audio", reason)

        raw_prob = ir.get("trust_score")
        if raw_prob is None or raw_prob != raw_prob:
            return neutral_result("audio",
                                  ir.get("rationale", "audio model returned NaN"))

        raw_prob = float(raw_prob)
        threshold = float(ir.get("threshold", self._threshold))
        fake_prob = _calibrate(raw_prob, threshold)

        rationale = ir.get("rationale")
        if self._experimental:
            caveat = (
                "EXPERIMENTAL CHECKPOINT: trained only on ASVspoof 2019 LA "
                "(studio recordings, 6 synthesis methods). Models trained on "
                "this corpus alone are known to misclassify genuine "
                "real-world recordings as fake — do not treat this verdict as "
                "reliable until it has been validated on diverse real audio. "
                "Set DEEPTRUTH_AUDIO_EXPERIMENTAL=false once that is done."
            )
            rationale = f"{caveat} | {rationale}" if rationale else caveat

        return EngineResult(
            modality="audio",
            fake_prob=fake_prob,
            real_prob=1.0 - fake_prob,
            confidence=float(ir.get("confidence", 1.0)),
            evidence={
                "rationale":      rationale,
                "experimental":   self._experimental,
                "per_checkpoint": ir.get("per_model"),
                "threshold":      threshold,
                "raw_fake_prob":  round(raw_prob, 6),
                "calibrated":     True,
                "sample_rate":    ir.get("sample_rate"),
                "max_audio_sec":  ir.get("max_audio_sec"),
                "heatmap_path":   None,
            },
            model_version="wavlm-large-asvspoof19",
        )
