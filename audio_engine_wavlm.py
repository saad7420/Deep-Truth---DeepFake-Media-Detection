"""
M6 — AudioFakeNet Engine, WavLM-Large implementation.

Replaces AudioFakeNetStub once DEEPTRUTH_AUDIO_CHECKPOINT is set. Selection
happens in app/registry.py, so wiring a trained checkpoint in is a config
change rather than a code change.

Like the visual and image engines, this goes through
`deeptruth_pipeline.Pipeline` rather than driving the inferencer directly.
That matters for audio specifically: `inp.artifact_path` is the *original
uploaded file* — an mp3, m4a, ogg, whatever the operator dropped in — but
inferencers/audio.py reads it with soundfile, which handles neither mp3 nor
m4a reliably. The pipeline's AudioPreprocessor runs ffmpeg first
(`-vn -ac 1 -ar 16000`) to normalise everything to the 16 kHz mono WAV the
model was trained on, and caches the result under DEEPTRUTH_CACHE so a
re-analysis of the same file skips the transcode.

Failure is soft throughout. A missing checkpoint, a corrupt WAV, a
`strict=True` state-dict mismatch — each returns a neutral result carrying
the reason rather than raising, so audio degrades to "no signal" the way the
stub did instead of failing the whole case. The console renders that as
"Nothing was measured", which is distinct from an inconclusive finding about
the media.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from app._dtp import get_pipeline
from ..base import Engine, EngineInput, EngineResult, neutral_result

log = logging.getLogger(__name__)


def _calibrate(raw: float, threshold: float) -> float:
    """Map a raw P(fake) onto a 0-1 scale whose midpoint is the model's own
    decision threshold.

    This is not cosmetic. `analyser._status()` bands every modality the same
    way — >=65% manipulated, <=35% authentic — which assumes a probability
    centred near 0.5. The audio model's equal-error point is nowhere near
    0.5: heavy class weighting (ASVspoof LA is ~1:9 bonafide:spoof) squeezes
    its raw scores toward zero, putting the measured threshold around 0.0003.

    Feeding raw scores into those bands inverts the verdict. A clip the model
    calls FAKE at p=0.01 is 35x above its own threshold, yet lands at "1%
    risk" and is reported to the operator as *authentic*. Verified across the
    range before this was added.

    The mapping is piecewise linear and strictly monotonic, so it preserves
    ranking and therefore EER exactly — it moves where the midpoint sits, it
    does not invent separation the model lacks. Genuine probability
    calibration (Platt scaling, isotonic regression) would need a held-out
    set and belongs in training, not here; the raw score is kept in evidence
    so nothing is lost.
    """
    threshold = min(max(threshold, 1e-9), 1.0 - 1e-9)
    raw = min(max(raw, 0.0), 1.0)

    if raw <= threshold:
        return 0.5 * (raw / threshold)
    return 0.5 + 0.5 * (raw - threshold) / (1.0 - threshold)


class WavLMAudioEngine(Engine):
    modality = "audio"

    def __init__(self, threshold: float | None = None):
        # Resolved from config (env var, else the checkpoint's metadata.json)
        # so it can be passed explicitly on every call. Without that,
        # Pipeline.analyze()'s own `threshold=0.5` default reaches
        # WavLMAudioInferencer.predict() via opts and silently overrides the
        # configured value — the measured EER point would be discarded on
        # every single analysis.
        if threshold is None:
            try:
                from app._dtp import bootstrap
                bootstrap()
                from deeptruth_pipeline.config import AUDIO_THRESHOLD
                threshold = float(AUDIO_THRESHOLD)
            except Exception:  # noqa: BLE001
                threshold = 0.5
        self._threshold = threshold

        # An ASVspoof-only checkpoint is a documented failure case, not a
        # hypothetical one: models trained solely on ASVspoof 2019 LA
        # routinely collapse toward chance on real-world audio recorded
        # outside the corpus's studio conditions (the "In-the-Wild"
        # benchmark literature reports in-domain EER around 6% rising to
        # ~50% out-of-domain). A checkpoint in that state must not present
        # itself as equivalent to the video and image engines, which are
        # past that stage.
        #
        # Default ON — opt-out, not opt-in — because the failure is silent:
        # the model returns confident, specific, wrong answers. The safe
        # assumption for any freshly wired checkpoint is that it has not
        # been validated on diverse real audio yet.
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
            # The stub inferencer is still registered — DEEPTRUTH_AUDIO_CHECKPOINT
            # was not set when the Registry was constructed.
            return neutral_result(
                "audio",
                "no audio checkpoint configured — set DEEPTRUTH_AUDIO_CHECKPOINT "
                "and restart the server.")
        except RuntimeError as exc:
            # The likeliest RuntimeError here is a state-dict mismatch out of
            # load_state_dict(strict=True). Worth naming, because the cause is
            # architecture drift between training and inferencers/audio.py
            # rather than a bad path.
            return neutral_result(
                "audio",
                f"audio model did not load ({exc}). If this mentions missing or "
                f"unexpected keys, the trained architecture has drifted from "
                f"inferencers/audio.py.")
        except Exception as exc:  # noqa: BLE001
            log.exception("[M6] audio analysis failed")
            return neutral_result("audio", f"WavLMAudioEngine failed: {exc}")

        ir = result.inferences.get("audio")
        if ir is None:
            reason = "; ".join(result.warnings) or "no audio inference produced"
            return neutral_result("audio", reason)

        raw_prob = ir.get("trust_score")
        if raw_prob is None or raw_prob != raw_prob:  # None or NaN
            return neutral_result("audio",
                                  ir.get("rationale", "audio model returned NaN"))

        raw_prob = float(raw_prob)
        threshold = float(ir.get("threshold", self._threshold))
        fake_prob = _calibrate(raw_prob, threshold)

        rationale = ir.get("rationale")
        if self._experimental:
            # explainEvidence() in the console renders `rationale` verbatim
            # above the risk score — the one place an operator cannot miss it.
            # Kept specific to this checkpoint's known limitation rather than
            # a generic hedge, so it does not become boilerplate people skim.
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
                # The model's untouched output, kept so the reported risk
                # score can always be traced back to what the network
                # actually produced.
                "raw_fake_prob":  round(raw_prob, 6),
                "calibrated":     True,
                "sample_rate":    ir.get("sample_rate"),
                "max_audio_sec":  ir.get("max_audio_sec"),
                # Reserved for a future spectrogram / attention overlay, so
                # adding one later does not change this engine's return shape.
                "heatmap_path":   None,
            },
            model_version="wavlm-large-asvspoof19",
        )
