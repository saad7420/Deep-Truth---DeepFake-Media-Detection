"""Wires preprocessors and inferencers to media kinds.

Video and image are implemented end-to-end. Audio is too, once a checkpoint
is configured: set DEEPTRUTH_AUDIO_CHECKPOINT and the real WavLM inferencer
is selected automatically in place of the stub. Without it, audio
preprocessing (ffmpeg WAV extract) still runs but inference stays stubbed,
so the channel reports "no signal" rather than a fabricated verdict.

Replace components with .replace_preprocessor() / .replace_inferencer() to
inject custom configurations.
"""
from __future__ import annotations

import logging
import os

from .preprocessors.base import Preprocessor
from .preprocessors.video import VideoPreprocessor
from .preprocessors.image import ImagePreprocessor
from .preprocessors.srm import SRMPreprocessor
from .preprocessors.stubs import AudioPreprocessor
from .inferencers.base import Inferencer
from .inferencers.video import VideoInferencer
from .inferencers.image import ImageInferencer
from .inferencers.srm import SRMInferencer
from .inferencers.stubs import AudioInferencer
from .storage import CacheStore
from .config import (AUDIO_CHECKPOINT, AUDIO_THRESHOLD,
                     SRM_CHECKPOINT, SRM_THRESHOLD)

log = logging.getLogger(__name__)


class _NeutralSRMInferencer(Inferencer):
    """Used when no SRM checkpoint is configured. Distinct from SRMInferencer
    itself (which also soft-fails to neutral without a checkpoint) so that a
    completely disabled SRM channel never even attempts feature extraction —
    useful for a deployment that wants zero SRM overhead rather than "compute
    features, then discard them" on every single video/image case."""
    modality = "srm"

    def supports(self, media_kind: str) -> bool:
        return media_kind in ("video", "image")

    def predict(self, media_key, preprocessed, **opts):
        from .inferencers.base import InferenceResult
        return InferenceResult(
            media_key=media_key, modality="srm", trust_score=float("nan"),
            verdict="UNKNOWN", confidence=0.0,
            rationale="SRM Noise Analysis (M8) not yet trained — no checkpoint configured")


def _default_srm_inferencer() -> Inferencer:
    """SRMInferencer always runs real feature extraction (cheap: pure numpy
    convolution over a handful of small frames, no GPU, no torch import
    needed for that half). It only needs a checkpoint for the classifier
    step, and soft-fails to a neutral result with the features attached as
    evidence when one isn't configured yet — so unlike audio, there is no
    reason to swap the whole inferencer out; the same object handles both
    states.

    DEEPTRUTH_SRM_DISABLE skips this entirely for a deployment that wants
    zero SRM overhead (e.g. very high case volume, resource-constrained).
    """
    if os.environ.get("DEEPTRUTH_SRM_DISABLE", "").lower() in ("1", "true", "yes"):
        return _NeutralSRMInferencer()

    log.info(
        "srm: feature extraction active" +
        (f" — checkpoint {SRM_CHECKPOINT}" if SRM_CHECKPOINT
         else " (no checkpoint: features only, no verdict)"))
    return SRMInferencer(checkpoint_path=SRM_CHECKPOINT, threshold=SRM_THRESHOLD)


def _default_audio_inferencer() -> Inferencer:
    """Real WavLM inferencer when a checkpoint is configured, else the stub.

    Import is deferred into this function because inferencers/audio.py pulls
    in torch and transformers at call time; a deployment running only the
    video branch should not pay that cost, and a broken audio install should
    not stop the whole registry from constructing.
    """
    if not AUDIO_CHECKPOINT:
        return AudioInferencer()

    try:
        from .inferencers.audio import WavLMAudioInferencer
        log.info(f"audio: using WavLM checkpoint {AUDIO_CHECKPOINT} "
                 f"(threshold {AUDIO_THRESHOLD})")
        return WavLMAudioInferencer(checkpoint_path=AUDIO_CHECKPOINT,
                                    threshold=AUDIO_THRESHOLD)
    except Exception as exc:  # noqa: BLE001
        log.warning(f"audio: could not build WavLM inferencer ({exc}); "
                    f"falling back to the stub")
        return AudioInferencer()


class Registry:
    def __init__(self, store: CacheStore | None = None,
                 video_inferencer: Inferencer | None = None,
                 image_inferencer: Inferencer | None = None,
                 audio_inferencer: Inferencer | None = None,
                 srm_inferencer: Inferencer | None = None,
                 image_preprocessor: Preprocessor | None = None):
        self.store = store or CacheStore()
        self._preprocessors: dict[str, Preprocessor] = {
            "video": VideoPreprocessor(store=self.store),
            "audio": AudioPreprocessor(store=self.store),
            "image": image_preprocessor or ImagePreprocessor(store=self.store),
            "srm":   SRMPreprocessor(store=self.store),
        }
        self._inferencers: dict[str, Inferencer] = {
            "video": video_inferencer or VideoInferencer(),
            "audio": audio_inferencer or _default_audio_inferencer(),
            "image": image_inferencer or ImageInferencer(),
            "srm":   srm_inferencer or _default_srm_inferencer(),
        }

    def preprocessor_for(self, media_kind: str) -> Preprocessor | None:
        return self._preprocessors.get(media_kind)

    def inferencer_for(self, media_kind: str) -> Inferencer | None:
        return self._inferencers.get(media_kind)

    def replace_preprocessor(self, media_kind: str, p: Preprocessor) -> None:
        self._preprocessors[media_kind] = p

    def replace_inferencer(self, media_kind: str, i: Inferencer) -> None:
        self._inferencers[media_kind] = i
