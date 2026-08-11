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

from .preprocessors.base import Preprocessor
from .preprocessors.video import VideoPreprocessor
from .preprocessors.image import ImagePreprocessor
from .preprocessors.stubs import AudioPreprocessor
from .inferencers.base import Inferencer
from .inferencers.video import VideoInferencer
from .inferencers.image import ImageInferencer
from .inferencers.stubs import AudioInferencer
from .storage import CacheStore
from .config import AUDIO_CHECKPOINT, AUDIO_THRESHOLD

log = logging.getLogger(__name__)


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
                 image_preprocessor: Preprocessor | None = None):
        self.store = store or CacheStore()
        self._preprocessors: dict[str, Preprocessor] = {
            "video": VideoPreprocessor(store=self.store),
            "audio": AudioPreprocessor(store=self.store),
            "image": image_preprocessor or ImagePreprocessor(store=self.store),
        }
        self._inferencers: dict[str, Inferencer] = {
            "video": video_inferencer or VideoInferencer(),
            "audio": audio_inferencer or _default_audio_inferencer(),
            "image": image_inferencer or ImageInferencer(),
        }

    def preprocessor_for(self, media_kind: str) -> Preprocessor | None:
        return self._preprocessors.get(media_kind)

    def inferencer_for(self, media_kind: str) -> Inferencer | None:
        return self._inferencers.get(media_kind)

    def replace_preprocessor(self, media_kind: str, p: Preprocessor) -> None:
        self._preprocessors[media_kind] = p

    def replace_inferencer(self, media_kind: str, i: Inferencer) -> None:
        self._inferencers[media_kind] = i
