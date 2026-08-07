"""Wires preprocessors and inferencers to media kinds.

Default registry has video and image implemented end-to-end and audio
inference stubbed (audio preprocessing — WAV extract — is real).
Replace components with .replace_preprocessor() / .replace_inferencer() once
the audio model is trained, or to inject custom configurations.
"""
from __future__ import annotations

from .preprocessors.base import Preprocessor
from .preprocessors.video import VideoPreprocessor
from .preprocessors.image import ImagePreprocessor
from .preprocessors.stubs import AudioPreprocessor
from .inferencers.base import Inferencer
from .inferencers.video import VideoInferencer
from .inferencers.image import ImageInferencer
from .inferencers.stubs import AudioInferencer
from .storage import CacheStore


class Registry:
    def __init__(self, store: CacheStore | None = None,
                 video_inferencer: Inferencer | None = None,
                 image_inferencer: Inferencer | None = None,
                 image_preprocessor: Preprocessor | None = None):
        self.store = store or CacheStore()
        self._preprocessors: dict[str, Preprocessor] = {
            "video": VideoPreprocessor(store=self.store),
            "audio": AudioPreprocessor(store=self.store),
            "image": image_preprocessor or ImagePreprocessor(store=self.store),
        }
        self._inferencers: dict[str, Inferencer] = {
            "video": video_inferencer or VideoInferencer(),
            "audio": AudioInferencer(),
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
