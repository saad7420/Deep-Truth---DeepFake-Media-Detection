"""Audio inferencer stub.

Replace .predict() with real model code once the audio model is trained.
The image inferencer has moved to its own file (inferencers/image.py).
"""
from __future__ import annotations

from .base import Inferencer, InferenceResult


class AudioInferencer(Inferencer):
    modality = "audio"

    def supports(self, media_kind: str) -> bool:
        return media_kind == "audio"

    def predict(self, media_key, preprocessed, **opts) -> InferenceResult:
        raise NotImplementedError(
            "AudioInferencer not yet implemented. "
            "Wire in your fine-tuned WavLM here."
        )
