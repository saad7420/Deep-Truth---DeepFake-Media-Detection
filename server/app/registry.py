"""
Wires each modality to its Engine implementation.

This is the ONLY file that should need editing when M6 or M8 go from
stub to real: swap the import + registration, nothing else in M3/M4/M5/M9
changes, since they all depend on app.engines.base's shapes, not on
these concrete classes directly.
"""
from __future__ import annotations

import os

from app.engines.base import Engine
from app.engines.visual.engine import VisualForensicsEngine
from app.engines.image.engine import ImageForensicsEngine
from app.engines.audio.stub import AudioFakeNetStub
from app.engines.audio.wavlm import WavLMAudioEngine
from app.engines.srm.stub import SRMNoiseStub


def _audio_engine() -> Engine:
    """Real WavLM engine when a checkpoint is configured, stub otherwise.

    Selection is by environment rather than by editing this file, so the same
    build runs with or without the audio model and nobody has to remember to
    revert a code change. WavLMAudioEngine loads lazily and degrades to a
    neutral result if the checkpoint turns out to be bad, so setting the
    variable can never stop the server from starting — the worst case is
    audio reporting "no signal", which is exactly what the stub does anyway.
    """
    if os.getenv("DEEPTRUTH_AUDIO_CHECKPOINT"):
        return WavLMAudioEngine()
    return AudioFakeNetStub()


class EngineRegistry:
    def __init__(self) -> None:
        self._engines: dict[str, Engine] = {
            "visual": VisualForensicsEngine(),   # M7  — real (video, ViViT)
            "image":  ImageForensicsEngine(),    # M7b — real (image, ViT-B/16)
            "audio":  _audio_engine(),           # M6  — real iff a checkpoint is set
            "srm":    SRMNoiseStub(),            # M8  — stub, swap when ready
        }

    def get(self, modality: str) -> Engine | None:
        return self._engines.get(modality)

    def replace(self, modality: str, engine: Engine) -> None:
        self._engines[modality] = engine

    def all_modalities(self) -> list[str]:
        return list(self._engines.keys())


_registry: EngineRegistry | None = None


def get_registry() -> EngineRegistry:
    global _registry
    if _registry is None:
        _registry = EngineRegistry()
    return _registry
