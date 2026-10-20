from __future__ import annotations

import os

from app.engines.base import Engine
from app.engines.visual.engine import VisualForensicsEngine
from app.engines.image.engine import ImageForensicsEngine
from app.engines.audio.stub import AudioFakeNetStub
from app.engines.audio.wavlm import WavLMAudioEngine
from app.engines.srm.stub import SRMNoiseStub
from app.engines.srm.engine import SRMEngine


def _srm_engine() -> Engine:
    if os.getenv("DEEPTRUTH_SRM_DISABLE", "").lower() in ("1", "true", "yes"):
        return SRMNoiseStub()
    return SRMEngine()


# stub unless an audio checkpoint is configured
def _audio_engine() -> Engine:
    if os.getenv("DEEPTRUTH_AUDIO_CHECKPOINT"):
        return WavLMAudioEngine()
    return AudioFakeNetStub()


class EngineRegistry:
    def __init__(self) -> None:
        self._engines: dict[str, Engine] = {
            "visual": VisualForensicsEngine(),
            "image":  ImageForensicsEngine(),
            "audio":  _audio_engine(),
            "srm":    _srm_engine(),
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
