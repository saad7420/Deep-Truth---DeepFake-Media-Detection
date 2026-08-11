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
from app.engines.srm.engine import SRMEngine


def _srm_engine() -> Engine:
    """SRMEngine always runs, checkpoint or not.

    Matches the pipeline-level fix in registry.py's _default_srm_inferencer():
    feature extraction is cheap (pure numpy, no GPU) and useful as evidence
    on its own, so it is not worth gating behind a checkpoint the way audio's
    whole engine is. SRMEngine -> SRMInferencer.predict() already handles
    "no checkpoint" internally by returning a neutral result with the
    computed features attached rather than a verdict — see
    inferencers/srm.py.

    Unlike audio, SRM is never the primary source of a case's verdict even
    once a checkpoint exists (see engines/srm/engine.py's docstring) — this
    selection only controls whether analyser.py's supplementary SRM pass
    runs the real pipeline or the original bare "not built" stub.

    DEEPTRUTH_SRM_DISABLE opts a deployment out entirely if even the small
    per-case feature-extraction cost is unwanted.
    """
    if os.getenv("DEEPTRUTH_SRM_DISABLE", "").lower() in ("1", "true", "yes"):
        return SRMNoiseStub()
    return SRMEngine()


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
            "srm":    _srm_engine(),             # M8  — real iff a checkpoint is set
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
