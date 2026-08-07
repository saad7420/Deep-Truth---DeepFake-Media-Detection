"""
M6 — AudioFakeNet Engine (STUB).

Replace this with a real implementation by subclassing `Engine` with
modality="audio" and registering it in app/registry.py. A working
starting point already exists in deeptruth_pipeline/inferencers/audio.py
(WavLM-Large + 3-layer head, trained on ASVspoof 2019 LA) — see the
note at the bottom of this file for how to adapt it.

Until then, this stub returns a neutral (confidence=0.0) result so M9's
fusion ignores the audio channel entirely rather than dragging every
verdict toward 0.5.
"""
from __future__ import annotations

from ..base import Engine, EngineInput, EngineResult, neutral_result


class AudioFakeNetStub(Engine):
    modality = "audio"

    def analyze(self, inp: EngineInput) -> EngineResult:
        return neutral_result("audio", "AudioFakeNet (M6) not yet wired in — stub result")


# ─────────────────────────────────────────────────────────────────────────
# Adapting the existing WavLM inferencer (deeptruth_pipeline/inferencers/audio.py)
# into this contract, once someone picks up M6 for real:
#
#   from app._dtp import bootstrap; bootstrap()
#   from deeptruth_pipeline.inferencers.audio import WavLMAudioInferencer
#
#   class AudioFakeNetEngine(Engine):
#       modality = "audio"
#       def __init__(self):
#           self._inner = WavLMAudioInferencer()
#
#       def analyze(self, inp: EngineInput) -> EngineResult:
#           try:
#               ir = self._inner.predict(inp.media_key,
#                                         {"audio_wav_path": inp.artifact_path})
#               return EngineResult(
#                   modality="audio",
#                   fake_prob=ir.trust_score,
#                   real_prob=1.0 - ir.trust_score,
#                   confidence=ir.confidence,
#                   evidence={"rationale": ir.rationale, **ir.extra},
#                   model_version="wavlm-large-asvspoof19",
#               )
#           except Exception as e:
#               return neutral_result("audio", f"AudioFakeNet failed: {e}")
#
# `inp.artifact_path` must be a 16kHz mono WAV — M4's demux step already
# produces this via ffmpeg (-vn -ac 1 -ar 16000).
