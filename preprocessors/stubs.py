"""Audio preprocessor.

Audio: we already know the universal first step for audio inputs is to
normalise to a 16 kHz mono WAV, so we do that. The model-specific feature
extraction (mel spectrograms etc.) lives in the audio inferencer.

The image preprocessor has moved to its own file (preprocessors/image.py)
now that it produces both whole-image and MTCNN-cropped branches.
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any

from .base import Preprocessor
from ..storage import CacheRecord
from ..demux import extract_audio_wav

log = logging.getLogger(__name__)


class AudioPreprocessor(Preprocessor):
    name = "audio"

    def supports(self, media_kind: str) -> bool:
        return media_kind == "audio"

    def run(self, source_path: Path, media_key: str,
            record: CacheRecord, *, force: bool = False) -> dict[str, Any]:
        out_dir = self.store.subdir(media_key, "audio")
        wav_path = out_dir / "audio_16k.wav"
        stats_path = out_dir / "stats.json"

        if not force and wav_path.exists() and stats_path.exists():
            stats = json.loads(stats_path.read_text())
            return {"audio_wav_path": str(wav_path), "cached": True, **stats}

        extract_audio_wav(source_path, wav_path, sr=16000)
        stats = {"sample_rate": 16000, "channels": 1}
        stats_path.write_text(json.dumps(stats, indent=2))

        record.completed["audio"] = True
        self.store.save(record)
        return {"audio_wav_path": str(wav_path), "cached": False, **stats}
