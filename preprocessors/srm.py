"""
M8 preprocessor — SRM Noise Analysis Stream.

Per the module's original design note: SRM needs no new decode pass. It
reuses exactly the frames VideoPreprocessor / ImagePreprocessor already
extracted and cached, and simply reduces them to feature vectors on top.

This preprocessor therefore *delegates* rather than reimplements: for a
video it hands off to VideoPreprocessor (whose cache means the second call
costs a stats.json read, not a re-decode), for an image to ImagePreprocessor,
and then runs the SRM filter bank over whichever frames come back.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from .base import Preprocessor
from .image import ImagePreprocessor
from .video import VideoPreprocessor
from .srm_filters import SRMFilterBank, patch_feature_vector, to_grayscale
from ..storage import CacheRecord

log = logging.getLogger(__name__)

# How many frames to featurise per video. SRM features are cheap (pure numpy
# convolution, no GPU), but there is no forensic benefit to running all 16
# decoded frames through it — noise/splice-boundary statistics are stable
# across a clip, and this keeps a video case's SRM pass to a fraction of a
# second rather than 16x that.
MAX_VIDEO_FRAMES = 4


class SRMPreprocessor(Preprocessor):
    name = "srm"

    def supports(self, media_kind: str) -> bool:
        return media_kind in ("video", "image")

    def run(self, source_path: Path, media_key: str,
            record: CacheRecord, *, force: bool = False) -> dict[str, Any]:
        out_dir = self.store.subdir(media_key, "srm")
        feat_path = out_dir / "features.npy"
        stats_path = out_dir / "stats.json"

        if not force and stats_path.exists() and feat_path.exists():
            stats = json.loads(stats_path.read_text())
            return {"features_path": str(feat_path), **stats, "cached": True}

        bank = SRMFilterBank.default()

        if record.media_kind == "video":
            frames = self._video_frames(source_path, media_key, record, force)
        else:
            frames = self._image_frame(source_path, media_key, record, force)

        vecs = np.stack([
            patch_feature_vector(to_grayscale(f), bank) for f in frames
        ])  # (n_frames, feature_dim)

        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(feat_path, vecs)

        stats = {
            "n_frames_used": int(vecs.shape[0]),
            "feature_dim": int(vecs.shape[1]),
            "n_filters": bank.n_filters,
        }
        stats_path.write_text(json.dumps(stats))

        return {"features_path": str(feat_path), **stats, "cached": False}

    def _video_frames(self, source_path: Path, media_key: str,
                      record: CacheRecord, force: bool) -> np.ndarray:
        # Delegates to VideoPreprocessor's own cache: if M7 already ran for
        # this media_key, this is a stats.json read, not a re-decode.
        inner = VideoPreprocessor(store=self.store)
        pre = inner.run(source_path, media_key, record, force=force)
        full = np.load(pre["full_frames_path"])  # (16, 224, 224, 3) uint8

        n = min(MAX_VIDEO_FRAMES, full.shape[0])
        idx = np.linspace(0, full.shape[0] - 1, n).round().astype(int)
        return full[idx]

    def _image_frame(self, source_path: Path, media_key: str,
                     record: CacheRecord, force: bool) -> np.ndarray:
        inner = ImagePreprocessor(store=self.store)
        pre = inner.run(source_path, media_key, record, force=force)
        whole = np.load(pre["image_path"])  # (224, 224, 3) uint8
        return whole[None, ...]  # (1, 224, 224, 3) — one "frame"