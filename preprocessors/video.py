"""Video preprocessor. Produces both branches needed by the inferencer:
    face_frames  -> (16, 224, 224, 3) uint8, MTCNN-cropped, for face checkpoints
    full_frames  -> (16, 224, 224, 3) uint8, center-cropped, for genvideo

Both are derived from a single decode call so the file is touched once.
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from .base import Preprocessor
from ..storage import CacheRecord
from .. import train_bridge

log = logging.getLogger(__name__)


class VideoPreprocessor(Preprocessor):
    name = "video"

    def supports(self, media_kind: str) -> bool:
        return media_kind == "video"

    def run(self, source_path: Path, media_key: str,
            record: CacheRecord, *, force: bool = False) -> dict[str, Any]:
        out_dir = self.store.subdir(media_key, "video")
        face_path = out_dir / "face_frames.npy"
        full_path = out_dir / "full_frames.npy"
        stats_path = out_dir / "stats.json"

        if not force and stats_path.exists() and full_path.exists():
            stats = json.loads(stats_path.read_text())
            return {
                "face_frames_path": (str(face_path)
                                     if stats.get("has_face_branch") and face_path.exists()
                                     else None),
                "full_frames_path": str(full_path),
                "n_face_detected": stats["n_face_detected"],
                "num_frames":      stats["num_frames"],
                "cached":          True,
            }

        pipeline = train_bridge.load()
        raw = pipeline._decode_video(source_path, pipeline.NUM_FRAMES)
        if raw is None:
            raise RuntimeError(f"failed to decode video: {source_path}")

        full_frames = pipeline._resize_only(raw)
        np.save(full_path, full_frames)

        face_frames = None
        n_detected = 0
        try:
            from PIL import Image
            detector = pipeline._get_face_detector()
            pil_frames = [Image.fromarray(f) for f in raw]
            boxes_list, _ = detector.detect(pil_frames)
            n_detected = sum(1 for b in boxes_list if b is not None and len(b) > 0)
            face_frames = pipeline._crop_faces(raw)
            np.save(face_path, face_frames)
        except Exception as e:
            log.warning(f"face preprocess failed for {source_path.name}: {e}")
            if face_path.exists():
                face_path.unlink()

        stats = {
            "num_frames":       int(pipeline.NUM_FRAMES),
            "n_face_detected":  int(n_detected),
            "has_face_branch":  face_frames is not None,
        }
        stats_path.write_text(json.dumps(stats, indent=2))

        record.completed["video"] = True
        record.extra.setdefault("video", {}).update(stats)
        self.store.save(record)

        return {
            "face_frames_path": str(face_path) if face_frames is not None else None,
            "full_frames_path": str(full_path),
            "n_face_detected":  n_detected,
            "num_frames":       int(pipeline.NUM_FRAMES),
            "cached":           False,
        }
