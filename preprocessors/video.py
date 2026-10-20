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


_MAX_DECODE_PER_SAMPLE = 400
_SEEK_MIN_DURATION_S = 20.0


# seek to each sample time instead of decoding the whole clip
def _decode_sparse(path: Path, n_frames: int) -> np.ndarray | None:
    import av

    try:
        container = av.open(str(path))
    except Exception as exc:
        log.warning(f"sparse decode: cannot open {path.name}: {exc}")
        return None

    try:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        time_base = stream.time_base

        duration = None
        if stream.duration and time_base:
            duration = float(stream.duration * time_base)
        elif container.duration:
            duration = float(container.duration) / av.time_base

        rate = float(stream.average_rate) if stream.average_rate else 0.0

        if not duration or duration <= 0 or not time_base or rate <= 0:
            return None

        if duration < _SEEK_MIN_DURATION_S:
            return None

        total = stream.frames or int(duration * rate)
        if total <= n_frames:
            return None

        targets = [
            float(idx) / rate
            # evenly spaced frame positions across the clip
            for idx in np.linspace(0, total - 1, num=n_frames, dtype=int).tolist()
        ]

        frames: list[np.ndarray] = []
        for target_sec in targets:
            offset = int(target_sec / time_base)
            try:
                container.seek(offset, stream=stream, backward=True, any_frame=False)
            except Exception as exc:
                log.warning(f"sparse decode: seek failed on {path.name}: {exc}")
                return None

            picked = None
            for count, frame in enumerate(container.decode(video=0)):
                picked = frame
                if frame.pts is None:
                    break
                if float(frame.pts * time_base) >= target_sec:
                    break
                if count >= _MAX_DECODE_PER_SAMPLE:
                    break

            if picked is None:
                return None
            frames.append(picked.to_ndarray(format="rgb24"))

        if not frames:
            return None
        return np.stack(frames)

    except Exception as exc:
        log.warning(f"sparse decode failed for {path.name}: {exc}")
        return None
    finally:
        container.close()


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

        raw = _decode_sparse(source_path, pipeline.NUM_FRAMES)
        if raw is None:
            log.info(f"sparse decode unavailable for {source_path.name}; "
                     f"falling back to linear decode")
            raw = pipeline._decode_video(source_path, pipeline.NUM_FRAMES)
        if raw is None:
            raise RuntimeError(f"failed to decode video: {source_path}")

        if len(raw) < pipeline.NUM_FRAMES:
            # repeat the last frame when the clip is too short
            pad = np.repeat(raw[-1:], pipeline.NUM_FRAMES - len(raw), axis=0)
            raw = np.concatenate([raw, pad], axis=0)

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
