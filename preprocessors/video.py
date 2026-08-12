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

# Upper bound on how many frames may be decoded while walking forward from a
# keyframe to a sample point. Typical keyframe intervals are 2-10 seconds
# (~60-300 frames), so this is generous; its job is only to stop a pathological
# file (no keyframes, or a seek that lands far away) from turning one sample
# into a full-file scan.
_MAX_DECODE_PER_SAMPLE = 400

# Below this duration, seeking is not worth it. Sixteen seeks each cost a
# keyframe walk, so on a short clip that is *more* work than one linear pass —
# measured at 2.5s seeking versus 1.7s linear on a 10-second clip, against
# 2.8s versus 39s on a 5-minute one. The crossover sits around 15-20 seconds.
_SEEK_MIN_DURATION_S = 20.0


def _decode_sparse(path: Path, n_frames: int) -> np.ndarray | None:
    """Decode `n_frames` evenly spaced frames by seeking to each one.

    Why this exists
    ---------------
    `deeptruth_train._decode_video` picks the same evenly-spaced frames, but it
    finds them by decoding the file from the beginning and counting: because
    the last sample point sits near the end, it decodes essentially the whole
    video to keep 16 frames. Cost therefore scales with the clip's *duration*
    rather than with the 16 frames actually wanted — measured at 1.4s for a
    10-second clip and 40s for a 5-minute one, on the same 16 frames.

    That is the difference between a usable and an unusable submission for
    anything longer than a minute, so inference seeks instead: jump to each
    sample timestamp, decode forward to the nearest frame, keep it. Work is
    then proportional to `n_frames` and the keyframe interval, and flat in
    duration.

    The *selection policy* is deliberately identical to the linear decoder's:
    `_sample_indices` takes `np.linspace(0, total - 1, n)`, and the timestamps
    below are those same frame indices divided by the frame rate. On a
    constant-rate file this lands on the same frames; on a variable-rate one it
    lands on the nearest decodable frame at or after each point. Keeping the
    policy identical matters because these frames feed checkpoints that were
    fine-tuned on frames chosen exactly this way.

    Returns None for anything it cannot handle confidently — no duration or
    rate metadata, a container that will not seek, a clip short enough that
    seeking is the slower option — and the caller falls back to the linear
    decoder, which is slow but always correct.
    """
    import av

    try:
        container = av.open(str(path))
    except Exception as exc:  # noqa: BLE001 — av raises a wide family
        log.warning(f"sparse decode: cannot open {path.name}: {exc}")
        return None

    try:
        stream = container.streams.video[0]
        # Let ffmpeg use all cores for the short forward walks.
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

        # Short clips are cheaper to decode straight through — see the constant.
        if duration < _SEEK_MIN_DURATION_S:
            return None

        total = stream.frames or int(duration * rate)
        if total <= n_frames:
            return None

        # The linear decoder's exact choice, expressed as timestamps.
        targets = [
            float(idx) / rate
            for idx in np.linspace(0, total - 1, num=n_frames, dtype=int).tolist()
        ]

        frames: list[np.ndarray] = []
        for target_sec in targets:
            offset = int(target_sec / time_base)
            try:
                container.seek(offset, stream=stream, backward=True, any_frame=False)
            except Exception as exc:  # noqa: BLE001
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

    except Exception as exc:  # noqa: BLE001
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

        # Seek to the frames we want; fall back to the training-time linear
        # decoder when the container will not cooperate. See _decode_sparse.
        raw = _decode_sparse(source_path, pipeline.NUM_FRAMES)
        if raw is None:
            log.info(f"sparse decode unavailable for {source_path.name}; "
                     f"falling back to linear decode")
            raw = pipeline._decode_video(source_path, pipeline.NUM_FRAMES)
        if raw is None:
            raise RuntimeError(f"failed to decode video: {source_path}")

        # Short clips can yield fewer frames than requested; the model needs
        # exactly NUM_FRAMES, so repeat the last one as the linear decoder does.
        if len(raw) < pipeline.NUM_FRAMES:
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
