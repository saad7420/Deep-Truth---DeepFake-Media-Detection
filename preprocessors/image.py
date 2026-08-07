"""Image preprocessor. Produces up to two branches:

    image_224.npy        (224, 224, 3) uint8  — whole image, fed to every
                                                 checkpoint except #8.
    image_face_224.npy   (224, 224, 3) uint8  — MTCNN crop with 30% margin,
                                                 only written when MTCNN
                                                 detects a face; fed to
                                                 image_ffpp_facecrop only.

The whole-image branch always exists. The face branch is optional: if MTCNN
isn't available, fails, or finds no face, we skip the face crop, set
stats["face_detected"] = False, and the inferencer will simply not score
image_ffpp_facecrop on this input.

Cropping policy matches training (ffpp_facecrop_pipeline.py):
  * Use the LARGEST detected face by box area when multiple are returned.
  * Pad the box by IMAGE_FACE_MARGIN (default 0.30) on each side.
  * Clip to image bounds, then resize to IMAGE_INPUT_SIZE × IMAGE_INPUT_SIZE.
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from .base import Preprocessor
from ..storage import CacheRecord
from ..config import IMAGE_INPUT_SIZE, IMAGE_FACE_MARGIN

log = logging.getLogger(__name__)

_MTCNN_SINGLETON = None


def _get_mtcnn(device: str | None = None):
    """Lazily build a single MTCNN instance per process. Returns None if
    facenet-pytorch isn't installed or the load fails — callers should
    handle that as "skip the face branch", not as a fatal error."""
    global _MTCNN_SINGLETON
    if _MTCNN_SINGLETON is not None:
        return _MTCNN_SINGLETON
    try:
        import torch
        from facenet_pytorch import MTCNN  # type: ignore
    except ImportError:
        log.info("facenet-pytorch not installed; face-crop branch disabled")
        return None
    try:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        # keep_all=True so we can pick the largest face when multiple appear;
        # post_process=False because we want raw pixel coords back, not the
        # MTCNN-internal normalised tensor.
        _MTCNN_SINGLETON = MTCNN(keep_all=True, post_process=False, device=device)
        log.info(f"MTCNN ready on {device}")
        return _MTCNN_SINGLETON
    except Exception as e:
        log.warning(f"MTCNN init failed: {e}; face-crop branch disabled")
        return None


def _pick_largest_box(boxes) -> list[float] | None:
    """boxes is whatever MTCNN.detect returned for one image: an (N, 4) array
    of [x1, y1, x2, y2] or None when no face was found. Return the box with
    the largest area, or None."""
    if boxes is None or len(boxes) == 0:
        return None
    best, best_area = None, -1.0
    for b in boxes:
        if b is None or len(b) < 4:
            continue
        x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if area > best_area:
            best, best_area = (x1, y1, x2, y2), area
    return list(best) if best is not None else None


def _crop_with_margin(pil_image, box: list[float], margin: float):
    """Pad the detected box by `margin` of its size on every side, clip to
    image bounds, return a PIL crop. Mirrors training-time behaviour."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    px, py = w * margin, h * margin
    img_w, img_h = pil_image.size
    x1 = max(0, int(x1 - px))
    y1 = max(0, int(y1 - py))
    x2 = min(img_w, int(x2 + px))
    y2 = min(img_h, int(y2 + py))
    if x2 <= x1 or y2 <= y1:
        return None
    return pil_image.crop((x1, y1, x2, y2))


class ImagePreprocessor(Preprocessor):
    name = "image"

    def __init__(self, store=None, *, detect_faces: bool = True,
                 face_margin: float | None = None):
        super().__init__(store=store)
        self.detect_faces = detect_faces
        self.face_margin = (face_margin if face_margin is not None
                            else IMAGE_FACE_MARGIN)

    def supports(self, media_kind: str) -> bool:
        return media_kind == "image"

    def run(self, source_path: Path, media_key: str,
            record: CacheRecord, *, force: bool = False) -> dict[str, Any]:
        out_dir = self.store.subdir(media_key, "image")
        whole_path = out_dir / "image_224.npy"
        face_path  = out_dir / "image_face_224.npy"
        stats_path = out_dir / "stats.json"

        # Cache short-circuit.
        if not force and stats_path.exists() and whole_path.exists():
            stats = json.loads(stats_path.read_text())
            return {
                "image_path":      str(whole_path),
                "face_image_path": (str(face_path)
                                    if stats.get("face_detected")
                                       and face_path.exists()
                                    else None),
                "face_detected":   bool(stats.get("face_detected", False)),
                "face_box":        stats.get("face_box"),
                "orig_size":       stats.get("orig_size"),
                "cached":          True,
            }

        from PIL import Image
        with Image.open(source_path) as im:
            im = im.convert("RGB")
            orig_w, orig_h = im.size

            # Whole-image branch (always written).
            whole = im.resize((IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE), Image.BICUBIC)
            np.save(whole_path, np.asarray(whole, dtype=np.uint8))

            # Face-crop branch (optional).
            face_detected = False
            face_box: list[float] | None = None
            if self.detect_faces:
                mtcnn = _get_mtcnn()
                if mtcnn is not None:
                    try:
                        boxes, _ = mtcnn.detect(im)
                        face_box = _pick_largest_box(boxes)
                        if face_box is not None:
                            crop = _crop_with_margin(im, face_box,
                                                     self.face_margin)
                            if crop is not None:
                                crop = crop.resize(
                                    (IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE),
                                    Image.BICUBIC)
                                np.save(face_path,
                                        np.asarray(crop, dtype=np.uint8))
                                face_detected = True
                    except Exception as e:
                        log.warning(
                            f"MTCNN detect failed on {source_path.name}: {e}")

            if not face_detected and face_path.exists():
                # Stale crop from an earlier run — wipe.
                face_path.unlink()

        stats = {
            "size":          [IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE],
            "orig_size":     [orig_w, orig_h],
            "face_detected": face_detected,
            "face_box":      face_box,
            "face_margin":   self.face_margin if face_detected else None,
        }
        stats_path.write_text(json.dumps(stats, indent=2))

        record.completed["image"] = True
        record.extra.setdefault("image", {}).update(stats)
        self.store.save(record)

        return {
            "image_path":      str(whole_path),
            "face_image_path": str(face_path) if face_detected else None,
            "face_detected":   face_detected,
            "face_box":        face_box,
            "orig_size":       [orig_w, orig_h],
            "cached":          False,
        }
