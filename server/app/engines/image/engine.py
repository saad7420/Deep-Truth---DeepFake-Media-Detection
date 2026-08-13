"""
M7b — Image Forensics Engine, backed by `deeptruth_pipeline`.

Before the merge this slot did not exist: the server's vendored copy of the
pipeline had only a stubbed ImageInferencer, so analyser.py returned a hard
"inconclusive / not implemented" row for every image case. The pipeline now
ships a real image path, and this engine is the adapter for it.

What runs underneath (deeptruth_pipeline/inferencers/image.py):

  * Eight ViT-B/16 + LoRA checkpoints under images_checkpoints/, discovered
    as image_<slug>_lora_best. Five are "generalist" (AI-generated image
    detection across content types), three are face-specialists.
  * image_ffpp_facecrop is fed the MTCNN crop; the other seven see the whole
    224x224 image. If no face is found, the crop branch is skipped and the
    face group is down-weighted by ensemble policy rather than dropped.
  * Scores are softmax probabilities, fused by ensemble.image_ensemble_decide.

Same contract as the visual engine: `inp.artifact_path` is the original
uploaded file, and preprocessing happens inside the pipeline (with its own
on-disk cache), not in M4.
"""
from __future__ import annotations

import logging
import os
import shutil
import uuid
from pathlib import Path

from app._dtp import get_pipeline
from ..base import Engine, EngineInput, EngineResult, neutral_result

log = logging.getLogger(__name__)

# Where rendered artifact maps are published so the console can load them.
# Separate from uploads/: these are derived, reproducible from the cache, and
# safe to delete wholesale, which is not true of the evidence files.
ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", "artifacts"))
ARTIFACT_BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")


def _publish_artifact_map(record: dict | None) -> dict | None:
    """Copy a rendered map out of the pipeline cache into the served directory.

    The pipeline writes the PNG beside its preprocessed tensors, under
    DEEPTRUTH_CACHE — deliberately, so it is cached and evicted on the same
    terms as everything else derived from that file. But that directory is not
    web-served, and mounting it would expose the whole preprocessing cache.
    Copying one file is the cheaper, narrower option.

    Returns the record with `url` added and the local `path` dropped: a
    filesystem path on the analysis host is of no use to a browser and only
    discloses server layout.
    """
    if not record:
        return None

    src = record.get("path")
    if not src or not Path(src).is_file():
        return None

    try:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        name = f"{uuid.uuid4().hex}.png"
        shutil.copyfile(src, ARTIFACT_DIR / name)
    except OSError as exc:
        log.warning("[M7b] could not publish artifact map: %s", exc)
        return None

    published = {k: v for k, v in record.items() if k != "path"}
    published["url"] = f"{ARTIFACT_BASE_URL}/artifacts/{name}"
    return published


class ImageForensicsEngine(Engine):
    modality = "image"

    def analyze(self, inp: EngineInput) -> EngineResult:
        try:
            pipeline = get_pipeline()
            result = pipeline.analyze(Path(inp.artifact_path),
                                      media_kind_hint="image",
                                      modalities=["image"],
                                      threshold=0.5)
        except Exception as e:
            log.exception("[M7b] image analysis failed")
            return neutral_result("image", f"ImageForensicsEngine failed: {e}")

        ir = result.inferences.get("image")
        if ir is None:
            reason = "; ".join(result.warnings) or "no image inference produced"
            return neutral_result("image", reason)

        trust = ir["trust_score"]
        if trust != trust:  # NaN — no checkpoint produced a usable score
            return neutral_result("image", ir.get("rationale",
                                                  "image ensemble returned NaN"))

        return EngineResult(
            modality="image",
            fake_prob=float(trust),
            real_prob=float(1.0 - trust),
            confidence=float(ir["confidence"]),
            evidence={
                "rationale":       ir.get("rationale"),
                "per_checkpoint":  ir.get("per_model"),
                "per_model_role":  ir.get("per_model_role"),
                "policy":          ir.get("policy"),
                "generalist_avg":  ir.get("generalist_avg"),
                "face_avg":        ir.get("face_avg"),
                "face_detected":   ir.get("face_detected"),
                "face_trusted":    ir.get("face_trusted"),
                "n_generalist":    ir.get("n_generalist"),
                "n_face":          ir.get("n_face"),
                "skipped":         ir.get("skipped"),
                # M7 FE-3. `artifact_map` carries the overlay URL plus the
                # region summary and, importantly, whether the evidence was
                # localised at all — see the report UI, which says so rather
                # than implying a region that does not exist.
                "artifact_map":    _publish_artifact_map(ir.get("artifact_map")),
            },
            model_version="vit-b16-lora-ensemble",
        )
