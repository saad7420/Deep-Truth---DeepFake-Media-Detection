"""ViViT + LoRA ensemble inferencer.

Discovers all *_lora_best checkpoints under CHECKPOINT_DIR, routes each one
to its required preprocessing branch (face vs full-frame), runs inference,
and combines per-checkpoint scores via ensemble_decide().

Loaded models are cached on the instance so subsequent predict() calls
across multiple videos don't pay reload cost. Call .unload() to free GPU
memory when done.
"""
from __future__ import annotations
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from .base import Inferencer, InferenceResult
from ..config import (
    CHECKPOINT_DIR, FACE_CHECKPOINT_NAMES, FULL_FRAME_CHECKPOINT_NAMES,
    USE_FP16, DEFAULT_THRESHOLD,
)
from ..ensemble import ensemble_decide
from .. import train_bridge

log = logging.getLogger(__name__)


def _discover_checkpoints(checkpoint_dir: Path) -> list[dict]:
    found, seen = [], set()
    if not checkpoint_dir.exists():
        return found
    for entry in sorted(checkpoint_dir.iterdir()):
        stem = entry.stem
        if not stem.endswith("_lora_best"):
            continue
        ds = stem[: -len("_lora_best")]
        if ds in seen:
            continue
        if entry.is_dir() and (entry / "adapter_config.json").exists():
            kind = "dir"
        elif entry.is_file() and entry.suffix == ".pt":
            kind = "pt"
        else:
            continue
        if ds in FACE_CHECKPOINT_NAMES:
            preprocess = "face"
        elif ds in FULL_FRAME_CHECKPOINT_NAMES:
            preprocess = "full"
        else:
            log.warning(f"unknown checkpoint {ds}; defaulting to face")
            preprocess = "face"
        found.append({"name": ds, "path": str(entry),
                      "kind": kind, "preprocess": preprocess})
        seen.add(ds)
    return found


class VideoInferencer(Inferencer):
    modality = "video"

    def __init__(self, checkpoint_dir: Path | None = None,
                 device: str | None = None,
                 threshold: float = DEFAULT_THRESHOLD,
                 keep_models_loaded: bool = True):
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else CHECKPOINT_DIR
        self._explicit_device = device
        self.threshold = threshold
        self.keep_loaded = keep_models_loaded
        self._processor = None
        self._device = None
        self._checkpoints: list[dict] | None = None
        self._loaded_models: dict[str, Any] = {}

    def supports(self, media_kind: str) -> bool:
        return media_kind == "video"

    def _setup(self):
        if self._processor is not None:
            return
        import torch
        from transformers import VivitImageProcessor

        pipeline = train_bridge.load()
        pipeline.set_seed(pipeline.SEED)

        if self._explicit_device:
            self._device = torch.device(self._explicit_device)
        else:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._processor = VivitImageProcessor.from_pretrained(pipeline.MODEL_ID)
        self._checkpoints = _discover_checkpoints(self.checkpoint_dir)
        if not self._checkpoints:
            raise RuntimeError(f"no checkpoints found in {self.checkpoint_dir}")

        log.info(f"VideoInferencer ready: device={self._device}, "
                 f"fp16={USE_FP16 and self._device.type == 'cuda'}, "
                 f"checkpoints={[c['name'] for c in self._checkpoints]}")

    def _get_model(self, ckpt: dict, pipeline):
        name = ckpt["name"]
        if name in self._loaded_models:
            return self._loaded_models[name]
        model = pipeline._build_model("lora")
        model = pipeline._load_checkpoint(model, ckpt["path"], "lora")
        model.eval().to(self._device)
        if self.keep_loaded:
            self._loaded_models[name] = model
        return model

    def _release_model(self, name: str):
        if name in self._loaded_models:
            del self._loaded_models[name]
            self._maybe_empty_cuda()

    def _maybe_empty_cuda(self):
        if self._device is not None and self._device.type == "cuda":
            import torch
            torch.cuda.empty_cache()

    def unload(self):
        self._loaded_models.clear()
        self._maybe_empty_cuda()

    def predict(self, media_key: str, preprocessed: dict[str, Any],
                **opts) -> InferenceResult:
        self._setup()

        face_path = preprocessed.get("face_frames_path")
        full_path = preprocessed.get("full_frames_path")
        n_face = int(preprocessed.get("n_face_detected") or 0)

        face_frames = np.load(face_path) if face_path else None
        full_frames = np.load(full_path) if full_path else None
        if face_frames is None and full_frames is None:
            raise RuntimeError("no frames available for video inference")

        pipeline = train_bridge.load()
        per_ckpt: dict[str, float] = {}

        for ckpt in self._checkpoints:
            frames = face_frames if ckpt["preprocess"] == "face" else full_frames
            if frames is None:
                log.warning(f"  skip {ckpt['name']}: no {ckpt['preprocess']} frames")
                continue
            t0 = time.time()
            try:
                model = self._get_model(ckpt, pipeline)
                p_fake = self._predict_one(model, frames)
                per_ckpt[ckpt["name"]] = p_fake
                log.info(f"  {ckpt['name']:<22s} P(fake)={p_fake:.4f}  "
                         f"({time.time() - t0:.1f}s)")
            except Exception as e:
                log.warning(f"  {ckpt['name']} failed: {e}")
                if not self.keep_loaded:
                    self._release_model(ckpt["name"])
            finally:
                if not self.keep_loaded:
                    self._release_model(ckpt["name"])

        decision = ensemble_decide(per_ckpt, n_face)
        threshold = float(opts.get("threshold", self.threshold))
        ens = decision["ensemble"]
        if ens != ens:
            verdict = "UNKNOWN"
        else:
            verdict = "FAKE" if ens >= threshold else "REAL"

        return InferenceResult(
            media_key=media_key,
            modality="video",
            trust_score=float(ens) if ens == ens else float("nan"),
            verdict=verdict,
            confidence=float(decision["confidence"]),
            per_model=per_ckpt,
            rationale=decision["rationale"],
            extra={
                "face_avg":         decision["face_avg"],
                "genvideo_score":   decision["genvideo_score"],
                "face_trusted":     decision["face_trusted"],
                "n_face_detected":  n_face,
                "threshold":        threshold,
            },
        )

    def _predict_one(self, model, frames: np.ndarray) -> float:
        import torch
        inputs = self._processor(list(frames), return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self._device)
        with torch.no_grad():
            if self._device.type == "cuda" and USE_FP16:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    out = model(pixel_values=pixel_values,
                                interpolate_pos_encoding=True)
            else:
                out = model(pixel_values=pixel_values,
                            interpolate_pos_encoding=True)
            logits = out.logits.float()
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        return float(probs[1])
