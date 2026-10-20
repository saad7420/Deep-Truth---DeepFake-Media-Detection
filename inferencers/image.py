from __future__ import annotations
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import artifact_map
from .base import Inferencer, InferenceResult
from ..config import (
    IMAGE_CHECKPOINT_DIR, IMAGE_CHECKPOINT_INFO, IMAGE_BACKBONE_ID,
    IMAGE_INPUT_SIZE, IMAGE_FACE_MARGIN, USE_FP16, DEFAULT_THRESHOLD,
    ARTIFACT_MAPS_ENABLED, ARTIFACT_MAP_MIN_SCORE,
)
from ..ensemble import image_ensemble_decide

log = logging.getLogger(__name__)

_CKPT_PREFIX = "image_"
_CKPT_SUFFIX = "_lora_best"


def _head_source(entry: Path) -> str:
    try:
        cfg = json.loads((entry / "adapter_config.json").read_text())
    except (OSError, json.JSONDecodeError):
        cfg = {}
    if "classifier" in (cfg.get("modules_to_save") or []):
        return "adapter"
    if (entry / "classifier_head.pt").exists():
        return "file"
    return "none"


def _discover_image_checkpoints(ckpt_dir: Path) -> list[dict]:
    if not ckpt_dir.exists():
        return []

    found: dict[str, dict] = {}
    for entry in ckpt_dir.iterdir():
        if not entry.is_dir():
            continue
        name = entry.name
        if not (name.startswith(_CKPT_PREFIX) and name.endswith(_CKPT_SUFFIX)):
            continue
        slug = name[len(_CKPT_PREFIX): -len(_CKPT_SUFFIX)]
        if slug in found:
            continue
        if not (entry / "adapter_config.json").exists():
            log.warning(f"skipping {name}: no adapter_config.json")
            continue
        head_path = entry / "classifier_head.pt"
        info = IMAGE_CHECKPOINT_INFO.get(slug)
        if info is None:
            log.warning(f"unknown image checkpoint slug '{slug}'; "
                        f"treating as generalist (whole-image)")
            info = {"role": "generalist", "needs_face_crop": False,
                    "auc": 0.5,
                    "role_in_curriculum": "unknown (not in curriculum)"}
        found[slug] = {
            "slug":            slug,
            "path":            str(entry),
            "head_path":       str(head_path) if head_path.exists() else None,
            "head_source":     _head_source(entry),
            "role":            info["role"],
            "needs_face_crop": bool(info["needs_face_crop"]),
            "auc":             float(info["auc"]),
            "curriculum":      info.get("role_in_curriculum", ""),
        }

    order = list(IMAGE_CHECKPOINT_INFO.keys())
    def keyfn(rec):
        s = rec["slug"]
        return (0, order.index(s)) if s in order else (1, s)
    return sorted(found.values(), key=keyfn)


def _build_base_model(backbone_id: str):
    from transformers import ViTForImageClassification
    return ViTForImageClassification.from_pretrained(
        backbone_id,
        num_labels=2,
        ignore_mismatched_sizes=True,
    )


def _load_classifier_head(model, head_path: str | None) -> bool:
    if head_path is None:
        return False
    import torch
    state = torch.load(head_path, map_location="cpu")

    wrapped = {k.split(".")[-1]: v for k, v in state.items()
               if k.startswith("modules_to_save.")}
    if wrapped:
        state = wrapped

    classifier = None
    if hasattr(model, "classifier"):
        classifier = model.classifier
    elif hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        classifier = getattr(model.base_model.model, "classifier", None)

    if classifier is None:
        log.warning("  could not locate classifier layer to load head into")
        return False

    if "weight" in state and "bias" in state and len(state) == 2:
        try:
            classifier.load_state_dict(state, strict=True)
            return True
        except Exception as e:
            log.warning(f"  plain head load failed: {e}")

    stripped = {k.split(".", 1)[1]: v
                for k, v in state.items() if k.startswith("classifier.")}
    if stripped:
        try:
            classifier.load_state_dict(stripped, strict=True)
            return True
        except Exception as e:
            log.warning(f"  stripped head load failed: {e}")

    try:
        missing, unexpected = model.load_state_dict(state, strict=False)
        if any("classifier" in k for k in (set(state.keys()))) and \
                not any("classifier" in k for k in missing):
            return True
    except Exception as e:
        log.warning(f"  full-state head load failed: {e}")

    log.warning(f"  classifier_head.pt had unexpected keys: {list(state.keys())[:5]}")
    return False


def _build_and_load(ckpt: dict, backbone_id: str):
    from peft import PeftModel
    base = _build_base_model(backbone_id)
    model = PeftModel.from_pretrained(base, ckpt["path"])

    source = ckpt.get("head_source", "none")
    if source == "adapter":
        log.debug(f"  {ckpt['slug']}: classifier head restored from adapter")
    elif source == "file":
        if not _load_classifier_head(model, ckpt["head_path"]):
            raise RuntimeError(
                f"classifier_head.pt present but unloadable "
                f"({ckpt['head_path']}) and the adapter carries no head")
        log.info(f"  {ckpt['slug']}: classifier head loaded from "
                 f"classifier_head.pt")
    else:
        raise RuntimeError(
            "no trained classifier head: adapter_config.json has no "
            "modules_to_save=['classifier'] and there is no "
            "classifier_head.pt next to it")

    model.eval()
    return model


class ImageInferencer(Inferencer):
    modality = "image"

    def __init__(self,
                 checkpoint_dir: Path | None = None,
                 backbone_id: str | None = None,
                 device: str | None = None,
                 threshold: float = DEFAULT_THRESHOLD,
                 keep_models_loaded: bool = True):
        self.checkpoint_dir = (Path(checkpoint_dir) if checkpoint_dir
                               else IMAGE_CHECKPOINT_DIR)
        self.backbone_id = backbone_id or IMAGE_BACKBONE_ID
        self._explicit_device = device
        self.threshold = threshold
        self.keep_loaded = keep_models_loaded

        self._processor = None
        self._device = None
        self._checkpoints: list[dict] | None = None
        self._loaded_models: dict[str, Any] = {}

    def supports(self, media_kind: str) -> bool:
        return media_kind == "image"

    def _setup(self):
        if self._processor is not None:
            return
        import torch
        from transformers import ViTImageProcessor

        if self._explicit_device:
            self._device = torch.device(self._explicit_device)
        else:
            self._device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")

        self._processor = ViTImageProcessor.from_pretrained(self.backbone_id)
        self._checkpoints = _discover_image_checkpoints(self.checkpoint_dir)
        if not self._checkpoints:
            raise RuntimeError(
                f"no image checkpoints found in {self.checkpoint_dir} "
                f"(looking for image_<slug>_lora_best directories with "
                f"adapter_config.json)")
        names = [c['slug'] for c in self._checkpoints]
        log.info(f"ImageInferencer ready: device={self._device}, "
                 f"fp16={USE_FP16 and self._device.type == 'cuda'}, "
                 f"checkpoints={names}")
        headless = [c["slug"] for c in self._checkpoints
                    if c.get("head_source") == "none"]
        if headless:
            log.warning(f"  checkpoints with no trained classifier head will be "
                        f"skipped at scoring time: {headless}")

    def _get_model(self, ckpt: dict):
        slug = ckpt["slug"]
        if slug in self._loaded_models:
            return self._loaded_models[slug]
        model = _build_and_load(ckpt, self.backbone_id).to(self._device)
        if self.keep_loaded:
            self._loaded_models[slug] = model
        return model

    def _release_model(self, slug: str):
        if slug in self._loaded_models:
            del self._loaded_models[slug]
            self._maybe_empty_cuda()

    def _maybe_empty_cuda(self):
        if self._device is not None and self._device.type == "cuda":
            import torch
            torch.cuda.empty_cache()

    def unload(self):
        self._loaded_models.clear()
        self._maybe_empty_cuda()

    def _preprocess_to_tensor(self, arr: np.ndarray):
        inputs = self._processor(images=arr, return_tensors="pt")
        return inputs["pixel_values"].to(self._device)

    def _forward(self, model, pixel_values) -> float:
        import torch
        # peft's wrapper breaks vision models, the base model still has the lora weights
        call_target = model.get_base_model() if hasattr(model, "get_base_model") else model
        with torch.no_grad():
            if self._device.type == "cuda" and USE_FP16:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    out = call_target(pixel_values=pixel_values)
            else:
                out = call_target(pixel_values=pixel_values)
            logits = out.logits.float()
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        # class 1 means fake
        return float(probs[1])

    @staticmethod
    def _face_rect_in_frame(preprocessed: dict[str, Any]) -> tuple[int, int, int, int] | None:
        box = preprocessed.get("face_box")
        orig = preprocessed.get("orig_size")
        if not box or not orig or len(box) != 4 or len(orig) != 2:
            return None

        try:
            x1, y1, x2, y2 = (float(v) for v in box)
            ow, oh = (float(v) for v in orig)
        except (TypeError, ValueError):
            return None
        if ow <= 0 or oh <= 0:
            return None

        pad_x = (x2 - x1) * IMAGE_FACE_MARGIN
        pad_y = (y2 - y1) * IMAGE_FACE_MARGIN
        cx1 = max(0.0, x1 - pad_x)
        cy1 = max(0.0, y1 - pad_y)
        cx2 = min(ow, x2 + pad_x)
        cy2 = min(oh, y2 + pad_y)
        if cx2 <= cx1 or cy2 <= cy1:
            return None

        sx = IMAGE_INPUT_SIZE / ow
        sy = IMAGE_INPUT_SIZE / oh
        return (int(cx1 * sx), int(cy1 * sy), int(cx2 * sx), int(cy2 * sy))

    # score every checkpoint, then ensemble the results
    def predict(self, media_key: str, preprocessed: dict[str, Any],
                **opts) -> InferenceResult:
        self._setup()

        whole_path = preprocessed.get("image_path")
        face_path  = preprocessed.get("face_image_path")
        face_detected = bool(preprocessed.get("face_detected", False))

        if not whole_path:
            raise RuntimeError("no whole-image tensor available for inference")

        whole_arr = np.load(whole_path)
        face_arr  = np.load(face_path) if face_path else None

        whole_px = self._preprocess_to_tensor(whole_arr)
        face_px  = (self._preprocess_to_tensor(face_arr)
                    if face_arr is not None else None)

        per_ckpt: dict[str, float] = {}
        per_ckpt_role: dict[str, str] = {}
        skipped: list[dict] = []

        want_maps = bool(opts.get("artifact_maps", ARTIFACT_MAPS_ENABLED))
        frame_maps: dict[str, np.ndarray] = {}
        face_rect = self._face_rect_in_frame(preprocessed) if want_maps else None

        for ckpt in self._checkpoints:
            slug = ckpt["slug"]
            if ckpt["needs_face_crop"]:
                if face_px is None:
                    skipped.append({"slug": slug,
                                    "reason": "no face crop available "
                                              "(MTCNN found nothing or "
                                              "disabled)"})
                    continue
                # the face-crop checkpoint gets the crop, the others the whole image
                pixel_values = face_px
            else:
                pixel_values = whole_px

            t0 = time.time()
            try:
                model = self._get_model(ckpt)
                p_fake = self._forward(model, pixel_values)
                per_ckpt[slug] = p_fake
                per_ckpt_role[slug] = ckpt["role"]
                log.info(f"  image_{slug:<16s}  P(fake)={p_fake:.4f}  "
                         f"role={ckpt['role']:<10s}  "
                         f"branch={'face' if ckpt['needs_face_crop'] else 'whole'}"
                         f"  ({time.time() - t0:.2f}s)")

                # heat map only for checkpoints that flag something, never changes the score
                if want_maps and p_fake >= ARTIFACT_MAP_MIN_SCORE:
                    cam = artifact_map.compute_cam(model, pixel_values)
                    if cam is not None and cam.max() > 0:
                        rect = face_rect if ckpt["needs_face_crop"] else None
                        if ckpt["needs_face_crop"] and rect is None:
                            log.debug(f"  {slug}: face rect unknown, map dropped")
                        else:
                            frame_maps[slug] = artifact_map.place_in_frame(
                                cam, IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE, rect)

            except Exception as e:
                log.warning(f"  image_{slug} failed: {e}")
                skipped.append({"slug": slug, "reason": f"inference error: {e}"})
            finally:
                if not self.keep_loaded:
                    self._release_model(slug)

        decision = image_ensemble_decide(per_ckpt, face_detected)
        threshold = float(opts.get("threshold", self.threshold))
        ens = decision["ensemble"]
        if ens != ens:
            verdict = "UNKNOWN"
        else:
            verdict = "FAKE" if ens >= threshold else "REAL"

        artifact = self._write_artifact_map(whole_path, whole_arr,
                                            frame_maps, per_ckpt)

        return InferenceResult(
            media_key=media_key,
            modality="image",
            trust_score=float(ens) if ens == ens else float("nan"),
            verdict=verdict,
            confidence=float(decision["confidence"]),
            per_model=per_ckpt,
            rationale=decision["rationale"],
            extra={
                "policy":          decision["policy"],
                "generalist_avg":  decision["generalist_avg"],
                "face_avg":        decision["face_avg"],
                "face_trusted":    decision["face_trusted"],
                "face_detected":   face_detected,
                "n_generalist":    decision["n_generalist"],
                "n_face":          decision["n_face"],
                "per_model_role":  per_ckpt_role,
                "skipped":         skipped,
                "threshold":       threshold,
                "artifact_map":    artifact,
            },
        )

    def _write_artifact_map(self, whole_path: str, whole_arr: np.ndarray,
                            frame_maps: dict[str, np.ndarray],
                            per_ckpt: dict[str, float]) -> dict | None:
        if not frame_maps:
            return None

        try:
            combined = artifact_map.combine(frame_maps, per_ckpt)
            if combined is None:
                return None

            out_path = Path(whole_path).parent / "artifact_map.png"
            artifact_map.render_overlay(whole_arr, combined, str(out_path))

            record = artifact_map.summarise(combined)
            record["path"] = str(out_path)
            record["method"] = "grad-cam/vit-last-block-prenorm"
            record["contributors"] = sorted(frame_maps.keys())
            log.info(f"  artifact map: {len(frame_maps)} checkpoint(s), "
                     f"concentration={record['concentration']:.2f}, "
                     f"localised={record['localised']}")
            return record
        except Exception as e:
            log.warning(f"  artifact map rendering failed: {e}")
            return None
