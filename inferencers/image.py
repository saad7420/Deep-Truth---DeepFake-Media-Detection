"""ViT-B/16 + LoRA image ensemble inferencer.

Discovers every directory under IMAGE_CHECKPOINT_DIR whose name matches
"image_<slug>_lora_best" and whose slug is registered in
IMAGE_CHECKPOINT_INFO. For each one:

  * Builds a fresh ViTForImageClassification base (num_labels=2).
  * Wraps it with the checkpoint's PEFT adapter via PeftModel.from_pretrained.
  * Makes sure the *trained* classifier head is in place.
    >>> This is the #1 ensemble bug from the training era: a ViT whose LoRA
        loaded but whose head didn't scores noise, and the ensemble has no way
        to tell that apart from a confident model. There are two ways a head
        gets restored, and a checkpoint must satisfy exactly one of them:

          a) adapter_config.json lists "classifier" in modules_to_save, so the
             head rides inside adapter_model.safetensors and PeftModel restores
             it. Seven of the eight checkpoints are built this way.
          b) The adapter has no modules_to_save, and the trained head sits in a
             separate classifier_head.pt written by the training script.
             image_ffpp_lora_best is the one checkpoint that needs this.

        A checkpoint satisfying neither would run with a randomly-initialised
        head, so it is dropped from the ensemble instead of being scored — see
        `_head_source()` and the "no trained classifier head" skip reason.

Routing:
  * Checkpoints with needs_face_crop=False (seven of the eight) → fed the
    whole-image tensor.
  * Checkpoint image_ffpp_facecrop_lora_best → fed the MTCNN-cropped tensor.
    If the preprocessor didn't produce one (no face / MTCNN unavailable), the
    checkpoint is skipped for this input.

Outputs are softmax probabilities (NEVER raw logits) so the ensemble mean
isn't biased by per-checkpoint logit temperature.

Loaded models cache on the instance across predict() calls. Call .unload()
to release GPU memory.
"""
from __future__ import annotations
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from .base import Inferencer, InferenceResult
from ..config import (
    IMAGE_CHECKPOINT_DIR, IMAGE_CHECKPOINT_INFO, IMAGE_BACKBONE_ID,
    IMAGE_INPUT_SIZE, USE_FP16, DEFAULT_THRESHOLD,
)
from ..ensemble import image_ensemble_decide

log = logging.getLogger(__name__)

_CKPT_PREFIX = "image_"
_CKPT_SUFFIX = "_lora_best"


# ────────────────────────────────────────────────────────────────────────────
# Checkpoint discovery
# ────────────────────────────────────────────────────────────────────────────

def _head_source(entry: Path) -> str:
    """Where this checkpoint's trained classifier head comes from.

    Returns "adapter" (inside adapter_model.*, via PEFT modules_to_save),
    "file" (a sibling classifier_head.pt), or "none" — which means the model
    would run on a randomly-initialised head and must not be scored.
    """
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
    """Walk ckpt_dir for image_<slug>_lora_best/ directories that look like
    valid PEFT adapter exports. Returns a list of records — ordered by the
    curriculum slot in IMAGE_CHECKPOINT_INFO when the slug is known, and
    appended at the end otherwise."""
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
        # Must look like a PEFT adapter dir.
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

    # Sort: known curriculum order first (matching IMAGE_CHECKPOINT_INFO
    # insertion order), unknowns last.
    order = list(IMAGE_CHECKPOINT_INFO.keys())
    def keyfn(rec):
        s = rec["slug"]
        return (0, order.index(s)) if s in order else (1, s)
    return sorted(found.values(), key=keyfn)


# ────────────────────────────────────────────────────────────────────────────
# Model build / load
# ────────────────────────────────────────────────────────────────────────────

def _build_base_model(backbone_id: str):
    """Fresh ViTForImageClassification with a 2-way head matching the
    training-time setup. ignore_mismatched_sizes=True lets us reinit the
    classifier (the in21k checkpoint comes with a 21k-way head we don't want)."""
    from transformers import ViTForImageClassification
    return ViTForImageClassification.from_pretrained(
        backbone_id,
        num_labels=2,
        ignore_mismatched_sizes=True,
    )


def _load_classifier_head(model, head_path: str | None) -> bool:
    """Try every reasonable way to put the trained head weights back into
    `model.classifier`. Returns True if a head was loaded, False otherwise.

    Only called for checkpoints whose adapter does NOT carry the head itself
    (head_source == "file"); when PEFT already restored it via modules_to_save
    this would at best be a no-op re-load of the same weights.

    The training pipeline saved classifier_head.pt as the state_dict of a
    single nn.Linear (in_features=768, out_features=2) — i.e. {"weight": ...,
    "bias": ...}. Some runs saved {"classifier.weight": ...} and some saved it
    straight off the PEFT wrapper, which prefixes the live copy with
    "modules_to_save.default.". Handle all three.
    """
    if head_path is None:
        return False
    import torch
    state = torch.load(head_path, map_location="cpu")

    # A head saved off a ModulesToSaveWrapper carries both the untouched
    # base copy (original_module.*) and the trained one
    # (modules_to_save.<adapter>.*). The trained copy is the one we want.
    wrapped = {k.split(".")[-1]: v for k, v in state.items()
               if k.startswith("modules_to_save.")}
    if wrapped:
        state = wrapped

    # PeftModel wraps the base under .base_model.model; raw HF model has
    # .classifier directly.
    classifier = None
    if hasattr(model, "classifier"):
        classifier = model.classifier
    elif hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        classifier = getattr(model.base_model.model, "classifier", None)

    if classifier is None:
        log.warning("  could not locate classifier layer to load head into")
        return False

    # Try plain {"weight","bias"} first.
    if "weight" in state and "bias" in state and len(state) == 2:
        try:
            classifier.load_state_dict(state, strict=True)
            return True
        except Exception as e:
            log.warning(f"  plain head load failed: {e}")

    # Try stripping "classifier." prefix.
    stripped = {k.split(".", 1)[1]: v
                for k, v in state.items() if k.startswith("classifier.")}
    if stripped:
        try:
            classifier.load_state_dict(stripped, strict=True)
            return True
        except Exception as e:
            log.warning(f"  stripped head load failed: {e}")

    # Try as a full-model partial state_dict.
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
    """Build base → wrap PEFT adapter → make sure the trained head is in
    place. Returns a fully-prepared, eval-mode model on CPU; caller moves it
    to the target device.

    Raises RuntimeError if the head cannot be restored, so the caller drops
    the checkpoint instead of feeding ensemble noise from a random head.
    """
    from peft import PeftModel  # imported lazily to keep the package optional
    base = _build_base_model(backbone_id)
    model = PeftModel.from_pretrained(base, ckpt["path"])

    source = ckpt.get("head_source", "none")
    if source == "adapter":
        # PeftModel.from_pretrained already restored the head from
        # adapter_model.* (modules_to_save). Nothing else to do.
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


# ────────────────────────────────────────────────────────────────────────────
# Inferencer
# ────────────────────────────────────────────────────────────────────────────

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

    # ── lifecycle ──────────────────────────────────────────────────────────

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

    # ── prediction ─────────────────────────────────────────────────────────

    def _preprocess_to_tensor(self, arr: np.ndarray):
        """uint8 (H, W, 3) → normalised pixel_values on self._device."""
        # ViTImageProcessor handles the in21k mean/std normalisation. We pass
        # a single PIL-style image (it accepts numpy uint8 too).
        inputs = self._processor(images=arr, return_tensors="pt")
        return inputs["pixel_values"].to(self._device)

    def _forward(self, model, pixel_values) -> float:
        import torch
        # Same PEFT TaskType.FEATURE_EXTRACTION forward-signature bug as
        # inferencers/video.py — see that file's _predict_one for the full
        # explanation. get_base_model() bypasses PeftModel's broken generic
        # forward() while still running with LoRA weights active.
        call_target = model.get_base_model() if hasattr(model, "get_base_model") else model
        with torch.no_grad():
            if self._device.type == "cuda" and USE_FP16:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    out = call_target(pixel_values=pixel_values)
            else:
                out = call_target(pixel_values=pixel_values)
            logits = out.logits.float()
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        # Class 1 = fake. Matches the training-time label convention used
        # across every image_*_lora_best checkpoint.
        return float(probs[1])

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

        # Pre-build the two pixel_values tensors (one each). Cheaper than
        # re-running the processor per checkpoint.
        whole_px = self._preprocess_to_tensor(whole_arr)
        face_px  = (self._preprocess_to_tensor(face_arr)
                    if face_arr is not None else None)

        per_ckpt: dict[str, float] = {}
        per_ckpt_role: dict[str, str] = {}
        skipped: list[dict] = []

        for ckpt in self._checkpoints:
            slug = ckpt["slug"]
            if ckpt["needs_face_crop"]:
                if face_px is None:
                    skipped.append({"slug": slug,
                                    "reason": "no face crop available "
                                              "(MTCNN found nothing or "
                                              "disabled)"})
                    continue
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
            except Exception as e:
                log.warning(f"  image_{slug} failed: {e}")
                skipped.append({"slug": slug, "reason": f"inference error: {e}"})
            finally:
                if not self.keep_loaded:
                    self._release_model(slug)

        decision = image_ensemble_decide(per_ckpt, face_detected)
        threshold = float(opts.get("threshold", self.threshold))
        ens = decision["ensemble"]
        if ens != ens:  # NaN check
            verdict = "UNKNOWN"
        else:
            verdict = "FAKE" if ens >= threshold else "REAL"

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
            },
        )