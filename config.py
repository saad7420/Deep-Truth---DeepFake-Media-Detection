"""Paths and constants. Override with environment variables."""
from __future__ import annotations
import os
from pathlib import Path

DEEPTRUTH_ROOT = Path(os.environ.get("DEEPTRUTH_ROOT", "/data/deeptruth")).resolve()

CHECKPOINT_DIR = Path(os.environ.get(
    "DEEPTRUTH_CHECKPOINTS", DEEPTRUTH_ROOT / "checkpoints")).resolve()

CACHE_DIR = Path(os.environ.get(
    "DEEPTRUTH_CACHE", DEEPTRUTH_ROOT / "preprocessed")).resolve()

LOG_DIR = Path(os.environ.get(
    "DEEPTRUTH_LOGS", DEEPTRUTH_ROOT / "logs")).resolve()

TRAIN_PIPELINE = Path(os.environ.get(
    "DEEPTRUTH_TRAIN_PIPELINE", Path.home() / "deeptruth_train.py")).resolve()

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus", ".aac"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# ─── Video model ──────────────────────────────────────────────────────────────

FACE_CHECKPOINT_NAMES = {"celebdf_v2", "ffpp", "dfdc", "wilddeepfake", "deeperforensics"}
FULL_FRAME_CHECKPOINT_NAMES = {"genvideo"}

# Lowered from 8 after the syntx_ai clip (7/16 face frames) was incorrectly
# trusted to genvideo while every face checkpoint flagged it as fake.
MIN_FACE_FRAMES = 6

DEFAULT_THRESHOLD = 0.5
USE_FP16 = True

# ─── Image model ──────────────────────────────────────────────────────────────
#
# Image checkpoints can either live alongside the video ones in CHECKPOINT_DIR
# or in a separate directory. They are distinguished by the "image_" prefix.
# Override with DEEPTRUTH_IMAGE_CHECKPOINTS if you keep them apart from video.

# Image checkpoints can either live alongside the video ones in CHECKPOINT_DIR
# or in a separate directory. They are distinguished by the "image_" prefix.
# Override with DEEPTRUTH_IMAGE_CHECKPOINTS if you keep them apart from video.
# (DEEPTRUTH_IMG_ROOT is the legacy training-time env var; honour it as a
# fallback so the same VM that trained the models can also serve them.)
def _resolve_image_checkpoint_dir() -> Path:
    explicit = os.environ.get("DEEPTRUTH_IMAGE_CHECKPOINTS")
    if explicit:
        return Path(explicit).resolve()
    img_root = os.environ.get("DEEPTRUTH_IMG_ROOT")
    if img_root:
        return (Path(img_root) / "checkpoints").resolve()
    return CHECKPOINT_DIR

IMAGE_CHECKPOINT_DIR = _resolve_image_checkpoint_dir()

# Hugging Face ID for the image backbone the LoRA adapters were trained against.
IMAGE_BACKBONE_ID = os.environ.get(
    "DEEPTRUTH_IMAGE_BACKBONE", "google/vit-base-patch16-224-in21k")

# Image side: side length the model expects (224×224 — ViT-B/16).
IMAGE_INPUT_SIZE = 224

# MTCNN crop margin around the detected face box. 0.30 = 30% extra context.
# Matches FCROP_MARGIN in ffpp_facecrop_pipeline.py — must stay aligned with
# the value used to train image_ffpp_facecrop_lora_best.
IMAGE_FACE_MARGIN = float(os.environ.get("DEEPTRUTH_IMAGE_FACE_MARGIN", "0.30"))

# Per-checkpoint metadata. Keyed by the slug between "image_" and "_lora_best".
#
#   role            "generalist" → AI-generated detection across content types
#                   "face"       → trained on face-deepfake distributions
#   needs_face_crop True  → must be fed the MTCNN-cropped tensor (only ckpt #8)
#                   False → fed the whole-image 224×224 tensor
#   auc             Best reported validation AUC on the dataset that ckpt was
#                   trained on. Used as the weight inside its role group.
#   role_in_curriculum   Human-readable explainer.
#
# Order in this dict reflects curriculum order (#1 → #8). Discovery in the
# inferencer uses the slugs as a whitelist so unknown directories are flagged.
IMAGE_CHECKPOINT_INFO: dict[str, dict] = {
    "genimage": {
        "role": "generalist",
        "needs_face_crop": False,
        "auc": 0.85,
        "role_in_curriculum":
            "#1 foundational — older GAN + early diffusion (NeurIPS 2023)",
    },
    "mscocoai": {
        "role": "generalist",
        "needs_face_crop": False,
        "auc": 0.90,
        "role_in_curriculum":
            "#2 modern commercial generators (DALL-E 3, MJv6, SD3)",
    },
    "wildrf": {
        "role": "generalist",
        "needs_face_crop": False,
        "auc": 0.89,
        "role_in_curriculum":
            "#3 real-world social-media (Reddit, X, Facebook)",
    },
    "commforensics": {
        "role": "generalist",
        "needs_face_crop": False,
        "auc": 0.99,
        "role_in_curriculum":
            "#4 generator breadth — 4,800+ generator variants",
    },
    "ntire": {
        "role": "generalist",
        "needs_face_crop": False,
        "auc": 0.94,
        "role_in_curriculum":
            "#5 newest gens + 12 training-time distortions (NTIRE 2026)",
    },
    "dff": {
        "role": "face",
        "needs_face_crop": False,
        "auc": 0.88,
        "role_in_curriculum":
            "#6 modern AI face gen / inpaint / swap (DeepFakeFace)",
    },
    "ffpp": {
        "role": "face",
        "needs_face_crop": False,
        "auc": 0.85,
        "role_in_curriculum":
            "#7 classical face deepfakes — whole-frame (FaceForensics++)",
    },
    "ffpp_facecrop": {
        "role": "face",
        "needs_face_crop": True,
        "auc": 0.92,
        "role_in_curriculum":
            "#8 FF++ MTCNN-cropped — +0.12 Celeb-DF cross-dataset AUC",
    },
}

# When grouping by role for the ensemble, these are the canonical sets.
IMAGE_GENERALIST_NAMES = {n for n, m in IMAGE_CHECKPOINT_INFO.items()
                          if m["role"] == "generalist"}
IMAGE_FACE_NAMES       = {n for n, m in IMAGE_CHECKPOINT_INFO.items()
                          if m["role"] == "face"}

# Down-weight applied to face-group scores when no face is detected in the
# image. Face-trained models still carry some signal on whole scenes (DFF and
# ffpp were trained on whole frames) but their out-of-domain reliability is
# lower, so they should not dominate the verdict.
IMAGE_NO_FACE_FACE_WEIGHT = float(os.environ.get(
    "DEEPTRUTH_IMAGE_NO_FACE_FACE_WEIGHT", "0.2"))
