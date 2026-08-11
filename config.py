"""Paths and constants. Override with environment variables."""
from __future__ import annotations
import os
from pathlib import Path

# DEEPTRUTH_ROOT used to default to a hardcoded "/data/deeptruth" — a fixed
# path that only ever made sense on the one Linux server this was first
# deployed to. Any entry point that imports this module without going
# through server/app/_dtp.py's bootstrap() first (which overrides these via
# os.environ.setdefault before config.py is ever imported) inherited that
# wrong default — this is what broke diagnose_video.py/diagnose_image.py
# outright on Windows, and would break `python cli.py` the same way.
#
# The fix: default to this file's own location. config.py lives at the
# project root, so its parent directory *is* DEEPTRUTH_ROOT on any machine,
# any OS, with no environment setup required — matching exactly what
# bootstrap() computes, just without needing bootstrap() to have run first.
# An explicit DEEPTRUTH_ROOT env var still overrides this, same as before.
_THIS_FILE_ROOT = Path(__file__).resolve().parent

DEEPTRUTH_ROOT = Path(os.environ.get("DEEPTRUTH_ROOT", str(_THIS_FILE_ROOT))).resolve()

CHECKPOINT_DIR = Path(os.environ.get(
    "DEEPTRUTH_CHECKPOINTS", DEEPTRUTH_ROOT / "videos_checkpoints")).resolve()

CACHE_DIR = Path(os.environ.get(
    "DEEPTRUTH_CACHE", DEEPTRUTH_ROOT / "preprocessed")).resolve()

LOG_DIR = Path(os.environ.get(
    "DEEPTRUTH_LOGS", DEEPTRUTH_ROOT / "logs")).resolve()

TRAIN_PIPELINE = Path(os.environ.get(
    "DEEPTRUTH_TRAIN_PIPELINE",
    str(DEEPTRUTH_ROOT / "train_pipeline" / "deeptruth_train.py"))).resolve()

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
    # Previously fell back to CHECKPOINT_DIR — the VIDEO checkpoint dir.
    # With no env vars set, that made image checkpoint discovery silently
    # search the video folder for image_*_lora_best directories, producing
    # a "no image checkpoints found in .../videos_checkpoints" error that
    # looks like a missing-files problem when it is actually two different
    # modalities being pointed at the same folder. Falls back to its own
    # sibling folder instead, matching server/app/_dtp.py's bootstrap()
    # convention (PROJECT_ROOT / "images_checkpoints").
    return DEEPTRUTH_ROOT / "images_checkpoints"

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


# ─── Audio model ──────────────────────────────────────────────────────────────
#
# WavLM-Large + 3-layer head, fine-tuned on ASVspoof 2019 LA.
# inferencers/audio.py rebuilds this architecture and loads with strict=True,
# so MODEL_ID / UNFREEZE_LAYERS / HEAD_DIMS there must match whatever produced
# the checkpoint. Those constants live in that module; only deployment-time
# settings live here.

# Directory containing model.pt (+ optional metadata.json), or a .pt file.
# Unset means the audio channel stays on the stub and reports "no signal"
# rather than guessing.
AUDIO_CHECKPOINT = os.environ.get("DEEPTRUTH_AUDIO_CHECKPOINT", "").strip() or None

# Decision threshold for P(fake). Heavily class-weighted training (ASVspoof LA
# is ~1:9 bonafide:spoof) pushes the operating point well away from 0.5 — the
# training script writes the measured EER threshold into metadata.json, and
# _resolve_audio_threshold() below picks it up automatically so the value does
# not have to be copied by hand into the environment.
AUDIO_THRESHOLD_ENV = os.environ.get("DEEPTRUTH_AUDIO_THRESHOLD", "").strip() or None


def _resolve_audio_threshold() -> float:
    """Explicit env var > metadata.json's measured EER point > 0.5.

    Reading metadata.json matters: a checkpoint trained with class weighting
    can have its equal-error point at 0.0003 rather than 0.5, and running it
    at 0.5 silently under-flags fake audio. Shipping the threshold alongside
    the weights keeps the two from drifting apart.
    """
    if AUDIO_THRESHOLD_ENV:
        try:
            return float(AUDIO_THRESHOLD_ENV)
        except ValueError:
            pass

    if AUDIO_CHECKPOINT:
        ckpt = Path(AUDIO_CHECKPOINT)
        meta = (ckpt / "metadata.json") if ckpt.is_dir() else ckpt.parent / "metadata.json"
        if meta.exists():
            try:
                import json
                value = json.loads(meta.read_text()).get("recommended_threshold")
                if isinstance(value, (int, float)):
                    return float(value)
            except Exception:
                pass

    return DEFAULT_THRESHOLD


AUDIO_THRESHOLD = _resolve_audio_threshold()


# ─── SRM noise-analysis model ───────────────────────────────────────────────
#
# 5-filter high-pass bank -> 15-dim statistics -> small MLP head, trained on
# CASIA v2 (splicing/copy-move). See preprocessors/srm_filters.py for the
# filter bank and inferencers/srm.py for the checkpoint contract those
# HEAD_DIMS constants must match.

SRM_CHECKPOINT = os.environ.get("DEEPTRUTH_SRM_CHECKPOINT", "").strip() or None
SRM_THRESHOLD_ENV = os.environ.get("DEEPTRUTH_SRM_THRESHOLD", "").strip() or None


def _resolve_srm_threshold() -> float:
    """Same precedence as audio: explicit env var > checkpoint's own measured
    value > 0.5. SRM's classifier is small and trained on a much smaller,
    more balanced dataset than the audio model, so unlike audio there is no
    a-priori reason to expect the threshold to sit far from 0.5 — but reading
    it from metadata.json rather than assuming keeps that an empirical
    question instead of an assumption."""
    if SRM_THRESHOLD_ENV:
        try:
            return float(SRM_THRESHOLD_ENV)
        except ValueError:
            pass

    if SRM_CHECKPOINT:
        ckpt = Path(SRM_CHECKPOINT)
        meta = (ckpt / "metadata.json") if ckpt.is_dir() else ckpt.parent / "metadata.json"
        if meta.exists():
            try:
                import json
                value = json.loads(meta.read_text()).get("recommended_threshold")
                if isinstance(value, (int, float)):
                    return float(value)
            except Exception:
                pass

    return DEFAULT_THRESHOLD


SRM_THRESHOLD = _resolve_srm_threshold()