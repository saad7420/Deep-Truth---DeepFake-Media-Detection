from __future__ import annotations
import os
from pathlib import Path

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


FACE_CHECKPOINT_NAMES = {"celebdf_v2", "ffpp", "dfdc", "wilddeepfake", "deeperforensics"}
FULL_FRAME_CHECKPOINT_NAMES = {"genvideo"}

# face models are only trusted when faces show up in this many frames
MIN_FACE_FRAMES = 6

DEFAULT_THRESHOLD = 0.5
USE_FP16 = True


def _resolve_image_checkpoint_dir() -> Path:
    explicit = os.environ.get("DEEPTRUTH_IMAGE_CHECKPOINTS")
    if explicit:
        return Path(explicit).resolve()
    img_root = os.environ.get("DEEPTRUTH_IMG_ROOT")
    if img_root:
        return (Path(img_root) / "checkpoints").resolve()
    return DEEPTRUTH_ROOT / "images_checkpoints"

IMAGE_CHECKPOINT_DIR = _resolve_image_checkpoint_dir()

IMAGE_BACKBONE_ID = os.environ.get(
    "DEEPTRUTH_IMAGE_BACKBONE", "google/vit-base-patch16-224-in21k")

IMAGE_INPUT_SIZE = 224

IMAGE_FACE_MARGIN = float(os.environ.get("DEEPTRUTH_IMAGE_FACE_MARGIN", "0.30"))

ARTIFACT_MAPS_ENABLED = os.environ.get(
    "DEEPTRUTH_ARTIFACT_MAPS", "1").lower() not in ("0", "false", "no")

ARTIFACT_MAP_MIN_SCORE = float(
    os.environ.get("DEEPTRUTH_ARTIFACT_MAP_MIN_SCORE", "0.10"))

# role and ensemble weight of each image checkpoint
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

IMAGE_GENERALIST_NAMES = {n for n, m in IMAGE_CHECKPOINT_INFO.items()
                          if m["role"] == "generalist"}
IMAGE_FACE_NAMES       = {n for n, m in IMAGE_CHECKPOINT_INFO.items()
                          if m["role"] == "face"}

IMAGE_NO_FACE_FACE_WEIGHT = float(os.environ.get(
    "DEEPTRUTH_IMAGE_NO_FACE_FACE_WEIGHT", "0.2"))


AUDIO_CHECKPOINT = os.environ.get("DEEPTRUTH_AUDIO_CHECKPOINT", "").strip() or None

AUDIO_THRESHOLD_ENV = os.environ.get("DEEPTRUTH_AUDIO_THRESHOLD", "").strip() or None


# the threshold comes from the checkpoint metadata, not a fixed 0.5
def _resolve_audio_threshold() -> float:
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


SRM_CHECKPOINT = os.environ.get("DEEPTRUTH_SRM_CHECKPOINT", "").strip() or None
SRM_THRESHOLD_ENV = os.environ.get("DEEPTRUTH_SRM_THRESHOLD", "").strip() or None


def _resolve_srm_threshold() -> float:
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
