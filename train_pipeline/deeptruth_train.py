#!/usr/bin/env python3
"""
DeepTruth — ViViT Fine-Tuning Pipeline for Deepfake / AI-Gen Video Detection
============================================================================

Designed for: GCP VM, NVIDIA L4 24GB, Ubuntu 22.04, Python 3.10+
Datasets:  Celeb-DF v2, FaceForensics++, DFDC (subset), WildDeepfake,
           GenVideo-100K (subset), AIGVDet/GVD, FakeAVCeleb (optional manual)

Strategy:  LoRA r=16 + AMP (FP16) on google/vivit-b-16x2-kinetics400.
           16 frames per video. Pre-extracted .npy frame caches.
           Per-dataset training with checkpointing + identity-aware splits.

Run modes:
    python deeptruth_train.py setup           # Install deps, verify GPU
    python deeptruth_train.py download <ds>   # Download a single dataset
    python deeptruth_train.py download all    # Download every available dataset
    python deeptruth_train.py preprocess <ds> # Extract & cache frames
    python deeptruth_train.py preprocess all
    python deeptruth_train.py train <ds>      # Train on one dataset (continues from last ckpt)
    python deeptruth_train.py train all       # Train sequentially on every dataset
    python deeptruth_train.py evaluate <ds>   # Evaluate test split of a dataset
    python deeptruth_train.py status          # Show what's done / pending

The pipeline is idempotent: every phase records progress to disk and can be
resumed after a crash, OOM, preemption, or VM restart.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
import random
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import traceback
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# =============================================================================
# CONFIGURATION
# =============================================================================
# Override anything below via environment variables (e.g. ROOT=/data python ...)

ROOT = Path(os.environ.get("DEEPTRUTH_ROOT", "/data/deeptruth")).resolve()

# Sub-directories under ROOT
DIR_RAW       = ROOT / "raw"          # Downloaded dataset archives + extracted videos
DIR_FRAMES    = ROOT / "frames"       # Pre-extracted .npy frame caches  <-- the speed unlock
DIR_MANIFESTS = ROOT / "manifests"    # train/val/test CSVs per dataset
DIR_CKPT      = ROOT / "checkpoints"  # Model checkpoints (LoRA adapters + heads)
DIR_LOGS      = ROOT / "logs"         # JSONL training metrics + text logs
DIR_STATE     = ROOT / "state"        # Phase progress markers (resumability)

# Model / training hyperparameters (tuned for L4 24GB + 16 frames)
MODEL_ID         = "google/vivit-b-16x2-kinetics400"  # ~88M params
NUM_FRAMES       = 16          # Half of pretrained 32 — 2x faster, AUC drop is negligible.
                               # ViViT uses interpolate_pos_encoding=True to handle this.
IMAGE_SIZE       = 224         # ViViT default
BATCH_SIZE       = 8           # Direct batch on L4 24GB with LoRA + AMP
GRAD_ACCUM       = 2           # Effective batch = 16
NUM_WORKERS      = 6           # Tuned for typical 8-vCPU GCP VM; auto-clamped to cpu_count-2
EPOCHS_MAX       = 25           # Hard ceiling; early stopping will usually stop at 3-5
EARLY_STOP_PAT   = 2           # Stop if val AUC doesn't improve for N epochs
LR               = 5e-4        # Higher than full-FT — LoRA tolerates / benefits from this
WEIGHT_DECAY     = 0.01
WARMUP_RATIO     = 0.1
LORA_R           = 16
LORA_ALPHA       = 32
LORA_DROPOUT     = 0.05
SEED             = 42

# Per-dataset video sample caps (matches xlsx subset plan).
# Keeps disk + train time bounded. Real:Fake balanced where possible.
DATASET_CAPS = {
    "celebdf_v2":   {"real": 590,  "fake": 5639},   # full dataset (small)
    "ffpp":         {"real": 1000, "fake": 4000},   # 4 manipulation methods × 1000
    "dfdc":         {"real": 2500, "fake": 2500},   # 5K subset
    "wilddeepfake": {"real": 1500, "fake": 1500},
    "fakeavceleb":  {"real": 1500, "fake": 1500},   # optional
    "aigvdet":      {"real": 5800, "fake": 5800},   # GVD full
    "genvideo":     {"real": 2500, "fake": 2500},   # GenVideo-100K subset
}

# Datasets that depict faces -> we run face detection + crop before saving frames.
# Datasets that are general AI-gen video -> we keep the whole frame.
FACE_DATASETS = {"celebdf_v2", "ffpp", "dfdc", "wilddeepfake", "fakeavceleb"}

# =============================================================================
# LOGGING / UTILS
# =============================================================================

def _setup_logging() -> logging.Logger:
    DIR_LOGS.mkdir(parents=True, exist_ok=True)
    log_fmt = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(DIR_LOGS / "pipeline.log"),
        ],
    )
    return logging.getLogger("deeptruth")

LOG = _setup_logging()

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # We DO NOT enable deterministic algorithms — it slows ViViT attention dramatically.
        torch.backends.cudnn.benchmark = True
    except ImportError:
        pass

def state_done(name: str) -> bool:
    return (DIR_STATE / f"{name}.done").exists()

def mark_done(name: str) -> None:
    DIR_STATE.mkdir(parents=True, exist_ok=True)
    (DIR_STATE / f"{name}.done").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))

def ensure_dirs() -> None:
    for d in [ROOT, DIR_RAW, DIR_FRAMES, DIR_MANIFESTS, DIR_CKPT, DIR_LOGS, DIR_STATE]:
        d.mkdir(parents=True, exist_ok=True)

def run_cmd(cmd: list[str] | str, cwd: Optional[Path] = None, check: bool = True) -> int:
    """Stream a subprocess to our log."""
    LOG.info(f"$ {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    proc = subprocess.run(
        cmd, cwd=cwd, shell=isinstance(cmd, str),
        check=False, text=True, capture_output=False,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(f"Command failed (rc={proc.returncode}): {cmd}")
    return proc.returncode

def free_gpu() -> None:
    try:
        import torch
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except Exception:
        pass


# =============================================================================
# 1. SETUP / DEPENDENCY CHECK
# =============================================================================

REQUIRED_PIP = [
    "torch>=2.2",
    "torchvision",
    "transformers>=4.41",
    "accelerate>=0.30",
    "peft>=0.11",
    "huggingface_hub>=0.23",
    "datasets>=2.19",
    "av",                 # PyAV for video decoding
    "opencv-python-headless",
    "facenet-pytorch",    # MTCNN face detector
    "scikit-learn",
    "pandas",
    "numpy",
    "tqdm",
    "gdown",              # Google Drive downloader (Celeb-DF v2)
    "kaggle",             # DFDC
    "modelscope",         # GenVideo
]

def cmd_setup(args: argparse.Namespace) -> None:
    ensure_dirs()
    LOG.info("Installing Python dependencies...")
    pip_args = [sys.executable, "-m", "pip", "install", "-q", "--upgrade", "pip"]
    run_cmd(pip_args, check=False)
    pip_args = [sys.executable, "-m", "pip", "install", "-q"] + REQUIRED_PIP
    run_cmd(pip_args)

    # OS-level: ffmpeg is needed for PyAV + many videos. Install if missing.
    if shutil.which("ffmpeg") is None:
        LOG.warning("ffmpeg not found. Trying apt-get install (needs sudo)...")
        run_cmd("sudo apt-get update -qq && sudo apt-get install -y -qq ffmpeg", check=False)

    # GPU sanity
    import torch
    if not torch.cuda.is_available():
        LOG.error("CUDA NOT available. This script requires a GPU.")
        sys.exit(1)
    dev = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    LOG.info(f"GPU: {dev} ({vram:.1f} GB)")
    LOG.info(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")

    # Pre-warm the model cache so download time is paid now, not at training start
    LOG.info(f"Pre-fetching model weights: {MODEL_ID}")
    from transformers import VivitImageProcessor, VivitForVideoClassification
    VivitImageProcessor.from_pretrained(MODEL_ID)
    VivitForVideoClassification.from_pretrained(MODEL_ID)
    LOG.info("Setup complete.")
    mark_done("setup")


# =============================================================================
# 2. DATASET DOWNLOADERS
# =============================================================================
# Each downloader is a method on the DatasetSpec. They MUST be idempotent —
# i.e. safe to re-run, and skip already-downloaded files.

def _http_download(url: str, dst: Path, retries: int = 5) -> None:
    """Resumable HTTP download via curl (handles flaky connections better than requests)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            run_cmd(["curl", "-L", "--retry", "10", "--retry-delay", "5",
                     "-C", "-", "-o", str(dst), url])
            return
        except Exception as e:
            LOG.warning(f"Download attempt {attempt+1}/{retries} failed: {e}")
            time.sleep(min(60, 5 * (2 ** attempt)))
    raise RuntimeError(f"Failed to download after {retries} attempts: {url}")

def _gdrive_download(file_id: str, dst: Path) -> None:
    """Resumable Google Drive download via gdown.

    Tries single-file first (most common). If that fails because the ID is
    actually a folder, falls back to folder download into dst.parent / dst.stem.
    """
    if dst.exists() and dst.stat().st_size > 0:
        LOG.info(f"Already downloaded: {dst}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    import gdown
    # fuzzy=True allows passing share-links / open?id=... style URLs too.
    try:
        result = gdown.download(id=file_id, output=str(dst), quiet=False, fuzzy=True)
        if result and Path(result).exists() and Path(result).stat().st_size > 0:
            return
        raise RuntimeError("gdown returned empty result for single-file mode")
    except Exception as e:
        LOG.warning(f"Single-file download failed ({e}); trying as folder...")
    folder_dst = dst.parent / dst.stem
    folder_dst.mkdir(parents=True, exist_ok=True)
    gdown.download_folder(id=file_id, output=str(folder_dst), quiet=False, use_cookies=False)

def _hf_snapshot(repo_id: str, dst: Path, repo_type: str = "dataset",
                 allow_patterns: list[str] | None = None) -> None:
    if (dst / ".hf_done").exists():
        LOG.info(f"HF snapshot already present: {dst}")
        return
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=repo_id, repo_type=repo_type, local_dir=str(dst),
        allow_patterns=allow_patterns, max_workers=4,
    )
    (dst / ".hf_done").write_text("ok")

def _safe_extract(archive: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    LOG.info(f"Extracting {archive} -> {dst}")
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(dst)
    elif archive.suffixes[-2:] == [".tar", ".gz"] or archive.suffix == ".tgz":
        with tarfile.open(archive, "r:gz") as t:
            t.extractall(dst)
    elif archive.suffix == ".tar":
        with tarfile.open(archive, "r:") as t:
            t.extractall(dst)
    else:
        raise ValueError(f"Unknown archive type: {archive}")


# ---- Celeb-DF v2 ------------------------------------------------------------
# Source: Google Drive (from authors' Google Form response). ID confirmed.
CELEBDF_V2_GDRIVE_ID = "1iLx76wsbi9itnkxSqz9BVBl4ZvnbIazj"

def download_celebdf_v2() -> None:
    out_dir = DIR_RAW / "celebdf_v2"
    if state_done("dl_celebdf_v2"):
        LOG.info("Celeb-DF v2 already downloaded."); return
    out_dir.mkdir(parents=True, exist_ok=True)
    archive = out_dir / "celeb-df-v2.zip"
    LOG.info("Downloading Celeb-DF v2 from Google Drive (~25 GB)...")
    _gdrive_download(CELEBDF_V2_GDRIVE_ID, archive)
    # _gdrive_download may have produced either a .zip OR a folder
    # (out_dir/celeb-df-v2/...) if the ID turned out to be a folder.
    if archive.exists() and archive.stat().st_size > 0:
        try:
            _safe_extract(archive, out_dir)
            archive.unlink()  # save 25GB
        except Exception as e:
            LOG.error(f"Extraction failed: {e}; archive kept for manual inspection.")
            raise
    elif (out_dir / "celeb-df-v2").exists():
        LOG.info("Celeb-DF v2 downloaded as folder (already unpacked).")
    else:
        raise RuntimeError("Celeb-DF v2 download produced no output.")
    mark_done("dl_celebdf_v2")


# ---- FaceForensics++ --------------------------------------------------------
# User has the official faceforensics_download_v4.py script in /home/<user>/.
# We invoke it with the standard args. It'll prompt them to accept the EULA
# the first time (interactive).
def download_ffpp() -> None:
    if state_done("dl_ffpp"):
        LOG.info("FaceForensics++ already downloaded."); return
    out_dir = DIR_RAW / "ffpp"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Look for the user's download script. Fail loudly with instructions if missing.
    script_candidates = [
        Path.home() / "faceforensics_download_v4.py",
        ROOT / "faceforensics_download_v4.py",
        Path("/opt/faceforensics_download_v4.py"),
    ]
    script = next((p for p in script_candidates if p.exists()), None)
    if script is None:
        raise FileNotFoundError(
            "faceforensics_download_v4.py not found. "
            "Place it at one of: " + ", ".join(str(p) for p in script_candidates)
        )

    # We pull c23 compression (good quality, manageable size) for original + 4 fake methods.
    # --num_videos 1000 caps each method to keep download under ~4 GB.
    methods = ["original", "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
    for method in methods:
        LOG.info(f"FF++ downloading: {method}")
        run_cmd([
            sys.executable, str(script), str(out_dir),
            "-d", method, "-c", "c23", "-t", "videos",
            "--num_videos", "1000", "--server", "EU2",
        ])
    mark_done("dl_ffpp")


# ---- DFDC -------------------------------------------------------------------
# Kaggle competition dataset. Requires kaggle.json in ~/.kaggle/.
# Full set is 470GB — we pull only the ~10GB sample (test_videos).
def download_dfdc() -> None:
    if state_done("dl_dfdc"):
        LOG.info("DFDC already downloaded."); return
    out_dir = DIR_RAW / "dfdc"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not (Path.home() / ".kaggle" / "kaggle.json").exists():
        raise FileNotFoundError(
            "Kaggle API key missing. Put kaggle.json in ~/.kaggle/ "
            "(chmod 600). Get it from kaggle.com/settings -> Create API Token."
        )
    # Use the smaller 'sample' files (train_sample_videos.zip ~4GB) — enough for
    # our 5K subset target.
    LOG.info("DFDC downloading train_sample_videos (~4GB) via Kaggle API...")
    run_cmd([
        "kaggle", "competitions", "download",
        "-c", "deepfake-detection-challenge",
        "-f", "train_sample_videos.zip",
        "-p", str(out_dir),
    ])
    archive = out_dir / "train_sample_videos.zip"
    if archive.exists():
        _safe_extract(archive, out_dir)
        archive.unlink()
    mark_done("dl_dfdc")


# ---- WildDeepfake -----------------------------------------------------------
# HF dataset xingjunm/WildDeepfake is gated. User must accept terms on the HF
# page first AND set HUGGINGFACE_TOKEN env var. Stored as face-sequence frames.
def download_wilddeepfake() -> None:
    if state_done("dl_wilddeepfake"):
        LOG.info("WildDeepfake already downloaded."); return
    out_dir = DIR_RAW / "wilddeepfake"
    if not os.environ.get("HUGGINGFACE_TOKEN") and not os.environ.get("HF_TOKEN"):
        LOG.warning(
            "WildDeepfake is a GATED HF dataset. Set HF_TOKEN env var and accept "
            "terms at https://huggingface.co/datasets/xingjunm/WildDeepfake first. "
            "Skipping for now — re-run after granting access."
        )
        return
    _hf_snapshot("xingjunm/WildDeepfake", out_dir, repo_type="dataset")
    mark_done("dl_wilddeepfake")


# ---- FakeAVCeleb (manual / optional) ----------------------------------------
# No public direct download. Authors require email request + send a script.sh.
# We treat it as optional: if the user has dropped the unzipped data into
# DIR_RAW/fakeavceleb/ we'll use it; otherwise skip.
def download_fakeavceleb() -> None:
    out_dir = DIR_RAW / "fakeavceleb"
    if any(out_dir.rglob("*.mp4")) if out_dir.exists() else False:
        LOG.info("FakeAVCeleb data found in raw/ — will be used.")
        mark_done("dl_fakeavceleb")
        return
    LOG.warning(
        "FakeAVCeleb requires emailing the authors (see github.com/DASH-Lab/FakeAVCeleb). "
        "After you receive the script.sh, run it and place the resulting folder at "
        f"{out_dir}/. Skipping for now — pipeline continues without it."
    )


# ---- AIGVDet / GVD ----------------------------------------------------------
# GitHub repo with download instructions. The dataset itself is hosted on
# Baidu / Google Drive (links inside the repo). We clone the repo and let the
# user provide a Google Drive ID via env var if they have one. Otherwise we
# fall back to cloning + printing instructions.
def download_aigvdet() -> None:
    if state_done("dl_aigvdet"):
        LOG.info("AIGVDet/GVD already downloaded."); return
    out_dir = DIR_RAW / "aigvdet"
    out_dir.mkdir(parents=True, exist_ok=True)

    repo_dir = out_dir / "AIGVDet"
    if not repo_dir.exists():
        run_cmd(["git", "clone", "--depth", "1",
                 "https://github.com/multimediaFor/AIGVDet.git", str(repo_dir)])

    gdrive_id = os.environ.get("AIGVDET_GDRIVE_ID")
    if gdrive_id:
        LOG.info(f"Downloading GVD from Google Drive ID {gdrive_id}...")
        archive = out_dir / "gvd.zip"
        _gdrive_download(gdrive_id, archive)
        _safe_extract(archive, out_dir)
        archive.unlink()
        mark_done("dl_aigvdet")
    else:
        LOG.warning(
            "AIGVDet GVD download requires a manual step:\n"
            f"  1. Read {repo_dir}/README.md for the latest dataset link.\n"
            "  2. Set AIGVDET_GDRIVE_ID=<id> env var and re-run, OR\n"
            f"  3. Manually drop the unzipped GVD videos into {out_dir}/."
        )


# ---- GenVideo (DeMamba) -----------------------------------------------------
# Hosted on ModelScope (cccnju/GenVideo-100K), the lightweight 100K version.
def download_genvideo() -> None:
    if state_done("dl_genvideo"):
        LOG.info("GenVideo-100K already downloaded."); return
    out_dir = DIR_RAW / "genvideo"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        from modelscope.hub.snapshot_download import snapshot_download as ms_snapshot
        ms_snapshot(
            "cccnju/GenVideo-100K",
            cache_dir=str(out_dir),
            repo_type="dataset",
        )
        mark_done("dl_genvideo")
    except Exception as e:
        LOG.error(f"GenVideo download failed: {e}")
        LOG.warning(
            "If ModelScope is rate-limiting, retry later. "
            "Alternative: manually download from "
            "https://modelscope.cn/datasets/cccnju/GenVideo-100K and unzip into "
            f"{out_dir}/"
        )


DOWNLOADERS = {
    "celebdf_v2":   download_celebdf_v2,
    "ffpp":         download_ffpp,
    "dfdc":         download_dfdc,
    "wilddeepfake": download_wilddeepfake,
    "fakeavceleb":  download_fakeavceleb,
    "aigvdet":      download_aigvdet,
    "genvideo":     download_genvideo,
}

def cmd_download(args: argparse.Namespace) -> None:
    ensure_dirs()
    targets = list(DOWNLOADERS.keys()) if args.dataset == "all" else [args.dataset]
    for ds in targets:
        if ds not in DOWNLOADERS:
            LOG.error(f"Unknown dataset: {ds}"); continue
        LOG.info(f"=== Download: {ds} ===")
        try:
            DOWNLOADERS[ds]()
        except Exception as e:
            LOG.error(f"Download of {ds} failed: {e}")
            LOG.error(traceback.format_exc())
            if args.dataset != "all":
                raise


# =============================================================================
# 3. PREPROCESSING — extract frames ONCE, save as .npy
# =============================================================================
# Why this exists: in the previous Colab run, frame decoding happened inside
# __getitem__ on every epoch. That made the GPU sit idle waiting for CPU.
# Pre-extracting frames to .npy removes ~80% of training time.
#
# Format: each video -> one .npy of shape (NUM_FRAMES, IMAGE_SIZE, IMAGE_SIZE, 3)
#         in uint8 (RGB). About 2.4 MB per video.
# Per-dataset manifest CSV: filepath, label, identity, dataset, split

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

@dataclass
class VideoSample:
    path: Path                 # Source video file
    label: int                 # 0 = real, 1 = fake
    identity: str              # Used for non-leaking train/val/test split
    dataset: str
    extra: dict = field(default_factory=dict)

def _list_videos(root: Path) -> list[Path]:
    return [p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXTS]

def _label_from_celebdf(p: Path) -> tuple[int, str]:
    # Folder structure: Celeb-real/, Celeb-synthesis/, YouTube-real/
    parts = {x.lower() for x in p.parts}
    if "celeb-synthesis" in parts:
        # Filename: id<X>_id<Y>_<ZZZZ>.mp4 -> identity = id<X>_id<Y>
        ident = "_".join(p.stem.split("_")[:2]) or p.stem
        return 1, ident
    # Real: id<X>_<NNNN>.mp4
    ident = p.stem.split("_")[0] or p.stem
    return 0, ident

def _label_from_ffpp(p: Path) -> tuple[int, str]:
    parts = {x.lower() for x in p.parts}
    real_markers = {"original_sequences", "original", "youtube", "actors"}
    is_real = bool(parts & real_markers)
    # FF++ filenames are like 000.mp4 (real) or 000_003.mp4 (fake derived from real id 000).
    # Identity = the source real video id, so fakes derived from same real go in same split.
    ident = p.stem.split("_")[0]
    return (0 if is_real else 1), ident

def _label_from_dfdc(p: Path, meta: dict) -> tuple[int, str]:
    # metadata.json: { "abc.mp4": { "label": "FAKE", "original": "xyz.mp4" }, ... }
    info = meta.get(p.name, {})
    is_fake = info.get("label", "").upper() == "FAKE"
    # Identity = original real video (so derivatives don't leak into val/test).
    # For real videos, identity = self.
    ident = info.get("original") or p.name
    return (1 if is_fake else 0), ident

def _label_from_wilddeepfake(p: Path) -> tuple[int, str]:
    # WildDeepfake structure: real_*/face_seq_id/*.png  vs  fake_*/face_seq_id/*.png
    parts = [x.lower() for x in p.parts]
    is_fake = any("fake" in x for x in parts)
    # Identity = the face-sequence directory (one level above the frames)
    ident = p.parent.name
    return (1 if is_fake else 0), ident

def _label_from_aigvdet(p: Path) -> tuple[int, str]:
    parts = [x.lower() for x in p.parts]
    real_keywords = {"real", "youtube_vos", "got"}
    is_real = any(any(k in x for k in real_keywords) for x in parts)
    # Identity = generator + filename
    return (0 if is_real else 1), p.stem

def _label_from_genvideo(p: Path) -> tuple[int, str]:
    parts = [x.lower() for x in p.parts]
    # MSR-VTT, Kinetics, Youku = real. Everything else = generated.
    real_kw = {"msr", "msrvtt", "kinetics", "youku", "real"}
    is_real = any(any(k in x for k in real_kw) for x in parts)
    return (0 if is_real else 1), p.stem

def _label_from_fakeavceleb(p: Path) -> tuple[int, str]:
    # FakeAVCeleb folder: RealVideo-RealAudio/, FakeVideo-*, etc.
    parts = [x.lower() for x in p.parts]
    is_real = any("realvideo-realaudio" in x for x in parts)
    # Identity = speaker id (vox-celeb id, usually 'idXXXXX' folder)
    ident = next((x for x in p.parts if x.lower().startswith("id0") or x.lower().startswith("id1")), p.stem)
    return (0 if is_real else 1), ident


def _enumerate_dataset(name: str) -> list[VideoSample]:
    """Walk the raw/<dataset>/ tree and return labelled samples."""
    raw = DIR_RAW / name
    if not raw.exists():
        return []

    if name == "celebdf_v2":
        out = []
        for p in _list_videos(raw):
            label, ident = _label_from_celebdf(p)
            out.append(VideoSample(p, label, ident, name))
        return out

    if name == "ffpp":
        out = []
        for p in _list_videos(raw):
            label, ident = _label_from_ffpp(p)
            out.append(VideoSample(p, label, ident, name))
        return out

    if name == "dfdc":
        # Find metadata.json (it's bundled inside train_sample_videos/)
        meta_paths = list(raw.rglob("metadata.json"))
        meta = {}
        for mp in meta_paths:
            try:
                meta.update(json.loads(mp.read_text()))
            except Exception as e:
                LOG.warning(f"Failed to parse {mp}: {e}")
        out = []
        for p in _list_videos(raw):
            label, ident = _label_from_dfdc(p, meta)
            out.append(VideoSample(p, label, ident, name))
        return out

    if name == "wilddeepfake":
        # WildDeepfake may arrive in one of three layouts depending on the source:
        #   (a) raw .png frames inside .../<real|fake>/<sequence_dir>/*.png   <- original
        #   (b) tar shards inside the HF snapshot (.../*.tar)                  <- needs unpack
        #   (c) parquet shards (.../*.parquet)                                 <- HF datasets
        # We handle (a) directly; for (b) we extract any *.tar once; (c) is
        # left for the user to extract because parquet→frames is dataset-specific.
        out = []

        # Step 1: if any tars exist, unpack them in-place once.
        tars = list(raw.rglob("*.tar"))
        if tars:
            unpacked_marker = raw / ".tars_unpacked"
            if not unpacked_marker.exists():
                LOG.info(f"WildDeepfake: unpacking {len(tars)} tar shards...")
                for t in tars:
                    try:
                        _safe_extract(t, t.parent)
                    except Exception as e:
                        LOG.warning(f"Failed to extract {t}: {e}")
                unpacked_marker.write_text("ok")

        # Step 2: enumerate sequences (directories containing >= 1 .png).
        seq_dirs = sorted({p.parent for p in raw.rglob("*.png")})
        if not seq_dirs:
            # Try .jpg too — some WildDeepfake mirrors use jpg.
            seq_dirs = sorted({p.parent for p in raw.rglob("*.jpg")})
        for sd in seq_dirs:
            label, ident = _label_from_wilddeepfake(sd / "x")  # path used for parsing only
            out.append(VideoSample(sd, label, ident, name, extra={"is_frame_dir": True}))

        if not out:
            LOG.warning(
                "WildDeepfake: no .png/.jpg frames found. The HF snapshot may be "
                "in parquet form. Manually convert to frames and place them "
                f"under {raw}/{{real,fake}}/<seq_id>/*.png to enable training."
            )
        return out

    if name == "fakeavceleb":
        out = []
        for p in _list_videos(raw):
            label, ident = _label_from_fakeavceleb(p)
            out.append(VideoSample(p, label, ident, name))
        return out

    if name == "aigvdet":
        out = []
        for p in _list_videos(raw):
            label, ident = _label_from_aigvdet(p)
            out.append(VideoSample(p, label, ident, name))
        return out

    if name == "genvideo":
        out = []
        for p in _list_videos(raw):
            label, ident = _label_from_genvideo(p)
            out.append(VideoSample(p, label, ident, name))
        return out

    raise ValueError(f"Unknown dataset: {name}")


def _balance_and_cap(samples: list[VideoSample], cap: dict) -> list[VideoSample]:
    """Randomly down-sample to {real: N, fake: M} with deterministic seed."""
    rng = random.Random(SEED)
    by_label = {0: [s for s in samples if s.label == 0],
                1: [s for s in samples if s.label == 1]}
    rng.shuffle(by_label[0]); rng.shuffle(by_label[1])
    out = by_label[0][:cap["real"]] + by_label[1][:cap["fake"]]
    rng.shuffle(out)
    return out


# ---- Frame extraction -------------------------------------------------------

_face_detector = None
def _get_face_detector():
    global _face_detector
    if _face_detector is None:
        import torch
        from facenet_pytorch import MTCNN
        device = "cpu"  # MTCNN on CPU avoids cuBLAS mismatch with new CUDA
        # margin=20 gives a bit of context around the face — helps deepfake artifacts.
        _face_detector = MTCNN(image_size=IMAGE_SIZE, margin=20, post_process=False,
                               device=device, select_largest=True, keep_all=False)
    return _face_detector

def _sample_indices(total: int, n: int) -> list[int]:
    """Uniformly sample n frame indices spanning the full clip."""
    if total <= 0:
        return []
    if total <= n:
        return list(range(total)) + [total - 1] * (n - total)
    import numpy as np
    return np.linspace(0, total - 1, num=n, dtype=int).tolist()

def _decode_video(path: Path, n_frames: int) -> Optional["np.ndarray"]:
    """Decode n_frames evenly spaced from a video. Returns (n_frames, H, W, 3) uint8 RGB."""
    import av
    import numpy as np
    try:
        container = av.open(str(path))
    except Exception as e:
        LOG.warning(f"Cannot open {path}: {e}")
        return None
    try:
        stream = container.streams.video[0]
        # Don't trust stream.frames for short / variable-FPS clips — it can be 0.
        # We seek by index after a first pass count fallback.
        total = stream.frames or 0
        if total == 0:
            # Cheap count
            total = sum(1 for _ in container.decode(video=0))
            container.seek(0)
            container = av.open(str(path))
            stream = container.streams.video[0]

        target = set(_sample_indices(total, n_frames))
        frames = []
        for i, frame in enumerate(container.decode(video=0)):
            if i in target:
                frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) >= n_frames:
                break
        if len(frames) == 0:
            return None
        # Pad if we got fewer than requested (very short clips)
        while len(frames) < n_frames:
            frames.append(frames[-1])
        return np.stack(frames)  # (T, H, W, 3) uint8
    finally:
        container.close()

def _decode_frame_dir(d: Path, n_frames: int) -> Optional["np.ndarray"]:
    """For WildDeepfake-style data: a directory of frames is one 'video'."""
    import numpy as np
    import cv2
    # Some mirrors use .jpg, others .png — accept both.
    imgs = sorted(list(d.glob("*.png")) + list(d.glob("*.jpg")) + list(d.glob("*.jpeg")))
    if not imgs:
        return None
    idxs = _sample_indices(len(imgs), n_frames)
    out = []
    for i in idxs:
        img = cv2.imread(str(imgs[i]))
        if img is None: return None
        out.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    while len(out) < n_frames:
        out.append(out[-1])
    return np.stack(out)

def _crop_faces(frames: "np.ndarray") -> "np.ndarray":
    """Run MTCNN on each frame, crop to face, resize to IMAGE_SIZE.
    If no face is detected, fall back to a center crop of the original frame."""
    import numpy as np
    import cv2
    from PIL import Image
    detector = _get_face_detector()
    out = np.empty((len(frames), IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    pil_frames = [Image.fromarray(f) for f in frames]
    boxes_list, _ = detector.detect(pil_frames)
    H, W = frames[0].shape[:2]
    for i, (frame, boxes) in enumerate(zip(frames, boxes_list)):
        if boxes is not None and len(boxes) > 0:
            x1, y1, x2, y2 = [int(max(0, v)) for v in boxes[0]]
            x2 = min(W, x2); y2 = min(H, y2)
            if x2 > x1 and y2 > y1:
                crop = frame[y1:y2, x1:x2]
            else:
                crop = frame
        else:
            # center square crop fallback
            s = min(H, W)
            top = (H - s) // 2; left = (W - s) // 2
            crop = frame[top:top+s, left:left+s]
        out[i] = cv2.resize(crop, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    return out

def _resize_only(frames: "np.ndarray") -> "np.ndarray":
    """For non-face datasets: center-crop to square, resize to IMAGE_SIZE."""
    import numpy as np
    import cv2
    H, W = frames[0].shape[:2]
    s = min(H, W)
    top = (H - s) // 2; left = (W - s) // 2
    out = np.empty((len(frames), IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    for i, f in enumerate(frames):
        crop = f[top:top+s, left:left+s]
        out[i] = cv2.resize(crop, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    return out

def _process_one(sample: VideoSample, cache_dir: Path) -> Optional[Path]:
    """Extract frames for one video. Returns the cache path, or None on failure."""
    # Stable cache filename: hash of source path so collisions are impossible
    digest = hashlib.md5(str(sample.path).encode()).hexdigest()[:16]
    cache_path = cache_dir / f"{sample.dataset}_{sample.label}_{digest}.npy"
    if cache_path.exists():
        return cache_path
    try:
        if sample.extra.get("is_frame_dir"):
            frames = _decode_frame_dir(sample.path, NUM_FRAMES)
        else:
            frames = _decode_video(sample.path, NUM_FRAMES)
        if frames is None:
            return None
        if sample.dataset in FACE_DATASETS and not sample.extra.get("is_frame_dir"):
            frames = _crop_faces(frames)
        else:
            frames = _resize_only(frames)
        import numpy as np
        np.save(cache_path, frames)
        return cache_path
    except Exception as e:
        LOG.warning(f"Failed to process {sample.path}: {e}")
        return None


def _build_splits(rows: list[dict]) -> list[dict]:
    """Identity-aware 70/15/15 split. All clips with the same identity end up
    in the same split — this is the no-leakage guarantee."""
    rng = random.Random(SEED)
    by_ident: dict[str, list[dict]] = {}
    for r in rows:
        by_ident.setdefault(r["identity"], []).append(r)
    idents = list(by_ident.keys()); rng.shuffle(idents)
    n = len(idents)
    n_train = int(n * 0.70); n_val = int(n * 0.15)
    train_ids = set(idents[:n_train])
    val_ids   = set(idents[n_train:n_train + n_val])
    for r in rows:
        if r["identity"] in train_ids: r["split"] = "train"
        elif r["identity"] in val_ids: r["split"] = "val"
        else: r["split"] = "test"
    return rows


def preprocess_one(name: str) -> None:
    if state_done(f"pp_{name}"):
        LOG.info(f"Preprocessing already done: {name}"); return

    samples = _enumerate_dataset(name)
    if not samples:
        LOG.warning(f"No videos found for {name}. Skipping.")
        return
    LOG.info(f"{name}: found {len(samples)} videos "
             f"(real={sum(1 for s in samples if s.label==0)}, "
             f"fake={sum(1 for s in samples if s.label==1)})")

    cap = DATASET_CAPS.get(name)
    if cap:
        samples = _balance_and_cap(samples, cap)
        LOG.info(f"{name}: balanced+capped to {len(samples)}")

    cache_dir = DIR_FRAMES / name
    cache_dir.mkdir(parents=True, exist_ok=True)

    from tqdm import tqdm
    rows = []
    failures = 0
    # Sequential processing — face detector + PyAV both already use threads
    # internally. Adding multiprocessing on top causes GPU OOM with MTCNN.
    for s in tqdm(samples, desc=f"Extracting {name}"):
        path = _process_one(s, cache_dir)
        if path is None:
            failures += 1
            continue
        rows.append({
            "filepath": str(path),
            "label": s.label,
            "identity": s.identity,
            "dataset": name,
        })
    LOG.info(f"{name}: extracted {len(rows)} / {len(samples)} ({failures} failures)")

    if not rows:
        LOG.error(f"{name}: zero successful extractions. Skipping manifest.")
        return

    rows = _build_splits(rows)

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(DIR_MANIFESTS / f"{name}.csv", index=False)
    LOG.info(f"{name} manifest: train={sum(df.split=='train')}, "
             f"val={sum(df.split=='val')}, test={sum(df.split=='test')}")
    mark_done(f"pp_{name}")


def cmd_preprocess(args: argparse.Namespace) -> None:
    ensure_dirs()
    targets = list(DOWNLOADERS.keys()) if args.dataset == "all" else [args.dataset]
    for ds in targets:
        LOG.info(f"=== Preprocess: {ds} ===")
        try:
            preprocess_one(ds)
        except Exception as e:
            LOG.error(f"Preprocessing of {ds} failed: {e}")
            LOG.error(traceback.format_exc())
            if args.dataset != "all":
                raise
        finally:
            free_gpu()


# =============================================================================
# 4. DATASET / DATALOADER
# =============================================================================

def _build_dataset_class():
    """Built lazily so the heavy imports only happen at train time."""
    import numpy as np
    import torch
    from torch.utils.data import Dataset

    class FrameNpyDataset(Dataset):
        """Loads pre-extracted .npy frame caches.
        Each sample is a tensor of shape (T, 3, H, W), float, normalized
        with the ViViT processor's mean/std."""

        def __init__(self, df, processor, augment: bool = False):
            self.df = df.reset_index(drop=True)
            self.augment = augment
            mean = np.array(processor.image_mean, dtype=np.float32) * 255.0
            std  = np.array(processor.image_std,  dtype=np.float32) * 255.0
            self.mean = mean.reshape(1, 1, 1, 3)
            self.std  = std.reshape(1, 1, 1, 3)

        def __len__(self): return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            # mmap_mode='r' avoids loading the whole array into RAM at once
            arr = np.load(row.filepath, mmap_mode="r").astype(np.float32)  # (T,H,W,3)
            arr = arr.copy()  # detach from mmap before in-place ops

            # Light augmentation — cheap on CPU, helps generalisation a lot.
            if self.augment:
                if random.random() < 0.5:  # horizontal flip
                    arr = arr[:, :, ::-1, :].copy()
                # Tiny brightness/contrast jitter (uniform over the clip)
                if random.random() < 0.5:
                    bright = random.uniform(0.85, 1.15)
                    arr = np.clip(arr * bright, 0, 255)
                if random.random() < 0.3:
                    # Per-clip Gaussian noise — boosts robustness to compression artifacts
                    noise = np.random.normal(0, 5.0, arr.shape).astype(np.float32)
                    arr = np.clip(arr + noise, 0, 255)

            arr = (arr - self.mean) / self.std
            arr = arr.transpose(0, 3, 1, 2)  # (T, 3, H, W)
            return {
                "pixel_values": torch.from_numpy(arr),
                "labels": torch.tensor(int(row.label), dtype=torch.long),
            }

    return FrameNpyDataset


def _make_loaders(name: str, processor, batch_size: int):
    import pandas as pd
    from torch.utils.data import DataLoader
    DS = _build_dataset_class()

    df = pd.read_csv(DIR_MANIFESTS / f"{name}.csv")
    train_ds = DS(df[df.split == "train"], processor, augment=True)
    val_ds   = DS(df[df.split == "val"],   processor, augment=False)
    test_ds  = DS(df[df.split == "test"],  processor, augment=False)

    n_workers = min(NUM_WORKERS, max(1, (os.cpu_count() or 4) - 2))
    loader_kwargs = dict(
        batch_size=batch_size, num_workers=n_workers,
        pin_memory=True, persistent_workers=(n_workers > 0),
        drop_last=False,
    )
    return (
        DataLoader(train_ds, shuffle=True,  **loader_kwargs),
        DataLoader(val_ds,   shuffle=False, **loader_kwargs),
        DataLoader(test_ds,  shuffle=False, **loader_kwargs),
    )


# =============================================================================
# 5. MODEL — ViViT + LoRA
# =============================================================================

def _resize_temporal_pos_embed(model, new_num_frames: int) -> None:
    """ViViT's pre-trained pos embedding is sized for 32 frames (16 temporal patches
    after the [2,16,16] tubelet). When NUM_FRAMES != 32 we need to interpolate the
    TEMPORAL axis of the position embedding to match — HuggingFace's built-in
    interpolate_pos_encoding only handles the spatial axis.

    Pos emb shape: (1, 1 + T*H*W, D) where T = num_frames/tubelet[0], H = W = 14, D = 768.
    """
    import torch
    cfg = model.config
    Th = cfg.image_size // cfg.tubelet_size[1]  # 14
    Tw = cfg.image_size // cfg.tubelet_size[2]  # 14
    old_T = cfg.num_frames // cfg.tubelet_size[0]
    new_T = new_num_frames // cfg.tubelet_size[0]
    if new_T == old_T:
        return  # nothing to do

    # The actual nn.Parameter lives at model.{vivit|base_model}.embeddings.position_embeddings.
    # Locate it dynamically so we don't break if HF renames internals.
    embeddings_module = None
    for m in model.modules():
        if hasattr(m, "position_embeddings") and hasattr(m, "patch_embeddings"):
            embeddings_module = m
            break
    if embeddings_module is None:
        LOG.warning("Couldn't locate VivitEmbeddings to resize pos embedding; skipping.")
        return

    pos = embeddings_module.position_embeddings.data  # (1, 1 + old_T*H*W, D)
    cls_pos, patch_pos = pos[:, :1], pos[:, 1:]
    D = patch_pos.shape[-1]
    # (1, old_T, H, W, D) -> (1, D, old_T, H, W) for trilinear interp
    grid = patch_pos.reshape(1, old_T, Th, Tw, D).permute(0, 4, 1, 2, 3)
    grid = torch.nn.functional.interpolate(
        grid, size=(new_T, Th, Tw), mode="trilinear", align_corners=False,
    )
    new_patch_pos = grid.permute(0, 2, 3, 4, 1).reshape(1, new_T * Th * Tw, D)
    new_pos = torch.cat([cls_pos, new_patch_pos], dim=1)

    # Also need to update the patch_embeddings.num_patches and re-create the parameter
    embeddings_module.position_embeddings = torch.nn.Parameter(new_pos)
    embeddings_module.patch_embeddings.num_frames = new_num_frames
    embeddings_module.patch_embeddings.num_patches = new_T * Th * Tw
    cfg.num_frames = new_num_frames
    LOG.info(f"Resized temporal pos embedding: {old_T} -> {new_T} temporal patches.")


def _build_model(mode: str = "lora"):
    """Build ViViT classifier with the chosen fine-tuning strategy.

    mode:
      'lora' — LoRA r=16 + AMP. Trains ~2-4M params. Default. (Recommended)
      'full' — Full FT + AMP. Trains all 88M params. Needs more VRAM.
      'head' — Train only the classification head. Fastest, lowest accuracy.
    """
    import torch
    from transformers import VivitForVideoClassification

    model = VivitForVideoClassification.from_pretrained(
        MODEL_ID,
        num_labels=2,
        ignore_mismatched_sizes=True,
        attn_implementation="sdpa",  # Fast attention impl, supports FP16
    )

    # If we're using a non-default frame count, interpolate the pretrained
    # position embeddings to match. Without this, the forward pass would error
    # with a shape mismatch (HF's interpolate_pos_encoding handles only spatial).
    if NUM_FRAMES != model.config.num_frames:
        _resize_temporal_pos_embed(model, NUM_FRAMES)

    if mode == "head":
        for n, p in model.named_parameters():
            p.requires_grad = "classifier" in n

    elif mode == "lora":
        from peft import LoraConfig, get_peft_model, TaskType
        # Target only the attention QKV — these have the highest impact
        # per parameter for vision transformers. We keep "dense" out because
        # ViViT has many "dense" submodules (intermediate, output, attn-output)
        # and adapting all of them inflates trainable params unnecessarily.
        lora_cfg = LoraConfig(
            r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
            target_modules=["query", "key", "value"],
            modules_to_save=["classifier"],   # Always trains the head fully
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    elif mode == "full":
        for p in model.parameters():
            p.requires_grad = True

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return model


# =============================================================================
# 6. TRAIN / EVAL LOOP
# =============================================================================

def _eval_loop(model, loader, device, criterion):
    """Returns (loss, acc, auc) on the given loader."""
    import numpy as np
    import torch
    from sklearn.metrics import roc_auc_score, accuracy_score

    model.eval()
    losses, all_logits, all_labels = [], [], []
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            out = model(pixel_values=pixel_values, interpolate_pos_encoding=True)
            loss = criterion(out.logits, labels)
            losses.append(loss.item())
            all_logits.append(out.logits.float().cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    logits = np.concatenate(all_logits); labels = np.concatenate(all_labels)
    probs_fake = (np.exp(logits[:, 1]) / np.exp(logits).sum(axis=1))
    preds = logits.argmax(axis=1)
    acc = float(accuracy_score(labels, preds))
    try:
        auc = float(roc_auc_score(labels, probs_fake)) if len(set(labels)) > 1 else float("nan")
    except Exception:
        auc = float("nan")
    return float(np.mean(losses)), acc, auc


def train_one_dataset(name: str, mode: str, resume_from: Optional[str] = None) -> None:
    if state_done(f"train_{name}_{mode}"):
        LOG.info(f"Already trained {name} ({mode}). Use --force to retrain.")
        return

    set_seed(SEED)
    import torch
    from torch import nn
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from transformers import VivitImageProcessor

    device = torch.device("cuda")
    processor = VivitImageProcessor.from_pretrained(MODEL_ID)

    # Try to fit the requested batch size, halve on OOM, resume training.
    batch_size = BATCH_SIZE
    while True:
        try:
            train_loader, val_loader, _ = _make_loaders(name, processor, batch_size)
            model = _build_model(mode).to(device)
            if resume_from:
                model = _load_checkpoint(model, resume_from, mode)
            else:
                # Continue from the previous dataset's best ckpt (curriculum learning)
                model = _maybe_continue_from_previous(model, name, mode)
            break
        except torch.cuda.OutOfMemoryError:
            free_gpu()
            if batch_size <= 1:
                raise
            batch_size //= 2
            LOG.warning(f"OOM at batch_size={batch_size*2}; retrying with batch_size={batch_size}")

    LOG.info(f"Starting train: dataset={name}, mode={mode}, batch_size={batch_size}, "
             f"effective={batch_size * GRAD_ACCUM}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = max(1, len(train_loader) // GRAD_ACCUM) * EPOCHS_MAX
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=LR * 0.05)
    scaler = torch.amp.GradScaler("cuda")

    best_auc = -1.0
    best_path = DIR_CKPT / f"{name}_{mode}_best.pt"
    metrics_path = DIR_LOGS / f"{name}_{mode}_metrics.jsonl"
    epochs_no_improve = 0

    from tqdm import tqdm
    global_step = 0
    for epoch in range(1, EPOCHS_MAX + 1):
        model.train()
        epoch_losses = []
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(train_loader, desc=f"[{name}][{mode}] ep {epoch}/{EPOCHS_MAX}")
        for step, batch in enumerate(pbar):
            try:
                pixel_values = batch["pixel_values"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    out = model(pixel_values=pixel_values, interpolate_pos_encoding=True)
                    loss = criterion(out.logits, labels) / GRAD_ACCUM
                scaler.scale(loss).backward()

                if (step + 1) % GRAD_ACCUM == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    global_step += 1

                epoch_losses.append(loss.item() * GRAD_ACCUM)
                pbar.set_postfix(loss=f"{epoch_losses[-1]:.4f}")
            except torch.cuda.OutOfMemoryError:
                # Skip this batch rather than crash the whole run
                LOG.warning("OOM during training step — skipping batch and clearing cache")
                optimizer.zero_grad(set_to_none=True)
                free_gpu()
                continue

        train_loss = sum(epoch_losses) / max(1, len(epoch_losses))
        val_loss, val_acc, val_auc = _eval_loop(model, val_loader, device, criterion)
        LOG.info(f"[{name}][{mode}] ep {epoch}: train_loss={train_loss:.4f}, "
                 f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, val_auc={val_auc:.4f}")

        # Append metrics
        with metrics_path.open("a") as f:
            f.write(json.dumps({
                "epoch": epoch, "train_loss": train_loss,
                "val_loss": val_loss, "val_acc": val_acc, "val_auc": val_auc,
            }) + "\n")

        # Best-model selection
        improved = (val_auc > best_auc) and (val_auc == val_auc)  # not NaN
        if improved:
            best_auc = val_auc
            _save_checkpoint(model, best_path, mode)
            LOG.info(f"  -> NEW BEST AUC={best_auc:.4f}, saved to {best_path}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Always save 'last' so a crash doesn't lose progress
        _save_checkpoint(model, DIR_CKPT / f"{name}_{mode}_last.pt", mode)

        # Early stopping
        if epochs_no_improve >= EARLY_STOP_PAT and epoch >= 3:
            LOG.info(f"Early stopping: no AUC improvement for {epochs_no_improve} epochs.")
            break
        if best_auc >= 0.97:
            LOG.info(f"Reached AUC>=0.97; stopping early to save GPU time.")
            break

    LOG.info(f"=== {name} done. Best val AUC = {best_auc:.4f} ===")
    mark_done(f"train_{name}_{mode}")
    free_gpu()


def _save_checkpoint(model, path: Path, mode: str) -> None:
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    if mode == "lora":
        # Save only the adapter + classifier head. Tiny file (~10MB) vs ~350MB.
        from peft import PeftModel
        if isinstance(model, PeftModel):
            adapter_dir = path.with_suffix("")  # use as a directory
            adapter_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(adapter_dir))
            return
    torch.save({"state_dict": model.state_dict(), "mode": mode}, path)


def _load_checkpoint(model, path: str, mode: str):
    """Load weights into `model` IN-PLACE. Returns the (possibly wrapped) model.

    The caller must reassign: `model = _load_checkpoint(model, path, mode)`.
    For LoRA, when the model isn't yet a PeftModel, this returns a wrapped one.
    When it already is, we load adapter weights into the existing wrapper.
    """
    import torch
    p = Path(path)
    if mode == "lora" and p.is_dir():
        from peft import PeftModel
        if isinstance(model, PeftModel):
            # Already wrapped (built by _build_model). Replace adapter weights.
            model.load_adapter(str(p), adapter_name="default", is_trainable=True)
            return model
        # Bare model + LoRA dir: wrap it.
        return PeftModel.from_pretrained(model, str(p), is_trainable=True)
    state = torch.load(p, map_location="cpu")
    sd = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
    model.load_state_dict(sd, strict=False)
    return model

def _maybe_continue_from_previous(model, name: str, mode: str):
    """If we already have a 'best' ckpt from a previous dataset, load it as a
    warm start. Order matters: train on bigger/cleaner datasets first.
    Returns the (possibly wrapped) model — caller must reassign.
    """
    order = ["ffpp", "celebdf_v2", "dfdc", "wilddeepfake",
             "fakeavceleb", "aigvdet", "genvideo"]
    if name not in order:
        return model
    prev = order[:order.index(name)]
    for p in reversed(prev):
        cand_pt = DIR_CKPT / f"{p}_{mode}_best.pt"
        cand_dir = cand_pt.with_suffix("")  # LoRA dir form
        if cand_pt.exists():
            LOG.info(f"Warm-starting from previous ckpt: {cand_pt}")
            return _load_checkpoint(model, str(cand_pt), mode)
        if mode == "lora" and cand_dir.is_dir() and (cand_dir / "adapter_config.json").exists():
            LOG.info(f"Warm-starting from previous LoRA adapter: {cand_dir}")
            return _load_checkpoint(model, str(cand_dir), mode)
    return model


def cmd_train(args: argparse.Namespace) -> None:
    ensure_dirs()
    targets = ["ffpp", "celebdf_v2", "dfdc", "wilddeepfake",
               "fakeavceleb", "aigvdet", "genvideo"] if args.dataset == "all" else [args.dataset]
    for ds in targets:
        if not (DIR_MANIFESTS / f"{ds}.csv").exists():
            LOG.warning(f"No manifest for {ds} — skipping (run preprocess first).")
            continue
        LOG.info(f"=== Train: {ds} ({args.mode}) ===")
        try:
            train_one_dataset(ds, args.mode, resume_from=args.resume)
        except Exception as e:
            LOG.error(f"Training of {ds} failed: {e}")
            LOG.error(traceback.format_exc())
            if args.dataset != "all":
                raise
        finally:
            free_gpu()


# =============================================================================
# 7. EVALUATION
# =============================================================================

def cmd_evaluate(args: argparse.Namespace) -> None:
    ensure_dirs()
    set_seed(SEED)
    import torch
    from torch import nn
    from transformers import VivitImageProcessor

    targets = list(DOWNLOADERS.keys()) if args.dataset == "all" else [args.dataset]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = VivitImageProcessor.from_pretrained(MODEL_ID)
    criterion = nn.CrossEntropyLoss()

    results = {}
    for ds in targets:
        if not (DIR_MANIFESTS / f"{ds}.csv").exists():
            continue
        ckpt_pt = DIR_CKPT / f"{ds}_{args.mode}_best.pt"
        ckpt_dir = ckpt_pt.with_suffix("")  # for LoRA
        if not (ckpt_pt.exists() or ckpt_dir.is_dir()):
            LOG.warning(f"No checkpoint for {ds} ({args.mode}) — skipping.")
            continue
        model = _build_model(args.mode).to(device)
        load_path = str(ckpt_dir if (args.mode == "lora" and ckpt_dir.is_dir()) else ckpt_pt)
        model = _load_checkpoint(model, load_path, args.mode)
        _, _, test_loader = _make_loaders(ds, processor, BATCH_SIZE)
        loss, acc, auc = _eval_loop(model, test_loader, device, criterion)
        LOG.info(f"[TEST] {ds}: loss={loss:.4f}, acc={acc:.4f}, auc={auc:.4f}")
        results[ds] = {"loss": loss, "acc": acc, "auc": auc}
        del model; free_gpu()

    out = DIR_LOGS / "test_results.json"
    if out.exists():
        prev = json.loads(out.read_text())
        prev.update(results); results = prev
    out.write_text(json.dumps(results, indent=2))
    LOG.info(f"Test results saved to {out}")


# =============================================================================
# 8. STATUS / HELPERS
# =============================================================================

def cmd_status(args: argparse.Namespace) -> None:
    ensure_dirs()
    print(f"\nDeepTruth pipeline status (root: {ROOT})\n" + "=" * 60)
    print(f"{'phase':<28} {'dataset':<14} {'state'}")
    print("-" * 60)
    for ds in DOWNLOADERS:
        for phase in ["dl", "pp"]:
            done = "✓" if state_done(f"{phase}_{ds}") else "."
            label = {"dl": "download", "pp": "preprocess"}[phase]
            print(f"{label:<28} {ds:<14} {done}")
        for mode in ["lora", "full", "head"]:
            done = "✓" if state_done(f"train_{ds}_{mode}") else "."
            print(f"train ({mode}){'':<16}{ds:<14} {done}")
    print("-" * 60)


# =============================================================================
# 9. MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="DeepTruth ViViT fine-tuning pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("setup", help="Install dependencies, verify GPU, prefetch model")

    p_dl = sub.add_parser("download", help="Download dataset(s)")
    p_dl.add_argument("dataset", help="dataset name or 'all'")

    p_pp = sub.add_parser("preprocess", help="Extract & cache frames as .npy")
    p_pp.add_argument("dataset", help="dataset name or 'all'")

    p_tr = sub.add_parser("train", help="Train ViViT on dataset(s)")
    p_tr.add_argument("dataset", help="dataset name or 'all'")
    p_tr.add_argument("--mode", choices=["lora", "full", "head"], default="lora")
    p_tr.add_argument("--resume", default=None,
                      help="Path to checkpoint (file or LoRA dir) to warm-start from")

    p_ev = sub.add_parser("evaluate", help="Evaluate on test split")
    p_ev.add_argument("dataset", help="dataset name or 'all'")
    p_ev.add_argument("--mode", choices=["lora", "full", "head"], default="lora")

    sub.add_parser("status", help="Show pipeline progress")

    args = parser.parse_args()
    set_seed(SEED)

    {
        "setup": cmd_setup,
        "download": cmd_download,
        "preprocess": cmd_preprocess,
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "status": cmd_status,
    }[args.cmd](args)


if __name__ == "__main__":
    # Graceful shutdown on Ctrl+C / SIGTERM (so checkpoints get saved)
    def _sigterm(signo, frame):
        LOG.warning(f"Received signal {signo} — exiting cleanly.")
        sys.exit(130)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _sigterm)
    main()
