from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SERVER_ROOT = PROJECT_ROOT / "server"

_done = False
_pipeline = None


# defaults only, real environment variables and .env still win
def bootstrap() -> Path:
    global _done
    if _done:
        return PROJECT_ROOT

    parent = str(PROJECT_ROOT.parent)
    if parent not in sys.path:
        # the repo root is itself the deeptruth_pipeline package
        sys.path.insert(0, parent)

    ckpts = PROJECT_ROOT / "checkpoints"
    os.environ.setdefault("DEEPTRUTH_ROOT", str(PROJECT_ROOT))
    os.environ.setdefault("DEEPTRUTH_CHECKPOINTS",
                          str(ckpts / "videos_checkpoints"))
    os.environ.setdefault("DEEPTRUTH_IMAGE_CHECKPOINTS",
                          str(ckpts / "images_checkpoints"))
    if (ckpts / "audios_checkpoints" / "model.pt").is_file():
        os.environ.setdefault("DEEPTRUTH_AUDIO_CHECKPOINT",
                              str(ckpts / "audios_checkpoints"))
    if (ckpts / "srm_checkpoints" / "model.pt").is_file():
        os.environ.setdefault("DEEPTRUTH_SRM_CHECKPOINT",
                              str(ckpts / "srm_checkpoints" / "model.pt"))
    os.environ.setdefault("DEEPTRUTH_TRAIN_PIPELINE",
                          str(PROJECT_ROOT / "train_pipeline" / "deeptruth_train.py"))
    os.environ.setdefault("DEEPTRUTH_CACHE", str(SERVER_ROOT / "_dtp_cache"))
    os.environ.setdefault("DEEPTRUTH_LOGS", str(SERVER_ROOT / "_dtp_logs"))

    _done = True
    return PROJECT_ROOT


def get_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    bootstrap()
    from deeptruth_pipeline import Pipeline, Registry

    _pipeline = Pipeline(registry=Registry())
    return _pipeline
