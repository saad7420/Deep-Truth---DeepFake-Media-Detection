"""Single place where the server binds to `deeptruth_pipeline`.

The server used to carry its own vendored copy of the pipeline under
`app/_dtp_src`. That copy drifted (it had no image support at all), so the
merged layout keeps exactly one source of truth: the `deeptruth_pipeline`
package that this `server/` directory now lives inside.

    <project root>/                 <- the deeptruth_pipeline package
        pipeline.py, registry.py, inferencers/, preprocessors/
        videos_checkpoints/         <- *_lora_best (ViViT + LoRA)
        images_checkpoints/         <- image_*_lora_best (ViT-B/16 + LoRA)
        train_pipeline/deeptruth_train.py
        server/   <- you are here
        client/

`bootstrap()` puts the package's *parent* on sys.path so `import
deeptruth_pipeline` resolves, and defaults every DEEPTRUTH_* path to the
in-repo location. Every value is `setdefault`, so anything already exported
in the environment (or server/.env) still wins.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# app/_dtp.py -> app/ -> server/ -> <project root == deeptruth_pipeline pkg>
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SERVER_ROOT = PROJECT_ROOT / "server"

_done = False
_pipeline = None


def bootstrap() -> Path:
    """Idempotent. Returns the project root."""
    global _done
    if _done:
        return PROJECT_ROOT

    parent = str(PROJECT_ROOT.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)

    # Weights live under checkpoints/. The old top-level videos_checkpoints/
    # and images_checkpoints/ paths made every case fail with
    # CheckpointsUnavailable unless env vars were exported by hand.
    ckpts = PROJECT_ROOT / "checkpoints"
    os.environ.setdefault("DEEPTRUTH_ROOT", str(PROJECT_ROOT))
    os.environ.setdefault("DEEPTRUTH_CHECKPOINTS",
                          str(ckpts / "videos_checkpoints"))
    os.environ.setdefault("DEEPTRUTH_IMAGE_CHECKPOINTS",
                          str(ckpts / "images_checkpoints"))
    # Audio and SRM only switch from stub to real model when these are set,
    # so set them only when the weights are actually on disk.
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
    """Build (once, lazily) the shared deeptruth_pipeline.Pipeline.

    One Pipeline instance backs both the visual and the image engine so the
    ViViT and ViT checkpoints are each loaded into memory only once, and both
    engines share the preprocessing cache under DEEPTRUTH_CACHE.

    The import is deferred to here so that importing an engine module doesn't
    drag in torch/transformers until something actually runs inference.
    """
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    bootstrap()
    from deeptruth_pipeline import Pipeline, Registry

    _pipeline = Pipeline(registry=Registry())
    return _pipeline
