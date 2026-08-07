"""Bridge to the training script.

Inference reuses the training-time functions for frame decoding, MTCNN crop,
center-resize, model build, and checkpoint load to guarantee identical
preprocessing. Adjust DEEPTRUTH_TRAIN_PIPELINE if the script lives elsewhere.
"""
from __future__ import annotations
import importlib.util
import sys

from .config import TRAIN_PIPELINE

_module = None


def load():
    global _module
    if _module is not None:
        return _module

    if not TRAIN_PIPELINE.exists():
        raise FileNotFoundError(
            f"Training pipeline script not found: {TRAIN_PIPELINE}\n"
            f"Set DEEPTRUTH_TRAIN_PIPELINE to its absolute path."
        )

    sys.path.insert(0, str(TRAIN_PIPELINE.parent))
    spec = importlib.util.spec_from_file_location("deeptruth_train", TRAIN_PIPELINE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["deeptruth_train"] = mod
    spec.loader.exec_module(mod)
    _module = mod
    return mod
