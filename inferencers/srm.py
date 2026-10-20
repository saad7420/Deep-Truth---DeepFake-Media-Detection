from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .base import Inferencer, InferenceResult

log = logging.getLogger(__name__)

FEATURE_DIM = 45
HEAD_DIMS = [FEATURE_DIM, 32, 16, 2]
DEFAULT_THRESHOLD = 0.5


def _build_classifier():
    import torch.nn as nn

    class SRMClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(HEAD_DIMS[0], HEAD_DIMS[1]),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(HEAD_DIMS[1], HEAD_DIMS[2]),
                nn.ReLU(),
                nn.Linear(HEAD_DIMS[2], HEAD_DIMS[3]),
            )

        def forward(self, x):
            return self.net(x)

    return SRMClassifier()


class SRMInferencer(Inferencer):
    modality = "srm"

    def __init__(self, checkpoint_path: str | None = None,
                threshold: float = DEFAULT_THRESHOLD,
                device: str | None = None):
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.threshold = threshold
        self._explicit_device = device
        self._model = None
        self._device = None
        self._feat_mean = None
        self._feat_std = None

    def supports(self, media_kind: str) -> bool:
        return media_kind in ("video", "image")

    def _setup(self) -> None:
        if self._model is not None:
            return
        if self.checkpoint_path is None:
            raise FileNotFoundError(
                "No SRM checkpoint configured. Set DEEPTRUTH_SRM_CHECKPOINT "
                "once train_pipeline/train_srm_casia.py has produced one.")

        import torch

        self._device = torch.device(
            self._explicit_device or ("cuda" if torch.cuda.is_available() else "cpu"))

        ckpt = self.checkpoint_path
        model_pt = ckpt / "model.pt" if ckpt.is_dir() else ckpt
        if not model_pt.exists():
            raise FileNotFoundError(
                f"SRM checkpoint not found: {model_pt}\n"
                f"Set DEEPTRUTH_SRM_CHECKPOINT to the correct path.")

        state = torch.load(model_pt, map_location=self._device)
        if any(k.startswith("module.") for k in state):
            state = {k.removeprefix("module."): v for k, v in state.items()}

        model = _build_classifier().to(self._device)
        model.load_state_dict(state, strict=True)
        model.eval()

        meta_path = (ckpt / "metadata.json") if ckpt.is_dir() else ckpt.parent / "metadata.json"
        if not meta_path.exists():
            raise RuntimeError(
                f"SRM checkpoint has no metadata.json beside it at {meta_path}. "
                f"feature_mean/feature_std are required to normalise inputs the "
                f"same way training did — without them this checkpoint's "
                f"predictions are meaningless. Retrain with the current "
                f"train_srm_casia.py, which writes these automatically.")
        import json
        meta = json.loads(meta_path.read_text())
        if "feature_mean" not in meta or "feature_std" not in meta:
            raise RuntimeError(
                f"{meta_path} is missing feature_mean/feature_std — this "
                f"checkpoint predates feature normalisation and cannot be "
                f"used safely. Retrain with the current train_srm_casia.py.")
        self._feat_mean = np.array(meta["feature_mean"], dtype=np.float32)
        self._feat_std = np.array(meta["feature_std"], dtype=np.float32)

        self._model = model
        log.info(f"SRMInferencer: loaded checkpoint {model_pt} "
                 f"device={self._device}")

    # the strongest frame represents the whole video
    def predict(self, media_key: str, preprocessed: dict, **opts) -> InferenceResult:
        threshold = float(opts.get("threshold", self.threshold))

        features_path = preprocessed.get("features_path")
        if not features_path or not Path(features_path).exists():
            return InferenceResult(
                media_key=media_key, modality="srm", trust_score=float("nan"),
                verdict="UNKNOWN", confidence=0.0,
                rationale="SRM features were not produced by preprocessing.")

        feats = np.load(features_path)
        frame_summary = {
            "n_frames_used": int(preprocessed.get("n_frames_used", feats.shape[0])),
            "feature_dim": int(feats.shape[1]),
        }

        try:
            self._setup()
        except FileNotFoundError as exc:
            return InferenceResult(
                media_key=media_key, modality="srm", trust_score=float("nan"),
                verdict="UNKNOWN", confidence=0.0,
                rationale=str(exc), extra={"features_computed": True, **frame_summary})
        except RuntimeError as exc:
            return InferenceResult(
                media_key=media_key, modality="srm", trust_score=float("nan"),
                verdict="UNKNOWN", confidence=0.0,
                rationale=str(exc), extra={"features_computed": True, **frame_summary})

        feats = (feats - self._feat_mean) / self._feat_std

        import torch
        with torch.no_grad():
            x = torch.from_numpy(feats).to(self._device)
            logits = self._model(x)
            probs = torch.softmax(logits, dim=-1)[:, 1]
            per_frame = probs.cpu().numpy()

        p_fake = float(per_frame.max())
        verdict = "FAKE" if p_fake >= threshold else "REAL"
        confidence = float(abs(p_fake - 0.5) * 2)

        return InferenceResult(
            media_key=media_key, modality="srm", trust_score=p_fake,
            verdict=verdict, confidence=confidence,
            per_model={f"frame_{i}": float(s) for i, s in enumerate(per_frame)},
            rationale=(
                f"SRM noise-residual classifier (5-filter bank -> "
                f"{FEATURE_DIM}-dim statistics -> MLP). "
                f"Strongest of {len(per_frame)} scored frame(s): "
                f"P(fake)={p_fake:.4f} vs threshold={threshold}."),
            extra={"features_computed": True, "threshold": threshold, **frame_summary},
        )
