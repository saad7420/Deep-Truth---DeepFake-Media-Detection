"""
M8 inferencer — SRM Noise Analysis Stream.

Two halves with very different maturity, same split as the audio module:

  feature extraction   real and working right now (preprocessors/srm_filters.py,
                        numerically verified in preprocessors/test_srm_filters.py)
  classifier head       needs training — see train_pipeline/train_srm_casia.py

Until DEEPTRUTH_SRM_CHECKPOINT is set, predict() returns a neutral result:
the features are computed (so the plumbing is exercised end to end and the
evidence is genuinely available for debugging), but no verdict is produced,
because an untrained classifier's output is not a signal, it is noise
wearing the shape of one.

CHECKPOINT CONTRACT
─────────────────────
A tiny MLP, not a CNN. The features are a fixed-length statistic vector, not
a spatial map, so there is nothing for a CNN's convolutions to slide over.

The vector is 45-dim, not the original 15: preprocessors/srm_filters.py
tiles each frame into patches and computes mean, std, and a within-image
outlier score per base statistic (patch_feature_vector /
patch_feature_names). Plain whole-image pooling was tried first and
measured to wash out a local splice's signal to an unlearnable Cohen's d of
~0.15-0.25 (see preprocessors/test_srm_filters.py) — a splice is local, and
averaging it together with the untouched majority of the frame erases
almost all of it. Comparing each patch against the other patches in its own
image, rather than against other photos, recovers d > 1.2 on the same test.

    Linear(45 -> 32)  ReLU  Dropout(0.2)
    Linear(32 -> 16)  ReLU
    Linear(16 -> 2)

`load_state_dict(strict=True)`. If FEATURE_DIM, HEAD_DIMS, or the layer
order below changes, a checkpoint trained against the old shape will not
load — that mismatch is intentional; a silent shape-mismatch load would be
worse than a loud one.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .base import Inferencer, InferenceResult

log = logging.getLogger(__name__)

FEATURE_DIM = 45   # patch-based: mean/std/outlier per base stat — see
                   # preprocessors/srm_filters.py::patch_feature_vector.
                   # Was 15 (plain whole-image mean) until that pooling was
                   # measured to wash out local splice signal down to an
                   # unlearnable Cohen's d of ~0.15-0.25; patch-based
                   # outlier detection recovers d > 1.2 on the same test.
                   # See preprocessors/test_srm_filters.py for the
                   # measurement this number is based on.
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

        # The 45-dim feature vector packs groups on very different scales
        # (patch-mean stats sit under ~3, outlier-ratio stats run into the
        # 30s-50s). The classifier was trained on features standardised
        # with these exact per-dimension mean/std values — scoring raw,
        # unnormalised features against it produces meaningless output, the
        # same way loading weights for the wrong architecture would.
        # Required, not optional: a checkpoint missing these is treated as
        # unusable rather than silently skipping normalisation.
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

    def predict(self, media_key: str, preprocessed: dict, **opts) -> InferenceResult:
        threshold = float(opts.get("threshold", self.threshold))

        features_path = preprocessed.get("features_path")
        if not features_path or not Path(features_path).exists():
            return InferenceResult(
                media_key=media_key, modality="srm", trust_score=float("nan"),
                verdict="UNKNOWN", confidence=0.0,
                rationale="SRM features were not produced by preprocessing.")

        feats = np.load(features_path)  # (n_frames, FEATURE_DIM)
        frame_summary = {
            "n_frames_used": int(preprocessed.get("n_frames_used", feats.shape[0])),
            "feature_dim": int(feats.shape[1]),
        }

        try:
            self._setup()
        except FileNotFoundError as exc:
            # Soft-fail: the features were computed (real signal, kept in
            # `extra` for evidence/debugging) even though no classifier
            # exists yet to score them.
            return InferenceResult(
                media_key=media_key, modality="srm", trust_score=float("nan"),
                verdict="UNKNOWN", confidence=0.0,
                rationale=str(exc), extra={"features_computed": True, **frame_summary})
        except RuntimeError as exc:
            # Checkpoint present but missing normalisation stats — same
            # "computed but can't be scored" situation as above, different
            # cause.
            return InferenceResult(
                media_key=media_key, modality="srm", trust_score=float("nan"),
                verdict="UNKNOWN", confidence=0.0,
                rationale=str(exc), extra={"features_computed": True, **frame_summary})

        feats = (feats - self._feat_mean) / self._feat_std

        import torch
        with torch.no_grad():
            x = torch.from_numpy(feats).to(self._device)
            logits = self._model(x)
            probs = torch.softmax(logits, dim=-1)[:, 1]  # P(fake) per frame
            per_frame = probs.cpu().numpy()

        # Per-frame scores are aggregated by max, not mean: SRM's whole
        # purpose is catching a LOCAL inconsistency (one spliced region, one
        # frame with a boundary artefact) — averaging it against frames with
        # nothing wrong dilutes exactly the signal this module exists to
        # surface. This mirrors the video engine's own face_avg/genvideo
        # max-of-branches logic for the identical reason (fusion.py).
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