#!/usr/bin/env python3
"""
Verify a trained SRM checkpoint, standalone — no deeptruth_pipeline repo
required. The filter bank and patch-based feature extraction are inlined
below (same reasoning as verify_audio_standalone.py: this should run
anywhere, not just inside a full pipeline checkout).

    pip install torch pillow scipy
    python verify_srm_checkpoint.py /path/to/final_model
    python verify_srm_checkpoint.py /path/to/final_model --image sample.jpg

A strict load proves the checkpoint's shapes and keys match this
architecture. --image additionally runs one real inference — including the
feature normalisation the checkpoint was trained with — so you see an
actual score, not just a successful load.

CONTRACT
─────────
45-dim patch-based features (5 filters x mean/std/within-image-outlier,
tiled over a 7x7 grid — see preprocessors/srm_filters.py), standardised
with the exact feature_mean/feature_std train_srm_casia.py saved into
metadata.json. Both the architecture AND the normalisation stats are
required for a checkpoint's predictions to mean anything; a checkpoint
missing either is treated as unusable, not silently scored wrong.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

FEATURE_DIM = 45
HEAD_DIMS = [FEATURE_DIM, 32, 16, 2]

GREEN, RED, YELLOW, DIM, RESET = (
    "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m")


# ── Filter bank + patch feature extraction, inlined ─────────────────────────
# Identical to preprocessors/srm_filters.py — kept in sync by hand.

_KV = np.array([
    [-1,  2, -2,  2, -1], [ 2, -6,  8, -6,  2], [-2,  8,-12,  8, -2],
    [ 2, -6,  8, -6,  2], [-1,  2, -2,  2, -1],
], dtype=np.float32) / 12.0


def _directional(dy, dx):
    k = np.zeros((5, 5), dtype=np.float32)
    k[2, 2] = -2
    k[2 + dy, 2 + dx] = 1
    k[2 - dy, 2 - dx] = 1
    return k


_BANK = np.stack([_KV, _directional(0, 1), _directional(1, 0),
                  _directional(1, 1), _directional(1, -1)])
_NAMES = ["kv", "horiz", "vert", "diag", "adiag"]
_TRUNCATION = 3.0
_PATCH_SIZE = 32  # 224/32 = 7x7 grid


def _base_stats(gray: np.ndarray) -> np.ndarray:
    from scipy.signal import convolve2d

    residuals = np.stack([convolve2d(gray, k, mode="valid") for k in _BANK])
    truncated = np.clip(residuals, -_TRUNCATION, _TRUNCATION)

    feats = []
    for i in range(len(_NAMES)):
        r = truncated[i]
        feats += [float(r.mean()), float(r.std()),
                  float((np.abs(residuals[i]) >= _TRUNCATION).mean())]
    return np.array(feats, dtype=np.float32)


def _feature_vector(gray: np.ndarray) -> np.ndarray:
    """45-dim: patch-mean / patch-std / within-image-outlier per base stat.
    See preprocessors/srm_filters.py's module docstring for why patch-based
    pooling is used instead of a plain whole-image average — global pooling
    was measured to wash a local splice's signal out almost entirely."""
    h, w = gray.shape
    margin = 4
    patches = []
    for y in range(0, h - _PATCH_SIZE + 1, _PATCH_SIZE):
        for x in range(0, w - _PATCH_SIZE + 1, _PATCH_SIZE):
            y0, y1 = max(0, y - margin), min(h, y + _PATCH_SIZE + margin)
            x0, x1 = max(0, x - margin), min(w, x + _PATCH_SIZE + margin)
            patch = gray[y0:y1, x0:x1]
            if patch.shape[0] >= 9 and patch.shape[1] >= 9:
                patches.append(patch)

    per_patch = np.stack([_base_stats(p) for p in patches])

    p_mean = per_patch.mean(axis=0)
    p_std = per_patch.std(axis=0)
    median = np.median(per_patch, axis=0)
    mad = np.median(np.abs(per_patch - median), axis=0) + 1e-6
    p_outlier = np.abs(per_patch - median).max(axis=0) / mad

    return np.concatenate([p_mean, p_std, p_outlier]).astype(np.float32)


def _to_grayscale(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)


def _check_imports() -> None:
    missing = []
    for module, pip_name in [("torch", "torch"), ("PIL", "pillow"), ("scipy", "scipy")]:
        try:
            __import__(module)
        except ImportError:
            missing.append(pip_name)
    if missing:
        sys.exit(f"\n{RED}Missing packages:{RESET} {', '.join(missing)}\n\n"
                 f"    pip install {' '.join(missing)}\n")


def build_model():
    import torch.nn as nn

    class SRMClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(HEAD_DIMS[0], HEAD_DIMS[1]), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(HEAD_DIMS[1], HEAD_DIMS[2]), nn.ReLU(),
                nn.Linear(HEAD_DIMS[2], HEAD_DIMS[3]),
            )

        def forward(self, x):
            return self.net(x)

    return SRMClassifier()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--image", default=None)
    ap.add_argument("--threshold", type=float, default=None)
    args = ap.parse_args()

    _check_imports()
    import torch

    ckpt = Path(args.checkpoint).expanduser().resolve()
    if not ckpt.exists():
        sys.exit(f"{RED}not found:{RESET} {ckpt}")

    print(f"\nDeep Truth — standalone SRM checkpoint check")
    print("(no deeptruth_pipeline codebase required)")
    print("=" * 64)

    pt = ckpt / "model.pt" if ckpt.is_dir() else ckpt
    if not pt.exists():
        sys.exit(f"{RED}FAIL{RESET} expected {pt}")
    print(f"weights {pt.name}  ({pt.stat().st_size / 1e3:.0f} KB)")

    meta_path = (ckpt / "metadata.json") if ckpt.is_dir() else ckpt.parent / "metadata.json"
    if not meta_path.exists():
        sys.exit(
            f"{RED}FAIL{RESET} no metadata.json beside the weights at {meta_path}\n"
            f"      feature_mean/feature_std live there and are required — "
            f"without them\n      this checkpoint's predictions are meaningless, "
            f"not just uncalibrated.")

    meta = json.loads(meta_path.read_text())
    print(f"{DIM}trained {meta.get('epochs_requested', '?')} epoch(s) on "
          f"{meta.get('trained_on', 'unknown data')} — "
          f"val EER {meta.get('val_eer', float('nan')):.2f}%{RESET}")

    if "feature_mean" not in meta or "feature_std" not in meta:
        sys.exit(
            f"{RED}FAIL{RESET} {meta_path} has no feature_mean/feature_std.\n"
            f"      This checkpoint predates feature normalisation and cannot "
            f"be used\n      safely — retrain with the current "
            f"train_srm_casia.py.")

    feat_mean = np.array(meta["feature_mean"], dtype=np.float32)
    feat_std = np.array(meta["feature_std"], dtype=np.float32)
    if feat_mean.shape[0] != FEATURE_DIM or feat_std.shape[0] != FEATURE_DIM:
        sys.exit(
            f"{RED}FAIL{RESET} feature_mean/feature_std are "
            f"{feat_mean.shape[0]}-dim, expected {FEATURE_DIM}.\n"
            f"      This checkpoint was trained against a different feature "
            f"layout than\n      this script expects — likely predates the "
            f"patch-based redesign.")
    print(f"{GREEN}ok{RESET}    normalisation stats present and correctly shaped")

    threshold = args.threshold
    if threshold is None:
        threshold = float(meta.get("recommended_threshold", 0.5))

    print("\n[1] Reading the state dict")
    try:
        state = torch.load(pt, map_location="cpu")
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"{RED}FAIL{RESET} torch.load failed: {exc}")

    if any(k.startswith("module.") for k in state):
        state = {k.removeprefix("module."): v for k, v in state.items()}
    print(f"{GREEN}ok{RESET}    {len(state)} tensors")

    print("\n[2] Loading into the SRMClassifier architecture (strict=True)")
    model = build_model()
    try:
        model.load_state_dict(state, strict=True)
        print(f"{GREEN}PASS{RESET}  checkpoint loads exactly.")
    except RuntimeError as exc:
        msg = str(exc)
        print(f"{RED}FAIL{RESET}  {msg[:300]}")
        if "size mismatch" in msg and "45" not in msg:
            print(f"\n{YELLOW}      This looks like a dimension mismatch — "
                  f"likely a checkpoint trained\n      against the older "
                  f"15-dim whole-image feature layout rather than the\n      "
                  f"current 45-dim patch-based one. Retrain with the current "
                  f"train_srm_casia.py.{RESET}")
        sys.exit(1)
    model.eval()

    if args.image:
        img_path = Path(args.image).expanduser().resolve()
        if not img_path.exists():
            sys.exit(f"{RED}FAIL{RESET} --image not found: {img_path}")

        print(f"\n[3] Running one inference on {img_path.name}")
        from PIL import Image

        with Image.open(img_path) as im:
            im = im.convert("RGB").resize((224, 224), Image.BICUBIC)
            rgb = np.asarray(im, dtype=np.float32)

        gray = _to_grayscale(rgb)
        vec = _feature_vector(gray)
        vec_norm = (vec - feat_mean) / feat_std

        with torch.no_grad():
            x = torch.from_numpy(vec_norm).unsqueeze(0)
            probs = torch.softmax(model(x), dim=-1)[0]

        p_fake = float(probs[1])
        verdict = "TAMPERED" if p_fake >= threshold else "AUTHENTIC"
        tone = RED if p_fake >= 0.65 else (GREEN if p_fake <= 0.35 else YELLOW)
        print(f"{GREEN}ok{RESET}    P(tampered) = {tone}{p_fake * 100:.2f}%{RESET}   "
              f"verdict {verdict}  (threshold {threshold})")

    print("\n" + "=" * 64)
    print(f"  export DEEPTRUTH_SRM_CHECKPOINT={ckpt}")
    if not (0.4 < threshold < 0.6):
        print(f"  export DEEPTRUTH_SRM_THRESHOLD={threshold:.4f}")
    print()


if __name__ == "__main__":
    main()
