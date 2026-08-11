#!/usr/bin/env python3
"""
Trace SRM's noise-analysis pass on one file, for either an image or a
video's sampled frames.

    python diagnose_srm.py path\\to\\photo.jpg
    python diagnose_srm.py path\\to\\clip.mp4

Same spirit as diagnose_video.py / diagnose_image.py: fast pass/fail, no
need to go through a full server case to see what SRM actually did.

Unlike those two, SRM has a real, meaningful "not broken, just not trained
yet" state — this script tells you which one you're looking at. Before
DEEPTRUTH_SRM_CHECKPOINT is set, seeing real feature numbers here (not an
error) means feature extraction is fully working and the only missing
piece is training.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"
RED, GREEN, YELLOW = "\033[31m", "\033[32m", "\033[33m"


def head(text: str) -> None:
    print(f"\n{BOLD}{text}{RESET}\n" + "─" * 66)


def die(msg: str, fix: str) -> None:
    print(f"\n{RED}STOPPED{RESET} {msg}\n{DIM}   fix: {fix}{RESET}\n")
    sys.exit(1)


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)

    src = Path(sys.argv[1]).expanduser().resolve()
    if not src.exists():
        die(f"no such file: {src}", "check the path")

    print(f"\n{BOLD}Deep Truth — SRM noise-analysis diagnosis{RESET}")
    print("=" * 66)
    print(f"file  {src.name}")

    # ── Dependencies ─────────────────────────────────────────────────────────
    head("1. Dependencies")

    try:
        import numpy  # noqa: F401
        import scipy  # noqa: F401

        print(f"{GREEN}ok{RESET}    numpy + scipy — feature extraction needs "
              f"nothing else")
    except ImportError as exc:
        die(f"missing: {exc}", "pip install numpy scipy")

    checkpoint_configured = False
    try:
        import torch  # noqa: F401

        from deeptruth_pipeline.config import SRM_CHECKPOINT, SRM_THRESHOLD

        if SRM_CHECKPOINT:
            checkpoint_configured = True
            print(f"{GREEN}ok{RESET}    torch available, "
                  f"DEEPTRUTH_SRM_CHECKPOINT={SRM_CHECKPOINT}")
            print(f"{DIM}      threshold={SRM_THRESHOLD}{RESET}")
        else:
            print(f"{YELLOW}note{RESET}  no DEEPTRUTH_SRM_CHECKPOINT set — "
                  f"feature extraction will run,\n      but there is no "
                  f"classifier yet to score the result. That's expected\n"
                  f"      if you haven't trained one — see "
                  f"train_pipeline/train_srm_casia.py.")
    except ImportError:
        print(f"{YELLOW}note{RESET}  torch not installed — feature "
              f"extraction still works (it needs\n      only numpy/scipy), "
              f"but no checkpoint could load even if configured.")

    # ── Get grayscale frame(s) ───────────────────────────────────────────────
    head("2. Preparing input")

    from deeptruth_pipeline.config import VIDEO_EXTS, IMAGE_EXTS
    from deeptruth_pipeline.preprocessors.srm_filters import to_grayscale

    is_video = src.suffix.lower() in VIDEO_EXTS
    is_image = src.suffix.lower() in IMAGE_EXTS
    if not (is_video or is_image):
        die(f"unrecognised extension {src.suffix!r}",
            "pass a video or image file")

    frames = []
    if is_image:
        from PIL import Image

        with Image.open(src) as im:
            im = im.convert("RGB").resize((224, 224), Image.BICUBIC)
            frames.append(__import__("numpy").asarray(im, dtype="uint8"))
        print(f"      image — 1 frame prepared")
    else:
        import numpy as np

        from deeptruth_pipeline import train_bridge

        pipeline = train_bridge.load()
        raw = pipeline._decode_video(src, pipeline.NUM_FRAMES)
        if raw is None:
            die("video decode failed", "PyAV may not support this codec")
        full = pipeline._resize_only(raw)
        n = min(4, full.shape[0])
        idx = np.linspace(0, full.shape[0] - 1, n).round().astype(int)
        frames = list(full[idx])
        print(f"      video — {len(frames)} frames sampled for SRM "
              f"(of {full.shape[0]} decoded)")

    # ── Real feature extraction ─────────────────────────────────────────────
    head("3. Feature extraction (real, runs regardless of checkpoint)")

    from deeptruth_pipeline.preprocessors.srm_filters import patch_feature_vector, patch_feature_names

    import numpy as np

    vecs = np.stack([patch_feature_vector(to_grayscale(f)) for f in frames])
    names = patch_feature_names()

    print(f"      {vecs.shape[0]} frame(s) x {vecs.shape[1]}-dim features")
    print()
    for i, name in enumerate(names):
        vals = vecs[:, i]
        print(f"      {name:<16} mean={vals.mean():8.4f}  "
              f"range=[{vals.min():.4f}, {vals.max():.4f}]")

    if np.allclose(vecs, 0):
        print(f"\n{YELLOW}warn{RESET}  every feature is exactly zero — "
              f"suspicious for a real photo/frame\n      (a flat/blank "
              f"image would do this; check the source file looks right).")
    else:
        print(f"\n{GREEN}ok{RESET}    non-trivial values — the filter bank "
              f"is genuinely reading structure\n      from this file, not "
              f"producing placeholder output.")

    # ── Classifier, if configured ────────────────────────────────────────────
    head("4. Classifier")

    if not checkpoint_configured:
        print(
            f"{YELLOW}Not yet trained.{RESET} This is expected, not a "
            f"failure — feature extraction above\nis the real, working "
            f"half of this module. Train the classifier with:\n\n"
            f"    train_pipeline/train_srm_casia.py  (Kaggle, CASIA v2, "
            f"~2 min on CPU)\n\n"
            f"then:\n\n"
            f"    export DEEPTRUTH_SRM_CHECKPOINT=/path/to/final_model\n"
        )
        return

    from deeptruth_pipeline.inferencers.srm import _build_classifier
    import torch

    ckpt = Path(__import__("deeptruth_pipeline.config", fromlist=["SRM_CHECKPOINT"]).SRM_CHECKPOINT)
    model_pt = ckpt / "model.pt" if ckpt.is_dir() else ckpt
    if not model_pt.exists():
        die(f"configured but not found: {model_pt}", "check the path")

    meta_path = (ckpt / "metadata.json") if ckpt.is_dir() else ckpt.parent / "metadata.json"
    if not meta_path.exists():
        die(f"no metadata.json beside {model_pt}",
            "feature_mean/feature_std live there and are required to score "
            "correctly — see inferencers/srm.py")
    import json
    meta = json.loads(meta_path.read_text())
    if "feature_mean" not in meta or "feature_std" not in meta:
        die(f"{meta_path} has no feature_mean/feature_std",
            "this checkpoint predates feature normalisation — retrain with "
            "the current train_srm_casia.py")
    feat_mean = np.array(meta["feature_mean"], dtype=np.float32)
    feat_std = np.array(meta["feature_std"], dtype=np.float32)
    if feat_mean.shape[0] != vecs.shape[1]:
        die(f"checkpoint's normalisation stats are {feat_mean.shape[0]}-dim, "
            f"features are {vecs.shape[1]}-dim",
            "checkpoint/feature layout mismatch — likely a stale checkpoint "
            "from before the patch-based redesign")

    try:
        state = torch.load(model_pt, map_location="cpu")
        model = _build_classifier()
        model.load_state_dict(state, strict=True)
        model.eval()
        print(f"{GREEN}ok{RESET}    checkpoint loads (strict=True), "
              f"normalisation stats present and correctly shaped")
    except Exception as exc:  # noqa: BLE001
        die(f"checkpoint failed to load: {exc}",
            "architecture drift between training and inferencers/srm.py — "
            "see that file's HEAD_DIMS")

    from deeptruth_pipeline.config import SRM_THRESHOLD

    vecs_norm = (vecs - feat_mean) / feat_std

    with torch.no_grad():
        logits = model(torch.from_numpy(vecs_norm))
        probs = torch.softmax(logits, dim=-1)[:, 1].numpy()

    p_fake = float(probs.max())
    verdict = "TAMPERED" if p_fake >= SRM_THRESHOLD else "AUTHENTIC"
    tone = RED if p_fake >= 0.65 else (GREEN if p_fake <= 0.35 else YELLOW)

    print(f"\n      per-frame P(tampered): "
          f"{', '.join(f'{p * 100:.1f}%' for p in probs)}")
    print(f"      {BOLD}strongest = {tone}{p_fake * 100:.1f}%{RESET}   "
          f"verdict {BOLD}{tone}{verdict}{RESET}  "
          f"(threshold {SRM_THRESHOLD})\n")


if __name__ == "__main__":
    main()
