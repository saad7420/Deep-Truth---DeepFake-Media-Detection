#!/usr/bin/env python3
"""
Trace exactly why an image got its verdict, and check whether the PEFT
forward-signature bug (see inferencers/image.py::_forward and
inferencers/video.py::_predict_one) is present on this machine.

    python diagnose_image.py /path/to/photo.jpg

Mirrors diagnose_video.py's structure. Where the two genuinely differ:

  * Image checkpoints load via PeftModel.from_pretrained(base, ckpt_path),
    reading whatever task_type each checkpoint's own adapter_config.json
    was saved with. Video's _build_model() hardcodes
    TaskType.FEATURE_EXTRACTION in code, used identically at train and
    inference time. This repo has no image training script (deeptruth_train.py
    is video-only), so there is no way to inspect what task_type the image
    checkpoints were actually saved with from code — only from running this.

  * The image ensemble is generalist/face weighted-average with a
    max-alarm rule when a face is trusted (image_ensemble_decide), not
    video's discrete "trusted / not trusted" cliff at MIN_FACE_FRAMES.
"""
from __future__ import annotations

import sys
import time
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

    print(f"\n{BOLD}Deep Truth — image diagnosis{RESET}")
    print("=" * 66)
    print(f"file  {src.name}")

    # ── Dependencies ─────────────────────────────────────────────────────────
    head("1. Dependencies")

    try:
        import peft

        print(f"{GREEN}ok{RESET}    peft {peft.__version__}")
        from peft import PeftModel

        if not hasattr(PeftModel, "get_base_model"):
            print(
                f"{YELLOW}warn{RESET}  no get_base_model() on this peft "
                f"version — the applied forward-signature fix relies on it")
    except ImportError:
        die("peft is not installed", "pip install -r server/requirements.txt")

    try:
        import torch

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"{GREEN}ok{RESET}    torch {torch.__version__} — device: {dev}")
    except ImportError:
        die("torch is not installed", "pip install -r server/requirements.txt")

    try:
        import facenet_pytorch  # noqa: F401

        face_available = True
        print(f"{GREEN}ok{RESET}    facenet-pytorch present")
    except ImportError:
        face_available = False
        print(f"{YELLOW}warn{RESET}  facenet-pytorch missing — face-crop "
              f"checkpoints will be skipped for every image")

    # ── Discovery ────────────────────────────────────────────────────────────
    head("2. Checkpoint discovery")

    from deeptruth_pipeline.config import IMAGE_CHECKPOINT_DIR
    from deeptruth_pipeline.inferencers.image import ImageInferencer

    inf = ImageInferencer(checkpoint_dir=IMAGE_CHECKPOINT_DIR)
    try:
        inf._setup()
    except Exception as exc:  # noqa: BLE001
        die(f"could not initialise the inferencer: {exc}",
            "run verify_setup.py — most likely the checkpoints aren't found")

    print(f"      {len(inf._checkpoints)} checkpoints discovered "
          f"at {IMAGE_CHECKPOINT_DIR}")

    # ── Load + score each checkpoint, catching the exact bug we're hunting ──
    head("3. Per-checkpoint load + score")

    from deeptruth_pipeline.preprocessors.image import ImagePreprocessor
    from deeptruth_pipeline.storage import CacheStore, CacheRecord

    store = CacheStore()
    rec = CacheRecord(media_kind="image", source_path=str(src),
                      media_key="diagnose_image")
    pre = ImagePreprocessor(store=store)
    preprocessed = pre.run(src, "diagnose_image", rec, force=True)

    import numpy as np
    whole = np.load(preprocessed["image_path"])
    face = (np.load(preprocessed["face_image_path"])
            if preprocessed.get("face_image_path") else None)
    face_detected = bool(preprocessed.get("face_detected"))
    print(f"      face_detected={face_detected}  "
          f"face_crop_available={face is not None}")

    per_ckpt: dict[str, float] = {}
    input_ids_bug_hit = False

    for ckpt in inf._checkpoints:
        arr = face if ckpt.get("needs_face_crop") else whole
        if arr is None:
            print(f"      {ckpt['slug']:<20} {DIM}skipped — no face crop "
                  f"for this image{RESET}")
            continue
        try:
            t = time.time()
            model = inf._get_model(ckpt)
            pixel_values = inf._preprocess_to_tensor(arr)
            p = inf._forward(model, pixel_values)
            per_ckpt[ckpt["slug"]] = p
            bar = "█" * int(p * 28)
            tone = RED if p >= 0.65 else (GREEN if p <= 0.35 else YELLOW)
            print(f"      {ckpt['slug']:<20} {tone}{p * 100:5.1f}%{RESET} "
                  f"{tone}{bar}{RESET} {DIM}({time.time() - t:.1f}s){RESET}")
        except TypeError as exc:
            if "input_ids" in str(exc):
                input_ids_bug_hit = True
                print(f"      {ckpt['slug']:<20} {RED}FAILED{RESET} "
                      f"the PEFT input_ids bug — {exc}")
            else:
                print(f"      {ckpt['slug']:<20} {RED}FAILED{RESET} {exc}")
        except Exception as exc:  # noqa: BLE001
            print(f"      {ckpt['slug']:<20} {RED}FAILED{RESET} {exc}")

    if input_ids_bug_hit:
        print(
            f"\n{RED}The PEFT forward-signature bug is still present.{RESET}\n"
            f"get_base_model() either isn't being reached (check that "
            f"inferencers/image.py\nwas actually replaced) or doesn't fully "
            f"resolve it on peft {peft.__version__} specifically.\n"
            f"Fallback to test: pip install \"peft==0.19.0\" and re-run this "
            f"script."
        )

    if not per_ckpt:
        die("no checkpoint produced a score", "see failures above")

    # ── Decision ─────────────────────────────────────────────────────────────
    head("4. Ensemble decision")

    from deeptruth_pipeline.ensemble import image_ensemble_decide

    d = image_ensemble_decide(per_ckpt, face_detected)
    ens = d["ensemble"]
    risk = ens * 100 if ens == ens else float("nan")

    ga = d["generalist_avg"]
    fa = d["face_avg"]
    ga_txt = f"{ga * 100:.1f}%" if ga is not None else "—"
    fa_txt = f"{fa * 100:.1f}%" if fa is not None else "—"

    print(f"      policy         {d['policy']}")
    print(f"      generalist_avg {ga_txt}")
    print(f"      face_avg       {fa_txt}")
    print(f"      face_trusted   {d['face_trusted']}")
    print(f"      confidence     {d['confidence']:.2f}")
    print(f"\n      {DIM}{d['rationale']}{RESET}")

    if ens != ens:
        verdict, colour = "INCONCLUSIVE (no usable output)", RED
    elif risk >= 65:
        verdict, colour = "MANIPULATED", RED
    elif risk <= 35:
        verdict, colour = "AUTHENTIC", GREEN
    else:
        verdict, colour = "INCONCLUSIVE (between thresholds)", YELLOW

    print(f"\n      risk score     {BOLD}{colour}{risk:.1f}%{RESET}")
    print(f"      verdict        {BOLD}{colour}{verdict}{RESET}\n")


if __name__ == "__main__":
    main()
