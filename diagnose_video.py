#!/usr/bin/env python3
"""
Trace exactly why one video got the verdict it did.

    python diagnose_video.py /path/to/clip.mp4

`cli.py` gives you the answer; this gives you the reasoning. It walks the same
path the server does — decode, face detection, per-checkpoint scoring, ensemble
— and reports what happened at each stage, including the two places a video can
silently end up "inconclusive":

  1. Something threw before any model scored. The engine catches it and returns
     a neutral result, which the API reports as inconclusive at exactly 50.0%.
     A missing PyAV install and an undecodable file both land here.

  2. Fewer than MIN_FACE_FRAMES faces were found. The five face-specialist
     checkpoints are then discarded entirely and the verdict comes from the
     single genvideo checkpoint. This is a cliff, not a slope: at 6 detected
     faces all five specialists count, at 5 none of them do.

Long videos are much more likely to hit (2), because exactly 16 frames are
sampled across the whole duration no matter how long it runs. A subject who is
on camera for a third of a five-minute clip can easily land under the cut.
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"
RED, GREEN, YELLOW, CYAN = "\033[31m", "\033[32m", "\033[33m", "\033[36m"


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

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    size_mb = src.stat().st_size / 1e6
    print(f"\n{BOLD}Deep Truth — video diagnosis{RESET}")
    print("=" * 66)
    print(f"file  {src.name}")
    print(f"size  {size_mb:.1f} MB")

    # ── Dependencies ─────────────────────────────────────────────────────────
    head("1. Dependencies on the video path")

    try:
        import peft

        print(f"{GREEN}ok{RESET}    peft {peft.__version__}")
        # The active bug as of this diagnostic: TaskType.FEATURE_EXTRACTION
        # has no vision-specific wrapper in PEFT, so get_peft_model() falls
        # back to the generic base PeftModel, whose forward() is hardcoded
        # for text models and unconditionally injects input_ids into the
        # call to the wrapped model — ViViT/ViT reject that outright with
        # "got an unexpected keyword argument 'input_ids'". This has been
        # observed specifically with peft 0.20.0; server/requirements.txt
        # only pins >=0.19 with no upper bound, so it was never validated
        # against 0.20.x. The fix (already applied in inferencers/video.py
        # and inferencers/image.py) calls model.get_base_model() instead of
        # model directly, bypassing the broken wrapper while keeping the
        # LoRA weights active. get_base_model() is one of PEFT's oldest,
        # most stable public methods, so this checks that it still exists
        # under whatever version is installed here.
        from peft import PeftModel

        if hasattr(PeftModel, "get_base_model"):
            print(f"{GREEN}ok{RESET}    PeftModel.get_base_model() available "
                  f"— the video/image PEFT-forward fix can apply")
        else:
            print(
                f"{YELLOW}warn{RESET}  PeftModel has no get_base_model() on "
                f"this peft version.\n      The applied fix relies on it; "
                f"if checkpoints still fail with an\n      input_ids error "
                f"after this, that is why — report the peft version above."
            )
    except ImportError:
        die("peft is not installed. Every LoRA checkpoint needs it to load.",
            "pip install -r server/requirements.txt")

    try:
        import av  # noqa: F401

        print(f"{GREEN}ok{RESET}    PyAV present — video decoding available")
    except ImportError:
        die(
            "PyAV is not installed. It is the only video decode path, so every "
            "video case returns a neutral result and reads as inconclusive at "
            "exactly 50.0%.",
            "pip install av",
        )

    try:
        import facenet_pytorch  # noqa: F401

        face_available = True
        print(f"{GREEN}ok{RESET}    facenet-pytorch present — face branch available")
    except ImportError:
        face_available = False
        print(
            f"{YELLOW}warn{RESET}  facenet-pytorch missing. No faces will ever be "
            f"detected,\n      so the five face checkpoints are always discarded "
            f"and every\n      verdict comes from genvideo alone."
        )

    try:
        import torch

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"{GREEN}ok{RESET}    torch {torch.__version__} — device: {dev}")
        if dev == "cpu":
            print(
                f"{DIM}      On CPU a 16-frame ViViT pass takes a few seconds per "
                f"checkpoint,\n      six checkpoints per clip. Expect this to be "
                f"slow, not stuck.{RESET}"
            )
    except ImportError:
        die("torch is not installed", "pip install -r server/requirements.txt")

    # ── Decode ───────────────────────────────────────────────────────────────
    head("2. Decode and frame sampling")

    from deeptruth_pipeline import train_bridge

    pipeline = train_bridge.load()

    try:
        import av

        container = av.open(str(src))
        stream = container.streams.video[0]
        total = stream.frames or 0
        duration = float(container.duration / 1e6) if container.duration else None
        fps = float(stream.average_rate) if stream.average_rate else None
        container.close()
    except Exception as exc:  # noqa: BLE001
        die(f"PyAV could not open this file: {exc}",
            "re-encode it, e.g. ffmpeg -i input -c:v libx264 output.mp4")

    print(f"      duration     {f'{duration:.1f}s' if duration else 'unknown'}")
    print(f"      fps          {f'{fps:.2f}' if fps else 'unknown'}")
    print(f"      total frames {total or 'unknown (needs a counting pass)'}")
    print(f"      sampled      {pipeline.NUM_FRAMES} frames, evenly spaced")

    if duration and duration > 60:
        gap = duration / pipeline.NUM_FRAMES
        print(
            f"\n{YELLOW}note{RESET}  This clip is {duration / 60:.1f} minutes long, so the "
            f"{pipeline.NUM_FRAMES} sampled frames\n"
            f"      sit about {gap:.1f}s apart. Two consequences worth knowing:\n"
            f"        · a manipulation confined to a short segment may appear in\n"
            f"          only one or two of the sampled frames, or none at all\n"
            f"        · if the subject is not on camera throughout, face detection\n"
            f"          can fall under the cut-off and discard the face models"
        )

    t0 = time.time()
    raw = pipeline._decode_video(src, pipeline.NUM_FRAMES)
    if raw is None:
        die("decode returned nothing", "the file may be truncated or use an "
            "unsupported codec")
    print(f"\n{GREEN}ok{RESET}    decoded {len(raw)} frames in {time.time() - t0:.1f}s "
          f"({raw.shape[2]}x{raw.shape[1]})")

    # ── Face detection ───────────────────────────────────────────────────────
    head("3. Face detection")

    from deeptruth_pipeline.config import MIN_FACE_FRAMES

    n_face = 0
    if face_available:
        try:
            from PIL import Image

            detector = pipeline._get_face_detector()
            boxes, _ = detector.detect([Image.fromarray(f) for f in raw])
            n_face = sum(1 for b in boxes if b is not None and len(b) > 0)
        except Exception as exc:  # noqa: BLE001
            print(f"{YELLOW}warn{RESET}  face detection failed: {exc}")

    trusted = n_face >= MIN_FACE_FRAMES
    colour = GREEN if trusted else RED
    print(f"      faces found  {colour}{n_face}/{len(raw)}{RESET}  "
          f"(cut-off is {MIN_FACE_FRAMES})")

    if trusted:
        print(f"{GREEN}ok{RESET}    face branch TRUSTED — all five face checkpoints count")
    else:
        print(
            f"{RED}!!{RESET}    face branch NOT trusted. The five face-deepfake "
            f"checkpoints\n      are discarded and the verdict rests entirely on "
            f"genvideo,\n      which is trained to spot fully-generated video "
            f"rather than\n      a real clip with a swapped face."
        )
        if n_face > 0:
            print(f"{DIM}      {n_face} face(s) were found — just under the cut-off "
                  f"of {MIN_FACE_FRAMES}.{RESET}")

    # ── Scoring ──────────────────────────────────────────────────────────────
    head("4. Per-checkpoint scores")

    from deeptruth_pipeline.config import CHECKPOINT_DIR
    from deeptruth_pipeline.inferencers.video import VideoInferencer
    from deeptruth_pipeline.ensemble import ensemble_decide

    inf = VideoInferencer(checkpoint_dir=CHECKPOINT_DIR)
    try:
        inf._setup()
    except Exception as exc:  # noqa: BLE001
        die(f"could not initialise the inferencer: {exc}",
            "run verify_setup.py — most likely the checkpoints aren't found")

    print(f"      {len(inf._checkpoints)} checkpoints discovered\n")

    full_frames = pipeline._resize_only(raw)
    face_frames = None
    if n_face > 0:
        try:
            face_frames = pipeline._crop_faces(raw)
        except Exception as exc:  # noqa: BLE001
            print(f"{YELLOW}warn{RESET}  face cropping failed: {exc}")

    per_ckpt: dict[str, float] = {}
    for ckpt in inf._checkpoints:
        frames = face_frames if ckpt["preprocess"] == "face" else full_frames
        if frames is None:
            print(f"      {ckpt['name']:<20} {DIM}skipped — no "
                  f"{ckpt['preprocess']} frames{RESET}")
            continue
        try:
            t = time.time()
            p = inf._predict_one(inf._get_model(ckpt, pipeline), frames)
            per_ckpt[ckpt["name"]] = p
            bar = "█" * int(p * 28)
            tone = RED if p >= 0.65 else (GREEN if p <= 0.35 else YELLOW)
            print(f"      {ckpt['name']:<20} {tone}{p * 100:5.1f}%{RESET} "
                  f"{tone}{bar}{RESET} {DIM}({ckpt['preprocess']}, "
                  f"{time.time() - t:.1f}s){RESET}")
        except Exception as exc:  # noqa: BLE001
            print(f"      {ckpt['name']:<20} {RED}FAILED{RESET} {exc}")

    if not per_ckpt:
        die("no checkpoint produced a score",
            "this is what makes a case inconclusive at exactly 50.0%")

    # ── Decision ─────────────────────────────────────────────────────────────
    head("5. Ensemble decision")

    d = ensemble_decide(per_ckpt, n_face)
    ens = d["ensemble"]
    risk = ens * 100 if ens == ens else float("nan")

    fa = d["face_avg"]
    gv = d["genvideo_score"]
    fa_txt = f"{fa * 100:.1f}%" if fa is not None else "—"
    gv_txt = f"{gv * 100:.1f}%" if gv is not None else "—"

    print(f"      face_avg       {fa_txt}")
    print(f"      genvideo       {gv_txt}")
    print(f"      face_trusted   {d['face_trusted']}")
    print(f"      confidence     {d['confidence']:.2f}   "
          f"{DIM}(1 − 2σ across checkpoints; 0 means the API reports "
          f"inconclusive){RESET}")
    print(f"\n      {DIM}{d['rationale']}{RESET}")

    if ens != ens:
        verdict, colour = "INCONCLUSIVE (no usable output)", RED
    elif d["confidence"] <= 0:
        verdict, colour = "INCONCLUSIVE (zero confidence)", RED
    elif risk >= 65:
        verdict, colour = "MANIPULATED", RED
    elif risk <= 35:
        verdict, colour = "AUTHENTIC", GREEN
    else:
        verdict, colour = "INCONCLUSIVE (between thresholds)", YELLOW

    print(f"\n      risk score     {BOLD}{colour}{risk:.1f}%{RESET}")
    print(f"      verdict        {BOLD}{colour}{verdict}{RESET}")

    # ── What to do ───────────────────────────────────────────────────────────
    head("6. Reading this result")

    if not trusted and any(
        per_ckpt.get(k, 0) >= 0.6
        for k in ("celebdf_v2", "ffpp", "dfdc", "wilddeepfake", "deeperforensics")
    ):
        print(
            f"{RED}This is the case to watch.{RESET} One or more face checkpoints "
            f"flagged this\nclip, but they were discarded because only {n_face} "
            f"frames had a detectable\nface. The reported verdict comes from "
            f"genvideo alone and does not\nreflect what the face models saw.\n\n"
            f"Try trimming to a segment where the subject is on camera "
            f"throughout\nand re-running — the sampled frames will then land on "
            f"the face."
        )
    elif ens == ens and 35 < risk < 65:
        lo, hi = min(per_ckpt.values()) * 100, max(per_ckpt.values()) * 100
        print(
            f"The checkpoints ran and disagreed — scores span {lo:.0f}% to "
            f"{hi:.0f}%.\nThe fused score landed between the thresholds, which "
            f"is a real finding\nabout this file rather than a fault. Each "
            f"checkpoint is tuned to a\ndifferent manipulation family, so a "
            f"split usually means the clip\nresembles one generator and not the "
            f"others."
        )
    elif duration and duration > 60:
        print(
            f"The verdict is based on {pipeline.NUM_FRAMES} frames spread across "
            f"{duration / 60:.1f} minutes.\nIf you suspect a specific passage, "
            f"trim to it and re-run — concentrating\nthe sampling on the "
            f"segment in question is far more sensitive than\nspreading it over "
            f"the whole clip."
        )
    else:
        print("Nothing anomalous in how this verdict was reached.")

    print()


if __name__ == "__main__":
    main()
