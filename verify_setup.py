#!/usr/bin/env python3
"""
Preflight check for the Deep Truth pipeline.

Run this from the project root (the directory that contains `server/`) before
starting the API. It answers the question that a bare "inconclusive" verdict
does not: is the pipeline actually wired up, or is it silently returning
neutral results?

    python verify_setup.py

It imports nothing heavy unless torch is already installed, so it is safe to
run on a machine that has not finished `pip install -r requirements.txt`.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

OK = "  \033[32mOK\033[0m  "
BAD = "  \033[31mFAIL\033[0m"
WARN = "  \033[33mWARN\033[0m"

problems: list[str] = []
warnings: list[str] = []


def ok(msg: str) -> None:
    print(f"{OK} {msg}")


def bad(msg: str, fix: str) -> None:
    print(f"{BAD} {msg}")
    problems.append(f"{msg}\n         fix: {fix}")


def warn(msg: str) -> None:
    print(f"{WARN} {msg}")
    warnings.append(msg)


print("\nDeep Truth — setup check")
print("=" * 62)

# ── 1. Package name ──────────────────────────────────────────────────────────
print("\n[1] Package layout")

if ROOT.name != "deeptruth_pipeline":
    bad(
        f"project root is named '{ROOT.name}', not 'deeptruth_pipeline'",
        f"mv '{ROOT}' '{ROOT.parent / 'deeptruth_pipeline'}'  —  "
        "server/app/_dtp.py does `import deeptruth_pipeline`, which fails on "
        "any other name. Every case then comes back inconclusive at 50%.",
    )
else:
    ok("project root is named 'deeptruth_pipeline'")

if (ROOT / "__init__.py").exists():
    ok("__init__.py present (root is an importable package)")
else:
    bad("no __init__.py at the project root", "restore it from the archive")

sys.path.insert(0, str(ROOT.parent))
try:
    import deeptruth_pipeline  # noqa: F401

    ok("`import deeptruth_pipeline` resolves")
except Exception as exc:  # noqa: BLE001
    bad(f"`import deeptruth_pipeline` failed: {exc}", "fix the package name above")

# ── 2. Checkpoints ───────────────────────────────────────────────────────────
print("\n[2] Checkpoints")

VIDEO_DIR = Path(os.environ.get("DEEPTRUTH_CHECKPOINTS", ROOT / "videos_checkpoints"))
IMAGE_DIR = Path(os.environ.get("DEEPTRUTH_IMAGE_CHECKPOINTS", ROOT / "images_checkpoints"))

EXPECTED_VIDEO = {
    "celebdf_v2", "deeperforensics", "dfdc", "ffpp", "genvideo", "wilddeepfake",
}
EXPECTED_IMAGE = {
    "genimage", "mscocoai", "wildrf", "commforensics", "ntire",
    "dff", "ffpp", "ffpp_facecrop",
}


def scan(directory: Path, prefix: str) -> dict[str, Path]:
    """Find <prefix><slug>_lora_best directories holding a PEFT adapter."""
    out: dict[str, Path] = {}
    if not directory.exists():
        return out
    for entry in sorted(directory.iterdir()):
        if not entry.is_dir() or not entry.name.endswith("_lora_best"):
            continue
        if prefix and not entry.name.startswith(prefix):
            continue
        if not prefix and entry.name.startswith("image_"):
            continue  # image adapters must not be picked up by the video branch
        slug = entry.name[len(prefix): -len("_lora_best")]
        if (entry / "adapter_config.json").exists():
            out[slug] = entry
    return out


video = scan(VIDEO_DIR, "")
image = scan(IMAGE_DIR, "image_")

print(f"      video dir: {VIDEO_DIR}")
if not VIDEO_DIR.exists():
    bad(
        "videos_checkpoints/ does not exist",
        "unzip videos_checkpoints.zip into the project root, or point "
        "DEEPTRUTH_CHECKPOINTS at wherever it lives",
    )
elif missing := EXPECTED_VIDEO - set(video):
    bad(
        f"found {len(video)}/6 video adapters, missing: {', '.join(sorted(missing))}",
        "each must be a directory named <slug>_lora_best containing "
        "adapter_config.json",
    )
else:
    ok(f"all 6 video adapters present: {', '.join(sorted(video))}")

print(f"      image dir: {IMAGE_DIR}")
if not IMAGE_DIR.exists():
    bad(
        "images_checkpoints/ does not exist",
        "unzip images_checkpoints.zip into the project root, or point "
        "DEEPTRUTH_IMAGE_CHECKPOINTS at it",
    )
elif missing := EXPECTED_IMAGE - set(image):
    bad(
        f"found {len(image)}/8 image adapters, missing: {', '.join(sorted(missing))}",
        "each must be a directory named image_<slug>_lora_best containing "
        "adapter_config.json",
    )
else:
    ok(f"all 8 image adapters present: {', '.join(sorted(image))}")

# The one checkpoint whose classifier head lives outside the adapter.
if "ffpp" in image:
    cfg_path = image["ffpp"] / "adapter_config.json"
    try:
        cfg = json.loads(cfg_path.read_text())
    except Exception:  # noqa: BLE001
        cfg = {}
    has_head_in_adapter = bool(cfg.get("modules_to_save"))
    has_head_file = (image["ffpp"] / "classifier_head.pt").exists()
    if has_head_in_adapter or has_head_file:
        src = "inside the adapter" if has_head_in_adapter else "classifier_head.pt"
        ok(f"image_ffpp classifier head found ({src})")
    else:
        bad(
            "image_ffpp_lora_best has neither modules_to_save nor "
            "classifier_head.pt",
            "without a head this adapter scores noise; the inferencer will "
            "drop it from the ensemble",
        )

if VIDEO_DIR.resolve() == IMAGE_DIR.resolve():
    warn(
        "video and image checkpoints share one directory. This is supported, "
        "but keep the image_ prefix intact — it is the only thing separating "
        "the two ensembles."
    )

# ── 3. Supporting paths ──────────────────────────────────────────────────────
print("\n[3] Supporting paths")

train = Path(
    os.environ.get(
        "DEEPTRUTH_TRAIN_PIPELINE", ROOT / "train_pipeline" / "deeptruth_train.py"
    )
)
if train.exists():
    ok(f"train pipeline found ({train.name})")
else:
    bad(
        f"train pipeline missing at {train}",
        "inference reuses the training module so preprocessing and model "
        "construction match; set DEEPTRUTH_TRAIN_PIPELINE if it moved",
    )

for name, path in [
    ("cache", Path(os.environ.get("DEEPTRUTH_CACHE", ROOT / "server" / "_dtp_cache"))),
    ("uploads", ROOT / "server" / "uploads"),
]:
    path.mkdir(parents=True, exist_ok=True)
    if os.access(path, os.W_OK):
        ok(f"{name} directory writable ({path.name}/)")
    else:
        bad(f"{name} directory not writable: {path}", "chmod u+w it")

# ── 4. Python dependencies ───────────────────────────────────────────────────
print("\n[4] Dependencies")

for mod, note in [
    ("torch", "core inference"),
    ("transformers", "ViViT / ViT backbones"),
    ("peft", "LoRA adapters — must be >= 0.19"),
    ("cv2", "frame extraction"),
    ("fastapi", "API"),
]:
    try:
        m = __import__(mod)
        ver = getattr(m, "__version__", "?")
        if mod == "peft":
            major_minor = tuple(int(x) for x in ver.split(".")[:2])
            if major_minor < (0, 19):
                bad(
                    f"peft {ver} is too old",
                    "pip install 'peft>=0.19' — these adapters were exported "
                    "by 0.19.x and older releases reject their config outright",
                )
                continue
            if major_minor >= (0, 20):
                # Reproduced directly: peft 0.20.0 raises
                # "VivitForVideoClassification.forward() got an unexpected
                # keyword argument 'input_ids'" on every checkpoint, every
                # time. TaskType.FEATURE_EXTRACTION has no vision-specific
                # wrapper class in peft, so get_peft_model() falls back to
                # the generic base PeftModel, whose forward() is hardcoded
                # for text models and unconditionally injects input_ids
                # into the call — ViViT/ViT reject that outright. The
                # inferencers now bypass this via get_base_model(), but
                # that fix was only confirmed against 0.19.0; flag 0.20+
                # until it's specifically re-tested.
                warn(
                    f"peft {ver} — 0.20.0 reproducibly hit the "
                    f"TaskType.FEATURE_EXTRACTION input_ids bug on this "
                    f"project's checkpoints (see inferencers/video.py, "
                    f"inferencers/image.py). The get_base_model() workaround "
                    f"there should handle it, but this combination is "
                    f"untested — 0.19.x is the confirmed-working version if "
                    f"checkpoints still fail with an input_ids error."
                )
        ok(f"{mod} {ver} ({note})")
    except ImportError:
        bad(f"{mod} not installed ({note})", "pip install -r server/requirements.txt")

# torch<2.3 lacks torch.float8_e4m3fnuz, which transformers>=4.49 references
# unconditionally somewhere in its internals — reproduced directly with
# torch 2.1.2 + transformers 4.49.0: "module 'torch' has no attribute
# 'float8_e4m3fnuz'" on every single checkpoint load, regardless of which
# one. server/requirements.txt's old torch>=2.1 floor was too low for
# transformers>=4.45's actual requirements; checked here explicitly rather
# than only in requirements.txt, since an already-installed environment
# won't re-read that file.
try:
    import torch
    import transformers

    torch_ver = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
    transformers_ver = tuple(int(x) for x in transformers.__version__.split(".")[:2])

    if torch_ver < (2, 3) and transformers_ver >= (4, 49):
        bad(
            f"torch {torch.__version__} is too old for transformers "
            f"{transformers.__version__}",
            "pip install 'torch>=2.3' 'torchvision>=0.18' — torch<2.3 lacks "
            "torch.float8_e4m3fnuz, which this transformers version reads "
            "unconditionally on model load",
        )
except ImportError:
    pass  # already reported as missing above

try:
    import facenet_pytorch  # noqa: F401

    ok("facenet-pytorch (MTCNN face detection)")
except ImportError:
    warn(
        "facenet-pytorch not installed. The pipeline still runs, but the face "
        "branch is skipped: video falls back to genvideo only, and the image "
        "ensemble drops ffpp_facecrop and re-weights onto the generalists. "
        "Expect weaker results on face deepfakes."
    )

# ── 5. Audio checkpoint (optional — only checked if configured) ─────────────
print("\n[5] Audio checkpoint (WavLM)")

audio_checkpoint = os.environ.get("DEEPTRUTH_AUDIO_CHECKPOINT", "").strip()
if not audio_checkpoint:
    warn(
        "DEEPTRUTH_AUDIO_CHECKPOINT is not set in this shell session. Audio "
        "will run on the stub and report 'no signal' for every case. If you "
        "already exported it elsewhere, remember env vars set in one "
        "terminal don't carry over to another — set it in the same window "
        "you launch the server from."
    )
else:
    audio_dir = Path(audio_checkpoint)
    print(f"      DEEPTRUTH_AUDIO_CHECKPOINT = {audio_dir}")

    model_pt = audio_dir / "model.pt" if audio_dir.is_dir() else audio_dir
    if not model_pt.exists():
        bad(
            f"no model.pt found at {model_pt}",
            "check the path — it should point at the FOLDER containing "
            "model.pt and metadata.json, not a file inside it (unless you "
            "set it directly to a .pt file)",
        )
    else:
        size_mb = model_pt.stat().st_size / 1e6
        if size_mb < 500:
            warn(
                f"model.pt is only {size_mb:.0f} MB — a full WavLM-Large "
                f"state dict is normally ~1.2 GB. This may be a partial "
                f"download."
            )
        else:
            ok(f"model.pt found ({size_mb:.0f} MB)")

        meta_path = (audio_dir / "metadata.json") if audio_dir.is_dir() else \
            audio_dir.parent / "metadata.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                thr = meta.get("recommended_threshold")
                dev = meta.get("dev_metrics", {})
                ok(f"metadata.json found — recommended_threshold={thr}")
                if dev:
                    print(f"        dev EER {dev.get('eer', float('nan')):.2f}%  "
                          f"bonafide {dev.get('accuracy_bonafide', float('nan')):.1f}%  "
                          f"spoof {dev.get('accuracy_spoof', float('nan')):.1f}%")
                env_thr = os.environ.get("DEEPTRUTH_AUDIO_THRESHOLD", "").strip()
                if env_thr and thr is not None and abs(float(env_thr) - float(thr)) > 1e-6:
                    warn(
                        f"DEEPTRUTH_AUDIO_THRESHOLD is set to {env_thr}, which "
                        f"differs from this checkpoint's measured threshold "
                        f"({thr}). The explicit env var wins — confirm that's "
                        f"intentional."
                    )
            except Exception as exc:  # noqa: BLE001
                warn(f"metadata.json exists but could not be parsed: {exc}")
        else:
            warn(
                "no metadata.json beside model.pt — the threshold will fall "
                "back to 0.5 unless DEEPTRUTH_AUDIO_THRESHOLD is set "
                "explicitly. For a checkpoint trained with class-weighted "
                "loss, 0.5 is very likely wrong."
            )

        experimental = os.environ.get("DEEPTRUTH_AUDIO_EXPERIMENTAL", "true").lower()
        if experimental not in ("false", "0", "no"):
            print("        DEEPTRUTH_AUDIO_EXPERIMENTAL is on (default) — "
                  "every audio verdict will be marked provisional in the "
                  "console and extension until this is set to false.")

    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        ok("torch + transformers available for audio inference")
    except ImportError as exc:
        bad(f"audio checkpoint is configured but a dependency is missing: {exc}",
            "pip install -r server/requirements.txt")

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
if problems:
    print(f"\n{len(problems)} problem(s) must be fixed:\n")
    for i, p in enumerate(problems, 1):
        print(f"  {i}. {p}\n")
else:
    print("\nNo blocking problems. The pipeline should produce real verdicts.")

if warnings:
    print(f"{len(warnings)} warning(s):\n")
    for w in warnings:
        print(f"  - {w}\n")

if not problems:
    print("A case can still come back 'inconclusive' for a legitimate reason:")
    print("  - the fused score landed between 35% and 65%, or")
    print("  - the adapters disagreed sharply with each other.")
    print("The console now says which of these applies on the case page.\n")

sys.exit(1 if problems else 0)