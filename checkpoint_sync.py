from __future__ import annotations
import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .config import IMAGE_CHECKPOINT_DIR, IMAGE_CHECKPOINT_INFO

log = logging.getLogger(__name__)


def _have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def _expected_dir_names(slugs: list[str] | None) -> list[str]:
    keep = slugs or list(IMAGE_CHECKPOINT_INFO.keys())
    return [f"image_{s}_lora_best" for s in keep]


def _verify_one(local: Path) -> tuple[bool, list[str]]:
    missing = []
    if not (local / "adapter_config.json").exists():
        missing.append("adapter_config.json")
    if not (local / "classifier_head.pt").exists():
        missing.append("classifier_head.pt")
    has_weights = (
        (local / "adapter_model.safetensors").exists()
        or (local / "adapter_model.bin").exists()
    )
    if not has_weights:
        missing.append("adapter_model.{safetensors,bin}")
    return len(missing) == 0, missing


def _sync_rclone(remote: str, dest: Path, names: list[str],
                 dry_run: bool) -> int:
    rc = 0
    for name in names:
        src = f"{remote.rstrip('/')}/{name}"
        dst = dest / name
        cmd = ["rclone", "copy", "--progress",
               "--transfers", "4", "--checkers", "8",
               src, str(dst)]
        if dry_run:
            cmd.insert(1, "--dry-run")
        log.info(f"  rclone: {' '.join(cmd)}")
        r = subprocess.run(cmd)
        if r.returncode != 0:
            log.warning(f"  rclone copy of {name} failed (rc={r.returncode})")
            rc = r.returncode
    return rc


def _sync_gdown(folder_id: str, dest: Path, dry_run: bool) -> int:
    if dry_run:
        log.info(f"  [dry-run] gdown --folder "
                 f"https://drive.google.com/drive/folders/{folder_id} "
                 f"-O {dest}")
        return 0
    try:
        import gdown
    except ImportError:
        log.error("  gdown not installed; pip install gdown")
        return 2
    cmd = ["gdown", "--folder", "--continue",
           f"https://drive.google.com/drive/folders/{folder_id}",
           "-O", str(dest)]
    log.info(f"  gdown: {' '.join(cmd)}")
    r = subprocess.run(cmd)
    return r.returncode


def sync_image_checkpoints(
    *,
    dest: Path | None = None,
    slugs: list[str] | None = None,
    remote: str | None = None,
    folder_id: str | None = None,
    dry_run: bool = False,
) -> dict:
    dest = (dest or IMAGE_CHECKPOINT_DIR).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    names = _expected_dir_names(slugs)

    remote = remote or os.environ.get("DEEPTRUTH_RCLONE_REMOTE")
    folder_id = folder_id or os.environ.get("DEEPTRUTH_DRIVE_FOLDER_ID")

    log.info(f"sync_image_checkpoints → {dest}")
    log.info(f"  targets: {names}")

    rc = 0
    if remote and _have("rclone"):
        log.info(f"  using rclone with remote: {remote}")
        rc = _sync_rclone(remote, dest, names, dry_run)
    elif folder_id:
        log.info(f"  using gdown with folder_id: {folder_id}")
        rc = _sync_gdown(folder_id, dest, dry_run)
    else:
        log.error("  no sync method configured. Set DEEPTRUTH_RCLONE_REMOTE "
                  "(preferred) or DEEPTRUTH_DRIVE_FOLDER_ID.")
        rc = 2

    report: dict[str, dict] = {}
    for name in names:
        local = dest / name
        if not local.exists():
            report[name] = {"present": False, "ok": False, "missing": ["dir"]}
            continue
        ok, missing = _verify_one(local)
        report[name] = {"present": True, "ok": ok, "missing": missing}

    log.info("")
    log.info("Verification:")
    for name, r in report.items():
        status = "OK" if r["ok"] else ("INCOMPLETE" if r["present"] else "MISSING")
        log.info(f"  {name:<40s} {status}  "
                 f"{('missing: ' + ', '.join(r['missing'])) if r['missing'] else ''}")

    return {"dest": str(dest), "rc": rc, "report": report}


def _cli():
    ap = argparse.ArgumentParser(description="Sync image checkpoints from "
                                              "Google Drive.")
    ap.add_argument("--dest", type=Path, default=None,
                    help="Destination dir (default: IMAGE_CHECKPOINT_DIR)")
    ap.add_argument("--slugs", nargs="+", default=None,
                    choices=list(IMAGE_CHECKPOINT_INFO.keys()),
                    help="Subset of checkpoint slugs to sync (default: all)")
    ap.add_argument("--remote", default=None,
                    help="Rclone remote path, e.g. gdrive:DeepTruth/ckpts")
    ap.add_argument("--folder-id", default=None,
                    help="Google Drive folder ID for gdown fallback")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    out = sync_image_checkpoints(
        dest=args.dest,
        slugs=args.slugs,
        remote=args.remote,
        folder_id=args.folder_id,
        dry_run=args.dry_run,
    )
    bad = [n for n, r in out["report"].items() if not r["ok"]]
    sys.exit(0 if not bad else 1)


if __name__ == "__main__":
    _cli()
