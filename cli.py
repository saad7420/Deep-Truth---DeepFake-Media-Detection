"""Command-line driver.

Usage:
    python -m deeptruth_pipeline.cli /path/to/video.mp4
    python -m deeptruth_pipeline.cli /videos/ --threshold 0.5
    python -m deeptruth_pipeline.cli /videos/ --json results.json
    python -m deeptruth_pipeline.cli /videos/ --device cpu
    python -m deeptruth_pipeline.cli /videos/ --queue-workers 2
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

from .config import VIDEO_EXTS, AUDIO_EXTS, IMAGE_EXTS, LOG_DIR
from .pipeline import Pipeline, make_queue
from .registry import Registry
from .task_queue import Task, TaskStatus
from .inferencers.video import VideoInferencer
from .inferencers.image import ImageInferencer

ALL_EXTS = VIDEO_EXTS | AUDIO_EXTS | IMAGE_EXTS


def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_DIR / "pipeline.log", mode="a"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger("cli")


def gather_inputs(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(p for p in path.iterdir()
                      if p.is_file() and p.suffix.lower() in ALL_EXTS)
    raise FileNotFoundError(path)


def build_pipeline(device: str | None, threshold: float) -> Pipeline:
    registry = Registry(
        video_inferencer=VideoInferencer(device=device, threshold=threshold),
        image_inferencer=ImageInferencer(device=device, threshold=threshold),
    )
    return Pipeline(registry=registry)


def run_inline(log, pipeline: Pipeline, files: list[Path],
               threshold: float, force: bool) -> list[dict]:
    results = []
    for f in files:
        log.info("")
        log.info("=" * 70)
        log.info(f"  {f.name}")
        log.info("=" * 70)
        try:
            r = pipeline.analyze(f, threshold=threshold, force=force)
            results.append(asdict(r))
            print_result(log, r)
        except Exception as e:
            log.exception(f"failed on {f.name}: {e}")
            results.append({"file": f.name, "error": str(e)})
    return results


def run_via_queue(log, pipeline: Pipeline, files: list[Path],
                  threshold: float, force: bool, workers: int) -> list[dict]:
    queue = make_queue(pipeline, async_workers=workers)
    task_ids = []
    for f in files:
        t = Task(media_path=f, options={"threshold": threshold, "force": force})
        task_ids.append(queue.submit(t))

    if workers > 0:
        import time as _time
        log.info(f"submitted {len(task_ids)} tasks to {workers} async worker(s); waiting...")
        while True:
            pending = [tid for tid in task_ids
                       if (queue.get(tid) and
                           queue.get(tid).status not in
                           (TaskStatus.COMPLETED, TaskStatus.FAILED))]
            if not pending:
                break
            _time.sleep(1.0)

    results = []
    for tid in task_ids:
        t = queue.get(tid)
        if t is None:
            continue
        if t.status == TaskStatus.COMPLETED:
            results.append(t.result)
        else:
            results.append({"task_id": tid, "status": t.status.value,
                            "error": t.error})
    return results


def print_result(log, result):
    log.info(f"media_key: {result.media_key}")
    log.info(f"media: kind={result.media_info.get('kind')} "
             f"duration={result.media_info.get('duration_seconds')}")
    for modality, ir in result.inferences.items():
        log.info(f"[{modality}] verdict={ir['verdict']} "
                 f"trust={ir['trust_score']:.3f} "
                 f"confidence={ir['confidence']:.3f}")
        log.info(f"        {ir['rationale']}")
        if ir.get("per_model"):
            cells = "  ".join(f"{k}={v:.3f}" for k, v in ir["per_model"].items())
            log.info(f"        per-model: {cells}")
    for w in result.warnings:
        log.info(f"  WARN: {w}")


def print_summary(log, results):
    log.info("")
    log.info("=" * 70)
    log.info("  SUMMARY")
    log.info("=" * 70)
    header = (f"{'FILE':<42s} {'KIND':<6s} {'VERDICT':<8s} "
              f"{'TRUST':>7s} {'CONF':>6s}")
    log.info(header)
    log.info("-" * len(header))
    for r in results:
        if "error" in r and "media_info" not in r:
            log.info(f"{(r.get('file') or r.get('task_id') or '?')[:40]:<42s} "
                     f"{'-':<6s} {'ERROR':<8s} {'-':>7s} {'-':>6s}")
            continue
        name = Path(r["media_info"]["path"]).name[:40]
        kind = r["media_info"].get("kind", "-")
        ir = (r["inferences"].get("video")
              or next(iter(r["inferences"].values()), None))
        if ir:
            log.info(f"{name:<42s} {kind:<6s} {ir['verdict']:<8s} "
                     f"{ir['trust_score']:>7.3f} {ir['confidence']:>6.3f}")
        else:
            log.info(f"{name:<42s} {kind:<6s} {'-':<8s} {'-':>7s} {'-':>6s}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="File or directory")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--json", default=None, help="Save full results JSON")
    ap.add_argument("--device", default=None,
                    help="Override device, e.g. 'cuda:0' or 'cpu'")
    ap.add_argument("--force", action="store_true",
                    help="Re-run preprocessing even if cached")
    ap.add_argument("--queue-workers", type=int, default=0,
                    help="If >0, route through TaskQueue with N async threads")
    args = ap.parse_args()
    log = setup_logging()

    files = gather_inputs(Path(args.input))
    if not files:
        log.error("no input files found")
        sys.exit(2)
    log.info(f"found {len(files)} input file(s)")

    pipeline = build_pipeline(args.device, args.threshold)

    if args.queue_workers > 0 or False:
        results = run_via_queue(log, pipeline, files, args.threshold,
                                args.force, args.queue_workers)
    else:
        results = run_inline(log, pipeline, files, args.threshold, args.force)

    print_summary(log, results)

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2, default=str))
        log.info(f"results saved to {args.json}")


if __name__ == "__main__":
    main()
