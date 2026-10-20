from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import sqlite3
import time
import traceback
import uuid
from pathlib import Path
from urllib.parse import urlsplit

import app.utils.env

from celery import Task
from celery.exceptions import SoftTimeLimitExceeded
from celery.signals import worker_ready

from app.queue import cache, state
from app.queue.celery_app import celery_app

log = logging.getLogger(__name__)

DB_PATH = os.getenv("DB_PATH", "forensics.db")

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads"))

MAX_ATTEMPTS = int(os.getenv("ANALYSIS_MAX_ATTEMPTS", "3"))
RETRY_BASE_DELAY = float(os.getenv("ANALYSIS_RETRY_BASE_DELAY", "10"))
RETRY_MAX_DELAY = float(os.getenv("ANALYSIS_RETRY_MAX_DELAY", "300"))


class PermanentFailure(Exception):
    pass


class CheckpointsUnavailable(Exception):
    pass


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=30.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=30000;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


# one transaction: replace the old rows with the new ones
def _store_success(case_db_id: str, status: str, risk: float,
                   likelihood: float, rows: list[dict]) -> None:
    conn = _connect()
    try:
        conn.execute("BEGIN IMMEDIATE;")
        conn.execute(
            """UPDATE cases
                  SET status = ?, risk_score = ?, synthetic_likelihood = ?,
                      updated_at = datetime('now')
                WHERE id = ?""",
            (status, risk, likelihood, case_db_id),
        )
        conn.execute("DELETE FROM analysis_results WHERE case_id = ?", (case_db_id,))
        for r in rows:
            conn.execute(
                """INSERT INTO analysis_results
                       (id, case_id, model_name, confidence, label, details)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (r["id"], case_db_id, r["model_name"], r["confidence"],
                 r["label"], r["details"]),
            )
        conn.execute("COMMIT;")
    except Exception:
        conn.execute("ROLLBACK;")
        raise
    finally:
        conn.close()


def _store_failure(case_db_id: str) -> None:
    conn = _connect()
    try:
        conn.execute(
            "UPDATE cases SET status='failed', updated_at=datetime('now') WHERE id=?",
            (case_db_id,),
        )
    finally:
        conn.close()


def _record_downloaded_file(case_db_id: str, *, file_name: str, file_url: str,
                            size: int) -> None:
    conn = _connect()
    try:
        conn.execute(
            """UPDATE cases
                  SET file_name = ?, file_url = ?, file_size = ?,
                      updated_at = datetime('now')
                WHERE id = ?""",
            (file_name, file_url, size, case_db_id),
        )
    finally:
        conn.close()


def _case_file_ref(case_db_id: str) -> dict | None:
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT file_name, file_url, file_size FROM cases WHERE id = ?",
            (case_db_id,),
        ).fetchone()
        if row is None or not row["file_url"]:
            return None
        return {
            "name": row["file_name"],
            "url": row["file_url"],
            "size": row["file_size"],
        }
    except sqlite3.Error:
        return None
    finally:
        conn.close()


def _public_case_id(case_db_id: str) -> str:
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT case_id FROM cases WHERE id = ?", (case_db_id,)
        ).fetchone()
        return row["case_id"] if row else ""
    except sqlite3.Error:
        return ""
    finally:
        conn.close()


def _mark_processing(case_db_id: str) -> None:
    conn = _connect()
    try:
        conn.execute(
            "UPDATE cases SET status='processing', updated_at=datetime('now') WHERE id=?",
            (case_db_id,),
        )
    finally:
        conn.close()


# exponential with jitter so retries do not pile up
def _backoff(attempt: int) -> float:
    ceiling = min(RETRY_BASE_DELAY * (2 ** (attempt - 1)), RETRY_MAX_DELAY)
    return round(random.uniform(RETRY_BASE_DELAY / 2, ceiling), 1)


# confidence 0 means no verdict, so it is not cached
def _has_signal(rows: list[dict]) -> bool:
    for r in rows:
        try:
            details = json.loads(r.get("details") or "{}")
        except (json.JSONDecodeError, TypeError):
            continue
        if details.get("tier") == "summary":
            return float(details.get("confidence") or 0.0) > 0.0
    return False


# permanent failures stop at once, transient ones retry with back-off
@celery_app.task(
    bind=True,
    name="app.queue.tasks.analyze_case",
    max_retries=MAX_ATTEMPTS - 1,
    acks_late=True,
)
def analyze_case(self: Task, case_db_id: str, media_type: str, file_path: str,
                 content_sha: str = "", source_url: str = "",
                 media_url: str = "") -> dict:
    attempt = self.request.retries + 1
    worker = getattr(self.request, "hostname", "") or ""

    state.mark_running(case_db_id, worker=worker, attempt=attempt)
    if attempt > 1:
        _mark_processing(case_db_id)

    log.info("analysing case=%s media=%s attempt=%d/%d on %s",
             case_db_id, media_type, attempt, MAX_ATTEMPTS, worker)

    started = time.time()
    try:
        if media_url and not file_path:
            file_path, content_sha, replayed = _ingest_url(
                case_db_id, media_type, media_url
            )
            if replayed is not None:
                log.info("case=%s served from cache after download (%s)",
                         case_db_id, media_url)
                return {"caseDbId": case_db_id, "status": replayed,
                        "cached": True, "attempts": attempt}

        risk, likelihood, status, rows = _run(case_db_id, media_type, file_path)

    except PermanentFailure as exc:
        log.error("case=%s permanently failed: %s", case_db_id, exc)
        _store_failure(case_db_id)
        state.mark_finished(case_db_id, state="failed", error=str(exc))
        return {"caseDbId": case_db_id, "status": "failed",
                "reason": str(exc), "retryable": False}

    except SoftTimeLimitExceeded:
        msg = (f"Analysis exceeded the {celery_app.conf.task_soft_time_limit}s "
               f"time limit.")
        log.error("case=%s %s", case_db_id, msg)
        _store_failure(case_db_id)
        state.mark_finished(case_db_id, state="failed", error=msg)
        return {"caseDbId": case_db_id, "status": "failed",
                "reason": msg, "retryable": False}

    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        log.warning("case=%s attempt %d failed: %s\n%s",
                    case_db_id, attempt, detail, traceback.format_exc())

        if attempt >= MAX_ATTEMPTS:
            final = f"Failed after {attempt} attempts. Last error — {detail}"
            _store_failure(case_db_id)
            state.mark_finished(case_db_id, state="failed", error=final)
            return {"caseDbId": case_db_id, "status": "failed",
                    "reason": final, "retryable": False}

        delay = _backoff(attempt)
        state.mark_retrying(case_db_id, attempt=attempt + 1,
                            error=detail, retry_in=delay)
        log.info("case=%s retrying in %.1fs (attempt %d/%d)",
                 case_db_id, delay, attempt + 1, MAX_ATTEMPTS)
        raise self.retry(exc=exc, countdown=delay)

    _store_success(case_db_id, status, risk, likelihood, rows)

    if content_sha and _has_signal(rows):
        cache.store(
            media_type,
            content_sha,
            cache.build_payload(media_type=media_type, risk=risk,
                                likelihood=likelihood, status=status,
                                rows=rows, source_case_id=case_db_id,
                                source_case_ref=_public_case_id(case_db_id),
                                source_file=_case_file_ref(case_db_id)),
            source_url=source_url or media_url or None,
        )

    state.mark_finished(case_db_id, state="succeeded")
    elapsed = time.time() - started
    log.info("case=%s done status=%s risk=%.1f in %.1fs",
             case_db_id, status, risk, elapsed)

    return {"caseDbId": case_db_id, "status": status, "risk": risk,
            "elapsedSeconds": round(elapsed, 1), "attempts": attempt}


def _ingest_url(case_db_id: str, media_type: str,
                media_url: str) -> tuple[str, str, str | None]:
    from app.security.urlfetch import UrlFetchFailed, UrlRejected, fetch

    try:
        result = fetch(media_url, media_type)
    except UrlRejected as exc:
        raise PermanentFailure(str(exc)) from exc
    except UrlFetchFailed as exc:
        raise RuntimeError(f"Download failed: {exc}") from exc

    content_sha = cache.content_hash(result.data)

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    suffix = _EXT_FOR_FORMAT.get(result.detected_format, ".bin")
    safe_name = f"{uuid.uuid4().hex}{suffix}"
    path = UPLOAD_DIR / safe_name
    path.write_bytes(result.data)

    origin_name = Path(urlsplit(result.final_url).path).name
    _record_downloaded_file(
        case_db_id,
        file_name=origin_name or f"download{suffix}",
        file_url=f"{BASE_URL}/uploads/{safe_name}",
        size=len(result.data),
    )
    log.info("case=%s fetched %.1f MB from %s", case_db_id,
             len(result.data) / 1e6, result.final_url)

    cache.remember_url(media_url, content_sha)

    cached = cache.lookup(media_type, content_sha)
    if cached:
        rows = cache.rehydrate_rows(cached, case_db_id)
        _store_success(case_db_id, cached["status"], cached["risk"],
                       cached["likelihood"], rows)
        state.mark_finished(case_db_id, state="cached", cache_hit=True)
        return str(path), content_sha, cached["status"]

    return str(path), content_sha, None


_EXT_FOR_FORMAT = {
    "jpeg": ".jpg", "png": ".png", "webp": ".webp", "bmp": ".bmp",
    "tiff": ".tiff", "mp4": ".mp4", "mov": ".mov", "avi": ".avi",
    "webm": ".webm", "mkv": ".mkv", "mp3": ".mp3", "wav": ".wav",
    "flac": ".flac", "ogg": ".ogg", "m4a": ".m4a",
}

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")


_CHECKPOINT_ENV = {
    "image": "DEEPTRUTH_IMAGE_CHECKPOINTS",
    "video": "DEEPTRUTH_CHECKPOINTS",
}


# fail loudly instead of returning a fake 'inconclusive'
def _preflight_checkpoints(media_type: str) -> None:
    var = _CHECKPOINT_ENV.get(media_type)
    if not var:
        return

    from app import _dtp
    _dtp.bootstrap()

    root = Path(os.environ.get(var, "")).expanduser()
    if not root.is_dir():
        raise CheckpointsUnavailable(
            f"{media_type} checkpoint directory does not exist: {root} "
            f"(set {var} if the weights live elsewhere)"
        )

    if not any(d.is_dir() and (d / "adapter_config.json").is_file()
               for d in root.iterdir()):
        raise CheckpointsUnavailable(
            f"No LoRA adapters found under {root} — expected subdirectories "
            f"containing adapter_config.json"
        )


def _run(case_db_id: str, media_type: str, file_path: str):
    if media_type not in ("image", "video", "audio"):
        raise PermanentFailure(f"Unsupported media type '{media_type}'.")

    path = Path(file_path)
    if not path.exists():
        raise PermanentFailure(f"Evidence file is missing: {file_path}")
    if path.stat().st_size == 0:
        raise PermanentFailure("Evidence file is empty.")

    _preflight_checkpoints(media_type)

    from app.services.analyser import run_analysis

    risk, likelihood, status, rows = asyncio.run(
        run_analysis(case_db_id, media_type, str(path))
    )

    if status == "failed" and not rows:
        raise RuntimeError(
            "The analysis engine returned no result. See the worker log for "
            "the engine-level traceback."
        )

    return risk, likelihood, status, rows


_SNAPSHOT_TTL = 5.0
_snapshot_cache: tuple[float, dict] | None = None


def cached_worker_snapshot() -> dict:
    global _snapshot_cache
    now = time.monotonic()
    if _snapshot_cache and now - _snapshot_cache[0] < _SNAPSHOT_TTL:
        return _snapshot_cache[1]
    snap = worker_snapshot()
    _snapshot_cache = (now, snap)
    return snap


# falls back to the heartbeat keys when celery inspect finds nobody
def worker_snapshot() -> dict:
    try:
        inspector = celery_app.control.inspect(timeout=1.0)
        active = inspector.active() or {}
        reserved = inspector.reserved() or {}
        stats_ = inspector.stats() or {}
    except Exception as exc:
        log.warning("worker inspect failed: %s", exc)
        return {"online": 0, "names": [], "active": 0, "reserved": 0,
                "concurrency": 0, "reachable": False}

    names = sorted(stats_.keys())
    concurrency = sum(
        (s.get("pool", {}) or {}).get("max-concurrency", 0) for s in stats_.values()
    )

    if not names:
        beats = _heartbeats()
        if beats:
            names = sorted(beats)
            concurrency = sum(beats.values())
    return {
        "online": len(names),
        "names": names,
        "active": sum(len(v) for v in active.values()),
        "reserved": sum(len(v) for v in reserved.values()),
        "concurrency": concurrency,
        "reachable": True,
    }


HEARTBEAT_INTERVAL = 10
HEARTBEAT_TTL = 30


def _heartbeats() -> dict[str, int]:
    from app.queue.redis_client import get_client, key
    out: dict[str, int] = {}
    try:
        client = get_client()
        for k in client.scan_iter(match=key("worker", "*"), count=100):
            raw = client.get(k)
            if raw is not None:
                out[k.split(":", 2)[2]] = int(raw or 1)
    except Exception:
        return {}
    return out


# the worker proves it is alive every few seconds
def _start_heartbeat(hostname: str, concurrency: int) -> None:
    import threading
    from app.queue.redis_client import get_client, key

    def beat() -> None:
        while True:
            try:
                get_client().setex(key("worker", hostname), HEARTBEAT_TTL,
                                   concurrency)
            except Exception as exc:
                log.warning("worker heartbeat failed: %s", exc)
            time.sleep(HEARTBEAT_INTERVAL)

    threading.Thread(target=beat, name="worker-heartbeat", daemon=True).start()


@worker_ready.connect
def _on_worker_ready(sender=None, **_kw) -> None:
    pool = getattr(sender, "pool", None)
    _start_heartbeat(getattr(sender, "hostname", "worker"),
                     int(getattr(pool, "limit", 1) or 1))


def active_jobs() -> list[dict]:
    try:
        active = celery_app.control.inspect(timeout=1.0).active() or {}
    except Exception:
        return []

    jobs: list[dict] = []
    for worker, entries in active.items():
        for e in entries:
            args = e.get("args") or []
            jobs.append({
                "worker": worker,
                "taskId": e.get("id"),
                "caseDbId": args[0] if args else None,
                "mediaType": args[1] if len(args) > 1 else None,
                "startedAt": e.get("time_start"),
            })
    return jobs


__all__ = ["analyze_case", "PermanentFailure", "worker_snapshot", "active_jobs",
           "MAX_ATTEMPTS"]
