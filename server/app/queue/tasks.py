"""The analysis task.

This is the only place inference is invoked from in the queued architecture.
The API process hands over a case id, a media type, and a path; everything
after that happens here, in a worker process, where blocking for four minutes
on a CPU-bound torch run costs nothing.

Retry policy (FE-3)
-------------------
Not every failure deserves a retry, and retrying the ones that don't is worse
than failing fast: each attempt is minutes of CPU, and a file that is corrupt
now will be corrupt on attempt three.

    PermanentFailure   the input itself is the problem — file gone, unknown
                       modality, no engine registered. Fails immediately.
    everything else    treated as transient — OOM under memory pressure, a
                       HuggingFace fetch that timed out, SQLite locked by a
                       concurrent writer, a worker killed mid-run. Retried
                       with exponential backoff and jitter.

Backoff is exponential *with jitter* because the realistic multi-failure case
is several jobs hitting the same external cause at the same moment (the model
hub is down, the box is out of memory). Retrying them all on an identical
schedule reproduces the pile-up that caused the failure; spreading them does
not.
"""
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

import app.utils.env  # noqa: F401  — load server/.env in the worker too

from celery import Task
from celery.exceptions import SoftTimeLimitExceeded

from app.queue import cache, state
from app.queue.celery_app import celery_app

log = logging.getLogger(__name__)

DB_PATH = os.getenv("DB_PATH", "forensics.db")

# Same expression as routers/cases.py. Both read the one environment variable
# so they cannot disagree, and the worker needs its own copy because importing
# the router here would drag FastAPI into every worker process.
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads"))

# Total attempts, not retries: 3 means one run plus two retries.
MAX_ATTEMPTS = int(os.getenv("ANALYSIS_MAX_ATTEMPTS", "3"))
RETRY_BASE_DELAY = float(os.getenv("ANALYSIS_RETRY_BASE_DELAY", "10"))
RETRY_MAX_DELAY = float(os.getenv("ANALYSIS_RETRY_MAX_DELAY", "300"))


class PermanentFailure(Exception):
    """A failure that retrying cannot fix."""


class CheckpointsUnavailable(Exception):
    """The model weights this modality needs are not on disk.

    Retryable on purpose. The realistic cause is a volume that has not
    finished mounting or a worker started before the weights were synced —
    conditions that resolve on their own within seconds.
    """


# ─────────────────────────────────────────────────────────────────────────────
# SQLite from a worker process
# ─────────────────────────────────────────────────────────────────────────────

def _connect() -> sqlite3.Connection:
    """Worker-side connection.

    `timeout` matters here in a way it does not in the single-process design
    this replaces. Several workers plus the API now write to one SQLite file,
    and without a busy timeout a concurrent writer raises "database is locked"
    immediately — which would surface as a retry storm on jobs that actually
    succeeded at inference and only failed to record themselves. WAL (set at
    init_db) plus a 30-second busy timeout lets readers and one writer proceed
    without tripping over each other.
    """
    conn = sqlite3.connect(DB_PATH, timeout=30.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=30000;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


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
        # A retry may have partially written rows before dying. Clearing first
        # keeps a retried case from showing each checkpoint twice.
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
    """Attach the fetched file to its case.

    A URL-submitted case is created before anything is downloaded, so it has no
    file details until the worker has them. Without this the report page shows
    a case with no media to preview.
    """
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


def _public_case_id(case_db_id: str) -> str:
    """The CASE-XXXXXXXX id for a row id, or "" if the row is gone.

    Cached alongside the verdict so a future cache hit can point a client at
    the case that originally produced it — the internal uuid the task works
    with is not addressable through the API.
    """
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
    """A retry re-enters from whatever status the failed attempt left behind;
    put the row back to 'processing' so the console never shows a stale
    'failed' for a case that is actively being retried."""
    conn = _connect()
    try:
        conn.execute(
            "UPDATE cases SET status='processing', updated_at=datetime('now') WHERE id=?",
            (case_db_id,),
        )
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# The task
# ─────────────────────────────────────────────────────────────────────────────

def _backoff(attempt: int) -> float:
    """Exponential with full jitter, capped."""
    ceiling = min(RETRY_BASE_DELAY * (2 ** (attempt - 1)), RETRY_MAX_DELAY)
    return round(random.uniform(RETRY_BASE_DELAY / 2, ceiling), 1)


def _has_signal(rows: list[dict]) -> bool:
    """True when some engine actually produced a verdict.

    Reads the tier-"summary" row's `confidence`, which is every engine's
    "ignore me" flag when it is 0.0 (see app/engines/base.neutral_result).
    """
    for r in rows:
        try:
            details = json.loads(r.get("details") or "{}")
        except (json.JSONDecodeError, TypeError):
            continue
        if details.get("tier") == "summary":
            return float(details.get("confidence") or 0.0) > 0.0
    return False


@celery_app.task(
    bind=True,
    name="app.queue.tasks.analyze_case",
    max_retries=MAX_ATTEMPTS - 1,
    # acks_late is set globally; repeated here as documentation of intent for
    # anyone reading just this task.
    acks_late=True,
)
def analyze_case(self: Task, case_db_id: str, media_type: str, file_path: str,
                 content_sha: str = "", source_url: str = "",
                 media_url: str = "") -> dict:
    """Run one case to completion and record it.

    Either `file_path` (a direct upload, already on disk) or `media_url` (a
    page URL the server fetches itself — Module 4 FE-1). The download happens
    here rather than in the API process for two reasons: a transfer the caller
    does not control must never block the event loop, and network failure is
    exactly the transient condition the retry policy above already handles.

    Returns a small dict for the Celery result backend; the durable record is
    the SQLite row and the Redis job state, both written before this returns.
    """
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
                # The URL was new but its bytes were not. Nothing to analyse.
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
        # The hard limit is close behind, so do the bookkeeping now while
        # there is still a process to do it in.
        msg = (f"Analysis exceeded the {celery_app.conf.task_soft_time_limit}s "
               f"time limit.")
        log.error("case=%s %s", case_db_id, msg)
        _store_failure(case_db_id)
        state.mark_finished(case_db_id, state="failed", error=msg)
        return {"caseDbId": case_db_id, "status": "failed",
                "reason": msg, "retryable": False}

    except Exception as exc:  # noqa: BLE001 — deliberately broad; see below
        # Anything not classified as permanent is assumed transient. That
        # default is the safe one: a genuinely deterministic failure costs
        # MAX_ATTEMPTS runs and then reports failed, whereas mis-classifying a
        # transient failure as permanent loses the case outright.
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

    # ── Success path ────────────────────────────────────────────────────────
    _store_success(case_db_id, status, risk, likelihood, rows)

    # Cache after the durable write, never before: an entry pointing at a
    # verdict that failed to persist would serve a result no case can show.
    #
    # And only cache an actual verdict. A confidence-0.0 result means no
    # engine contributed anything — an audio stub, an undecodable file, a
    # degraded run. Entries never expire (see cache.py), so storing one of
    # those would freeze "we could not tell" as this file's permanent answer,
    # and the case that finally gets a real checkpoint would still be served
    # the empty one.
    if content_sha and _has_signal(rows):
        cache.store(
            media_type,
            content_sha,
            cache.build_payload(media_type=media_type, risk=risk,
                                likelihood=likelihood, status=status,
                                rows=rows, source_case_id=case_db_id,
                                source_case_ref=_public_case_id(case_db_id)),
            # `source_url` is the URL an uploaded file came from; `media_url`
            # is the URL the server fetched. Either identifies where these
            # bytes live, and only one is ever set.
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
    """Fetch `media_url` server-side and land it where an upload would be.

    Returns (file_path, content_sha, replayed_status). `replayed_status` is
    non-None when the downloaded bytes turned out to be already cached, in
    which case the verdict has been written and there is nothing to analyse.

    That second cache check is not redundant with the API's. The API can only
    look the URL up, and a URL is a weak key — the same image is served from
    countless URLs. Once the bytes are in hand the strong key is available, so
    a first-ever sighting of a URL can still avoid inference entirely.
    """
    from app.security.urlfetch import UrlFetchFailed, UrlRejected, fetch

    try:
        result = fetch(media_url, media_type)
    except UrlRejected as exc:
        # The URL itself is the problem: bad host, 404, wrong content, too
        # large. Retrying cannot change any of those.
        raise PermanentFailure(str(exc)) from exc
    except UrlFetchFailed as exc:
        # Timeout, reset, 5xx. Exactly what retries are for, so this is left
        # to propagate into the generic handler.
        raise RuntimeError(f"Download failed: {exc}") from exc

    content_sha = cache.content_hash(result.data)

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    suffix = _EXT_FOR_FORMAT.get(result.detected_format, ".bin")
    safe_name = f"{uuid.uuid4().hex}{suffix}"
    path = UPLOAD_DIR / safe_name
    path.write_bytes(result.data)

    # Name from the URL's *path* only. Taking it from the whole URL drags the
    # query string in, and signed CDN links put a long hmac there — the case
    # would be labelled "300.jpg?hmac=0aXbwBEwdw...".
    origin_name = Path(urlsplit(result.final_url).path).name
    _record_downloaded_file(
        case_db_id,
        file_name=origin_name or f"download{suffix}",
        file_url=f"{BASE_URL}/uploads/{safe_name}",
        size=len(result.data),
    )
    log.info("case=%s fetched %.1f MB from %s", case_db_id,
             len(result.data) / 1e6, result.final_url)

    # Index the URL against these bytes immediately, on every path. Doing it
    # only on the cache-hit branch (or leaving it to cache.store at the end)
    # means a URL's very first fetch never teaches the index, so the second
    # submission of that URL downloads and analyses it all over again — the
    # exact saving this module exists to make.
    cache.remember_url(media_url, content_sha)

    cached = cache.lookup(media_type, content_sha)
    if cached:
        rows = cache.rehydrate_rows(cached, case_db_id)
        _store_success(case_db_id, cached["status"], cached["risk"],
                       cached["likelihood"], rows)
        state.mark_finished(case_db_id, state="cached", cache_hit=True)
        return str(path), content_sha, cached["status"]

    return str(path), content_sha, None


#: Extension to give a downloaded file, so the pipeline's suffix-based
#: dispatch sees something sensible. A CDN path often ends in a hash with no
#: extension at all, so the detected container is what decides.
_EXT_FOR_FORMAT = {
    "jpeg": ".jpg", "png": ".png", "webp": ".webp", "bmp": ".bmp",
    "tiff": ".tiff", "mp4": ".mp4", "mov": ".mov", "avi": ".avi",
    "webm": ".webm", "mkv": ".mkv", "mp3": ".mp3", "wav": ".wav",
    "flac": ".flac", "ogg": ".ogg", "m4a": ".m4a",
}

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")


#: Checkpoint directory each modality needs, by environment variable.
#: Audio is absent deliberately — it is designed to run without a checkpoint
#: and report "no signal", so a missing audio model is not a failure.
_CHECKPOINT_ENV = {
    "image": "DEEPTRUTH_IMAGE_CHECKPOINTS",
    "video": "DEEPTRUTH_CHECKPOINTS",
}


def _preflight_checkpoints(media_type: str) -> None:
    """Fail loudly when the weights are missing, instead of quietly neutral.

    This check exists because of how the engines behave without it. Per
    `app/engines/base.Engine.analyze`'s contract they never raise — a missing
    checkpoint directory comes back as a confidence-0.0 neutral result, which
    `analyser.py` correctly renders as `inconclusive`. That is right for the
    fusion layer and wrong for the queue: "no models were loaded" is an
    infrastructure fault that should be retried, and instead the operator was
    handed an authoritative-looking "inconclusive" verdict for a run in which
    nothing was actually evaluated.

    Checking the directory directly rather than parsing the engine's prose
    keeps this from breaking the next time a log message is reworded.
    """
    var = _CHECKPOINT_ENV.get(media_type)
    if not var:
        return

    # bootstrap() is idempotent and imports nothing heavy; it is what puts the
    # DEEPTRUTH_* defaults in the environment, so the variable is only
    # readable after it has run.
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
    """Validate the inputs we can, then hand off to the existing analyser.

    The pre-checks exist to convert the failures that *are* deterministic into
    PermanentFailure before minutes of model loading are spent discovering
    them.
    """
    if media_type not in ("image", "video", "audio"):
        raise PermanentFailure(f"Unsupported media type '{media_type}'.")

    path = Path(file_path)
    if not path.exists():
        raise PermanentFailure(f"Evidence file is missing: {file_path}")
    if path.stat().st_size == 0:
        raise PermanentFailure("Evidence file is empty.")

    _preflight_checkpoints(media_type)

    # Imported here rather than at module scope so that `celery inspect` and
    # the API's import of this module do not drag in torch.
    from app.services.analyser import run_analysis

    risk, likelihood, status, rows = asyncio.run(
        run_analysis(case_db_id, media_type, str(path))
    )

    # run_analysis swallows engine exceptions and reports this shape instead.
    # Raising here routes it into the retry path above — the engine blew up
    # for a reason the analyser could not attribute, and those are usually
    # resource-related and worth one more go.
    if status == "failed" and not rows:
        raise RuntimeError(
            "The analysis engine returned no result. See the worker log for "
            "the engine-level traceback."
        )

    return risk, likelihood, status, rows


# ─────────────────────────────────────────────────────────────────────────────
# Introspection used by /api/queue
# ─────────────────────────────────────────────────────────────────────────────

def worker_snapshot() -> dict:
    """Live worker census, straight from Celery's control API.

    Returns zeroed counts when no worker answers within the timeout, which is
    the honest reading: from the API's point of view a worker that cannot be
    pinged is a worker that cannot take work.
    """
    try:
        inspector = celery_app.control.inspect(timeout=1.0)
        active = inspector.active() or {}
        reserved = inspector.reserved() or {}
        stats_ = inspector.stats() or {}
    except Exception as exc:  # noqa: BLE001 — control plane must never 500 the API
        log.warning("worker inspect failed: %s", exc)
        return {"online": 0, "names": [], "active": 0, "reserved": 0,
                "concurrency": 0, "reachable": False}

    names = sorted(stats_.keys())
    concurrency = sum(
        (s.get("pool", {}) or {}).get("max-concurrency", 0) for s in stats_.values()
    )
    return {
        "online": len(names),
        "names": names,
        "active": sum(len(v) for v in active.values()),
        "reserved": sum(len(v) for v in reserved.values()),
        "concurrency": concurrency,
        "reachable": True,
    }


def active_jobs() -> list[dict]:
    """What each worker is running right now."""
    try:
        active = celery_app.control.inspect(timeout=1.0).active() or {}
    except Exception:  # noqa: BLE001
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
