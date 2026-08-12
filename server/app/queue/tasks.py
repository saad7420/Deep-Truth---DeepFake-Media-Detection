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
from pathlib import Path

import app.utils.env  # noqa: F401  — load server/.env in the worker too

from celery import Task
from celery.exceptions import SoftTimeLimitExceeded

from app.queue import cache, state
from app.queue.celery_app import celery_app

log = logging.getLogger(__name__)

DB_PATH = os.getenv("DB_PATH", "forensics.db")

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
                 content_sha: str = "", source_url: str = "") -> dict:
    """Run one case to completion and record it.

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
            source_url=source_url or None,
        )

    state.mark_finished(case_db_id, state="succeeded")
    elapsed = time.time() - started
    log.info("case=%s done status=%s risk=%.1f in %.1fs",
             case_db_id, status, risk, elapsed)

    return {"caseDbId": case_db_id, "status": status, "risk": risk,
            "elapsedSeconds": round(elapsed, 1), "attempts": attempt}


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
