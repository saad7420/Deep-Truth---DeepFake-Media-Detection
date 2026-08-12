"""Job lifecycle state, held in Redis.

SQLite already records what a case *concluded*. What it cannot record is what
a case is *doing right now* — waiting behind six other uploads, on its second
attempt after a transient failure, or executing on worker `celery@box-2`. That
is what this module keeps, and it lives in Redis because both the API process
and every worker must read and write it.

Two structures:

    dt:job:<case_db_id>   hash    the full record for one job
    dt:queue:pending      zset    case ids still waiting, scored by enqueue
                                  time, so ZRANK is literally queue position

and one broadcast channel, `dt:events`, which the API's SSE endpoint tails so
the browser is *told* about a transition rather than discovering it on its
next poll.

Every write also publishes. A caller that forgets to publish leaves a browser
hanging until its fallback poll fires, so the publish is folded into the same
functions that mutate rather than left to the caller to remember.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Literal

import redis

from app.queue.redis_client import get_client, key

log = logging.getLogger(__name__)

JobState = Literal["queued", "running", "retrying", "succeeded", "failed", "cached"]

#: States a job will not leave on its own.
TERMINAL: frozenset[str] = frozenset({"succeeded", "failed", "cached"})

PENDING_ZSET = key("queue", "pending")
EVENTS_CHANNEL = key("events")

# How long a finished job's record survives. Long enough for a browser that
# was closed during the run to come back and see how it ended; short enough
# that Redis does not accumulate history that SQLite already holds durably.
FINISHED_TTL = 24 * 3600

# A job that never reaches a terminal state (worker killed with acks_late
# disabled, Redis restored from an old dump) would otherwise sit in the
# pending set forever and inflate every queue-position readout.
ACTIVE_TTL = 7 * 24 * 3600


def job_key(case_db_id: str) -> str:
    return key("job", case_db_id)


# ─────────────────────────────────────────────────────────────────────────────
# Writes
# ─────────────────────────────────────────────────────────────────────────────

def _publish(client: redis.Redis, case_db_id: str, record: dict[str, Any]) -> None:
    """Announce a transition. Best-effort: a failure to publish must never
    fail the job, because the record itself is already committed and polling
    will still surface it."""
    try:
        client.publish(EVENTS_CHANNEL, json.dumps({"caseDbId": case_db_id, **record}))
    except redis.RedisError as exc:
        log.warning("could not publish state for %s: %s", case_db_id, exc)


def _write(client: redis.Redis, case_db_id: str, fields: dict[str, Any],
           *, ttl: int) -> dict[str, Any]:
    flat = {k: ("" if v is None else str(v)) for k, v in fields.items()}
    pipe = client.pipeline()
    pipe.hset(job_key(case_db_id), mapping=flat)
    pipe.expire(job_key(case_db_id), ttl)
    pipe.execute()
    return flat


def mark_queued(case_db_id: str, *, case_id: str, media_type: str,
                celery_task_id: str = "", max_attempts: int = 3,
                content_hash: str = "", source_url: str = "") -> dict[str, Any]:
    """Record a job as accepted and waiting. Called by the API immediately
    after the Celery message is published, so the very first status read a
    client makes already knows its position."""
    client = get_client()
    now = time.time()
    record = {
        "state": "queued",
        "caseId": case_id,
        "mediaType": media_type,
        "celeryTaskId": celery_task_id,
        "attempt": 1,
        "maxAttempts": max_attempts,
        "contentHash": content_hash,
        "sourceUrl": source_url,
        "queuedAt": now,
        "startedAt": "",
        "finishedAt": "",
        "error": "",
        "cacheHit": "0",
    }
    _write(client, case_db_id, record, ttl=ACTIVE_TTL)
    client.zadd(PENDING_ZSET, {case_db_id: now})
    _publish(client, case_db_id, {"state": "queued"})
    return record


def mark_running(case_db_id: str, *, worker: str = "", attempt: int = 1) -> None:
    """Worker has picked the job up. Leaving the pending set here is what
    makes everyone else's queue position tick down."""
    client = get_client()
    _write(client, case_db_id, {
        "state": "running",
        "worker": worker,
        "attempt": attempt,
        "startedAt": time.time(),
        "error": "",
    }, ttl=ACTIVE_TTL)
    client.zrem(PENDING_ZSET, case_db_id)
    _publish(client, case_db_id, {"state": "running", "attempt": attempt})


def mark_retrying(case_db_id: str, *, attempt: int, error: str,
                  retry_in: float) -> None:
    """A transient failure; the job goes back to waiting rather than dying.

    It is re-added to the pending set because that is the truth — it is once
    again a job with no worker executing it — and a user watching the console
    should see it queued rather than silently gone until the backoff expires.
    """
    client = get_client()
    _write(client, case_db_id, {
        "state": "retrying",
        "attempt": attempt,
        "error": error[:500],
        "retryAt": time.time() + retry_in,
    }, ttl=ACTIVE_TTL)
    client.zadd(PENDING_ZSET, {case_db_id: time.time() + retry_in})
    _publish(client, case_db_id, {
        "state": "retrying", "attempt": attempt,
        "error": error[:500], "retryIn": round(retry_in, 1),
    })


def mark_finished(case_db_id: str, *, state: JobState, error: str = "",
                  cache_hit: bool = False) -> None:
    """Terminal transition — succeeded, failed, or answered from cache."""
    client = get_client()
    _write(client, case_db_id, {
        "state": state,
        "finishedAt": time.time(),
        "error": error[:500],
        "cacheHit": "1" if cache_hit else "0",
    }, ttl=FINISHED_TTL)
    client.zrem(PENDING_ZSET, case_db_id)
    _publish(client, case_db_id, {
        "state": state, "error": error[:500], "cacheHit": cache_hit,
    })


def mark_cached(case_db_id: str, *, case_id: str, media_type: str,
                content_hash: str = "", source_url: str = "") -> None:
    """A job that never needed a worker. Recorded with the same shape as a
    real run so the console can render it without a special case — it simply
    arrives already terminal, with cacheHit set."""
    client = get_client()
    now = time.time()
    _write(client, case_db_id, {
        "state": "cached",
        "caseId": case_id,
        "mediaType": media_type,
        "attempt": 0,
        "maxAttempts": 0,
        "contentHash": content_hash,
        "sourceUrl": source_url,
        "queuedAt": now,
        "startedAt": now,
        "finishedAt": now,
        "error": "",
        "cacheHit": "1",
    }, ttl=FINISHED_TTL)
    _publish(client, case_db_id, {"state": "cached", "cacheHit": True})


# ─────────────────────────────────────────────────────────────────────────────
# Reads
# ─────────────────────────────────────────────────────────────────────────────

def _as_float(raw: str | None) -> float | None:
    try:
        return float(raw) if raw else None
    except (TypeError, ValueError):
        return None


def get_job(case_db_id: str) -> dict[str, Any] | None:
    """Full status for one job, including its live queue position.

    Returns None when Redis has no record — a case older than FINISHED_TTL,
    or one created before this module existed. Callers treat that as "no
    queue information", not as an error, so old cases still render.
    """
    client = get_client()
    try:
        raw = client.hgetall(job_key(case_db_id))
    except redis.RedisError as exc:
        log.warning("could not read job %s: %s", case_db_id, exc)
        return None
    if not raw:
        return None

    state = raw.get("state", "queued")
    position: int | None = None
    if state in ("queued", "retrying"):
        try:
            rank = client.zrank(PENDING_ZSET, case_db_id)
            position = None if rank is None else rank + 1
        except redis.RedisError:
            position = None

    return {
        "state": state,
        "caseId": raw.get("caseId") or None,
        "mediaType": raw.get("mediaType") or None,
        "position": position,
        "attempt": int(raw.get("attempt") or 0),
        "maxAttempts": int(raw.get("maxAttempts") or 0),
        "worker": raw.get("worker") or None,
        "queuedAt": _as_float(raw.get("queuedAt")),
        "startedAt": _as_float(raw.get("startedAt")),
        "finishedAt": _as_float(raw.get("finishedAt")),
        "retryAt": _as_float(raw.get("retryAt")),
        "error": raw.get("error") or None,
        "cacheHit": raw.get("cacheHit") == "1",
        "contentHash": raw.get("contentHash") or None,
        "sourceUrl": raw.get("sourceUrl") or None,
    }


def pending_depth() -> int:
    """How many jobs are waiting for a worker."""
    try:
        return int(get_client().zcard(PENDING_ZSET))
    except redis.RedisError:
        return 0


def pending_ids(limit: int = 50) -> list[str]:
    """Waiting jobs, oldest first — the order they will actually run in."""
    try:
        return list(get_client().zrange(PENDING_ZSET, 0, limit - 1))
    except redis.RedisError:
        return []
