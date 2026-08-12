"""Queue observability, and the channel the backend pushes state on.

Three endpoints:

    GET /api/queue                     system-wide: depth, workers, cache stats
    GET /api/queue/cases/{case_id}     one case's live job record
    GET /api/queue/stream              Server-Sent Events; the backend tells
                                       the browser about every transition

SSE rather than WebSockets, for two reasons specific to this app. The console
only ever *listens* — it has nothing to send back over a socket — and SSE
gives that for free over plain HTTP, through the same CORS config, with the
browser's own `EventSource` handling reconnection. And the Chrome extension's
service worker cannot hold a long-lived socket open (it is terminated when
idle), so it keeps polling `GET /cases/{id}`, which now carries the same job
record. One state source, two delivery mechanisms, no divergence.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncIterator, Optional

import redis
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from app.database import get_db
from app.queue import cache, state
from app.queue.redis_client import get_client, ping
from app.queue.tasks import active_jobs, worker_snapshot

log = logging.getLogger(__name__)

router = APIRouter()

# How often the stream emits a keep-alive when nothing is happening. Proxies
# and browsers drop idle connections; a comment line costs nothing and keeps
# the socket open through them.
HEARTBEAT_SECONDS = 20


@router.get("/queue")
async def queue_overview():
    """System health for the orchestration layer.

    Deliberately answers even when Redis is down — `redisOnline: false` is the
    single most useful thing this endpoint can report, and it cannot report it
    by raising.
    """
    online = ping()
    if not online:
        return {
            "redisOnline": False,
            "pendingDepth": 0,
            "workers": {"online": 0, "names": [], "active": 0,
                        "reserved": 0, "concurrency": 0, "reachable": False},
            "activeJobs": [],
            "cache": {"hits": 0, "misses": 0, "entries": 0,
                      "hitRate": 0.0, "version": cache.CACHE_VERSION},
            "message": "Redis is not reachable; uploads will be rejected with 503.",
        }

    workers = worker_snapshot()
    depth = state.pending_depth()

    message = None
    if workers["online"] == 0:
        message = ("No Celery workers are online. Jobs will queue and run as "
                   "soon as a worker starts.")
    elif depth > workers["concurrency"] * 3 and workers["concurrency"]:
        message = (f"{depth} jobs waiting for {workers['concurrency']} worker "
                   f"slots. Consider starting another worker.")

    return {
        "redisOnline": True,
        "pendingDepth": depth,
        "workers": workers,
        "activeJobs": active_jobs(),
        "cache": cache.stats(),
        "message": message,
    }


@router.get("/queue/pending")
async def queue_pending(limit: int = Query(25, ge=1, le=100)):
    """Waiting jobs in execution order, resolved to their public case ids."""
    ids = state.pending_ids(limit)
    out = []
    for i, db_id in enumerate(ids, start=1):
        job = state.get_job(db_id)
        out.append({
            "position": i,
            "caseDbId": db_id,
            "caseId": (job or {}).get("caseId"),
            "mediaType": (job or {}).get("mediaType"),
            "state": (job or {}).get("state"),
            "attempt": (job or {}).get("attempt"),
        })
    return {"pending": out, "depth": state.pending_depth()}


@router.get("/queue/cases/{case_id}")
async def queue_case(case_id: str):
    """One case's job record, addressed by the public `CASE-XXXXXXXX` id.

    Accepts the public id (not the internal uuid) because that is the only id
    a client ever holds, and resolves it through SQLite to the row id the
    queue keys on.
    """
    db_id = await _resolve_db_id(case_id)
    job = state.get_job(db_id)
    if job is None:
        raise HTTPException(
            404,
            f"No queue record for '{case_id}'. The case may predate the queue "
            f"or its record may have expired; its verdict is still on the case.",
        )
    return {"caseDbId": db_id, **job}


@router.get("/cache/lookup")
async def cache_lookup(
    media_type: str = Query(..., description="image | video | audio"),
    url: Optional[str] = Query(None, description="Media URL seen on a page"),
    sha256: Optional[str] = Query(None, description="sha256 of the file bytes"),
):
    """Ask whether a verdict already exists, without submitting anything.

    This is the cheap path for the browser extension. Its normal flow is
    download the media, upload it, wait — and for media it has already seen,
    every one of those steps is wasted. Asking by URL first turns a repeat
    encounter into a single GET.

    Either key works:
      * `sha256` — exact, when the caller already holds the bytes
      * `url`    — resolves through the URL index to a content hash, so it
                   answers only for bytes actually fetched from that URL
                   before. A URL whose content changed misses rather than
                   returning a stale verdict.

    A miss is a 200 with `hit: false`, not a 404 — "we have not seen this" is
    a successful answer to the question asked.
    """
    if media_type not in ("image", "video", "audio"):
        raise HTTPException(400, "media_type must be one of: image, video, audio")
    if not url and not sha256:
        raise HTTPException(400, "Provide either 'url' or 'sha256'.")

    if not ping():
        raise HTTPException(503, "Redis is not reachable; cache cannot be queried.")

    resolved_sha = sha256
    payload = None

    if sha256:
        payload = cache.lookup(media_type, sha256)
    if payload is None and url:
        resolved_sha, payload = cache.lookup_by_url(media_type, url)

    if payload is None:
        return {
            "hit": False,
            "mediaType": media_type,
            # Returned even on a miss: a known URL with no verdict still tells
            # the caller these bytes were seen, which is worth knowing.
            "contentHash": resolved_sha,
            "normalisedUrl": cache.normalise_url(url) if url else None,
        }

    return {
        "hit": True,
        "mediaType": media_type,
        "contentHash": resolved_sha,
        "normalisedUrl": cache.normalise_url(url) if url else None,
        "status": payload["status"],
        "riskScore": payload["risk"],
        "syntheticLikelihood": payload["likelihood"],
        "computedAt": payload.get("computedAt"),
        "sourceCaseDbId": payload.get("sourceCaseId"),
        # The public id of the case that produced this verdict — the only one
        # a client can navigate to. Empty for entries cached before this field
        # existed, so callers must treat it as optional.
        "sourceCaseId": payload.get("sourceCaseRef") or None,
        "analysisResults": payload.get("rows", []),
    }


async def _resolve_db_id(case_id: str) -> str:
    db = await get_db()
    try:
        async with db.execute(
            "SELECT id FROM cases WHERE case_id = ?", (case_id,)
        ) as cur:
            row = await cur.fetchone()
    finally:
        await db.close()
    if row is None:
        raise HTTPException(404, f"Case '{case_id}' not found.")
    return row["id"]


# ─────────────────────────────────────────────────────────────────────────────
# Server-Sent Events
# ─────────────────────────────────────────────────────────────────────────────

def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.get("/queue/stream")
async def queue_stream(request: Request, case_id: Optional[str] = Query(None)):
    """Push every job transition to the browser as it happens.

    With `?case_id=CASE-XXXXXXXX` the stream is filtered to that one case —
    what a report page wants. Without it, every transition is forwarded, which
    is what a dashboard showing the whole queue wants.

    The first event is always a `snapshot`, so a client that connects halfway
    through a run renders the correct state immediately instead of waiting for
    the next transition — which, for a job that is already finished, would
    never arrive.
    """
    if not ping():
        raise HTTPException(503, "Redis is not reachable; no event stream available.")

    watch_db_id = await _resolve_db_id(case_id) if case_id else None

    async def events() -> AsyncIterator[str]:
        client = get_client()
        pubsub = client.pubsub(ignore_subscribe_messages=True)
        try:
            pubsub.subscribe(state.EVENTS_CHANNEL)
        except redis.RedisError as exc:
            yield _sse("error", {"message": f"Could not subscribe: {exc}"})
            return

        try:
            # ── Opening snapshot ────────────────────────────────────────────
            if watch_db_id:
                job = state.get_job(watch_db_id)
                yield _sse("snapshot", {"caseDbId": watch_db_id,
                                        "caseId": case_id, **(job or {})})
            else:
                yield _sse("snapshot", {
                    "pendingDepth": state.pending_depth(),
                    "workers": worker_snapshot(),
                })

            last_beat = asyncio.get_event_loop().time()

            while True:
                # A disconnected browser is only detectable between reads;
                # checking each iteration keeps closed tabs from leaking a
                # subscription per reload.
                if await request.is_disconnected():
                    break

                # `get_message` is blocking C code, so it runs in a thread —
                # otherwise a quiet queue would stall the whole event loop for
                # the full timeout and every other request with it.
                message = await asyncio.to_thread(
                    pubsub.get_message, timeout=1.0
                )

                if message and message.get("type") == "message":
                    try:
                        payload = json.loads(message["data"])
                    except (json.JSONDecodeError, TypeError):
                        continue

                    if watch_db_id and payload.get("caseDbId") != watch_db_id:
                        continue

                    # Re-read rather than trusting the published delta: the
                    # published event carries only what changed, while the
                    # client wants position and attempt counts alongside it.
                    db_id = payload.get("caseDbId")
                    full = state.get_job(db_id) if db_id else None
                    yield _sse("job", {"caseDbId": db_id, **(full or payload)})

                now = asyncio.get_event_loop().time()
                if now - last_beat >= HEARTBEAT_SECONDS:
                    last_beat = now
                    yield ": keep-alive\n\n"

        except asyncio.CancelledError:
            raise
        finally:
            try:
                pubsub.close()
            except redis.RedisError:
                pass

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            # nginx buffers proxied responses by default, which would hold
            # every event until the buffer fills — i.e. defeat the stream.
            "X-Accel-Buffering": "no",
        },
    )
