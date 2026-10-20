from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import AsyncIterator, Optional

import redis
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from app.database import get_db
from app.queue import cache, state
from app.queue.redis_client import get_client, ping
from starlette.concurrency import run_in_threadpool

from app.queue.tasks import active_jobs, cached_worker_snapshot
from app.security import ratelimit

log = logging.getLogger(__name__)

router = APIRouter()

HEARTBEAT_SECONDS = 20

# close long streams so shutdown never hangs, the browser reconnects
MAX_STREAM_SECONDS = int(os.getenv("SSE_MAX_STREAM_SECONDS", "600"))


@router.get("/queue", dependencies=[Depends(ratelimit.read)])
async def queue_overview():
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

    workers = await run_in_threadpool(cached_worker_snapshot)
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
        "activeJobs": await run_in_threadpool(active_jobs),
        "cache": cache.stats(),
        "message": message,
    }


@router.get("/queue/pending", dependencies=[Depends(ratelimit.read)])
async def queue_pending(limit: int = Query(25, ge=1, le=100)):
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


@router.get("/queue/cases/{case_id}", dependencies=[Depends(ratelimit.read)])
async def queue_case(case_id: str):
    db_id = await _resolve_db_id(case_id)
    job = state.get_job(db_id)
    if job is None:
        raise HTTPException(
            404,
            f"No queue record for '{case_id}'. The case may predate the queue "
            f"or its record may have expired; its verdict is still on the case.",
        )
    return {"caseDbId": db_id, **job}


# lets the extension skip the download for media we already know
@router.get("/cache/lookup", dependencies=[Depends(ratelimit.read)])
async def cache_lookup(
    media_type: str = Query(..., description="image | video | audio"),
    url: Optional[str] = Query(None, description="Media URL seen on a page"),
    sha256: Optional[str] = Query(None, description="sha256 of the file bytes"),
):
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


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# server-sent events: a snapshot first, then live job updates
@router.get("/queue/stream", dependencies=[Depends(ratelimit.stream)])
async def queue_stream(request: Request, case_id: Optional[str] = Query(None)):
    if not ping():
        raise HTTPException(503, "Redis is not reachable; no event stream available.")

    watch_db_id = await _resolve_db_id(case_id) if case_id else None

    slot = ratelimit.StreamSlot(request).__enter__()

    async def events() -> AsyncIterator[str]:
        client = get_client()
        pubsub = client.pubsub(ignore_subscribe_messages=True)
        try:
            pubsub.subscribe(state.EVENTS_CHANNEL)
        except redis.RedisError as exc:
            slot.__exit__(None, None, None)
            yield _sse("error", {"message": f"Could not subscribe: {exc}"})
            return

        try:
            if watch_db_id:
                job = state.get_job(watch_db_id)
                yield _sse("snapshot", {"caseDbId": watch_db_id,
                                        "caseId": case_id, **(job or {})})
            else:
                yield _sse("snapshot", {
                    "pendingDepth": state.pending_depth(),
                    "workers": await run_in_threadpool(cached_worker_snapshot),
                })

            started = asyncio.get_event_loop().time()
            last_beat = started

            while True:
                if await request.is_disconnected():
                    break

                if asyncio.get_event_loop().time() - started >= MAX_STREAM_SECONDS:
                    yield _sse("reconnect", {"reason": "max stream lifetime reached"})
                    break

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
            slot.__exit__(None, None, None)
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
            "X-Accel-Buffering": "no",
        },
    )
