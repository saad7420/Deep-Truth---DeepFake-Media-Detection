"""Redis-backed rate limiting.

Why this endpoint needs it more than most APIs do: `POST /cases` accepts a
file up to 500 MB and queues a CPU-bound job that occupies a worker slot for
minutes. There are two slots on a default install. A client looping that
endpoint does not merely generate load — it takes the analysis pipeline away
from every other user with a handful of requests, and it costs them almost
nothing to do it. Read endpoints are cheap by comparison and are limited only
to stop a runaway poller.

Algorithm
---------
Sliding-window log: one Redis sorted set per (scope, client), scored by
timestamp. Admission trims entries older than the window, counts what is left,
and adds the current request if there is room.

The obvious alternative, a fixed-window counter (INCR + EXPIRE), is cheaper but
lets a caller send `limit` requests at 0:59 and `limit` again at 1:01 — double
the intended rate at exactly the moment a burst hurts most. The sliding window
has no such boundary, and since the limits here are small (tens of entries) the
memory the log costs is irrelevant.

The whole admission decision runs as one Lua script so it is atomic. Done as
separate round trips, two concurrent requests both read a count below the limit
and both proceed — precisely the case a rate limiter exists to stop.

Distributed by construction: the counters live in Redis, not in process memory,
so running several API workers does not multiply the effective limit.
"""
# NOTE: deliberately no `from __future__ import annotations` here, unlike the
# rest of this codebase. It would turn `RateLimit.__call__`'s parameter
# annotations into strings, and FastAPI resolves a dependency's signature
# through pydantic, which cannot evaluate `Request`/`Response` forward
# references from this module's namespace. The failure is a startup crash
# ("name 'Request' is not defined"), not a subtle bug — but it is a long way
# from the import that caused it, so it is called out here.
import logging
import math
import os
import time
import uuid

import redis
from fastapi import HTTPException, Request, Response

from app.queue.redis_client import get_client, key

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

#: Master switch. Off only for load testing against a local instance.
ENABLED = os.getenv("RATE_LIMIT_ENABLED", "1").lower() not in ("0", "false", "no")

#: How many reverse proxies sit in front of this service.
#:
#: 0 (the default) means the socket peer *is* the client, and the
#: X-Forwarded-For header is ignored entirely. That is the only safe reading
#: when nothing is in front of the app: the header is caller-supplied, so
#: honouring it would let anyone reset their own bucket by inventing an IP.
#: Set this to the real hop count when deploying behind nginx or a load
#: balancer, and the client is read that many positions from the right of the
#: header — the entries a trusted proxy appended, not the ones a caller sent.
TRUSTED_PROXY_HOPS = int(os.getenv("TRUSTED_PROXY_HOPS", "0"))


def _limit_from_env(name: str, default: str) -> tuple[int, int]:
    """Parse a "<count>/<seconds>" limit, e.g. UPLOAD_RATE_LIMIT="10/60"."""
    raw = os.getenv(name, default)
    try:
        count, window = raw.split("/", 1)
        return max(1, int(count)), max(1, int(window))
    except (ValueError, AttributeError):
        log.warning("invalid %s=%r; falling back to %s", name, raw, default)
        count, window = default.split("/", 1)
        return int(count), int(window)


#: Uploads. Each one can occupy a worker for minutes, so this is the strict
#: one. Ten a minute is far above any human operator's pace and far below what
#: it takes to starve the pool.
UPLOAD_LIMIT = _limit_from_env("UPLOAD_RATE_LIMIT", "10/60")

#: Reads. Generous, because the console legitimately polls: a dashboard with
#: several panels open refreshes queue state every 5s and case lists while
#: anything is processing.
READ_LIMIT = _limit_from_env("READ_RATE_LIMIT", "300/60")

#: Mutations other than upload (PATCH, DELETE). Cheap, but not something a
#: legitimate client does in a tight loop.
WRITE_LIMIT = _limit_from_env("WRITE_RATE_LIMIT", "60/60")

#: Event-stream opens. Distinct from the concurrent-stream cap below: this
#: bounds reconnect churn, that bounds how many stay open.
STREAM_LIMIT = _limit_from_env("STREAM_RATE_LIMIT", "30/60")


# ─────────────────────────────────────────────────────────────────────────────
# Admission script
# ─────────────────────────────────────────────────────────────────────────────

# KEYS[1] = bucket key
# ARGV    = now_ms, window_ms, limit, unique member id
# returns   {allowed, remaining, retry_after_ms}
_ADMIT_LUA = """
local bucket = KEYS[1]
local now    = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local limit  = tonumber(ARGV[3])
local member = ARGV[4]

redis.call('ZREMRANGEBYSCORE', bucket, 0, now - window)
local used = redis.call('ZCARD', bucket)

if used < limit then
  redis.call('ZADD', bucket, now, member)
  redis.call('PEXPIRE', bucket, window)
  return {1, limit - used - 1, 0}
end

-- Rejected. The caller may retry once the oldest entry leaves the window.
local oldest = redis.call('ZRANGE', bucket, 0, 0, 'WITHSCORES')
local retry = window
if oldest[2] then
  retry = (tonumber(oldest[2]) + window) - now
  if retry < 0 then retry = 0 end
end
redis.call('PEXPIRE', bucket, window)
return {0, 0, retry}
"""

_script = None


def _admit(bucket: str, limit: int, window_s: int) -> tuple[bool, int, float]:
    """(allowed, remaining, retry_after_seconds). Never raises."""
    global _script
    client = get_client()
    try:
        if _script is None:
            _script = client.register_script(_ADMIT_LUA)
        allowed, remaining, retry_ms = _script(
            keys=[bucket],
            args=[int(time.time() * 1000), window_s * 1000, limit, uuid.uuid4().hex],
            client=client,
        )
        return bool(allowed), int(remaining), float(retry_ms) / 1000.0
    except redis.RedisError as exc:
        # Fail open. A limiter outage must not become an API outage: the
        # limiter is a guard on a healthy system, not part of serving a
        # request. The expensive path is protected anyway — POST /cases
        # already returns 503 when Redis is unreachable, because it cannot
        # queue work without it.
        log.warning("rate limiter unavailable, allowing request: %s", exc)
        return True, -1, 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Client identity
# ─────────────────────────────────────────────────────────────────────────────

def client_id(request: Request) -> str:
    """Who to charge this request to.

    IP-based, which is the only identity this API has — it carries no auth
    (see app/core.py). That has known limits: callers behind one NAT share a
    bucket, and an attacker with many addresses gets many buckets. It is still
    the right control here, because the resource being protected is a small
    fixed worker pool and the realistic threat is one misbehaving client, not
    a distributed attack.
    """
    peer = request.client.host if request.client else "unknown"

    if TRUSTED_PROXY_HOPS <= 0:
        return peer

    forwarded = request.headers.get("x-forwarded-for", "")
    if not forwarded:
        return peer

    # "client, proxy1, proxy2" — rightmost entries were appended by the
    # infrastructure closest to us, so counting in from the right skips
    # exactly the hops we trust and lands on what they saw.
    hops = [h.strip() for h in forwarded.split(",") if h.strip()]
    index = len(hops) - TRUSTED_PROXY_HOPS
    if 0 <= index < len(hops):
        return hops[index]
    return hops[0] if hops else peer


# ─────────────────────────────────────────────────────────────────────────────
# Dependency
# ─────────────────────────────────────────────────────────────────────────────

class RateLimit:
    """FastAPI dependency enforcing one limit on one group of routes.

        @router.post("/cases", dependencies=[Depends(RateLimit.upload)])

    Scopes are named rather than derived from the path so that related routes
    share a bucket deliberately: every read endpoint draws on one read budget,
    instead of a caller getting a fresh allowance per URL by rotating between
    `/cases`, `/stats` and `/queue`.
    """

    def __init__(self, scope: str, limit_window: tuple[int, int]):
        self.scope = scope
        self.limit, self.window = limit_window

    async def __call__(self, request: Request, response: Response) -> None:
        if not ENABLED:
            return

        bucket = key("rl", self.scope, client_id(request))
        allowed, remaining, retry_after = _admit(bucket, self.limit, self.window)

        # -1 means the limiter was unreachable and the request was let through;
        # advertising a budget we did not actually check would be a lie.
        if remaining >= 0:
            response.headers["RateLimit-Limit"] = str(self.limit)
            response.headers["RateLimit-Remaining"] = str(remaining)
            response.headers["RateLimit-Reset"] = str(self.window)

        if allowed:
            return

        seconds = max(1, math.ceil(retry_after))
        raise HTTPException(
            status_code=429,
            detail=(
                f"Rate limit exceeded: {self.limit} request"
                f"{'' if self.limit == 1 else 's'} per {self.window}s for this "
                f"endpoint. Retry in {seconds}s."
            ),
            headers={
                "Retry-After": str(seconds),
                "RateLimit-Limit": str(self.limit),
                "RateLimit-Remaining": "0",
                "RateLimit-Reset": str(seconds),
            },
        )


#: Prebuilt dependencies, so routes name an intent rather than repeat numbers.
upload = RateLimit("upload", UPLOAD_LIMIT)
read = RateLimit("read", READ_LIMIT)
write = RateLimit("write", WRITE_LIMIT)
stream = RateLimit("stream", STREAM_LIMIT)


# ─────────────────────────────────────────────────────────────────────────────
# Concurrent stream cap
# ─────────────────────────────────────────────────────────────────────────────

#: How many SSE connections one client may hold open at once.
#:
#: Rate limiting opens is not enough on its own. Every live stream holds a
#: Redis pub/sub connection and occupies a thread from the default executor
#: (see routers/queue.py, which reads pub/sub via asyncio.to_thread), so a
#: client that opens streams and never closes them exhausts the thread pool
#: while staying comfortably inside any per-minute limit. A handful covers the
#: legitimate case of several console tabs.
MAX_CONCURRENT_STREAMS = int(os.getenv("MAX_CONCURRENT_STREAMS", "5"))


class StreamSlot:
    """Context manager holding one client's claim on a stream slot.

    Uses a plain counter with a TTL rather than a set of connection ids: a
    process killed mid-stream never runs its release, so the count has to be
    able to heal on its own. The TTL is refreshed while the stream is alive by
    the acquire path of any new connection, and a leaked slot expires rather
    than permanently costing that client capacity.
    """

    def __init__(self, request: Request):
        self.bucket = key("rl", "streams", client_id(request))
        self.held = False

    def __enter__(self) -> "StreamSlot":
        if not ENABLED:
            return self
        try:
            client = get_client()
            count = client.incr(self.bucket)
            # Re-arm on every acquire so a slot cannot outlive an hour of
            # inactivity even if a release is missed.
            client.expire(self.bucket, 3600)
            if count > MAX_CONCURRENT_STREAMS:
                client.decr(self.bucket)
                raise HTTPException(
                    status_code=429,
                    detail=(
                        f"Too many open event streams ({MAX_CONCURRENT_STREAMS} "
                        f"max). Close an existing one, or poll "
                        f"GET /api/cases/{{case_id}} instead."
                    ),
                    headers={"Retry-After": "5"},
                )
            self.held = True
        except redis.RedisError as exc:
            log.warning("stream slot accounting unavailable: %s", exc)
        return self

    def __exit__(self, *exc_info) -> None:
        if not self.held:
            return
        try:
            client = get_client()
            if client.decr(self.bucket) < 0:
                client.set(self.bucket, 0)
        except redis.RedisError as exc:
            log.warning("could not release stream slot: %s", exc)
