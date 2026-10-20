import logging
import math
import os
import time
import uuid

import redis
from fastapi import HTTPException, Request, Response

from app.queue.redis_client import get_client, key

log = logging.getLogger(__name__)


ENABLED = os.getenv("RATE_LIMIT_ENABLED", "1").lower() not in ("0", "false", "no")

TRUSTED_PROXY_HOPS = int(os.getenv("TRUSTED_PROXY_HOPS", "0"))


def _limit_from_env(name: str, default: str) -> tuple[int, int]:
    raw = os.getenv(name, default)
    try:
        count, window = raw.split("/", 1)
        return max(1, int(count)), max(1, int(window))
    except (ValueError, AttributeError):
        log.warning("invalid %s=%r; falling back to %s", name, raw, default)
        count, window = default.split("/", 1)
        return int(count), int(window)


UPLOAD_LIMIT = _limit_from_env("UPLOAD_RATE_LIMIT", "10/60")

READ_LIMIT = _limit_from_env("READ_RATE_LIMIT", "300/60")

WRITE_LIMIT = _limit_from_env("WRITE_RATE_LIMIT", "60/60")

STREAM_LIMIT = _limit_from_env("STREAM_RATE_LIMIT", "30/60")


# one atomic lua script so concurrent requests cannot slip past the limit
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


# fails open: a limiter outage must not become an api outage
def _admit(bucket: str, limit: int, window_s: int) -> tuple[bool, int, float]:
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
        log.warning("rate limiter unavailable, allowing request: %s", exc)
        return True, -1, 0.0


# ignore X-Forwarded-For unless we trust the proxy in front
def client_id(request: Request) -> str:
    peer = request.client.host if request.client else "unknown"

    if TRUSTED_PROXY_HOPS <= 0:
        return peer

    forwarded = request.headers.get("x-forwarded-for", "")
    if not forwarded:
        return peer

    hops = [h.strip() for h in forwarded.split(",") if h.strip()]
    index = len(hops) - TRUSTED_PROXY_HOPS
    if 0 <= index < len(hops):
        return hops[index]
    return hops[0] if hops else peer


class RateLimit:

    def __init__(self, scope: str, limit_window: tuple[int, int]):
        self.scope = scope
        self.limit, self.window = limit_window

    async def __call__(self, request: Request, response: Response) -> None:
        if not ENABLED:
            return

        bucket = key("rl", self.scope, client_id(request))
        allowed, remaining, retry_after = _admit(bucket, self.limit, self.window)

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


upload = RateLimit("upload", UPLOAD_LIMIT)
read = RateLimit("read", READ_LIMIT)
write = RateLimit("write", WRITE_LIMIT)
stream = RateLimit("stream", STREAM_LIMIT)


MAX_CONCURRENT_STREAMS = int(os.getenv("MAX_CONCURRENT_STREAMS", "5"))


class StreamSlot:

    def __init__(self, request: Request):
        self.bucket = key("rl", "streams", client_id(request))
        self.held = False

    def __enter__(self) -> "StreamSlot":
        if not ENABLED:
            return self
        try:
            client = get_client()
            count = client.incr(self.bucket)
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
