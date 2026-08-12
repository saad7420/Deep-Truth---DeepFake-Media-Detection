"""Redis connection management.

One pool per process, created lazily. Both the FastAPI process and every
Celery worker import this module, so it must be safe to import when Redis is
down — construction never connects, only the first command does.

`REDIS_URL` is the single knob. It is also what celery_app.py derives the
broker and result-backend URLs from, so a deployment points one variable at
its Redis and everything follows.
"""
from __future__ import annotations

import logging
import os

import redis

log = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")

# Namespace for every key this application writes, so a shared Redis can host
# other tenants without collision and `DEL dt:*` cleans up exactly ours.
NS = os.getenv("REDIS_NAMESPACE", "dt")

_pool: redis.ConnectionPool | None = None


def get_client() -> redis.Redis:
    """Shared client. `decode_responses=True` so every read is `str`, never
    `bytes` — the cache stores JSON and the state hashes store plain strings,
    and mixing the two representations is a reliable source of subtle bugs.
    """
    global _pool
    if _pool is None:
        _pool = redis.ConnectionPool.from_url(
            REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=5,
            health_check_interval=30,
            retry_on_timeout=True,
        )
    return redis.Redis(connection_pool=_pool)


class BrokerUnavailable(RuntimeError):
    """Redis could not be reached.

    The API turns this into a 503 rather than falling back to running the
    analysis inline. An inline fallback would look like it worked while
    quietly abandoning the ordering, parallelism and retry guarantees the
    queue exists to provide, and it would block the event loop for minutes
    per request. Failing loudly means work waits for the broker to come back
    instead of being silently downgraded.
    """


def ping(timeout_note: str = "") -> bool:
    """True when Redis answers. Never raises — callers branch on the bool."""
    try:
        return bool(get_client().ping())
    except redis.RedisError as exc:
        log.warning("redis unreachable at %s: %s %s", REDIS_URL, exc, timeout_note)
        return False


def require() -> redis.Redis:
    """Client, or `BrokerUnavailable` if Redis is not answering."""
    client = get_client()
    try:
        client.ping()
    except redis.RedisError as exc:
        raise BrokerUnavailable(
            f"Redis is not reachable at {REDIS_URL}. "
            "Start it (`sudo systemctl start redis-server`) and retry."
        ) from exc
    return client


def key(*parts: str) -> str:
    """Build a namespaced key: key('job', case_id) -> 'dt:job:<case_id>'."""
    return ":".join((NS, *parts))
