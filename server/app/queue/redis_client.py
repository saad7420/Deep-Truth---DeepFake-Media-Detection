from __future__ import annotations

import logging
import os

import redis

log = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")

NS = os.getenv("REDIS_NAMESPACE", "dt")

_pool: redis.ConnectionPool | None = None


def get_client() -> redis.Redis:
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
    pass


def ping(timeout_note: str = "") -> bool:
    try:
        return bool(get_client().ping())
    except redis.RedisError as exc:
        log.warning("redis unreachable at %s: %s %s", REDIS_URL, exc, timeout_note)
        return False


# answer 503 instead of silently running without the queue
def require() -> redis.Redis:
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
    return ":".join((NS, *parts))
