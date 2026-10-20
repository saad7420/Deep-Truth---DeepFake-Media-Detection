from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import redis

from app.queue.redis_client import get_client, key

log = logging.getLogger(__name__)

# bump this to invalidate every cached verdict
CACHE_VERSION = os.getenv("DEEPTRUTH_CACHE_VERSION", "v1")

CACHE_TTL = int(os.getenv("DEEPTRUTH_CACHE_TTL", "0"))

_CHUNK = 1 << 20

_TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "utm_id", "utm_source_platform", "fbclid", "gclid", "dclid", "msclkid",
    "igshid", "mc_cid", "mc_eid", "ref", "ref_src", "spm", "_ga",
}

HITS_KEY = key("cache", "stats", "hits")
MISSES_KEY = key("cache", "stats", "misses")


def content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def content_hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


# drop tracking params so the same media gets one key
def normalise_url(raw: str) -> str | None:
    if not raw or not raw.strip():
        return None
    try:
        parts = urlsplit(raw.strip())
    except ValueError:
        return None
    if parts.scheme.lower() not in ("http", "https") or not parts.netloc:
        return None

    query = urlencode(
        sorted((k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
               if k.lower() not in _TRACKING_PARAMS)
    )
    path = parts.path or "/"
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, query, ""))


def _result_key(media_type: str, sha: str) -> str:
    return key("cache", "res", CACHE_VERSION, media_type, sha)


def _url_key(normalised: str) -> str:
    return key("cache", "url", CACHE_VERSION,
               hashlib.sha1(normalised.encode("utf-8")).hexdigest())


def build_payload(*, media_type: str, risk: float, likelihood: float,
                  status: str, rows: list[dict], source_case_id: str,
                  source_case_ref: str = "",
                  source_file: dict | None = None) -> dict:
    return {
        "mediaType": media_type,
        "risk": risk,
        "likelihood": likelihood,
        "status": status,
        "rows": [
            {
                "model_name": r["model_name"],
                "confidence": r["confidence"],
                "label": r["label"],
                "details": r.get("details"),
            }
            for r in rows
        ],
        "sourceCaseId": source_case_id,
        "sourceCaseRef": source_case_ref,
        "sourceFile": source_file or None,
        "computedAt": time.time(),
        "version": CACHE_VERSION,
    }


def rehydrate_rows(payload: dict, case_db_id: str) -> list[dict]:
    rows: list[dict] = []
    for r in payload.get("rows", []):
        rows.append({
            "id": str(uuid.uuid4()),
            "case_id": case_db_id,
            "model_name": r["model_name"],
            "confidence": r["confidence"],
            "label": r["label"],
            "details": r.get("details"),
        })
    return rows


def lookup(media_type: str, sha: str) -> dict | None:
    client = get_client()
    try:
        raw = client.get(_result_key(media_type, sha))
    except redis.RedisError as exc:
        log.warning("cache lookup failed for %s: %s", sha[:12], exc)
        return None

    if not raw:
        _bump(client, MISSES_KEY)
        return None

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        log.warning("discarding corrupt cache entry %s", sha[:12])
        try:
            client.delete(_result_key(media_type, sha))
        except redis.RedisError:
            pass
        return None

    _bump(client, HITS_KEY)
    return payload


# a url only resolves to a hash, never straight to a verdict
def lookup_by_url(media_type: str, url: str) -> tuple[str | None, dict | None]:
    normalised = normalise_url(url)
    if not normalised:
        return (None, None)

    client = get_client()
    try:
        sha = client.get(_url_key(normalised))
    except redis.RedisError as exc:
        log.warning("url cache lookup failed: %s", exc)
        return (None, None)

    if not sha:
        return (None, None)
    return (sha, lookup(media_type, sha))


# best effort: a cache failure must not fail the job
def store(media_type: str, sha: str, payload: dict, *,
          source_url: str | None = None) -> None:
    client = get_client()
    blob = json.dumps(payload)
    try:
        if CACHE_TTL > 0:
            client.setex(_result_key(media_type, sha), CACHE_TTL, blob)
        else:
            client.set(_result_key(media_type, sha), blob)

        normalised = normalise_url(source_url or "")
        if normalised:
            if CACHE_TTL > 0:
                client.setex(_url_key(normalised), CACHE_TTL, sha)
            else:
                client.set(_url_key(normalised), sha)
    except redis.RedisError as exc:
        log.warning("could not cache result for %s: %s", sha[:12], exc)


def remember_url(url: str, sha: str) -> None:
    normalised = normalise_url(url)
    if not normalised:
        return
    try:
        client = get_client()
        if CACHE_TTL > 0:
            client.setex(_url_key(normalised), CACHE_TTL, sha)
        else:
            client.set(_url_key(normalised), sha)
    except redis.RedisError as exc:
        log.warning("could not record url alias: %s", exc)


def invalidate(media_type: str, sha: str) -> None:
    try:
        get_client().delete(_result_key(media_type, sha))
    except redis.RedisError as exc:
        log.warning("could not invalidate %s: %s", sha[:12], exc)


def _bump(client: redis.Redis, k: str) -> None:
    try:
        client.incr(k)
    except redis.RedisError:
        pass


def stats() -> dict[str, Any]:
    client = get_client()
    try:
        hits = int(client.get(HITS_KEY) or 0)
        misses = int(client.get(MISSES_KEY) or 0)
        entries = 0
        pattern = key("cache", "res", CACHE_VERSION, "*")
        for _ in client.scan_iter(match=pattern, count=500):
            entries += 1
    except redis.RedisError:
        return {"hits": 0, "misses": 0, "entries": 0,
                "hitRate": 0.0, "version": CACHE_VERSION}

    total = hits + misses
    return {
        "hits": hits,
        "misses": misses,
        "entries": entries,
        "hitRate": round(hits / total, 3) if total else 0.0,
        "version": CACHE_VERSION,
    }
