"""Hot-result cache.

The expensive thing in this system is inference: eight ViT adapters over an
image, six ViViT adapters over a video, on CPU. The cheap thing is a sha256 of
the bytes. So before anything is queued, the bytes are hashed and Redis is
asked whether this exact content has ever been analysed. If it has, the stored
verdict is replayed onto the new case and no worker is involved at all.

Two key spaces:

    dt:cache:res:<version>:<media_type>:<sha256>   the stored verdict
    dt:cache:url:<version>:<sha1(normalised url)>  url -> sha256

The URL space exists for the browser extension, which analyses media it finds
on a page. The same image served from the same URL usually *is* the same
bytes, but not always (CDN re-encoding, watermarking), so a URL never returns
a verdict directly — it resolves to a content hash, and that hash is then
looked up in the real cache. A URL whose bytes changed simply misses, which is
the correct outcome rather than a stale answer.

Invalidation is by `version`, not by expiry. A cached verdict is only wrong if
the models that produced it changed, and that is a deploy-time event, so
bumping DEEPTRUTH_CACHE_VERSION retires every entry at once. Entries do not
otherwise expire: a file analysed a year ago has the same content hash today
and the same models still produce the same answer.
"""
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

# Bump this whenever a checkpoint, ensemble policy, or threshold changes.
# Every previously cached verdict becomes unreachable in one step.
CACHE_VERSION = os.getenv("DEEPTRUTH_CACHE_VERSION", "v1")

# 0 (the default) means entries never expire. See the module docstring: the
# version prefix is the invalidation mechanism, not time.
CACHE_TTL = int(os.getenv("DEEPTRUTH_CACHE_TTL", "0"))

_CHUNK = 1 << 20

# Query parameters that identify the *click*, not the *resource*. Leaving them
# in would make the same image cache-miss for every visitor who arrived from a
# different campaign link.
_TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "utm_id", "utm_source_platform", "fbclid", "gclid", "dclid", "msclkid",
    "igshid", "mc_cid", "mc_eid", "ref", "ref_src", "spm", "_ga",
}

HITS_KEY = key("cache", "stats", "hits")
MISSES_KEY = key("cache", "stats", "misses")


# ─────────────────────────────────────────────────────────────────────────────
# Keys
# ─────────────────────────────────────────────────────────────────────────────

def content_hash(data: bytes) -> str:
    """sha256 of the raw bytes. Deliberately the full digest, not the 32-char
    truncation `storage.media_key_from_bytes` uses — that one keys a local
    preprocessing directory, this one keys a verdict that is served to users,
    and a collision here would show one file's result for another's."""
    return hashlib.sha256(data).hexdigest()


def content_hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def normalise_url(raw: str) -> str | None:
    """Reduce a URL to the resource it names.

    Lowercases scheme and host, drops the fragment (never sent to a server, so
    it cannot affect the bytes), strips tracking parameters, and sorts what
    remains so parameter order does not fork the cache. Returns None for
    anything that is not a usable absolute http(s) URL — including the
    `blob:` and `data:` URLs the extension can encounter, which are per-page
    identifiers with no cross-session meaning.
    """
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


# ─────────────────────────────────────────────────────────────────────────────
# Payload shape
# ─────────────────────────────────────────────────────────────────────────────

def build_payload(*, media_type: str, risk: float, likelihood: float,
                  status: str, rows: list[dict], source_case_id: str,
                  source_case_ref: str = "") -> dict:
    """Freeze an analysis outcome into something replayable.

    `id` and `case_id` are stripped from every row: they belong to the case
    that happened to be analysed first, and replaying them onto a second case
    would either collide on the analysis_results primary key or attach rows to
    the wrong case. They are regenerated on every hit by `rehydrate_rows`.

    Both ids of the originating case are kept, and they are not
    interchangeable. `sourceCaseId` is the internal uuid — useful for tracing
    in the database. `sourceCaseRef` is the public CASE-XXXXXXXX id, and is
    the only one a client can actually navigate to, which is what lets the
    extension link a cache hit back to the case that produced the verdict.
    """
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
        "computedAt": time.time(),
        "version": CACHE_VERSION,
    }


def rehydrate_rows(payload: dict, case_db_id: str) -> list[dict]:
    """Turn a cached payload back into insertable analysis_results rows for a
    different case."""
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


# ─────────────────────────────────────────────────────────────────────────────
# Lookup / store
# ─────────────────────────────────────────────────────────────────────────────

def lookup(media_type: str, sha: str) -> dict | None:
    """The stored verdict for this exact content, or None.

    Never raises. A Redis hiccup on the read path must degrade to a cache
    miss — recomputing is slow but correct, whereas failing the request would
    turn a cache outage into an outage.
    """
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


def lookup_by_url(media_type: str, url: str) -> tuple[str | None, dict | None]:
    """Resolve a URL to (content_hash, verdict).

    A URL we have seen before yields its content hash even when the verdict is
    gone, which is still useful: it tells the caller these bytes were
    previously fetched from here. The verdict is only returned when the real
    content-keyed entry is present.
    """
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


def store(media_type: str, sha: str, payload: dict, *,
          source_url: str | None = None) -> None:
    """Persist a verdict, and optionally remember which URL produced it.

    Best-effort by design: this runs after the analysis has already been
    written to SQLite, so failing to cache costs a recomputation next time and
    nothing else. It must not turn a completed job into a failed one.
    """
    client = get_client()
    blob = json.dumps(payload)
    try:
        if CACHE_TTL > 0:
            client.setex(_result_key(media_type, sha), CACHE_TTL, blob)
        else:
            client.set(_result_key(media_type, sha), blob)

        normalised = normalise_url(source_url or "")
        if normalised:
            # The URL alias tracks the *latest* bytes seen at that URL. If a
            # CDN re-encodes the image, the next visitor's fetch produces a
            # new hash and this pointer moves — which is what keeps the URL
            # path from ever serving a verdict for content that no longer
            # lives there.
            if CACHE_TTL > 0:
                client.setex(_url_key(normalised), CACHE_TTL, sha)
            else:
                client.set(_url_key(normalised), sha)
    except redis.RedisError as exc:
        log.warning("could not cache result for %s: %s", sha[:12], exc)


def remember_url(url: str, sha: str) -> None:
    """Record url -> content hash without having a verdict yet.

    Called at upload time so that even a job which later fails has taught the
    system what bytes that URL serves.
    """
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


# ─────────────────────────────────────────────────────────────────────────────
# Observability
# ─────────────────────────────────────────────────────────────────────────────

def _bump(client: redis.Redis, k: str) -> None:
    try:
        client.incr(k)
    except redis.RedisError:
        pass


def stats() -> dict[str, Any]:
    """Hit/miss counters and the number of live entries, for /api/queue."""
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
