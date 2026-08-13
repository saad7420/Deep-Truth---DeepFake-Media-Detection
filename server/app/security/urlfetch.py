"""Server-side media download (Module 4, FE-1).

The extension used to fetch page media in the browser and upload the bytes.
Fetching it here instead saves the user's bandwidth — they never pay to
download a video only to immediately send it back up — and lets the server
reach media the page's own CSP would block.

It also hands an attacker a request forgery primitive, which is the entire
difficulty of this module. "Download this URL for me" run by a server sitting
inside a private network is the classic SSRF setup: the caller picks a URL,
our host resolves it, and anything the host can reach becomes reachable by
someone who cannot reach it themselves — cloud instance metadata at
169.254.169.254, Redis on localhost:6379, an admin panel on a 10.x address.

So every fetch below is constrained on four axes:

    scheme      http and https only. No file://, gopher://, ftp://, data:.
    address     the resolved IP must be publicly routable. Checked per hop.
    size        streamed with a running cap; the transfer is aborted mid-body
                rather than trusting Content-Length.
    time        connect and total deadlines, so a slowloris cannot pin a
                worker indefinitely.

The address check is the one that is easy to get wrong. Validating the
hostname before connecting is not enough on its own — DNS can return a private
address (rebinding), and a redirect can move a public URL onto an internal one.
Redirects are therefore followed manually, one at a time, with the destination
re-resolved and re-validated at every hop.
"""
from __future__ import annotations

import ipaddress
import logging
import os
import socket
from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx

from app.security.media import SNIFF_BYTES, detect, validate_signature

log = logging.getLogger(__name__)

# Matches MAX_FILE_SIZE_MB in services/analyser.py so a URL cannot smuggle in
# something larger than a direct upload is allowed to be.
MAX_DOWNLOAD_MB = int(os.getenv("MAX_FILE_SIZE_MB", "500"))
MAX_DOWNLOAD_BYTES = MAX_DOWNLOAD_MB * 1024 * 1024

#: Smallest plausible media file. The signature check only reads the first few
#: bytes, so a response consisting of nothing but a valid magic number passes
#: it — an origin declaring `Content-Length: 10` and sending `\xff\xd8\xff...`
#: yields a "JPEG" that no decoder will open. Real media carries headers and at
#: least one coded frame; nothing legitimate is this small.
MIN_DOWNLOAD_BYTES = int(os.getenv("URL_FETCH_MIN_BYTES", "256"))

CONNECT_TIMEOUT = float(os.getenv("URL_FETCH_CONNECT_TIMEOUT", "10"))
READ_TIMEOUT = float(os.getenv("URL_FETCH_READ_TIMEOUT", "30"))
TOTAL_TIMEOUT = float(os.getenv("URL_FETCH_TOTAL_TIMEOUT", "300"))
MAX_REDIRECTS = int(os.getenv("URL_FETCH_MAX_REDIRECTS", "5"))

#: Opt-in escape hatch for deployments that genuinely host media internally
#: (a lab fileserver on a private range). Off by default: the safe posture has
#: to be the one you get without reading this file.
ALLOW_PRIVATE_ADDRESSES = os.getenv(
    "URL_FETCH_ALLOW_PRIVATE", "0"
).lower() in ("1", "true", "yes")

#: Optional host allowlist, comma-separated. When set, nothing else is
#: fetchable regardless of the checks below — the strongest control available,
#: and worth using if the extension only ever needs a handful of CDNs.
_raw_allowlist = os.getenv("URL_FETCH_ALLOWED_HOSTS", "").strip()
ALLOWED_HOSTS = {h.strip().lower() for h in _raw_allowlist.split(",") if h.strip()}

# A browser-ish UA. Many CDNs serve 403 to unrecognised agents, and this is a
# user-initiated fetch of media the user is already looking at.
USER_AGENT = os.getenv(
    "URL_FETCH_USER_AGENT",
    "Mozilla/5.0 (compatible; DeepTruth/1.0; +forensic-media-analysis)",
)


class UrlRejected(ValueError):
    """The URL is not one this service will fetch. Never retried."""


class UrlFetchFailed(RuntimeError):
    """The URL looked acceptable but the transfer failed. May be transient."""


# ─────────────────────────────────────────────────────────────────────────────
# Address validation
# ─────────────────────────────────────────────────────────────────────────────

def _is_public(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Whether an address is safe to connect to on a caller's behalf.

    `is_global` covers most of this, but it is checked alongside the specific
    categories rather than alone: the named properties document *what* is being
    excluded, and they are what a reader needs to confirm the metadata endpoint
    (link-local) and localhost (loopback) are handled.
    """
    if ip.is_private or ip.is_loopback or ip.is_link_local:
        return False
    if ip.is_multicast or ip.is_reserved or ip.is_unspecified:
        return False
    # IPv4-mapped and 6to4 addresses can wrap a private v4 address in a v6 one
    # that none of the properties above flag; unwrap and re-check.
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped and not _is_public(ip.ipv4_mapped):
            return False
        if ip.sixtofour and not _is_public(ip.sixtofour):
            return False
    return ip.is_global


def _resolve(host: str, port: int) -> list[str]:
    """Every address `host` resolves to. Raises UrlRejected if it resolves to
    nothing usable."""
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise UrlRejected(f"Could not resolve '{host}': {exc}") from exc
    return list({info[4][0] for info in infos})


def validate_url(raw: str) -> str:
    """Check a URL is one we are willing to fetch, and return it normalised.

    Raises UrlRejected with a message safe to show a user. Called both at
    submit time — so an obviously bad URL fails immediately rather than after a
    trip through the queue — and again per redirect hop during the download,
    because neither DNS nor the redirect chain is stable between those points.
    """
    if not raw or not raw.strip():
        raise UrlRejected("No media URL was provided.")

    try:
        parts = urlsplit(raw.strip())
    except ValueError as exc:
        raise UrlRejected(f"Malformed URL: {exc}") from exc

    scheme = parts.scheme.lower()
    if scheme not in ("http", "https"):
        raise UrlRejected(
            f"Only http and https URLs can be fetched by the server "
            f"(got '{scheme or 'no scheme'}'). Upload the file directly instead."
        )

    host = parts.hostname
    if not host:
        raise UrlRejected("URL has no host.")

    if ALLOWED_HOSTS and host.lower() not in ALLOWED_HOSTS:
        raise UrlRejected(f"Host '{host}' is not on this server's allowlist.")

    port = parts.port or (443 if scheme == "https" else 80)

    # A literal IP skips DNS but not the address check.
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None

    if literal is not None:
        addresses = [str(literal)]
    else:
        addresses = _resolve(host, port)

    if not addresses:
        raise UrlRejected(f"'{host}' did not resolve to any address.")

    if not ALLOW_PRIVATE_ADDRESSES:
        for addr in addresses:
            ip = ipaddress.ip_address(addr)
            if not _is_public(ip):
                # Deliberately does not say which address, or that others were
                # fine: a precise answer here is a working port scanner for
                # internal networks, one URL at a time.
                raise UrlRejected(
                    f"'{host}' resolves to an address this server will not "
                    f"fetch from. Only publicly routable hosts are allowed."
                )

    return parts.geturl()


# ─────────────────────────────────────────────────────────────────────────────
# Download
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FetchResult:
    data: bytes
    final_url: str
    content_type: str
    detected_format: str


def fetch(url: str, media_type: str) -> FetchResult:
    """Download `url`, enforcing every constraint in the module docstring.

    Synchronous on purpose: this runs inside a Celery worker, where blocking is
    free and the retry policy already handles transient network failure. Doing
    it in the API process would tie up the event loop for the length of a
    transfer the caller does not control.

    Raises UrlRejected (permanent — bad URL, wrong content, too large) or
    UrlFetchFailed (possibly transient — timeout, connection reset, 5xx).
    """
    current = validate_url(url)
    seen: list[str] = []

    timeout = httpx.Timeout(TOTAL_TIMEOUT, connect=CONNECT_TIMEOUT, read=READ_TIMEOUT)
    headers = {"User-Agent": USER_AGENT, "Accept": "*/*"}

    # follow_redirects is off so each hop can be validated. httpx would happily
    # follow a 302 from a public host to http://169.254.169.254/ otherwise, and
    # every check above would have been for nothing.
    with httpx.Client(timeout=timeout, follow_redirects=False,
                      headers=headers, http2=False) as client:
        for hop in range(MAX_REDIRECTS + 1):
            seen.append(current)
            try:
                with client.stream("GET", current) as response:
                    if response.is_redirect:
                        location = response.headers.get("location")
                        if not location:
                            raise UrlFetchFailed(
                                f"Server returned {response.status_code} with no "
                                f"redirect target."
                            )
                        nxt = str(response.url.join(location))
                        if nxt in seen:
                            raise UrlRejected("Redirect loop.")
                        # The whole point of the manual loop.
                        current = validate_url(nxt)
                        continue

                    if response.status_code == 404:
                        raise UrlRejected(
                            "The media is no longer available at that URL (404)."
                        )
                    if response.status_code in (401, 403):
                        raise UrlRejected(
                            "That media requires authentication the server does "
                            "not have. It has to be uploaded from the browser "
                            "instead."
                        )
                    if response.status_code >= 500:
                        raise UrlFetchFailed(
                            f"The origin server returned {response.status_code}."
                        )
                    if response.status_code != 200:
                        raise UrlRejected(
                            f"Unexpected response {response.status_code} for that URL."
                        )

                    # Content-Length is a hint, not a guarantee, so it is used
                    # only to fail early on an obviously oversized transfer.
                    declared = response.headers.get("content-length")
                    if declared and declared.isdigit() and int(declared) > MAX_DOWNLOAD_BYTES:
                        raise UrlRejected(
                            f"That media is {int(declared) / 1e6:.0f} MB; the "
                            f"limit is {MAX_DOWNLOAD_MB} MB."
                        )

                    chunks: list[bytes] = []
                    total = 0
                    for chunk in response.iter_bytes(chunk_size=1 << 16):
                        total += len(chunk)
                        if total > MAX_DOWNLOAD_BYTES:
                            # Abort mid-body. A server that lies about (or
                            # omits) Content-Length cannot make us buffer more
                            # than the cap.
                            raise UrlRejected(
                                f"That media exceeds the {MAX_DOWNLOAD_MB} MB limit."
                            )
                        chunks.append(chunk)

                    data = b"".join(chunks)
                    content_type = (
                        response.headers.get("content-type", "").split(";")[0].strip()
                    )
                    final_url = str(response.url)

            except httpx.TimeoutException as exc:
                raise UrlFetchFailed(f"Timed out fetching the media: {exc}") from exc
            except httpx.HTTPError as exc:
                raise UrlFetchFailed(f"Could not fetch the media: {exc}") from exc

            # ── Validate what actually arrived ──────────────────────────────
            if not data:
                raise UrlRejected("That URL returned an empty response.")

            # A body shorter than it promised means the connection dropped
            # mid-transfer. That is a transport failure, not a bad URL, so it
            # is retryable — unlike everything else rejected in this block.
            if declared and declared.isdigit() and total < int(declared):
                raise UrlFetchFailed(
                    f"Transfer truncated: got {total} bytes of {declared}."
                )

            if total < MIN_DOWNLOAD_BYTES:
                raise UrlRejected(
                    f"That URL returned only {total} bytes, too small to be "
                    f"real media. The origin may be serving a placeholder or "
                    f"an error page."
                )

            # The origin's Content-Type is no more trustworthy than a client's,
            # so the bytes decide. Same check a direct upload goes through.
            error = validate_signature(media_type, data)
            if error:
                raise UrlRejected(error)

            return FetchResult(
                data=data,
                final_url=final_url,
                content_type=content_type,
                detected_format=detect(data[:SNIFF_BYTES]) or "unknown",
            )

    raise UrlRejected(f"Too many redirects (more than {MAX_REDIRECTS}).")
