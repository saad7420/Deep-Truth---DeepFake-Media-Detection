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

MAX_DOWNLOAD_MB = int(os.getenv("MAX_FILE_SIZE_MB", "500"))
MAX_DOWNLOAD_BYTES = MAX_DOWNLOAD_MB * 1024 * 1024

# a bare magic number with no content is not real media
MIN_DOWNLOAD_BYTES = int(os.getenv("URL_FETCH_MIN_BYTES", "256"))

CONNECT_TIMEOUT = float(os.getenv("URL_FETCH_CONNECT_TIMEOUT", "10"))
READ_TIMEOUT = float(os.getenv("URL_FETCH_READ_TIMEOUT", "30"))
TOTAL_TIMEOUT = float(os.getenv("URL_FETCH_TOTAL_TIMEOUT", "300"))
MAX_REDIRECTS = int(os.getenv("URL_FETCH_MAX_REDIRECTS", "5"))

ALLOW_PRIVATE_ADDRESSES = os.getenv(
    "URL_FETCH_ALLOW_PRIVATE", "0"
).lower() in ("1", "true", "yes")

_raw_allowlist = os.getenv("URL_FETCH_ALLOWED_HOSTS", "").strip()
ALLOWED_HOSTS = {h.strip().lower() for h in _raw_allowlist.split(",") if h.strip()}

USER_AGENT = os.getenv(
    "URL_FETCH_USER_AGENT",
    "Mozilla/5.0 (compatible; DeepTruth/1.0; +forensic-media-analysis)",
)


class UrlRejected(ValueError):
    pass


class UrlFetchFailed(RuntimeError):
    pass


# blocks localhost, private ranges and the cloud metadata address
def _is_public(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if ip.is_private or ip.is_loopback or ip.is_link_local:
        return False
    if ip.is_multicast or ip.is_reserved or ip.is_unspecified:
        return False
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped and not _is_public(ip.ipv4_mapped):
            return False
        if ip.sixtofour and not _is_public(ip.sixtofour):
            return False
    return ip.is_global


def _resolve(host: str, port: int) -> list[str]:
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise UrlRejected(f"Could not resolve '{host}': {exc}") from exc
    return list({info[4][0] for info in infos})


def validate_url(raw: str) -> str:
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
                raise UrlRejected(
                    f"'{host}' resolves to an address this server will not "
                    f"fetch from. Only publicly routable hosts are allowed."
                )

    return parts.geturl()


@dataclass
class FetchResult:
    data: bytes
    final_url: str
    content_type: str
    detected_format: str


def fetch(url: str, media_type: str) -> FetchResult:
    current = validate_url(url)
    seen: list[str] = []

    timeout = httpx.Timeout(TOTAL_TIMEOUT, connect=CONNECT_TIMEOUT, read=READ_TIMEOUT)
    headers = {"User-Agent": USER_AGENT, "Accept": "*/*"}

    # redirects are followed by hand so each hop can be validated
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
                        # re-check every redirect hop, not just the first url
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

            if not data:
                raise UrlRejected("That URL returned an empty response.")

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
