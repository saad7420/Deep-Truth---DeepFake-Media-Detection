"""Content sniffing for uploads.

`validate_file` in services/analyser.py checks the `Content-Type` the *client
sent*. That header is not evidence of anything — it is a string chosen by the
caller, and the extension already sets it from `blob.type`, which browsers
derive from the server's response header on the original page. So a file can
declare `image/jpeg` and be a zip, a 500 MB block of zeros, or an ELF binary,
and it passes.

The cost of trusting it is not a security breach — nothing here executes the
file — but it is real: undecodable content reaches a worker, occupies a slot
for the length of a model load, and comes back "inconclusive", which reads to
an operator like a considered verdict rather than "this was never an image".
The queue's own test run produced exactly that result from 40 KB of urandom.

So the first bytes are checked against the container signatures for the formats
the pipeline actually supports, and a mismatch is refused at the gateway with
an error naming what the file really appears to be.

Deliberately signature-only. Detecting *format* is a fixed, well-documented
byte comparison; detecting whether a stream will fully decode is the decoder's
job, and the engines already handle that path.
"""
from __future__ import annotations

# How much of the file the checks below need. The furthest any signature
# reaches is the ISO-BMFF brand at byte 12, plus room for RIFF/EBML variants.
SNIFF_BYTES = 64


def _iso_bmff_brand(head: bytes) -> str | None:
    """Major brand of an ISO base media file (MP4, MOV, M4A) or None.

    Layout is [4-byte box size][b'ftyp'][4-byte major brand], so the brand is
    what separates an .mp4 from a .mov from an .m4a — all three share the
    container and differ only here.
    """
    if len(head) < 12 or head[4:8] != b"ftyp":
        return None
    return head[8:12].decode("ascii", errors="replace")


def detect(head: bytes) -> str | None:
    """Best-effort format label for a file's first bytes.

    Returns one of the pipeline's accepted formats ("jpeg", "png", "mp4", ...),
    or None when nothing matches.
    """
    if len(head) < 4:
        return None

    # ── Images ──────────────────────────────────────────────────────────────
    if head[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if head[:2] == b"BM":
        return "bmp"
    if head[:4] in (b"II*\x00", b"MM\x00*"):
        return "tiff"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp"

    # ── RIFF family: WAV and AVI share the container ────────────────────────
    if head[:4] == b"RIFF":
        if head[8:12] == b"WAVE":
            return "wav"
        if head[8:12] == b"AVI ":
            return "avi"

    # ── Audio ───────────────────────────────────────────────────────────────
    if head[:4] == b"fLaC":
        return "flac"
    if head[:4] == b"OggS":
        # Ogg carries Vorbis/Opus audio and, rarely, Theora video. The
        # pipeline accepts it only as audio, so it is labelled as such.
        return "ogg"
    if head[:3] == b"ID3":
        return "mp3"
    # MPEG audio frame sync: 11 set bits. Catches MP3s with no ID3 tag.
    if len(head) >= 2 and head[0] == 0xFF and (head[1] & 0xE0) == 0xE0:
        return "mp3"

    # ── Matroska / WebM ─────────────────────────────────────────────────────
    if head[:4] == b"\x1a\x45\xdf\xa3":
        # Both use EBML; the doctype string appears within the first header.
        return "webm" if b"webm" in head[:SNIFF_BYTES] else "mkv"

    # ── ISO base media (MP4 / MOV / M4A) ────────────────────────────────────
    brand = _iso_bmff_brand(head)
    if brand is not None:
        if brand.startswith("qt"):
            return "mov"
        if brand in ("M4A ", "M4B "):
            return "m4a"
        return "mp4"

    return None


#: Which detected formats are legitimate for each declared modality. Mirrors
#: ALLOWED_TYPES in services/analyser.py, expressed in container terms.
ALLOWED_FORMATS: dict[str, set[str]] = {
    "image": {"jpeg", "png", "webp", "bmp", "tiff"},
    "video": {"mp4", "mov", "avi", "webm", "mkv"},
    "audio": {"mp3", "wav", "flac", "ogg", "m4a"},
}

#: Human-readable names for the error message.
_FRIENDLY = {
    "jpeg": "a JPEG image", "png": "a PNG image", "webp": "a WebP image",
    "bmp": "a BMP image", "tiff": "a TIFF image",
    "mp4": "an MP4 video", "mov": "a QuickTime video", "avi": "an AVI video",
    "webm": "a WebM video", "mkv": "a Matroska video",
    "mp3": "an MP3 audio file", "wav": "a WAV audio file",
    "flac": "a FLAC audio file", "ogg": "an Ogg audio file",
    "m4a": "an M4A audio file",
}


def validate_signature(media_type: str, data: bytes) -> str | None:
    """Error message if the bytes are not `media_type`, else None.

    Cross-modality mismatches are reported specifically ("this is an MP4
    video, submitted as image") because that one is usually an honest mistake
    in a client, and a precise message saves the caller guessing.
    """
    allowed = ALLOWED_FORMATS.get(media_type)
    if allowed is None:
        return f"Unknown media_type '{media_type}'. Must be image, video, or audio."

    detected = detect(data[:SNIFF_BYTES])

    if detected is None:
        return (
            "This file's contents do not match any supported media format. "
            "It may be corrupt, empty, or not a media file at all — the "
            "declared type is not enough on its own."
        )

    if detected in allowed:
        return None

    for other, formats in ALLOWED_FORMATS.items():
        if detected in formats:
            return (
                f"This file is {_FRIENDLY.get(detected, detected)}, but it was "
                f"submitted as {media_type}. Resubmit it with "
                f"media_type={other}."
            )

    return (
        f"{_FRIENDLY.get(detected, detected).capitalize()} is not accepted for "
        f"{media_type}."
    )
