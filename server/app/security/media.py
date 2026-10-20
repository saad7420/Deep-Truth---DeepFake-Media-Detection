from __future__ import annotations

SNIFF_BYTES = 64


def _iso_bmff_brand(head: bytes) -> str | None:
    if len(head) < 12 or head[4:8] != b"ftyp":
        return None
    return head[8:12].decode("ascii", errors="replace")


# compare the first bytes with known file signatures
def detect(head: bytes) -> str | None:
    if len(head) < 4:
        return None

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

    if head[:4] == b"RIFF":
        if head[8:12] == b"WAVE":
            return "wav"
        if head[8:12] == b"AVI ":
            return "avi"

    if head[:4] == b"fLaC":
        return "flac"
    if head[:4] == b"OggS":
        return "ogg"
    if head[:3] == b"ID3":
        return "mp3"
    if len(head) >= 2 and head[0] == 0xFF and (head[1] & 0xE0) == 0xE0:
        return "mp3"

    if head[:4] == b"\x1a\x45\xdf\xa3":
        return "webm" if b"webm" in head[:SNIFF_BYTES] else "mkv"

    brand = _iso_bmff_brand(head)
    if brand is not None:
        if brand.startswith("qt"):
            return "mov"
        if brand in ("M4A ", "M4B "):
            return "m4a"
        return "mp4"

    return None


ALLOWED_FORMATS: dict[str, set[str]] = {
    "image": {"jpeg", "png", "webp", "bmp", "tiff"},
    "video": {"mp4", "mov", "avi", "webm", "mkv"},
    "audio": {"mp3", "wav", "flac", "ogg", "m4a"},
}

_FRIENDLY = {
    "jpeg": "a JPEG image", "png": "a PNG image", "webp": "a WebP image",
    "bmp": "a BMP image", "tiff": "a TIFF image",
    "mp4": "an MP4 video", "mov": "a QuickTime video", "avi": "an AVI video",
    "webm": "a WebM video", "mkv": "a Matroska video",
    "mp3": "an MP3 audio file", "wav": "a WAV audio file",
    "flac": "a FLAC audio file", "ogg": "an Ogg audio file",
    "m4a": "an M4A audio file",
}


# refuse files that are not the media type they claim to be
def validate_signature(media_type: str, data: bytes) -> str | None:
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
