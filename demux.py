"""ffprobe + ffmpeg helpers.

For video files we use ffmpeg to pull a 16 kHz mono WAV; frame extraction
itself goes through the training pipeline's _decode_video (decord/torchvision)
to match training. For pure audio inputs we still normalise via ffmpeg.
For images, we never invoke ffmpeg.
"""
from __future__ import annotations
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .config import VIDEO_EXTS, AUDIO_EXTS, IMAGE_EXTS


@dataclass
class MediaInfo:
    path: Path
    kind: str
    has_video: bool
    has_audio: bool
    duration_seconds: float | None
    container: str | None
    width: int | None
    height: int | None


def detect_kind_by_ext(path: Path) -> str | None:
    s = path.suffix.lower()
    if s in VIDEO_EXTS:
        return "video"
    if s in AUDIO_EXTS:
        return "audio"
    if s in IMAGE_EXTS:
        return "image"
    return None


def probe(path: Path) -> MediaInfo:
    ext_kind = detect_kind_by_ext(path)

    if ext_kind == "image":
        return MediaInfo(path=path, kind="image", has_video=False, has_audio=False,
                         duration_seconds=None,
                         container=path.suffix.lstrip(".") or None,
                         width=None, height=None)

    cmd = ["ffprobe", "-v", "error", "-print_format", "json",
           "-show_format", "-show_streams", str(path)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=30, check=True)
        data = json.loads(out.stdout)
    except (subprocess.CalledProcessError, FileNotFoundError,
            subprocess.TimeoutExpired, json.JSONDecodeError):
        if ext_kind:
            return MediaInfo(path=path, kind=ext_kind,
                             has_video=(ext_kind == "video"),
                             has_audio=(ext_kind in ("audio", "video")),
                             duration_seconds=None,
                             container=path.suffix.lstrip(".") or None,
                             width=None, height=None)
        raise ValueError(f"ffprobe failed and extension is unknown: {path}")

    streams = data.get("streams", [])
    has_video = any(s.get("codec_type") == "video" for s in streams)
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    fmt = data.get("format", {})
    duration = float(fmt.get("duration", 0)) or None

    width = height = None
    for s in streams:
        if s.get("codec_type") == "video":
            width, height = s.get("width"), s.get("height")
            break

    if has_video:
        kind = "video"
    elif has_audio:
        kind = "audio"
    else:
        kind = ext_kind or "unknown"

    return MediaInfo(path=path, kind=kind,
                     has_video=has_video, has_audio=has_audio,
                     duration_seconds=duration,
                     container=fmt.get("format_name"),
                     width=width, height=height)


def extract_audio_wav(input_path: Path, output_path: Path,
                      sr: int = 16000) -> Path:
    """Mono PCM WAV at the given sample rate. Used for both video tracks
    and audio-only inputs."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-i", str(input_path),
           "-vn", "-ac", "1", "-ar", str(sr),
           "-c:a", "pcm_s16le", str(output_path)]
    subprocess.run(cmd, check=True, timeout=180)
    return output_path
