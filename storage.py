"""Preprocessed-artifact cache.

Layout:
    <CACHE_DIR>/<media_key>/
        meta.json
        video/   face_frames.npy, full_frames.npy, stats.json
        audio/   audio_16k.wav, stats.json
        image/   image_224.npy, stats.json

media_key is sha256 of the raw file bytes (truncated to 32 hex chars), prefixed
with 'mk_'. Two uploads of the same file collide deliberately.
"""
from __future__ import annotations
import hashlib
import json
import shutil
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from .config import CACHE_DIR

_CHUNK = 1 << 20


def media_key_from_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(_CHUNK)
            if not b:
                break
            h.update(b)
    return "mk_" + h.hexdigest()[:32]


def media_key_from_bytes(b: bytes) -> str:
    return "mk_" + hashlib.sha256(b).hexdigest()[:32]


@dataclass
class CacheRecord:
    media_key: str
    source_path: str
    media_kind: str
    duration_seconds: float | None = None
    completed: dict[str, bool] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "CacheRecord":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class CacheStore:
    def __init__(self, root: Path | None = None):
        self.root = (root or CACHE_DIR).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def dir_for(self, media_key: str) -> Path:
        return self.root / media_key

    def meta_path(self, media_key: str) -> Path:
        return self.dir_for(media_key) / "meta.json"

    def load(self, media_key: str) -> CacheRecord | None:
        mp = self.meta_path(media_key)
        if not mp.exists():
            return None
        try:
            return CacheRecord.from_dict(json.loads(mp.read_text()))
        except (json.JSONDecodeError, TypeError):
            return None

    def save(self, rec: CacheRecord) -> None:
        d = self.dir_for(rec.media_key)
        d.mkdir(parents=True, exist_ok=True)
        self.meta_path(rec.media_key).write_text(json.dumps(asdict(rec), indent=2))

    def subdir(self, media_key: str, name: str) -> Path:
        d = self.dir_for(media_key) / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def clear(self, media_key: str) -> None:
        d = self.dir_for(media_key)
        if d.exists():
            shutil.rmtree(d)
