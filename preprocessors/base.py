from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ..storage import CacheStore, CacheRecord


class Preprocessor(ABC):
    """Turns a raw input file into model-ready artifacts.

    Implementations cache outputs under <CACHE_DIR>/<media_key>/<name>/ and
    short-circuit when called again with the same media_key (unless force=True).
    """
    name: str = "base"

    def __init__(self, store: CacheStore | None = None):
        self.store = store or CacheStore()

    @abstractmethod
    def supports(self, media_kind: str) -> bool: ...

    @abstractmethod
    def run(self, source_path: Path, media_key: str,
            record: CacheRecord, *, force: bool = False) -> dict[str, Any]:
        """Returns a dict pointing at produced artifacts (paths, stats)."""
