from __future__ import annotations

import logging
import os
import shutil
import uuid
from pathlib import Path

log = logging.getLogger(__name__)

ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", "artifacts"))
ARTIFACT_BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")


def publish(record: dict | None) -> dict | None:
    if not record:
        return None

    src = record.get("path")
    if not src or not Path(src).is_file():
        return None

    try:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        name = f"{uuid.uuid4().hex}.png"
        shutil.copyfile(src, ARTIFACT_DIR / name)
    except OSError as exc:
        log.warning("could not publish artifact map: %s", exc)
        return None

    published = {k: v for k, v in record.items() if k != "path"}
    published["url"] = f"{ARTIFACT_BASE_URL}/artifacts/{name}"
    return published
