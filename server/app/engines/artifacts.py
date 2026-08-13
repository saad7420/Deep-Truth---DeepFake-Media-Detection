"""Publishing artifact maps (M7 FE-3) for the console to load.

The pipeline writes a rendered map beside its preprocessed tensors, under
DEEPTRUTH_CACHE — deliberately, so it is cached and evicted on the same terms
as everything else derived from that file. That directory is not web-served,
and mounting it would expose the whole preprocessing cache; copying one file
out is the narrower option.

Shared by the image and visual engines because both now produce maps and the
only difference between them is what the picture contains.
"""
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
    """Copy a rendered map into the served directory and swap path for URL.

    The local `path` is dropped: a filesystem path on the analysis host is of
    no use to a browser and only discloses server layout. Returns None if
    there is nothing to publish, which is the normal case for a modality with
    no map — never an error.
    """
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
