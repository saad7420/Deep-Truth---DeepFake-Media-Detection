"""
Frame extractor — pulls frames from a video file using OpenCV.

Extracts a uniform sample of MAX_EXTRACT_FRAMES frames from anywhere
in the video, which the predictor then sub-samples down to NUM_FRAMES=32
for the ViViT model.

OpenCV 4.x works fine with NumPy 2.x — no downgrade needed.
"""

from __future__ import annotations

import os
import uuid

import cv2


# How many frames to extract from the video before handing off to predictor.
# Must be >= NUM_FRAMES (32).  64 gives the predictor good coverage for
# short clips while keeping memory usage low.
MAX_EXTRACT_FRAMES = 64

FRAME_DIR = "uploads/frames"


def extract_frames(
    video_path: str,
    output_dir: str = FRAME_DIR,
    max_frames: int = MAX_EXTRACT_FRAMES,
) -> list[str]:
    """
    Extract up to `max_frames` uniformly-spaced frames from a video.

    Args:
        video_path: Path to the video file.
        output_dir: Directory where JPEG frames are written.
        max_frames: Maximum number of frames to extract (default 64).

    Returns:
        Sorted list of paths to the extracted frame files.

    Raises:
        RuntimeError: If the video file cannot be opened.
    """

    os.makedirs(output_dir, exist_ok=True)

    # Use a unique subdirectory per call so parallel requests don't collide.
    session_id = uuid.uuid4().hex[:8]
    session_dir = os.path.join(output_dir, session_id)
    os.makedirs(session_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = max(total_frames, 1)          # guard against 0

    # Compute which frame indices to grab
    if total_frames <= max_frames:
        indices = list(range(total_frames))
    else:
        import numpy as np
        indices = list(
            np.linspace(0, total_frames - 1, max_frames, dtype=int)
        )

    index_set = set(indices)
    saved_paths: list[str] = []
    frame_idx = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        if frame_idx in index_set:
            filename = f"frame_{frame_idx:06d}.jpg"
            path = os.path.join(session_dir, filename)
            cv2.imwrite(path, frame)
            saved_paths.append(path)

        frame_idx += 1

        # Early exit once we have everything we need
        if frame_idx > max(index_set):
            break

    cap.release()

    if not saved_paths:
        raise RuntimeError(
            f"No frames could be extracted from '{video_path}'. "
            "The file may be corrupt or in an unsupported codec."
        )

    return sorted(saved_paths)
