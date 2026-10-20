from __future__ import annotations

import os
import uuid

import cv2


MAX_EXTRACT_FRAMES = 64

FRAME_DIR = "uploads/frames"


def extract_frames(
    video_path: str,
    output_dir: str = FRAME_DIR,
    max_frames: int = MAX_EXTRACT_FRAMES,
) -> list[str]:

    os.makedirs(output_dir, exist_ok=True)

    session_id = uuid.uuid4().hex[:8]
    session_dir = os.path.join(output_dir, session_id)
    os.makedirs(session_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = max(total_frames, 1)

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

        if frame_idx > max(index_set):
            break

    cap.release()

    if not saved_paths:
        raise RuntimeError(
            f"No frames could be extracted from '{video_path}'. "
            "The file may be corrupt or in an unsupported codec."
        )

    return sorted(saved_paths)
