import os
import ffmpeg

FRAME_DIR = "uploads/frames"
AUDIO_DIR = "uploads/audio"


def extract_frames(video_path: str):

    os.makedirs(FRAME_DIR, exist_ok=True)

    output_pattern = f"{FRAME_DIR}/frame_%04d.jpg"

    (
        ffmpeg
        .input(video_path)
        .output(
            output_pattern,
            vf="fps=1"
        )
        .run(overwrite_output=True)
    )

    frames = sorted([
        os.path.join(FRAME_DIR, file)
        for file in os.listdir(FRAME_DIR)
        if file.endswith(".jpg")
    ])

    return frames


def extract_audio(video_path: str):

    os.makedirs(AUDIO_DIR, exist_ok=True)

    output_audio = f"{AUDIO_DIR}/audio.wav"

    (
        ffmpeg
        .input(video_path)
        .output(output_audio)
        .run(overwrite_output=True)
    )

    return output_audio