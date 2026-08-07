import mimetypes


def detect_media_type(file_path: str):

    mime_type, _ = mimetypes.guess_type(file_path)

    if mime_type is None:
        return "unknown"

    if "video" in mime_type:
        return "video"

    if "image" in mime_type:
        return "image"

    if "audio" in mime_type:
        return "audio"

    return "unknown"