from __future__ import annotations

import asyncio
import json
import uuid
import os
from typing import Tuple

from app.engines.base import EngineInput, EngineResult
from app.registry import get_registry


AnalysisTuple = Tuple[float, float, str, list[dict]]


# 65 and above is manipulated, 35 and below is authentic
def _status(fake_prob_0_to_1: float) -> str:
    pct = fake_prob_0_to_1 * 100
    if pct >= 65:
        return "manipulated"
    if pct <= 35:
        return "authentic"
    return "inconclusive"


CHECKPOINT_LABELS = {
    "celebdf_v2":       "Celeb-DF v2",
    "deeperforensics":  "DeeperForensics",
    "dfdc":             "DFDC",
    "ffpp":             "FaceForensics++",
    "genvideo":         "GenVideo",
    "wilddeepfake":     "WildDeepfake",
    "genimage":         "GenImage",
    "mscocoai":         "MS-COCO AI",
    "wildrf":           "WildRF",
    "commforensics":    "CommunityForensics",
    "ntire":            "NTIRE",
    "dff":              "DiffusionFace",
    "ffpp_facecrop":    "FaceForensics++ (face crop)",
}


def _checkpoint_rows(case_id: str, per_checkpoint: dict | None,
                     roles: dict | None = None) -> list[dict]:
    rows = []
    for name, score in (per_checkpoint or {}).items():
        details = {
            "tier":      "checkpoint",
            "fake_prob": round(float(score), 4),
            "label_text": CHECKPOINT_LABELS.get(name, name),
        }
        role = (roles or {}).get(name)
        if role:
            details["role"] = role
        rows.append({
            "id":         str(uuid.uuid4()),
            "case_id":    case_id,
            "model_name": name,
            "confidence": round(float(score) * 100, 2),
            "label":      "SYNTHETIC" if score >= 0.5 else "AUTHENTIC",
            "details":    json.dumps(details),
        })
    return rows


def _engine_result_row(case_id: str, label: str, result: EngineResult) -> dict:
    details = {
        "tier":       "summary",
        "modality":   result.modality,
        "fake_prob":  round(result.fake_prob, 4),
        "real_prob":  round(result.real_prob, 4),
        "confidence": result.confidence,
        "model_version": result.model_version,
        **{k: v for k, v in result.evidence.items() if k != "per_checkpoint"},
    }
    if result.error:
        details["error"] = result.error

    if result.confidence <= 0:
        row_label = "INCONCLUSIVE"
    elif result.fake_prob >= 0.5:
        row_label = "SYNTHETIC"
    else:
        row_label = "AUTHENTIC"

    return {
        "id":         str(uuid.uuid4()),
        "case_id":    case_id,
        "model_name": label,
        "confidence": round(result.fake_prob * 100, 2),
        "label":      row_label,
        "details":    json.dumps(details),
    }


def _srm_row(case_id: str, result: EngineResult) -> dict:
    details = {
        "tier":       "secondary",
        "signal":     "srm_noise",
        "modality":   result.modality,
        "confidence": result.confidence,
        "model_version": result.model_version,
        **{k: v for k, v in result.evidence.items() if k != "per_checkpoint"},
    }
    if result.confidence > 0:
        details["fake_prob"] = round(result.fake_prob, 4)

    return {
        "id":         str(uuid.uuid4()),
        "case_id":    case_id,
        "model_name": "SRM Noise Analysis (secondary)",
        "confidence": round(result.fake_prob * 100, 2) if result.confidence > 0 else 0.0,
        "label":      "INFO",
        "details":    json.dumps(details),
    }


async def run_analysis(
    case_db_id: str,
    media_type: str,
    file_path: str,
) -> AnalysisTuple:
    await asyncio.sleep(0)

    if media_type == "video":
        return await _analyse_with_engine(case_db_id, file_path, "visual",
                                          label="Video Ensemble (fused)")
    elif media_type == "image":
        return await _analyse_with_engine(case_db_id, file_path, "image",
                                          label="Image Ensemble (fused)")
    elif media_type == "audio":
        return await _analyse_audio(case_db_id, file_path)

    return (0.0, 0.0, "failed", [])


async def _analyse_with_engine(case_db_id: str, file_path: str,
                               modality: str, *, label: str) -> AnalysisTuple:
    engine = get_registry().get(modality)
    if engine is None:
        print(f"[Analyser] No engine registered for modality '{modality}'")
        return (0.0, 0.0, "failed", [])

    try:
        result: EngineResult = await asyncio.get_event_loop().run_in_executor(
            None,
            engine.analyze,
            EngineInput(media_key=case_db_id, modality=modality,
                        artifact_path=file_path, task_id=case_db_id),
        )
    except Exception as exc:
        print(f"[Analyser] {modality} engine raised unexpectedly: {exc}")
        return (0.0, 0.0, "failed", [])

    rows = _checkpoint_rows(case_db_id,
                            result.evidence.get("per_checkpoint"),
                            result.evidence.get("per_model_role"))
    rows.append(_engine_result_row(case_db_id, label, result))

    risk = round(result.fake_prob * 100, 2)
    status = _status(result.fake_prob) if result.confidence > 0 else "inconclusive"

    print(f"[Analyser] {modality} done — risk={risk:.1f}%  status={status}  "
          f"confidence={result.confidence:.2f}  "
          f"checkpoints={len(result.evidence.get('per_checkpoint') or {})}")

    if modality == "visual":
        risk, status = await _fuse_audio_visual(case_db_id, file_path,
                                                result, rows)

    media_kind = "video" if modality == "visual" else modality
    try:
        # srm is extra evidence, it never changes the verdict
        srm_engine = get_registry().get("srm")
        if srm_engine is not None:
            srm_result: EngineResult = await asyncio.get_event_loop().run_in_executor(
                None,
                srm_engine.analyze,
                EngineInput(media_key=case_db_id, modality="srm",
                            artifact_path=file_path, task_id=case_db_id,
                            extra={"media_kind": media_kind}),
            )
            rows.append(_srm_row(case_db_id, srm_result))
    except Exception as exc:
        print(f"[Analyser] srm supplementary pass failed (non-fatal): {exc}")

    return (risk, risk, status, rows)


# video only: also run the audio engine and fuse both streams
async def _fuse_audio_visual(case_db_id: str, file_path: str,
                             visual: EngineResult, rows: list[dict]
                             ) -> tuple[float, str]:
    from app.services.fusion import AUDIO_EXPERIMENTAL_RELIABILITY, Stream, fuse

    audio: EngineResult | None = None
    audio_note = ""
    engine = get_registry().get("audio")
    if engine is None:
        audio_note = "no audio engine registered"
    else:
        try:
            audio = await asyncio.get_event_loop().run_in_executor(
                None, engine.analyze,
                EngineInput(media_key=case_db_id, modality="audio",
                            artifact_path=file_path, task_id=case_db_id))
        except Exception as exc:
            audio_note = f"audio engine failed: {exc}"
    if audio is not None and audio.confidence <= 0:
        audio_note = (audio.error or audio.evidence.get("note")
                      or "no soundtrack or no audio model")

    experimental = bool(audio and audio.evidence.get("experimental"))
    v_stream = Stream("visual", visual.fake_prob, visual.confidence)
    a_stream = Stream(
        "audio",
        audio.fake_prob if audio else float("nan"),
        audio.confidence if audio else 0.0,
        AUDIO_EXPERIMENTAL_RELIABILITY if experimental else 1.0,
        audio_note[:200],
    )
    fused = fuse(v_stream, a_stream)

    for r in rows:
        d = json.loads(r["details"])
        if d.get("tier") == "summary":
            d.update(tier="secondary", signal="visual_stream")
            r["details"] = json.dumps(d)
            r["model_name"] = "Video frames (visual stream)"
    if audio is not None and audio.confidence > 0:
        a_row = _engine_result_row(case_db_id, "Soundtrack (audio stream)", audio)
        d = json.loads(a_row["details"])
        d.update(tier="secondary", signal="audio_stream")
        a_row["details"] = json.dumps(d)
        rows.append(a_row)

    details = {
        "tier": "summary",
        "modality": "visual",
        "fake_prob": round(fused.fake_prob, 4),
        "real_prob": round(1 - fused.fake_prob, 4),
        "confidence": fused.confidence,
        "model_version": f"{visual.model_version}+audio-fusion",
        **{k: v for k, v in visual.evidence.items() if k != "per_checkpoint"},
        "rationale": fused.rationale,
        "policy": "audio_visual_max_weighted",
        "manipulation_type": fused.manipulation_type,
        "deciding_stream": fused.deciding_stream,
        "streams": fused.streams,
    }
    if fused.confidence <= 0:
        label = "INCONCLUSIVE"
    else:
        label = "SYNTHETIC" if fused.fake_prob >= 0.5 else "AUTHENTIC"
    rows.append({
        "id": str(uuid.uuid4()),
        "case_id": case_db_id,
        "model_name": "Audio-Visual Fusion (fused)",
        "confidence": round(fused.fake_prob * 100, 2),
        "label": label,
        "details": json.dumps(details),
    })

    risk = round(fused.fake_prob * 100, 2)
    status = _status(fused.fake_prob) if fused.confidence > 0 else "inconclusive"
    print(f"[Analyser] fusion — {fused.manipulation_type}  risk={risk:.1f}%  "
          f"status={status}  deciding={fused.deciding_stream}")
    return risk, status


async def _analyse_audio(case_db_id: str, file_path: str) -> AnalysisTuple:
    engine = get_registry().get("audio")
    if engine is None:
        print("[Analyser] No engine registered for modality 'audio'")
        return (0.0, 0.0, "failed", [])

    try:
        result: EngineResult = await asyncio.get_event_loop().run_in_executor(
            None,
            engine.analyze,
            EngineInput(media_key=case_db_id, modality="audio",
                        artifact_path=file_path, task_id=case_db_id),
        )
    except Exception as exc:
        print(f"[Analyser] audio engine raised unexpectedly: {exc}")
        return (0.0, 0.0, "failed", [])

    label = ("AudioFakeNet (stub)" if result.confidence <= 0
             else "Audio Ensemble (fused)")
    row = _engine_result_row(case_db_id, label, result)

    if result.confidence <= 0:
        return (50.0, 50.0, "inconclusive", [row])

    risk = round(result.fake_prob * 100, 2)
    print(f"[Analyser] audio done — risk={risk:.1f}%  "
          f"confidence={result.confidence:.2f}")
    return (risk, risk, _status(result.fake_prob), [row])


ALLOWED_TYPES = {
    "image": ["image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff"],
    "video": ["video/mp4", "video/quicktime", "video/x-msvideo", "video/webm", "video/x-matroska"],
    "audio": ["audio/mpeg", "audio/wav", "audio/x-wav", "audio/flac", "audio/ogg", "audio/mp4"],
}

MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "500"))


def validate_file(
    media_type: str,
    content_type: str,
    file_size: int,
) -> str | None:
    allowed = ALLOWED_TYPES.get(media_type)
    if allowed is None:
        return f"Unknown media_type '{media_type}'. Must be image, video, or audio."

    if content_type not in allowed:
        return (
            f"File type '{content_type}' is not allowed for {media_type}. "
            f"Allowed: {', '.join(allowed)}"
        )

    max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    if file_size > max_bytes:
        return f"File too large ({file_size / 1e6:.1f} MB). Max is {MAX_FILE_SIZE_MB} MB."

    return None
