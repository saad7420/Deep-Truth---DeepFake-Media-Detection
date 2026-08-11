"""
analyser.py — Forensic analysis pipeline.

Video now goes through app.registry's EngineRegistry (M7's fixed
VisualForensicsEngine, built on deeptruth_pipeline) instead of the old
app/services/inference/predictor.py + model_loader/detector.py, which
had the checkpoint-weight-scaling bug and the frame-count/position-
embedding architecture mismatch (see app/engines/visual/engine.py's
docstring for the full explanation).

Image analysis goes the same way as of the pipeline merge: the eight
ViT-B/16 + LoRA image checkpoints are real, so `image` now resolves to
ImageForensicsEngine (app/engines/image/engine.py) instead of the old
"not yet implemented" placeholder row. Both branches share
`_analyse_with_engine` so the DB rows, status mapping, and evidence
handling stay identical across modalities.

Audio is still M6's stub and deliberately reports confidence 0.0.
"""

from __future__ import annotations

import asyncio
import json
import uuid
import os
from typing import Tuple

from app.engines.base import EngineInput, EngineResult
from app.registry import get_registry

# ─────────────────────────────────────────────────────────────────────────────
# TYPES
# ─────────────────────────────────────────────────────────────────────────────

# (risk_score, synthetic_likelihood, status, analysis_results_rows)
AnalysisTuple = Tuple[float, float, str, list[dict]]


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _status(fake_prob_0_to_1: float) -> str:
    pct = fake_prob_0_to_1 * 100
    if pct >= 65:
        return "manipulated"
    if pct <= 35:
        return "authentic"
    return "inconclusive"


# Training corpus behind each LoRA adapter. The slug is what the pipeline
# emits; the label is what a human reading the report should see. A slug that
# isn't listed here still renders — the client falls back to the raw slug —
# so adding a checkpoint doesn't require touching this table first.
CHECKPOINT_LABELS = {
    # video (ViViT)
    "celebdf_v2":       "Celeb-DF v2",
    "deeperforensics":  "DeeperForensics",
    "dfdc":             "DFDC",
    "ffpp":             "FaceForensics++",
    "genvideo":         "GenVideo",
    "wilddeepfake":     "WildDeepfake",
    # image (ViT-B/16)
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
    """One DB row per individual LoRA checkpoint, for the bar-chart
    breakdown the dashboard shows (report FR10.1).

    `roles` is the image branch's {slug: "generalist"|"face"} map; it rides
    along in `details` so the report can explain why a given checkpoint was
    weighted the way it was. The video branch passes None.

    `tier` is written into every row so the frontend can group rows without
    string-matching on `model_name` — the client used to guess from names
    that no longer matched anything this function emits.
    """
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
    """Summary row for a whole-modality EngineResult (visual/image/audio/srm)."""
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

    # confidence == 0.0 is every engine's "ignore me" signal. Such a result
    # carries fake_prob 0.5, which used to fall through the `>= 0.5` test and
    # get stamped SYNTHETIC — a stub engine reporting nothing was being shown
    # to the operator as a positive detection.
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
    """Supplementary-tier row for the SRM noise-analysis pass.

    Deliberately NOT `tier: "summary"` — that tier is what the client reads
    as a case's verdict (`readAnalysis()` takes the first `summary` row it
    finds), and SRM must never be mistaken for that, whether it is a stub
    note, an untrained-but-feature-computed result, or eventually a real
    trained score. `tier: "secondary"` is a distinct, additive signal the
    console can choose to surface without it ever competing with the
    primary modality's fused risk score.
    """
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
        "label":      "INFO",  # never SYNTHETIC/AUTHENTIC — not a verdict
        "details":    json.dumps(details),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

async def run_analysis(
    case_db_id: str,
    media_type: str,
    file_path: str,
) -> AnalysisTuple:
    """
    Full detection pipeline. Returns:
        (risk_score, synthetic_likelihood, status, analysis_results)
    """
    await asyncio.sleep(0)   # yield so FastAPI stays responsive

    if media_type == "video":
        return await _analyse_with_engine(case_db_id, file_path, "visual",
                                          label="Video Ensemble (fused)")
    elif media_type == "image":
        return await _analyse_with_engine(case_db_id, file_path, "image",
                                          label="Image Ensemble (fused)")
    elif media_type == "audio":
        return await _analyse_audio(case_db_id, file_path)

    return (0.0, 0.0, "failed", [])


# ─────────────────────────────────────────────────────────────────────────────
# VIDEO (M7, ViViT) and IMAGE (M7b, ViT-B/16) — both real, same shape
# ─────────────────────────────────────────────────────────────────────────────

async def _analyse_with_engine(case_db_id: str, file_path: str,
                               modality: str, *, label: str) -> AnalysisTuple:
    """Run one registered engine and turn its EngineResult into DB rows.

    Engines are synchronous by contract (see app/engines/base.py) and load
    torch models, so they run in the default executor — never on the event
    loop — and the whole call is guarded: a broken engine fails one case, it
    doesn't take the API down.
    """
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
    # confidence == 0.0 is every engine's "ignore me" signal (neutral_result,
    # stubs), so never let it drive an authentic/manipulated verdict.
    status = _status(result.fake_prob) if result.confidence > 0 else "inconclusive"

    print(f"[Analyser] {modality} done — risk={risk:.1f}%  status={status}  "
          f"confidence={result.confidence:.2f}  "
          f"checkpoints={len(result.evidence.get('per_checkpoint') or {})}")

    # ── SRM supplementary pass ──────────────────────────────────────────────
    # Runs after the primary verdict is already computed above and can never
    # change it — `risk` and `status` were captured from `result` before this
    # block runs. A failure here is caught and logged, never re-raised: SRM
    # not being trained yet (the normal case today) must not make an
    # otherwise-successful video/image case fail.
    media_kind = "video" if modality == "visual" else modality
    try:
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
    except Exception as exc:  # noqa: BLE001
        print(f"[Analyser] srm supplementary pass failed (non-fatal): {exc}")

    return (risk, risk, status, rows)


# ─────────────────────────────────────────────────────────────────────────────
# AUDIO — M6, stub (see app/engines/audio/stub.py)
# ─────────────────────────────────────────────────────────────────────────────

async def _analyse_audio(case_db_id: str, file_path: str) -> AnalysisTuple:
    """Audio is still the M6 stub, but it is guarded exactly like the real
    engines so that swapping AudioFakeNetStub for the WavLM implementation
    (see app/engines/audio/stub.py) needs no change here.

    Once the real engine lands it reports a non-zero confidence, and the
    branch below starts producing genuine authentic/manipulated verdicts
    instead of the fixed 50/inconclusive the stub yields.
    """
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
        # Stub: report the neutral midpoint rather than a fabricated verdict.
        return (50.0, 50.0, "inconclusive", [row])

    risk = round(result.fake_prob * 100, 2)
    print(f"[Analyser] audio done — risk={risk:.1f}%  "
          f"confidence={result.confidence:.2f}")
    return (risk, risk, _status(result.fake_prob), [row])


# ─────────────────────────────────────────────────────────────────────────────
# FILE VALIDATION  (called by cases.py router) — unchanged
# ─────────────────────────────────────────────────────────────────────────────

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
    """
    Returns an error message string if invalid, or None if OK.
    Called by the cases router before saving the file.
    """
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
