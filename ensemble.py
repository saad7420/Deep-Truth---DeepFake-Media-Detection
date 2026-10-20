from __future__ import annotations
import statistics

from .config import (
    FACE_CHECKPOINT_NAMES, MIN_FACE_FRAMES,
    IMAGE_CHECKPOINT_INFO, IMAGE_GENERALIST_NAMES, IMAGE_FACE_NAMES,
    IMAGE_NO_FACE_FACE_WEIGHT,
)


# video rule: any specialist alarm wins once faces are trusted
def ensemble_decide(per_ckpt: dict[str, float], n_face_detected: int) -> dict:
    face_scores = {n: s for n, s in per_ckpt.items() if n in FACE_CHECKPOINT_NAMES}
    gv_score = per_ckpt.get("genvideo")

    face_avg = (sum(face_scores.values()) / len(face_scores)
                if face_scores else None)
    face_trusted = (n_face_detected >= MIN_FACE_FRAMES) and (face_avg is not None)

    if face_trusted and gv_score is not None:
        # take the stronger of the two alarms
        ensemble = max(face_avg, gv_score)
        rationale = (f"face detected {n_face_detected}/16; "
                     f"max(face_avg={face_avg:.3f}, genvideo={gv_score:.3f})")
    elif face_trusted:
        ensemble = face_avg
        rationale = (f"face detected {n_face_detected}/16; "
                     f"genvideo unavailable, using face_avg")
    elif gv_score is not None:
        ensemble = gv_score
        rationale = (f"face detected {n_face_detected}/16 "
                     f"(<{MIN_FACE_FRAMES}); using genvideo only")
    else:
        ensemble = float("nan")
        rationale = "no usable checkpoint outputs"

    all_scores = list(per_ckpt.values())
    if len(all_scores) >= 2:
        std = statistics.pstdev(all_scores)
        # confidence is how much the checkpoints agree
        confidence = max(0.0, min(1.0, 1.0 - 2.0 * std))
    elif len(all_scores) == 1:
        confidence = 0.5
    else:
        confidence = 0.0

    return {
        "face_avg":       face_avg,
        "genvideo_score": gv_score,
        "ensemble":       ensemble,
        "face_trusted":   face_trusted,
        "confidence":     confidence,
        "rationale":      rationale,
    }


# votes are weighted by each checkpoint's AUC
def _weighted_mean(scored: dict[str, float]) -> float | None:
    if not scored:
        return None
    total_w = 0.0
    total_v = 0.0
    for name, p in scored.items():
        w = float(IMAGE_CHECKPOINT_INFO.get(name, {}).get("auc", 1.0))
        total_w += w
        total_v += w * p
    return total_v / total_w if total_w > 0 else None


# image rule: generalist and face groups, combined by face presence
def image_ensemble_decide(per_ckpt: dict[str, float],
                          face_detected: bool) -> dict:
    gen_scored  = {n: p for n, p in per_ckpt.items() if n in IMAGE_GENERALIST_NAMES}
    face_scored = {n: p for n, p in per_ckpt.items() if n in IMAGE_FACE_NAMES}

    gen_avg  = _weighted_mean(gen_scored)
    face_avg = _weighted_mean(face_scored)

    face_trusted = face_detected and len(face_scored) >= 2

    if gen_avg is None and face_avg is None:
        ensemble = float("nan")
        policy = "n/a"
        rationale = "no usable checkpoint outputs"
    elif face_trusted and gen_avg is not None:
        ensemble = max(gen_avg, face_avg)
        policy = "max_alarm"
        rationale = (f"face detected; "
                     f"max(gen={gen_avg:.3f}, face={face_avg:.3f})")
    elif face_detected and face_scored and gen_avg is not None:
        ensemble = 0.5 * gen_avg + 0.5 * face_avg
        policy = "blend_50_50"
        rationale = (f"face detected (only {len(face_scored)} face ckpt(s)); "
                     f"0.5·gen + 0.5·face")
    elif gen_avg is not None and face_avg is not None:
        w = IMAGE_NO_FACE_FACE_WEIGHT
        ensemble = (1 - w) * gen_avg + w * face_avg
        policy = f"weighted_{int(100*(1-w))}_{int(100*w)}"
        rationale = (f"no face detected; {1-w:.0%}·gen + {w:.0%}·face "
                     f"(face-group down-weighted, off-distribution)")
    elif gen_avg is not None:
        ensemble = gen_avg
        policy = "generalist_only"
        rationale = "no face detected and no face-group scores; generalists only"
    else:
        ensemble = face_avg
        policy = "face_only_fallback"
        rationale = ("no generalist scores; falling back to face group "
                     "(low confidence)")

    all_scores = list(per_ckpt.values())
    if len(all_scores) >= 2:
        std = statistics.pstdev(all_scores)
        # confidence is how much the checkpoints agree
        confidence = max(0.0, min(1.0, 1.0 - 2.0 * std))
    elif len(all_scores) == 1:
        confidence = 0.5
    else:
        confidence = 0.0

    return {
        "ensemble":       float(ensemble) if ensemble == ensemble else float("nan"),
        "generalist_avg": gen_avg,
        "face_avg":       face_avg,
        "face_trusted":   face_trusted,
        "confidence":     confidence,
        "rationale":      rationale,
        "policy":         policy,
        "n_generalist":   len(gen_scored),
        "n_face":         len(face_scored),
    }
