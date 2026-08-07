"""Multi-checkpoint ensemble for the video model.

Policy:
    face_avg     = mean P(fake) across face-aware checkpoints
    genvideo     = P(fake) from the genvideo (full-frame) checkpoint
    trusted      = (n_face_detected >= MIN_FACE_FRAMES) and face_avg is not None

    if trusted and genvideo present:
        ensemble = max(face_avg, genvideo)        # any specialist alarm wins
    elif trusted:
        ensemble = face_avg
    elif genvideo present:
        ensemble = genvideo
    else:
        ensemble = NaN

Confidence (independent of verdict):
    If at least 2 checkpoints scored, confidence = max(0, 1 - 2*std(scores)).
    Low std means models agree; high std flags "needs human review".
"""
from __future__ import annotations
import statistics

from .config import (
    FACE_CHECKPOINT_NAMES, MIN_FACE_FRAMES,
    IMAGE_CHECKPOINT_INFO, IMAGE_GENERALIST_NAMES, IMAGE_FACE_NAMES,
    IMAGE_NO_FACE_FACE_WEIGHT,
)


def ensemble_decide(per_ckpt: dict[str, float], n_face_detected: int) -> dict:
    face_scores = {n: s for n, s in per_ckpt.items() if n in FACE_CHECKPOINT_NAMES}
    gv_score = per_ckpt.get("genvideo")

    face_avg = (sum(face_scores.values()) / len(face_scores)
                if face_scores else None)
    face_trusted = (n_face_detected >= MIN_FACE_FRAMES) and (face_avg is not None)

    if face_trusted and gv_score is not None:
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


# ────────────────────────────────────────────────────────────────────────────
# Image ensemble
# ────────────────────────────────────────────────────────────────────────────
#
# Eight checkpoints, two role groups:
#
#   generalist:  genimage, mscocoai, wildrf, commforensics, ntire
#                Trained on broad AI-image distributions. Run on the
#                whole 224×224 image. Strongest evidence for synthetic content
#                that doesn't centre on a face.
#
#   face:        dff, ffpp, ffpp_facecrop
#                Trained on face-deepfake distributions. The first two were
#                trained on whole frames; ffpp_facecrop was trained on MTCNN
#                crops with 30% margin and MUST be fed the cropped tensor.
#
# Within each group we take a weighted mean using each checkpoint's reported
# best-val AUC as the weight (commforensics' 0.99 will dominate the generalist
# vote; ffpp_facecrop's 0.92 will dominate the face vote when it ran).
#
# When the input contains a face, the two groups disagree often enough that we
# fall back to the same "any specialist alarm wins" rule the video ensemble
# uses — take the max. When the input has no detectable face, we down-weight
# the face group sharply (they are off-distribution) but do not zero them out;
# DFF and ffpp_whole were trained on whole frames so they retain some signal.
#
# Confidence is the same shape as the video formula: 1 − 2·σ across all
# scoring checkpoints. High variance ⇒ flag for human review.


def _weighted_mean(scored: dict[str, float]) -> float | None:
    """AUC-weighted mean over the supplied checkpoints. Missing scores are
    skipped; missing weights default to 1.0 (so unknown ckpts still vote)."""
    if not scored:
        return None
    total_w = 0.0
    total_v = 0.0
    for name, p in scored.items():
        w = float(IMAGE_CHECKPOINT_INFO.get(name, {}).get("auc", 1.0))
        total_w += w
        total_v += w * p
    return total_v / total_w if total_w > 0 else None


def image_ensemble_decide(per_ckpt: dict[str, float],
                          face_detected: bool) -> dict:
    """Combine per-checkpoint P(fake) scores into a single decision.

    Args:
        per_ckpt:       {checkpoint_slug: P(fake)} — slug is what comes between
                        "image_" and "_lora_best" (e.g. "ffpp_facecrop"). Only
                        include scores that actually ran; the policy adapts to
                        whichever subset is present.
        face_detected:  True if MTCNN found a face in the input. Drives whether
                        the face-specialist group is trusted at full weight.

    Returns:
        Dict with keys:
            ensemble        float in [0,1] or NaN
            generalist_avg  float in [0,1] or None
            face_avg        float in [0,1] or None
            face_trusted    bool — face_detected and ≥2 face-group scores
            confidence      float in [0,1], independent of verdict
            rationale       human-readable explanation
            policy          short tag: "max_alarm" / "blend_50_50" /
                            "weighted_80_20" / "generalist_only" / "n/a"
            n_generalist    int — how many generalist ckpts ran
            n_face          int — how many face ckpts ran
    """
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
        # Conservative on fakes: any group's alarm wins.
        ensemble = max(gen_avg, face_avg)
        policy = "max_alarm"
        rationale = (f"face detected; "
                     f"max(gen={gen_avg:.3f}, face={face_avg:.3f})")
    elif face_detected and face_scored and gen_avg is not None:
        # Face is present but only one face checkpoint scored — blend evenly
        # so the lone face vote isn't drowned but also can't sweep the call.
        ensemble = 0.5 * gen_avg + 0.5 * face_avg
        policy = "blend_50_50"
        rationale = (f"face detected (only {len(face_scored)} face ckpt(s)); "
                     f"0.5·gen + 0.5·face")
    elif gen_avg is not None and face_avg is not None:
        # No face detected. The face group is off-distribution; let it whisper
        # at IMAGE_NO_FACE_FACE_WEIGHT (default 0.2), not shout.
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
        # Only face scores available, and no face was detected — weird, but
        # honour what we have.
        ensemble = face_avg
        policy = "face_only_fallback"
        rationale = ("no generalist scores; falling back to face group "
                     "(low confidence)")

    all_scores = list(per_ckpt.values())
    if len(all_scores) >= 2:
        std = statistics.pstdev(all_scores)
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
