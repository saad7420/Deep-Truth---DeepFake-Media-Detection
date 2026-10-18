"""Cross-modal fusion for video cases (Module 9).

A video is judged on two independent streams: its frames (VisualForensicsEngine)
and its soundtrack (the audio engine). Either one can be manipulated on its
own — a face swap over the original voice, or genuine footage with a cloned
voice — so the fused verdict has to raise an alarm when *either* stream does,
and say which one did.

Policy
------
Each usable stream (confidence > 0) is first shrunk toward 0.5 by how much
we trust that stream's model:

    evidence_s = 0.5 + reliability_s * (p_s - 0.5)

then the fused probability is the strongest evidence of any stream (an
alarm from one stream is not cancelled by the other stream looking clean).

Reliability is 1.0 for a validated model. The current audio checkpoint is
flagged experimental (trained only on ASVspoof 2019 LA, known to call real
phone audio fake), so it gets AUDIO_EXPERIMENTAL_RELIABILITY = 0.25: at most
0.5 + 0.25 * 0.5 = 0.625. That is below the 0.65 "manipulated" line, so an
unvalidated audio model can raise a case to "inconclusive" but can never
convict a video on its own. Once the audio model is validated
(DEEPTRUTH_AUDIO_EXPERIMENTAL=false) it carries full weight.

The manipulation type is read from the streams' own probabilities, so the
report can say *what* was found even when the fused score is moderated.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

#: Verdict bands, identical to analyser._status().
FAKE_LINE = 0.65
REAL_LINE = 0.35

AUDIO_EXPERIMENTAL_RELIABILITY = float(
    os.getenv("DEEPTRUTH_AUDIO_EXPERIMENTAL_WEIGHT", "0.25"))


@dataclass
class Stream:
    """One modality's result, as the fusion layer sees it."""
    name: str                 # "visual" | "audio"
    fake_prob: float          # calibrated, 0..1
    confidence: float         # 0 means "no information"
    reliability: float = 1.0  # how far we trust this model, 0..1
    note: str = ""            # why the stream is unusable, if it is

    @property
    def usable(self) -> bool:
        return self.confidence > 0 and self.fake_prob == self.fake_prob

    @property
    def evidence(self) -> float:
        return 0.5 + self.reliability * (self.fake_prob - 0.5)


@dataclass
class FusionResult:
    fake_prob: float
    confidence: float
    manipulation_type: str
    rationale: str
    deciding_stream: str | None
    streams: dict = field(default_factory=dict)


def _band(p: float) -> str:
    if p >= FAKE_LINE:
        return "fake"
    if p <= REAL_LINE:
        return "real"
    return "uncertain"


_TYPE_TEXT = {
    ("fake", "fake"): ("audio_and_visual",
                       "Both the frames and the soundtrack look synthetic."),
    ("fake", "real"): ("visual_only",
                       "The frames look manipulated while the soundtrack looks "
                       "genuine (e.g. a face swap over the original voice)."),
    ("real", "fake"): ("audio_only",
                       "The frames look genuine but the soundtrack looks "
                       "synthetic (e.g. authentic footage with a cloned voice)."),
    ("real", "real"): ("none", "Neither the frames nor the soundtrack show "
                               "signs of manipulation."),
}


def fuse(visual: Stream, audio: Stream | None) -> FusionResult:
    """Combine the visual and audio streams of one video."""
    streams = [s for s in (visual, audio) if s is not None]
    usable = [s for s in streams if s.usable]
    summary = {
        s.name: {
            "fake_prob": round(s.fake_prob, 4) if s.usable else None,
            "confidence": round(s.confidence, 4),
            "reliability": s.reliability,
            "evidence": round(s.evidence, 4) if s.usable else None,
            "band": _band(s.fake_prob) if s.usable else "unavailable",
            **({"note": s.note} if s.note else {}),
        }
        for s in streams
    }

    if not usable:
        return FusionResult(0.5, 0.0, "unknown",
                            "No stream produced a usable score.", None, summary)

    deciding = max(usable, key=lambda s: s.evidence)
    fused = deciding.evidence

    if len(usable) == 1:
        only = usable[0]
        missing = "audio" if only.name == "visual" else "visual"
        reason = summary.get(missing, {}).get("note") or "not available"
        mtype = {"fake": f"{only.name}_only", "real": "none",
                 "uncertain": "uncertain"}[_band(only.fake_prob)]
        rationale = (f"Only the {only.name} stream was analysed "
                     f"({missing}: {reason}); fused score = {fused:.3f}.")
        return FusionResult(fused, only.confidence, mtype, rationale,
                            only.name, summary)

    v_band, a_band = _band(visual.fake_prob), _band(audio.fake_prob)
    mtype, text = _TYPE_TEXT.get((v_band, a_band),
                                 ("uncertain", "At least one stream is in the "
                                               "uncertain band."))
    if audio.reliability < 1.0 and a_band == "fake":
        text += (" The audio model is experimental, so its alarm is down-"
                 "weighted and cannot by itself mark the video manipulated.")
    rationale = (f"{text} visual={visual.fake_prob:.3f}, "
                 f"audio={audio.fake_prob:.3f} (weight {audio.reliability:g}); "
                 f"fused = strongest weighted evidence = {fused:.3f} "
                 f"from the {deciding.name} stream.")
    return FusionResult(fused, deciding.confidence, mtype, rationale,
                        deciding.name, summary)
