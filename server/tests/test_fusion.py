"""Tests for the audio-visual fusion layer (Module 9).

Run from server/:  python -m pytest tests/test_fusion.py -q
"""
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.engines.base import EngineResult  # noqa: E402
from app.services.fusion import Stream, fuse  # noqa: E402

EXP = 0.25  # experimental audio weight (default)


def test_face_swap_over_real_voice_is_visual_only():
    r = fuse(Stream("visual", 0.90, 0.8), Stream("audio", 0.10, 0.9))
    assert r.manipulation_type == "visual_only"
    assert r.deciding_stream == "visual"
    assert abs(r.fake_prob - 0.90) < 1e-9


def test_real_footage_with_cloned_voice_is_audio_only():
    """The scenario the frames alone can never catch (BO-7)."""
    r = fuse(Stream("visual", 0.10, 0.8), Stream("audio", 0.95, 0.9))
    assert r.manipulation_type == "audio_only"
    assert r.deciding_stream == "audio"
    assert r.fake_prob >= 0.65          # validated audio can convict


def test_both_streams_fake():
    r = fuse(Stream("visual", 0.80, 0.7), Stream("audio", 0.90, 0.9))
    assert r.manipulation_type == "audio_and_visual"
    assert abs(r.fake_prob - 0.90) < 1e-9


def test_both_streams_clean():
    r = fuse(Stream("visual", 0.10, 0.8), Stream("audio", 0.20, 0.9))
    assert r.manipulation_type == "none"
    assert r.fake_prob <= 0.35


def test_experimental_audio_cannot_convict_alone():
    """Unvalidated audio at 100% fake may raise doubt, never 'manipulated'."""
    r = fuse(Stream("visual", 0.10, 0.8), Stream("audio", 1.0, 1.0, EXP))
    assert r.manipulation_type == "audio_only"
    assert 0.35 < r.fake_prob < 0.65    # inconclusive band
    assert "experimental" in r.rationale


def test_experimental_audio_does_not_hide_visual_alarm():
    r = fuse(Stream("visual", 0.723, 0.4), Stream("audio", 1.0, 1.0, EXP))
    assert r.deciding_stream == "visual"
    assert abs(r.fake_prob - 0.723) < 1e-9


def test_no_soundtrack_falls_back_to_visual():
    r = fuse(Stream("visual", 0.70, 0.6),
             Stream("audio", float("nan"), 0.0, note="no soundtrack"))
    assert r.deciding_stream == "visual"
    assert abs(r.fake_prob - 0.70) < 1e-9
    assert "no soundtrack" in r.rationale


def test_no_usable_stream_is_unknown():
    r = fuse(Stream("visual", 0.5, 0.0), Stream("audio", 0.5, 0.0))
    assert r.confidence == 0.0
    assert r.manipulation_type == "unknown"


class _FakeEngine:
    def __init__(self, result):
        self.result = result

    def analyze(self, _inp):
        return self.result


class _FakeRegistry:
    def __init__(self, engines):
        self.engines = engines

    def get(self, name):
        return self.engines.get(name)


def test_analyser_writes_one_fused_summary_row(monkeypatch):
    """End-to-end through analyser: one summary row, per-stream rows kept."""
    from app.services import analyser

    visual = EngineResult("visual", 0.2, 0.8, 0.7,
                          evidence={"per_checkpoint": {"ffpp": 0.2},
                                    "n_face_detected": 12},
                          model_version="vivit-lora-ensemble")
    audio = EngineResult("audio", 0.95, 0.05, 0.9,
                         evidence={"experimental": False},
                         model_version="wavlm")
    srm = EngineResult("srm", 0.5, 0.5, 0.0, evidence={"note": "info"})
    reg = _FakeRegistry({"visual": _FakeEngine(visual),
                         "audio": _FakeEngine(audio),
                         "srm": _FakeEngine(srm)})
    monkeypatch.setattr(analyser, "get_registry", lambda: reg)

    risk, _lik, status, rows = asyncio.run(
        analyser.run_analysis("case-1", "video", "x.mp4"))

    tiers = [json.loads(r["details"])["tier"] for r in rows]
    assert tiers.count("summary") == 1
    summary = next(json.loads(r["details"]) for r in rows
                   if json.loads(r["details"])["tier"] == "summary")
    assert summary["manipulation_type"] == "audio_only"
    assert summary["n_face_detected"] == 12      # visual evidence carried over
    assert status == "manipulated" and risk == 95.0
    signals = {json.loads(r["details"]).get("signal") for r in rows}
    assert {"visual_stream", "audio_stream"} <= signals
