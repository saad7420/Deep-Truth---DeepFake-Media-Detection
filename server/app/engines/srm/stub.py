"""
M8 — Steganalysis Rich Model (SRM) Noise Analysis Stream (STUB).

Nothing in either uploaded zip implements this — it's a clean slate.
Per the report (Module 8 / AD-06), the real version should:
  1. Convert keyframes to grayscale float32
  2. High-pass filter to isolate the noise residual layer
  3. Apply the SRM 30-filter bank -> 686-dim feature vector per frame
  4. Run a small CNN classifier on the SRM features
  5. Flag inter-region noise inconsistency (spliced/GAN regions lack
     uniform camera-sensor noise)

`inp.artifact_path` will be the same frame directory M7 (visual) reads
from — M4 dispatches both from the same decode pass (see Fig. 15 in
the report), so whoever builds this doesn't need a new preprocessing
step, just a new consumer of the existing keyframe directory.

Until that's built, this stub keeps M9's fusion working in
audio+visual-only mode (confidence=0.0 -> SRM's weight contributes
nothing to the fused score).
"""
from __future__ import annotations

from ..base import Engine, EngineInput, EngineResult, neutral_result


class SRMNoiseStub(Engine):
    modality = "srm"

    def analyze(self, inp: EngineInput) -> EngineResult:
        return neutral_result("srm", "SRM Noise Analysis (M8) not yet built — stub result")
