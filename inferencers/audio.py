"""WavLM-Large audio deepfake inferencer.

Plugs into the DeepTruth pipeline as the real AudioInferencer.
Mirrors the architecture trained in the FYP:
  - WavLM-Large backbone, top-6 transformer layers unfrozen
  - 3-layer progressive-dropout classifier head (1024 → 256 → 64 → 2)
  - Trained on ASVspoof 2019 LA; checkpoint at /audiomodel/final_model

Audio contract from AudioPreprocessor:
    preprocessed["audio_wav_path"] = path to mono 16 kHz WAV

Usage (drop-in replacement for the stub):
    from deeptruth_pipeline.inferencers.audio import WavLMAudioInferencer
    from deeptruth_pipeline import Registry, Pipeline

    registry = Registry()
    registry.replace_inferencer("audio", WavLMAudioInferencer())
    pipeline = Pipeline(registry=registry)
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

from .base import Inferencer, InferenceResult
from ..config import USE_FP16, DEFAULT_THRESHOLD

log = logging.getLogger(__name__)

# ── constants matching training ──────────────────────────────────────────────
MODEL_ID        = "microsoft/wavlm-large"
SAMPLE_RATE     = 16_000
MAX_AUDIO_SEC   = 4                          # pad / truncate to 4 s
MAX_SAMPLES     = SAMPLE_RATE * MAX_AUDIO_SEC   # 64 000 samples
UNFREEZE_LAYERS = 6                          # top-N transformer layers trained
HEAD_DIMS       = [1024, 256, 64, 2]        # classifier widths (includes logit dim)

# Default checkpoint location; override with DEEPTRUTH_AUDIO_CHECKPOINT env var
DEFAULT_AUDIO_CKPT = os.environ.get(
    "DEEPTRUTH_AUDIO_CHECKPOINT",
    "/audiomodel/final_model",
)


# ── model builder ────────────────────────────────────────────────────────────

def _build_wavlm_model(device, fp16: bool):
    """Reconstruct the WavLM + classifier head used during training."""
    import torch
    import torch.nn as nn
    from transformers import WavLMModel

    class WavLMClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.wavlm = WavLMModel.from_pretrained(MODEL_ID)

            # Freeze all params, then unfreeze the top-N transformer layers
            for p in self.wavlm.parameters():
                p.requires_grad_(False)
            n_layers = len(self.wavlm.encoder.layers)
            for layer in self.wavlm.encoder.layers[n_layers - UNFREEZE_LAYERS:]:
                for p in layer.parameters():
                    p.requires_grad_(True)

            # Progressive-dropout classifier head
            hidden = HEAD_DIMS[0]
            self.classifier = nn.Sequential(
                nn.Linear(hidden, HEAD_DIMS[1]),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(HEAD_DIMS[1], HEAD_DIMS[2]),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(HEAD_DIMS[2], HEAD_DIMS[3]),
            )

        def forward(self, input_values, attention_mask=None):
            out = self.wavlm(input_values=input_values,
                             attention_mask=attention_mask)
            # mean-pool hidden states along time axis
            hidden = out.last_hidden_state.mean(dim=1)
            return self.classifier(hidden)

    model = WavLMClassifier()
    model = model.to(device)
    if fp16 and device.type == "cuda":
        model = model.half()
    return model


# ── inferencer ───────────────────────────────────────────────────────────────

class WavLMAudioInferencer(Inferencer):
    """Real audio inferencer – replaces AudioInferencer stub."""

    modality = "audio"

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        device: str | None = None,
        threshold: float = DEFAULT_THRESHOLD,
        keep_models_loaded: bool = True,
    ):
        self.checkpoint_path = Path(
            checkpoint_path if checkpoint_path else DEFAULT_AUDIO_CKPT
        )
        self._explicit_device = device
        self.threshold = threshold
        self.keep_loaded = keep_models_loaded

        self._model = None
        self._feature_extractor = None
        self._device = None

    # ── Inferencer protocol ──────────────────────────────────────────────────

    def supports(self, media_kind: str) -> bool:
        return media_kind == "audio"

    def predict(self, media_key: str, preprocessed: dict[str, Any],
                **opts) -> InferenceResult:
        self._setup()

        wav_path = preprocessed.get("audio_wav_path")
        if not wav_path or not Path(wav_path).exists():
            raise RuntimeError(f"audio_wav_path missing or not found: {wav_path}")

        waveform = self._load_waveform(wav_path)
        p_fake   = self._run_model(waveform)

        threshold = float(opts.get("threshold", self.threshold))
        verdict   = "FAKE" if p_fake >= threshold else "REAL"

        log.info(
            f"  audio WavLM  P(fake)={p_fake:.4f}  verdict={verdict}  "
            f"(threshold={threshold})"
        )

        return InferenceResult(
            media_key   = media_key,
            modality    = "audio",
            trust_score = p_fake,
            verdict     = verdict,
            confidence  = 1.0,                # single-model; no ensemble spread
            per_model   = {"wavlm_large": p_fake},
            rationale   = (
                f"WavLM-Large (top-{UNFREEZE_LAYERS} layers fine-tuned on "
                f"ASVspoof 2019 LA). "
                f"P(fake)={p_fake:.4f} vs threshold={threshold}."
            ),
            extra={
                "checkpoint":   str(self.checkpoint_path),
                "max_audio_sec": MAX_AUDIO_SEC,
                "sample_rate":  SAMPLE_RATE,
                "threshold":    threshold,
            },
        )

    # ── internals ───────────────────────────────────────────────────────────

    def _setup(self):
        if self._model is not None:
            return

        import torch
        from transformers import Wav2Vec2FeatureExtractor

        if self._explicit_device:
            self._device = torch.device(self._explicit_device)
        else:
            self._device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        fp16 = USE_FP16 and self._device.type == "cuda"
        log.info(
            f"WavLMAudioInferencer: device={self._device}  fp16={fp16}  "
            f"checkpoint={self.checkpoint_path}"
        )

        self._feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID)
        self._model = _build_wavlm_model(self._device, fp16)

        ckpt = self.checkpoint_path
        if not ckpt.exists():
            raise FileNotFoundError(
                f"Audio checkpoint not found: {ckpt}\n"
                f"Set DEEPTRUTH_AUDIO_CHECKPOINT to the correct path."
            )

        # Support both .pt files and HuggingFace-style directories
        if ckpt.is_dir():
            import torch
            state_dict = torch.load(ckpt / "model.pt",
                                    map_location=self._device)
        else:
            import torch
            state_dict = torch.load(ckpt, map_location=self._device)

        # Strip optional "module." prefix (DataParallel-saved checkpoints)
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k[len("module."):]: v for k, v in state_dict.items()}

        self._model.load_state_dict(state_dict, strict=True)
        self._model.eval()
        log.info("WavLMAudioInferencer: model loaded and ready.")

    def _load_waveform(self, wav_path: str) -> "torch.Tensor":
        """Load, resample if needed, pad/truncate to MAX_SAMPLES."""
        from scipy.signal import resample_poly
        import soundfile as sf

        audio, sr = sf.read(wav_path, dtype="float32", always_2d=False)

        # Mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Resample to 16 kHz if the WAV isn't already
        if sr != SAMPLE_RATE:
            from math import gcd
            g = gcd(SAMPLE_RATE, sr)
            audio = resample_poly(audio, SAMPLE_RATE // g, sr // g).astype("float32")

        # Pad / truncate
        if len(audio) < MAX_SAMPLES:
            audio = np.pad(audio, (0, MAX_SAMPLES - len(audio)))
        else:
            audio = audio[:MAX_SAMPLES]

        import torch
        return torch.from_numpy(audio)

    def _run_model(self, waveform: "torch.Tensor") -> float:
        import torch

        inputs = self._feature_extractor(
            waveform.numpy(),
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=False,
        )
        input_values = inputs["input_values"].to(self._device)

        fp16 = USE_FP16 and self._device.type == "cuda"
        with torch.no_grad():
            if fp16:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    logits = self._model(input_values)
            else:
                logits = self._model(input_values)

            probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()[0]

        # label 1 = fake (matches ASVspoof 2019 convention used during training)
        return float(probs[1])

    def unload(self):
        self._model = None
        self._feature_extractor = None
        if self._device is not None and self._device.type == "cuda":
            import torch
            torch.cuda.empty_cache()
        log.info("WavLMAudioInferencer: model unloaded.")
