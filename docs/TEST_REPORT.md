# DeepTruth — Model Test Report

Date: 2026-10-16 · Hardware: CPU only (torch 2.13 +cpu, no CUDA) · Tester: Claude Code

## 1. Test set

| File | Ground truth | Details |
|---|---|---|
| `valor_ai_image.jpg` | **AI-generated** image | 1222×864, illustrated sci-fi scene, multiple faces |
| `2nd video.mp4` | **AI-generated** video | 47.3 s, 1920×1080, 86 MB, has audio |
| `WhatsApp Video 2026-05-06 at 03.35.16.mp4` | **Real** (phone recording) | 9.1 s, 478×850, WhatsApp-recompressed, has audio |

Every model was run on every file it supports, through `Pipeline.analyze()` (modules 4–9).
The image was also run through the full server path (API → Redis → Celery worker → SQLite)
to confirm the deployed verdict matches.

## 2. Results — per model

P(fake) shown. ✅ = correct at the 0.5 threshold, ❌ = wrong.

### Image (Module 7, ViT-B/16 + LoRA) — truth: AI

| Checkpoint | Role | P(fake) | |
|---|---|---|---|
| genimage | generalist | 0.341 | ❌ |
| mscocoai | generalist | 0.332 | ❌ |
| wildrf | generalist | 0.587 | ✅ |
| commforensics | generalist | 0.536 | ✅ (barely) |
| ntire | generalist | **0.997** | ✅ |
| dff | face | 0.432 | ❌ |
| ffpp | face | 0.874 | ✅ |
| ffpp_facecrop | face | 0.281 | ❌ |
| **Ensemble** (`max_alarm`) | | **0.564** | ⚠️ server verdict **INCONCLUSIVE** (35–65% band) |
| SRM (Module 8) | secondary | 0.409 | ❌ |

### Video (Module 7, ViViT + LoRA)

| Checkpoint | AI video | Real phone video |
|---|---|---|
| celebdf_v2 | 0.856 ✅ | 0.871 ❌ |
| deeperforensics | 0.833 ✅ | 0.909 ❌ |
| dfdc | 0.579 ✅ | 0.631 ❌ |
| ffpp | 0.602 ✅ | 0.476 ✅ |
| genvideo (full-frame, the only model for fully generated video) | **0.037 ❌** | 0.032 ✅ |
| wilddeepfake | 0.393 ❌ | 0.729 ❌ |
| **Ensemble** | **0.652** → MANIPULATED ✅ | **0.723** → MANIPULATED ❌ |
| Faces detected | 9/16 frames | 15/16 frames |

### Audio (Module 6, WavLM-Large) and SRM (Module 8)

| Model | AI video audio | Real phone audio |
|---|---|---|
| WavLM | 0.9996 → FAKE | **1.000 → FAKE ❌** |
| SRM | 0.299 → REAL ❌ | 0.508 → FAKE ❌ |

## 3. Accuracy summary

Only 3 files, so these show behaviour, not statistical accuracy.

| Module | Correct (final verdict) | Notes |
|---|---|---|
| Image ensemble | 0/1 decisive (inconclusive) | 4 of 8 checkpoints wrong |
| Video ensemble | 1/2 | the **real video scored more fake (0.72) than the AI video (0.65)**, so the models don't separate the two |
| Audio | 1/2 | scores both ≈1.0, so it marks every sample "fake" |
| SRM | 0/3 | val EER 34% at training, consistent with this |
| **Overall** | **2 of 8 module verdicts correct** | |

## 4. Why: root causes

1. **The video ensemble catches faces, not generated video.** The face models fire on *any* face, real or not (5/6 say the real
   phone video is fake). `genvideo` is the only model meant for fully generated video, and it gave the AI video 3.7%.
   With the `max(face_avg, genvideo)` rule the verdict comes from the face models, which explains the false positive.
2. **Domain shift.** The models were trained on FF++, Celeb-DF and DFDC (2019–2020, desktop video). A WhatsApp-recompressed
   portrait phone clip is out of distribution, and the report itself (LI-*) says compression degrades detection.
3. **Audio is saturated.** Trained 1 epoch on ASVspoof2019 LA (studio-quality English TTS from 2019). Any real-world
   microphone audio looks "spoof", so P(fake) ≈ 1.0 on everything. Validation showed it gets 99.8% of bona fide clips right on
   ASVspoof but not on phone audio. Also, the CLI and `Pipeline.analyze()` default `threshold=0.5` overrides the
   tuned 0.00028.
4. **Image ensemble dilution.** One model is near-certain (ntire 0.997), but averaging with weak models pulls the
   result to 0.56, which lands in the inconclusive band. The config.py AUC weights are also wrong (see earlier review).
5. **SRM is too weak to use** (34% EER, trained 0.7 min on CASIA v2 tampering, not AI generation).
6. **Fusion (Module 9) was missing at the time of this test; it has since been added.** Video cases now run the audio
   engine on the soundtrack and fuse it with the frames (`server/app/services/fusion.py`): the strongest weighted alarm
   wins, the manipulation type is reported (visual_only / audio_only / audio_and_visual / none), and the experimental
   audio model is weighted 0.25 so it can raise doubt but never convict alone. Re-run on the phone video: fused 72.3%,
   type audio_and_visual, so the verdict is still wrong because both inputs are wrong. SRM stays info-only.

## 5. Report-vs-implementation gaps found while testing

| Report says | Reality |
|---|---|
| Verdict in 5–15 s (LI-4) | image ~2 min, 9 s video ~8 min, 47 s video ~8 min (CPU) |
| Samples 1–2 frames/sec | fixed 16 frames per video, whatever its length |
| FR9 fusion combines audio + visual | implemented after this test; see root cause 6 |
| "Fusion must improve overall accuracy" | not measured anywhere |

## 6. Bugs found and fixed during testing

| Bug | Effect | Fix |
|---|---|---|
| `tokenizers 0.23` installed, `transformers 4.49` needs `<0.22` | **no model could load** | pinned `tokenizers 0.21.x` (environment) |
| Server default checkpoint paths pointed to `./images_checkpoints`, `./videos_checkpoints` (moved into `checkpoints/`) | **every case failed** with `CheckpointsUnavailable` | `server/app/_dtp.py` now defaults to `checkpoints/…` |
| Audio/SRM checkpoints never configured by the server | server silently used the **audio stub** and SRM without its classifier | `_dtp.py` sets them when `model.pt` exists |
| `start.ps1` / `start.sh` checked the old checkpoint paths | start script reported "no adapters" | paths corrected |
| Extension gave up polling after 5 min | every video case (≈8 min on CPU) shown as **failed** | timeout 35 min, extension rebuilt |

## 7. Extension connection

Verified: API `/api/health` OK, Redis OK, CORS accepts `chrome-extension://…` origins, `/api/cache/lookup` and
multipart `POST /api/cases` work, and the worker processes the case and returns the verdict (CASE-92996BA9).

To run it:

1. Start Redis. Docker Desktop failed to start on this PC ("unable to start"), so install
   **Memurai** (`winget install Memurai.MemuraiDeveloper`) or repair Docker/WSL.
2. `.\scripts\start.ps1` (starts the API on :8000, the worker and the web console on :3000).
3. Chrome → `chrome://extensions` → Developer mode → **Load unpacked** → `S:\deeptruth_pipeline\extension`
   (reload it if already loaded, to pick up the new `background.js`).
4. Popup → server URL `http://localhost:8000`.

## 8. What would raise accuracy (next phase)

1. Stop the face models overriding when there are no face artefacts (a calibrated weighted average instead of `max`),
   and measure whether audio-visual fusion beats the best single stream.
2. Retrain or fine-tune `genvideo` on modern generators (Sora/Veo/Kling/Runway samples) — the actual threat in your report.
3. Audio: retrain on ASVspoof5 / In-the-Wild / real phone recordings, more than 1 epoch, and use the calibrated threshold.
4. Build a labelled test set (≥100 real + ≥100 fake per modality, including phone and WhatsApp-compressed media) and report
   accuracy, AUC, precision and recall per module. Your Chapter 8 (testing) needs this.
