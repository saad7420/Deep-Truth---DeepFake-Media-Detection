# Audio module — wiring the trained checkpoint in

The audio channel is now integrated across server, client and extension. It
stays on the stub until you point it at a checkpoint; nothing else changes.

---

## Turn it on

```bash
export DEEPTRUTH_AUDIO_CHECKPOINT=/path/to/final_model   # dir with model.pt
```

Restart the server. That's the whole change — no code edit. `app/registry.py`
selects `WavLMAudioEngine` when that variable is set and `AudioFakeNetStub`
when it isn't.

The decision threshold is picked up automatically from `metadata.json` next to
`model.pt` (`recommended_threshold`), so you don't have to copy it by hand.
Override only if you want something other than the measured EER point:

```bash
export DEEPTRUTH_AUDIO_THRESHOLD=0.0002785743272397667
```

This matters more than it looks. ASVspoof LA is roughly 1:9 bonafide:spoof, and
the class weighting that corrects for it squeezes raw scores toward zero — the
equal-error point on your current checkpoint is **0.00028**, not 0.5. Running at
0.5 would silently under-flag fake audio. Reading it from the checkpoint keeps
the weights and their operating point from drifting apart.

---

## Three pipeline-level bugs found and fixed

Wiring the registry was only half the job. Testing the actual numbers surfaced
these:

### 1. The verdict was inverted (serious)

`analyser._status()` bands every modality identically — >=65% manipulated,
<=35% authentic — which assumes a probability centred near 0.5. The audio
model's equal-error point is **0.00028**, because class weighting for
ASVspoof's 1:9 imbalance squeezes its raw scores toward zero.

Feeding raw scores into those bands inverted the result:

```
raw P(fake)   model says   server reported
0.000300      FAKE         authentic     <-- wrong
0.001000      FAKE         authentic     <-- wrong
0.010000      FAKE         authentic     <-- wrong
0.200000      FAKE         authentic     <-- wrong
```

A clip the model flagged at 35x its own threshold was shown to the operator as
authentic. Fixed with `_calibrate()` in `wavlm.py`: a piecewise-linear remap
anchoring the model's threshold to the 50% mark.

```
raw P(fake)   model says   server reports now
0.000300      FAKE         inconclusive
0.010000      FAKE         inconclusive
0.500000      FAKE         manipulated
0.999000      FAKE         manipulated
```

Two properties, both verified: it is **strictly monotonic**, so EER is
preserved exactly — it moves the midpoint, it does not invent separation the
model lacks. And it is the **identity function when the threshold is 0.5**, so
retraining to a properly calibrated model needs no code change here. The raw
score is kept in evidence as `raw_fake_prob`, and the console explains the
rescaling rather than hiding it.

This is not a substitute for real calibration (Platt scaling, isotonic
regression) — that needs a held-out set and belongs in training.

### 2. The configured threshold was being discarded

`Pipeline.analyze()` passes its own `threshold=0.5` default into
`predict(**opts)`, and `WavLMAudioInferencer.predict()` reads
`opts.get("threshold", self.threshold)` — so the pipeline's 0.5 won on every
call and the measured EER point was thrown away. The engine now resolves the
threshold from config and passes it explicitly.

### 3. Videos never had their audio analysed

`_default_modalities()` returned `["video"]` for video files, with a comment
saying to add audio "when the audio model lands". It has landed. A clip can be
visually untouched but carry a cloned voice, and analysing only the frames
reports it clean. Now returns `["video", "audio"]` when `info.has_audio`.

This governs `cli.py` and direct `Pipeline` use only —
`VisualForensicsEngine` passes `modalities=["video"]` explicitly, so the
server's per-case routing is unchanged.

---

## What was added

### Server

| File | Change |
|---|---|
| `config.py` | Audio section: `AUDIO_CHECKPOINT`, `AUDIO_THRESHOLD`, and `_resolve_audio_threshold()` which reads `metadata.json` |
| `registry.py` (pipeline) | `_default_audio_inferencer()` — real `WavLMAudioInferencer` when configured, stub otherwise; import deferred so torch isn't pulled in unless used |
| `server/app/engines/audio/wavlm.py` | **New.** The `Engine` wrapper |
| `server/app/registry.py` | `_audio_engine()` selects by environment |

The engine goes through `Pipeline.analyze()` rather than driving the inferencer
directly, matching the visual and image engines. That's deliberate: the
analyser hands the engine the *original uploaded file* — possibly an mp3 or m4a
— but `inferencers/audio.py` reads with soundfile, which handles neither
reliably. The pipeline's `AudioPreprocessor` runs ffmpeg first
(`-vn -ac 1 -ar 16000`) to produce the 16 kHz mono WAV the model was trained
on, and caches it so re-analysis skips the transcode.

Every failure path returns a neutral result carrying its reason rather than
raising — missing checkpoint, unreadable audio, `strict=True` key mismatch.
Setting the variable can't stop the server starting; worst case audio reports
"no signal", exactly as the stub does.

### Client

- `wavlm_large` checkpoint label and description
- `AudioEvidence` shape (`threshold`, `sample_rate`, `max_audio_sec`, `experimental`)
- Audio explanations in `explainEvidence()` — including a plain-language note
  about the non-0.5 threshold, so a reader doesn't judge "0.4%" against a
  mental 50% baseline and conclude the opposite of what the model found
- `SummaryRow.isExperimental`
- Orange banner in `ResultBarChart`, placed **above** the score
- Admin page resolves audio Live/Stub at runtime from actual case data rather
  than a hardcoded `false`

### Extension

Audio was already supported end to end (`<audio>` detection, MIME allow-list,
extension mapping). Added:

- `isExperimental()` in `api.ts`
- The long `EXPERIMENTAL CHECKPOINT:` prefix stripped from badge tooltips and
  carried as a flag instead, so the tooltip shows the actual finding
- 🧪 marker on the in-page badge
- `unvalidated` pill in the popup's scan list

---

## The experimental flag

While `DEEPTRUTH_AUDIO_EXPERIMENTAL` is unset or true (**the default**), every
audio verdict is marked provisional in all three surfaces.

This is opt-out rather than opt-in because the failure mode is silent. Your
current checkpoint scores a genuine recording and an AI-generated one both at
~100% fake — it isn't discriminating, and models trained on ASVspoof 2019 LA
alone are documented to collapse toward chance on real-world audio (in-domain
EER around 6% rising to ~50% out-of-domain on the In-the-Wild benchmark). A
result like that must not look identical to the video and image engines, which
are past that stage.

Turn it off once the model has been validated against several diverse real
recordings, not just one:

```bash
export DEEPTRUTH_AUDIO_EXPERIMENTAL=false
```

---

## Verified

Against a live server on `127.0.0.1:8000`:

```
created: CASE-EC9B1E74 | status: processing
settled: inconclusive | risk: 50.0
 row: AudioFakeNet (stub) | INCONCLUSIVE | tier: summary
   note: audio inference failed: No module named 'torch'
```

That is the soft-failure path working: no torch in this sandbox, so the engine
degraded to a neutral result and the row was written `INCONCLUSIVE` rather than
being stamped `SYNTHETIC` at 50%.

Also confirmed:

- Engine selection both ways — stub without the variable, `WavLMAudioEngine`
  with it, and a bogus path degrades instead of raising
- The analyser writes a correct row for a successful audio result
  (`Audio Ensemble (fused)`, tier `summary`, `SYNTHETIC` at 99.87%,
  `experimental` preserved)
- `readAnalysis` / `explainEvidence` on real audio row shapes, both the
  experimental and stub paths
- Client `tsc --noEmit` clean; extension `tsc --noEmit` clean and builds

**Not verified:** real inference. No torch, no GPU, and no disk headroom for
them here, so the model has never actually scored audio in this environment.
The plumbing around it is tested; the inference call itself is reached and
correctly caught when it fails. Your first real audio case will confirm the
rest — and if the checkpoint can't load, it surfaces as a named reason in the
case evidence rather than failing silently.

---

## Note on the package name

`server/app/_dtp.py` does `import deeptruth_pipeline`, resolved from the
project root's own folder name. The uploaded archive was named `server`, and
with that name every audio, video and image case fails with
`No module named 'deeptruth_pipeline'` and returns neutral. Confirmed
reproducible and fixed by the rename. The folder containing `config.py`,
`inferencers/`, `server/` and `client/` must be named exactly
`deeptruth_pipeline`.
