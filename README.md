# DeepTruth

Deepfake and synthetic-media detection: the inference pipeline, the FastAPI
backend, the Next.js frontend, and the browser extension in one repo.

This tree is the result of merging the site
(`Deep-Truth-Deep-Fake-Media-Detector`) into the image-capable pipeline. The
backend used to carry its own vendored copy of the pipeline under
`server/app/_dtp_src`, which had drifted and had no image support at all.
There is now exactly one copy of the pipeline — this package — and the server
imports it.

## Layout

```
deeptruth_pipeline/            the Python package AND the project root
├── pipeline.py, registry.py   orchestrator: probe → preprocess → infer
├── preprocessors/             video (frames + MTCNN), image, audio (wav)
├── inferencers/               video (ViViT), image (ViT-B/16), audio (WavLM)
├── ensemble.py                per-modality fusion policy
├── videos_checkpoints/        *_lora_best              — 6 ViViT adapters
├── images_checkpoints/        image_*_lora_best        — 8 ViT adapters
├── train_pipeline/            deeptruth_train.py, reused at inference time
│                              so preprocessing/model build match training
├── server/                    FastAPI: cases, uploads, analysis, SQLite
├── client/                    Next.js 16 frontend
├── extension/                 Chrome extension
└── docs/                      pipeline.md (internals), platform.md (site)
```

## What runs for each modality

| Media | Engine | Models | State |
|-------|--------|--------|-------|
| Video | `VisualForensicsEngine` | 6 ViViT + LoRA checkpoints, `max(face_avg, genvideo)` with a face-frame trust gate | real |
| Image | `ImageForensicsEngine` | 8 ViT-B/16 + LoRA checkpoints, 5 generalist + 3 face, MTCNN crop branch | real |
| Audio | `AudioFakeNetStub` | — | stub, returns confidence 0.0 |

Image support is the new half of this merge: before it, every image case came
back "inconclusive — not yet implemented".

## Running it

Backend (from `server/`):

```bash
pip install -r requirements.txt
```

```bash
python main.py
```

Frontend (from `client/`):

```bash
npm install && npm run dev
```

The API listens on `http://localhost:8000`, the app on
`http://localhost:3000`. The frontend reads `NEXT_PUBLIC_API_URL` from
`client/.env.local`; the backend reads `server/.env`.

CLI, without the web stack:

```bash
python -m deeptruth_pipeline.cli /path/to/media --threshold 0.5
```

## Model paths

`server/app/_dtp.py` is the single place the server binds to the pipeline. It
puts this package on `sys.path` and defaults every `DEEPTRUTH_*` path to the
in-repo location, so no configuration is needed for a local run. Every value
is a `setdefault`, so anything exported in the environment or set in
`server/.env` still wins — point `DEEPTRUTH_CHECKPOINTS` and
`DEEPTRUTH_IMAGE_CHECKPOINTS` elsewhere when the weights live on a mounted
volume.

Both engines share one `Pipeline` instance, so each checkpoint is loaded into
memory once and both reuse the preprocessing cache under `DEEPTRUTH_CACHE`
(`server/_dtp_cache` by default; keyed by a sha256 of the file, so
re-analysing the same upload skips straight to inference).

## Notes for whoever picks this up next

- **peft must be >= 0.19.** The adapters were exported by 0.19.x and their
  `adapter_config.json` carries fields older releases reject. The previous
  `peft<0.15` pin in requirements.txt could not load them at all.
- **Two ways a classifier head is restored.** Seven image checkpoints keep the
  head inside the adapter (`modules_to_save: ["classifier"]`);
  `image_ffpp_lora_best` has no `modules_to_save` and depends on its sibling
  `classifier_head.pt`. A checkpoint with neither would score noise, so
  `inferencers/image.py` drops it from the ensemble instead of trusting it.
- **`confidence == 0.0` means "ignore me"** across every engine. The analyser
  never lets a zero-confidence result produce an authentic/manipulated verdict
  — it reports `inconclusive`.
- **Audio is wired but not switched on.** `inferencers/audio.py` has a real
  WavLM-Large implementation; `server/app/engines/audio/stub.py` documents the
  three-line swap once a checkpoint is available.
