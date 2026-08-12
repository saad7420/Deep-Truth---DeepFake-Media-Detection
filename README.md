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
│   └── app/queue/             Celery + Redis orchestration (see below)
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

## Asynchronous task orchestration

Analysis does not run in the API process. `POST /cases` hashes the upload,
asks Redis whether that exact content has been analysed before, and either
replays the stored verdict immediately or publishes a Celery message. Workers
do the inference.

This is not decoration. Before it, a single upload blocked the FastAPI event
loop for minutes — a second user's request simply waited, and the server could
not even shut down cleanly while an analysis was in flight.

```
POST /cases ──> sha256(bytes) ──> Redis cache?
                                    │
                    hit ────────────┤────────── miss
                     │                            │
        replay verdict, case is                Celery ──> worker pool
        terminal before the response           (queue)     (N parallel)
        is sent, no worker involved                          │
                                                   SQLite + cache + SSE
```

**Queue** (FE-1) — one Redis-backed Celery queue, FIFO. `worker_prefetch_multiplier=1`
so a job is reserved only when a process is genuinely free; the default of 4
would let the first worker hoard four multi-minute jobs while another sits idle.

**Parallel workers** (FE-2) — `./run_worker.sh` starts a worker; run it more
than once with different `WORKER_NAME` to add capacity. Concurrency is bounded
by RAM, not cores, because each prefork child loads its own copy of the ViT/ViViT
checkpoints. Two slots is the default on a CPU-only box.

**Retries** (FE-3) — transient failures are retried with exponential backoff and
jitter, up to `ANALYSIS_MAX_ATTEMPTS` (3). Failures that retrying cannot fix —
missing file, empty file, unknown modality — raise `PermanentFailure` and fail
at once rather than burning three runs. `task_acks_late` means a worker killed
mid-run has its job redelivered rather than lost.

**Live state** — the backend pushes every transition (`queued → running →
retrying → succeeded/failed/cached`) on `GET /api/queue/stream` (SSE). The
console subscribes; the extension keeps polling, because a service worker
cannot hold a stream open. Both read the same record, which also rides along on
every case response as `case.job`.

### Cache

Keyed on the sha256 of the file bytes, with a secondary index from normalised
media URL to content hash. The URL index lets the extension ask
`GET /api/cache/lookup?url=…` *before* downloading anything — a repeat sighting
costs one GET instead of a download, an upload and a wait. A URL whose bytes
changed misses rather than returning a stale verdict.

Entries do not expire. A cached verdict is only wrong if the models changed, so
invalidation is by version: bump `DEEPTRUTH_CACHE_VERSION` and every entry is
retired at once. Zero-confidence results are never cached — freezing "we could
not tell" as a file's permanent answer is worse than recomputing it.

### Running the stack

Redis is required. Without it `POST /cases` returns 503 rather than silently
falling back to in-process analysis, which would abandon the ordering,
parallelism and retry guarantees the queue exists to provide.

```bash
sudo apt install -y redis-server
```

Backend (from `server/`), in three terminals:

```bash
pip install -r requirements.txt
```

```bash
python main.py
```

```bash
./run_worker.sh
```

Frontend (from `client/`):

```bash
npm install && npm run dev
```

The API listens on `http://localhost:8000`, the app on
`http://localhost:3000`. The frontend reads `NEXT_PUBLIC_API_URL` from
`client/.env.local`; the backend reads `server/.env`.

`GET /api/health` reports `ok`, `idle` (Redis up, no workers — uploads queue
but nothing runs), or `degraded` (database or Redis unreachable).
`GET /api/queue` gives depth, the worker census and the cache hit rate.

Environment knobs, all optional:

| Variable | Default | Effect |
|----------|---------|--------|
| `REDIS_URL` | `redis://127.0.0.1:6379/0` | Broker, result backend, and cache |
| `CONCURRENCY` | 2 | Parallel slots per worker (`run_worker.sh`) |
| `ANALYSIS_MAX_ATTEMPTS` | 3 | Total attempts before a job is failed |
| `ANALYSIS_RETRY_BASE_DELAY` | 10 | Seconds; backoff base |
| `DEEPTRUTH_CACHE_VERSION` | `v1` | Bump to retire every cached verdict |
| `DEEPTRUTH_CACHE_TTL` | 0 | Seconds; 0 means entries never expire |

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
