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

## Artifact maps

A score is not evidence. "84% synthetic" says what the model concluded, not
what it saw, and gives an operator no way to separate a real detection from a
confident mistake. Image cases now carry a heat map of the regions that drove
the verdict, rendered over the 224×224 tensor the models actually saw.

Grad-CAM, adapted for a ViT: the 196 patch tokens are a 14×14 grid, so the
usual arithmetic applies once the sequence is reshaped. Each checkpoint that
claimed something (P(fake) ≥ 0.10) contributes one map, and they are fused
weighted by that claim — a checkpoint reporting 1% synthetic is not asserting
anything is wrong anywhere, so averaging its map in would only dilute the ones
that are.

**The layer to hook is the one non-obvious detail.** `ViTForImageClassification`
classifies from the CLS token alone, so at the *output* of the last encoder
block the patch tokens feed nothing downstream and `d(logit)/d(patch)` is
exactly zero — hooking there yields an all-zero map, silently, on every image.
The hook goes on `layernorm_before` of the last block instead, ahead of that
block's attention, where patch tokens still reach CLS.

The face-crop checkpoint sees a crop, so its map is in crop coordinates and is
projected back into frame coordinates before fusion. Compositing it directly
would draw the mouth over somebody's shoulder, and a forensic tool pointing at
the wrong region is worse than one pointing nowhere.

### How much to trust them

Measured by deletion test — occlude the top 10% of the map and see whether
P(fake) drops more than occluding a random 10%:

| | |
|---|---|
| Clearly faithful | 5 of 12 checkpoint/image pairs |
| No measurable effect either way | 6 |
| Worse than random | 1 (`ffpp_facecrop` on one image) |

Nearly all the inconclusive cases are on an image every checkpoint scored above
0.95, where occluding *anything* barely moves the score — the evidence is
global texture, not a local edit. That is a real property of the input, not a
defect in the map, and it is why the UI reads `localised` before choosing its
wording: a concentrated map is described as the region that drove the verdict,
a diffuse one explicitly as attention rather than a marked-up area.

Grad-CAM is a saliency approximation, not a segmentation, and 14×14 localises
to about a sixteenth of the frame per cell — enough to say "the mouth region",
not enough to trace a splice boundary. It also explains the *model*, not the
image: a checkpoint keying on a JPEG artefact will produce a confident map over
that artefact.

Costs one backward pass per contributing checkpoint. Set
`DEEPTRUTH_ARTIFACT_MAPS=0` to skip it; the verdict is identical either way,
since the maps are computed separately and never feed back into the score.

## Server-side media fetch

`POST /cases` takes **either** `file` (upload the bytes) **or** `media_url`
(the server downloads it). The extension prefers the URL: it already had to
download the media in order to upload it, so the user was paying for the
transfer twice, once each way, on every scan.

The download runs in the **worker**, not the API. A transfer whose size and
speed the caller does not control must never occupy the event loop, and
network failure is precisely the transient condition the retry policy already
handles — a 5xx or a timeout is retried, a 404 or a blocked host is not.

### SSRF

"Fetch this URL for me", run by a server inside a private network, is a
request-forgery primitive: the caller picks the URL and anything the host can
reach becomes reachable by someone who cannot reach it directly — instance
metadata at `169.254.169.254`, Redis on `localhost:6379`, an admin panel on a
`10.x` address. Every fetch is therefore constrained on four axes:

| | |
|---|---|
| scheme | `http`/`https` only — no `file:`, `gopher:`, `data:` |
| address | must resolve to a publicly routable IP, **re-checked per redirect hop** |
| size | streamed with a running cap, aborted mid-body; `Content-Length` is a hint, not a promise |
| time | connect, read and total deadlines |

Redirects are followed **manually**, one hop at a time, each destination
re-resolved and re-validated. Letting the HTTP client follow them would make
every other check pointless: a public URL is free to 302 onto
`http://169.254.169.254/`. Re-validating also closes DNS rebinding, where the
name resolves publicly at submit time and privately a moment later.

Rejections are deliberately vague about *which* address was refused — a
precise answer turns this endpoint into a working internal port scanner, one
URL at a time.

`URL_FETCH_ALLOW_PRIVATE=1` opts a deployment out for genuinely internal media;
`URL_FETCH_ALLOWED_HOSTS` is stronger still and worth setting if only a handful
of CDNs are ever needed.

### Fallbacks

Server-side fetch is not always possible, so the extension keeps the upload
path: `blob:`/`data:` URLs mean nothing outside the page, and media behind a
login is fetchable by the browser (which holds the session) but not by the
server (which arrives anonymous). The extension tries the URL first and falls
back to uploading the bytes without troubling the user.

Repeat sightings skip everything: a URL already in the index resolves to a
content hash and from there to a stored verdict, measured at **57 ms against
~10 s** for download plus inference.

## Abuse controls

Two independent guards on the gateway, both in `server/app/security/`.

**Rate limiting.** `POST /cases` accepts a 500 MB file and queues a job that
occupies one of two worker slots for minutes, so a client looping it takes the
pipeline away from everyone else with a handful of requests. Limits are
enforced as a sliding-window log in Redis, admitted by a single Lua script so
the decision is atomic — verified by admitting exactly 25 of 200 requests fired
50-wide. Counters live in Redis rather than process memory, so running several
API workers does not multiply the effective limit.

A fixed-window counter would have been cheaper, but it lets a caller send the
full allowance at 0:59 and again at 1:01 — double the intended rate at the
moment a burst hurts most.

| Bucket | Default | Routes |
|--------|---------|--------|
| upload | 10/60s  | `POST /cases` |
| write  | 60/60s  | `PATCH`, `DELETE /cases/{id}` |
| read   | 300/60s | case reads, `/stats`, `/queue*`, `/cache/lookup`, `/health` |
| stream | 30/60s  | opening `/queue/stream` |

Responses carry `RateLimit-Limit`, `-Remaining` and `-Reset`; a 429 also carries
`Retry-After`. If Redis is unreachable the limiter **fails open** and omits
those headers rather than advertising a budget it did not check — a limiter
outage must not become an API outage, and the expensive path is already refused
with a 503 in that state.

`/queue/stream` additionally caps **concurrent** open streams per client
(`MAX_CONCURRENT_STREAMS`, default 5). Limiting opens is not sufficient on its
own: each live stream holds a Redis pub/sub connection and a thread, so a
client that opens streams and never closes them exhausts the pool while staying
inside any per-minute limit.

**Content validation.** `validate_file` checks the `Content-Type` the *caller
sent*, which is a string they chose. Uploads are now also matched against the
container signatures of the formats the pipeline supports, and a mismatch is
refused at the gateway. This is not about code execution — nothing here runs
the file — it is that undecodable content otherwise reaches a worker, occupies
a slot for a model load, and returns "inconclusive", which reads like a
considered verdict rather than "this was never an image". 40 KB of `urandom`
declared as `image/jpeg` produced exactly that before the check existed.

Cross-modality mistakes get a specific message (*"this is an MP4 video, but it
was submitted as image"*) since that one is usually an honest client bug.

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
| `RATE_LIMIT_ENABLED` | 1 | Set to 0 only for local load testing |
| `UPLOAD_RATE_LIMIT` | `10/60` | `<count>/<seconds>`; same form for the others |
| `READ_RATE_LIMIT` | `300/60` | Reads, including `/health` |
| `WRITE_RATE_LIMIT` | `60/60` | PATCH and DELETE |
| `STREAM_RATE_LIMIT` | `30/60` | SSE opens |
| `MAX_CONCURRENT_STREAMS` | 5 | Simultaneous open streams per client |
| `TRUSTED_PROXY_HOPS` | 0 | See below — leave at 0 unless behind a proxy |
| `URL_FETCH_ALLOW_PRIVATE` | 0 | Allow fetching from private/loopback addresses |
| `URL_FETCH_ALLOWED_HOSTS` | — | Comma-separated host allowlist; nothing else is fetchable |
| `URL_FETCH_MAX_REDIRECTS` | 5 | Each hop is re-validated |
| `URL_FETCH_TOTAL_TIMEOUT` | 300 | Seconds for a whole transfer |
| `SSE_MAX_STREAM_SECONDS` | 600 | Stream lifetime before the client reconnects |
| `DEEPTRUTH_ARTIFACT_MAPS` | 1 | Set to 0 to skip artifact-map generation |
| `DEEPTRUTH_ARTIFACT_MAP_MIN_SCORE` | 0.10 | P(fake) below which a checkpoint gets no map |
| `ARTIFACT_DIR` | `artifacts` | Where rendered maps are published |

`TRUSTED_PROXY_HOPS` deserves a note: at 0 the socket peer is treated as the
client and `X-Forwarded-For` is ignored entirely, because that header is
caller-supplied and honouring it would let anyone reset their own bucket by
inventing an IP. Set it to the real number of reverse proxies when deploying
behind nginx or a load balancer, and the client is read that many entries in
from the right of the header — the hops trusted infrastructure appended, not
the ones a caller sent.

No new dependency was needed: the limiter uses the `redis` client already
required by the queue.

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
