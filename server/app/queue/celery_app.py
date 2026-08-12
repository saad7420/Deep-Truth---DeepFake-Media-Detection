"""Celery application.

Imported by both sides:
  * the API process, only to `.delay()` work onto the queue
  * `celery -A app.queue.celery_app worker`, which actually runs it

The settings below are not defaults worth skimming past — several of them are
what make the queue behave correctly for *this* workload, where one task is a
multi-minute CPU-bound torch job rather than a fast IO call.
"""
from __future__ import annotations

import os

from celery import Celery

from app.queue.redis_client import REDIS_URL

# Broker and result backend both default to REDIS_URL. They are split out so a
# deployment can move results onto a different database (or drop them entirely)
# without touching the broker.
BROKER_URL = os.getenv("CELERY_BROKER_URL", REDIS_URL)
RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", REDIS_URL)

# Queue names. Analysis is deliberately its own queue so that a future light
# task (thumbnailing, notification) is never stuck behind a 4-minute ViViT run.
ANALYSIS_QUEUE = os.getenv("CELERY_ANALYSIS_QUEUE", "analysis")

celery_app = Celery("deeptruth", broker=BROKER_URL, backend=RESULT_BACKEND)

celery_app.conf.update(
    # ── Serialization ────────────────────────────────────────────────────────
    # JSON only. Pickle would let a task argument execute code on the worker,
    # and every argument here is a plain id or path anyway.
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    timezone="UTC",
    enable_utc=True,

    # ── Routing ──────────────────────────────────────────────────────────────
    task_default_queue=ANALYSIS_QUEUE,
    task_routes={"app.queue.tasks.analyze_case": {"queue": ANALYSIS_QUEUE}},

    # ── Fair dispatch across workers (FE-2) ──────────────────────────────────
    # Celery's default is to prefetch 4 tasks per worker process the moment
    # they appear. With tasks this long that is actively harmful: the first
    # worker to connect grabs four jobs and sits on three of them while a
    # second, idle worker has nothing to do. Prefetching exactly one means a
    # job is reserved only when a process is genuinely free, so N workers
    # really do process N files at once.
    worker_prefetch_multiplier=1,

    # ── Crash safety (FE-3) ──────────────────────────────────────────────────
    # Acknowledge only after the task returns. If a worker is killed mid-run
    # (OOM on a large video is the realistic case) the broker still holds the
    # message and redelivers it to another worker, instead of the job
    # vanishing with the process.
    task_acks_late=True,
    task_reject_on_worker_lost=True,

    # `started` is not tracked by default, which would leave every running job
    # indistinguishable from a queued one — exactly the distinction the UI
    # needs to show.
    task_track_started=True,

    # ── Memory hygiene ───────────────────────────────────────────────────────
    # Each child holds several ViT/ViViT models plus torch's allocator arenas.
    # Recycling the process after a number of tasks returns that memory to the
    # OS and bounds the damage from any per-run leak in the inference stack.
    worker_max_tasks_per_child=int(os.getenv("CELERY_MAX_TASKS_PER_CHILD", "8")),

    # ── Time limits ──────────────────────────────────────────────────────────
    # CPU-only inference over a long video is slow, so these are generous. The
    # soft limit raises SoftTimeLimitExceeded inside the task, which tasks.py
    # catches and records as a real failure reason; the hard limit is the
    # backstop that kills a truly wedged process.
    task_soft_time_limit=int(os.getenv("CELERY_SOFT_TIME_LIMIT", "1800")),
    task_time_limit=int(os.getenv("CELERY_TIME_LIMIT", "2100")),

    # Results are a debugging aid here — the durable record is SQLite plus the
    # job state hash — so they do not need to outlive the day.
    result_expires=int(os.getenv("CELERY_RESULT_EXPIRES", "86400")),

    broker_connection_retry_on_startup=True,
)

# Import the task module for its side effect of registering the task. Done at
# the bottom so `celery -A app.queue.celery_app` finds it without the caller
# having to pass `--include`.
celery_app.autodiscover_tasks(["app.queue"], related_name="tasks", force=True)
