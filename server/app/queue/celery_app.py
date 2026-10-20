from __future__ import annotations

import os

from celery import Celery

from app.queue.redis_client import REDIS_URL

BROKER_URL = os.getenv("CELERY_BROKER_URL", REDIS_URL)
RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", REDIS_URL)

ANALYSIS_QUEUE = os.getenv("CELERY_ANALYSIS_QUEUE", "analysis")

celery_app = Celery("deeptruth", broker=BROKER_URL, backend=RESULT_BACKEND)

celery_app.conf.update(
    # json only, so a message can never run pickled code
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    timezone="UTC",
    enable_utc=True,

    task_default_queue=ANALYSIS_QUEUE,
    task_routes={"app.queue.tasks.analyze_case": {"queue": ANALYSIS_QUEUE}},

    # take one job at a time so a free worker is never starved
    worker_prefetch_multiplier=1,

    # redeliver the job if the worker dies mid-run
    task_acks_late=True,
    task_reject_on_worker_lost=True,

    task_track_started=True,

    # recycle child processes to give memory back
    worker_max_tasks_per_child=int(os.getenv("CELERY_MAX_TASKS_PER_CHILD", "8")),

    task_soft_time_limit=int(os.getenv("CELERY_SOFT_TIME_LIMIT", "1800")),
    task_time_limit=int(os.getenv("CELERY_TIME_LIMIT", "2100")),

    result_expires=int(os.getenv("CELERY_RESULT_EXPIRES", "86400")),

    broker_connection_retry_on_startup=True,
)

celery_app.autodiscover_tasks(["app.queue"], related_name="tasks", force=True)
