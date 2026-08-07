"""Task abstraction with an in-memory queue.

For the 30% demo we run inline. Swap InMemoryQueue for a Celery / Redis-backed
queue later. The contract is: submit(task) -> task_id; get(task_id) -> Task.

If async_workers > 0, tasks run in worker threads (good enough for I/O-bound
queueing during the demo). For real concurrency across many GPU jobs you'll
want process-level workers (Celery, RQ).
"""
from __future__ import annotations
import logging
import threading
import time
import traceback
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from queue import Queue
from typing import Any, Callable

log = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    id: str = ""
    media_path: Path = Path()
    media_kind_hint: str | None = None
    options: dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.QUEUED
    stage: str = ""
    progress: float = 0.0
    result: Any = None
    error: dict[str, str] | None = None
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None


class TaskQueue(ABC):
    @abstractmethod
    def submit(self, task: Task) -> str: ...

    @abstractmethod
    def get(self, task_id: str) -> Task | None: ...


class InMemoryQueue(TaskQueue):
    def __init__(self, runner: Callable[[Task], None],
                 async_workers: int = 0,
                 max_records: int = 1000):
        self._runner = runner
        self._tasks: dict[str, Task] = {}
        self._lock = threading.Lock()
        self._max = max_records
        self._async = async_workers > 0
        if self._async:
            self._q: Queue = Queue()
            for _ in range(async_workers):
                threading.Thread(target=self._worker_loop, daemon=True).start()

    def submit(self, task: Task) -> str:
        if not task.id:
            task.id = "t_" + uuid.uuid4().hex[:24]
        with self._lock:
            self._tasks[task.id] = task
            self._evict()
        if self._async:
            self._q.put(task.id)
        else:
            self._run(task)
        return task.id

    def get(self, task_id: str) -> Task | None:
        with self._lock:
            return self._tasks.get(task_id)

    def _evict(self):
        if len(self._tasks) <= self._max:
            return
        finished = sorted(
            (t for t in self._tasks.values()
             if t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED)),
            key=lambda t: t.completed_at or t.created_at,
        )
        for t in finished[: len(self._tasks) - self._max]:
            self._tasks.pop(t.id, None)

    def _worker_loop(self):
        while True:
            tid = self._q.get()
            t = self.get(tid)
            if t is not None:
                self._run(t)

    def _run(self, task: Task):
        task.status = TaskStatus.RUNNING
        try:
            self._runner(task)
            task.status = TaskStatus.COMPLETED
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = {"type": type(e).__name__,
                          "message": str(e),
                          "trace": traceback.format_exc()}
            log.exception(f"task {task.id} failed")
        finally:
            task.completed_at = time.time()
            task.progress = 1.0
