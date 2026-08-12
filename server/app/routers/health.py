"""Health check endpoint."""

from fastapi import APIRouter
from app.models import HealthResponse
from app.database import DB_PATH, get_db
from app.queue.redis_client import ping
from app.queue.tasks import worker_snapshot
import os

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health():
    db_status = "unavailable"
    try:
        db = await get_db()
        await db.execute("SELECT 1")
        await db.close()
        db_status = "ok"
    except Exception as exc:
        db_status = f"error: {exc}"

    # Redis and the worker pool are now hard dependencies of the upload path
    # (POST /cases returns 503 without them), so a health check that reports
    # only the database would call the service healthy while it is unable to
    # accept a single case.
    redis_ok = ping()
    workers = worker_snapshot() if redis_ok else {"online": 0}

    if db_status != "ok" or not redis_ok:
        overall = "degraded"
    elif workers.get("online", 0) == 0:
        # Uploads are still accepted — they queue, and run when a worker
        # appears — so this is not "down", but nothing is being processed.
        overall = "idle"
    else:
        overall = "ok"

    return {
        "status": overall,
        "version": "1.0.0",
        "db": db_status,
        "redis": "ok" if redis_ok else "unavailable",
        "workers": workers.get("online", 0),
    }
