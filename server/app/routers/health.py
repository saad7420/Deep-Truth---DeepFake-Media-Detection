from fastapi import APIRouter, Depends
from app.models import HealthResponse
from app.database import DB_PATH, get_db
from app.queue.redis_client import ping
from starlette.concurrency import run_in_threadpool

from app.queue.tasks import cached_worker_snapshot
from app.security import ratelimit
import os

router = APIRouter()


@router.get("/health", response_model=HealthResponse,
            dependencies=[Depends(ratelimit.read)])
async def health():
    db_status = "unavailable"
    try:
        db = await get_db()
        await db.execute("SELECT 1")
        await db.close()
        db_status = "ok"
    except Exception as exc:
        db_status = f"error: {exc}"

    redis_ok = ping()
    workers = (await run_in_threadpool(cached_worker_snapshot)) if redis_ok else {"online": 0}

    if db_status != "ok" or not redis_ok:
        # db or redis is down
        overall = "degraded"
    # idle: uploads queue up but nothing runs them
    elif workers.get("online", 0) == 0:
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
