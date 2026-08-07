"""Health check endpoint."""

from fastapi import APIRouter
from app.models import HealthResponse
from app.database import DB_PATH, get_db
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

    return {
        "status": "ok",
        "version": "1.0.0",
        "db": db_status,
    }
