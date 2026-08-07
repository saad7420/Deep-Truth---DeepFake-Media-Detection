"""
Cases router — full CRUD + multipart file upload + async analysis.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import Optional

import aiosqlite
from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.responses import JSONResponse

from app.database import get_db
from app.models import (
    CaseListResponse,
    CaseResponse,
    CaseUpdate,
    DashboardStats,
)
from app.services.analyser import run_analysis, validate_file

router = APIRouter()

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _row_to_dict(row: aiosqlite.Row) -> dict:
    return dict(row)


def _build_response(row: dict, results: list[dict] | None = None) -> dict:
    """Map snake_case DB row → camelCase CaseResponse dict."""
    return {
        "id": row["id"],
        "caseId": row["case_id"],
        "title": row["title"],
        "mediaType": row["media_type"],
        "status": row["status"],
        "riskScore": row["risk_score"],
        "syntheticLikelihood": row["synthetic_likelihood"],
        "fileName": row.get("file_name"),
        "fileUrl": row.get("file_url"),
        "fileSize": row.get("file_size"),
        "userId": row.get("user_id"),
        "notes": row.get("notes"),
        "createdAt": row.get("created_at"),
        "updatedAt": row.get("updated_at"),
        "analysisResults": results or [],
    }


async def _fetch_analysis(db: aiosqlite.Connection, case_db_id: str) -> list[dict]:
    async with db.execute(
        "SELECT * FROM analysis_results WHERE case_id = ? ORDER BY created_at",
        (case_db_id,),
    ) as cur:
        rows = await cur.fetchall()
    return [
        {
            "id": r["id"],
            "case_id": r["case_id"],
            "model_name": r["model_name"],
            "confidence": r["confidence"],
            "label": r["label"],
            "details": json.loads(r["details"]) if r["details"] else None,
            "created_at": r["created_at"],
        }
        for r in rows
    ]


# ── Background task ───────────────────────────────────────────────────────────

async def _analyse_and_store(case_db_id: str, media_type: str, file_path: str):
    """Runs analysis in the background and writes results back to the DB."""
    try:
        risk, likelihood, status, results = await run_analysis(
            case_db_id, media_type, file_path
        )
    except Exception as exc:
        import traceback
        print(f"[Analysis ERROR] case={case_db_id}")
        traceback.print_exc()
        # Mark case as failed so frontend doesn't wait forever
        db = await get_db()
        try:
            await db.execute(
                "UPDATE cases SET status='failed', updated_at=datetime('now') WHERE id=?",
                (case_db_id,),
            )
            await db.commit()
        finally:
            await db.close()
        return

    db = await get_db()
    try:
        await db.execute("PRAGMA foreign_keys=ON;")
        await db.execute(
            """
            UPDATE cases
               SET status = ?, risk_score = ?, synthetic_likelihood = ?,
                   updated_at = datetime('now')
             WHERE id = ?
            """,
            (status, risk, likelihood, case_db_id),
        )
        for r in results:
            await db.execute(
                """
                INSERT INTO analysis_results
                    (id, case_id, model_name, confidence, label, details)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (r["id"], case_db_id, r["model_name"], r["confidence"],
                 r["label"], r["details"]),
            )
        await db.commit()
        print(f"[Analysis DONE] case={case_db_id} status={status} risk={risk}")
    finally:
        await db.close()


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/cases", response_model=CaseListResponse)
async def list_cases(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    media_type: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    q: Optional[str] = Query(None, description="Search title"),
):
    """Return a paginated list of all cases."""
    filters, params = [], []

    if media_type:
        filters.append("media_type = ?")
        params.append(media_type)
    if status:
        filters.append("status = ?")
        params.append(status)
    if q:
        filters.append("title LIKE ?")
        params.append(f"%{q}%")

    where = ("WHERE " + " AND ".join(filters)) if filters else ""
    offset = (page - 1) * page_size

    db = await get_db()
    try:
        async with db.execute(f"SELECT COUNT(*) FROM cases {where}", params) as cur:
            total = (await cur.fetchone())[0]

        async with db.execute(
            f"""
            SELECT * FROM cases {where}
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            """,
            params + [page_size, offset],
        ) as cur:
            rows = await cur.fetchall()

        cases = [_build_response(_row_to_dict(r)) for r in rows]
    finally:
        await db.close()

    return {"cases": cases, "total": total, "page": page, "page_size": page_size}


@router.post("/cases", response_model=CaseResponse, status_code=201)
async def create_case(
    background_tasks: BackgroundTasks,
    title: str = Form(...),
    media_type: str = Form(...),
    user_id: Optional[str] = Form(None),
    notes: Optional[str] = Form(None),
    file: UploadFile = File(...),
):
    """
    Create a new forensic case.
    Accepts multipart/form-data with the evidence file.
    """
    # ── Validate ──────────────────────────────────────────────────────────────
    if media_type not in ("image", "video", "audio"):
        raise HTTPException(400, "media_type must be one of: image, video, audio")

    content = await file.read()
    error = validate_file(media_type, file.content_type or "", len(content))
    if error:
        raise HTTPException(422, error)

    # ── Persist file ──────────────────────────────────────────────────────────
    ext = Path(file.filename or "upload").suffix or ".bin"
    safe_name = f"{uuid.uuid4().hex}{ext}"
    file_path = UPLOAD_DIR / safe_name
    file_path.write_bytes(content)
    file_url = f"{BASE_URL}/uploads/{safe_name}"

    # ── Write case row ────────────────────────────────────────────────────────
    case_id = f"CASE-{uuid.uuid4().hex[:8].upper()}"
    db_id = str(uuid.uuid4())

    db = await get_db()
    try:
        await db.execute("PRAGMA foreign_keys=ON;")
        await db.execute(
            """
            INSERT INTO cases
                (id, case_id, title, media_type, status, risk_score,
                 synthetic_likelihood, file_name, file_url, file_size, user_id, notes)
            VALUES (?, ?, ?, ?, 'processing', 0, 0, ?, ?, ?, ?, ?)
            """,
            (db_id, case_id, title, media_type,
             file.filename, file_url, len(content), user_id, notes),
        )
        await db.commit()

        async with db.execute("SELECT * FROM cases WHERE id = ?", (db_id,)) as cur:
            row = _row_to_dict(await cur.fetchone())
    finally:
        await db.close()

    # ── Queue analysis in background ──────────────────────────────────────────
    background_tasks.add_task(_analyse_and_store, db_id, media_type, str(file_path))

    return _build_response(row)


@router.get("/cases/{case_id}", response_model=CaseResponse)
async def get_case(case_id: str):
    """Fetch a single case including its analysis results."""
    db = await get_db()
    try:
        async with db.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)) as cur:
            row = await cur.fetchone()
        if row is None:
            raise HTTPException(404, f"Case '{case_id}' not found.")
        row_dict = _row_to_dict(row)
        results = await _fetch_analysis(db, row_dict["id"])
    finally:
        await db.close()

    return _build_response(row_dict, results)


@router.patch("/cases/{case_id}", response_model=CaseResponse)
async def update_case(case_id: str, body: CaseUpdate):
    """Partially update a case (title, status, scores, notes)."""
    db = await get_db()
    try:
        async with db.execute("SELECT id FROM cases WHERE case_id = ?", (case_id,)) as cur:
            found = await cur.fetchone()
            if found is None:
                raise HTTPException(404, f"Case '{case_id}' not found.")
            internal_id = found["id"]

        updates, vals = [], []
        if body.title is not None:
            updates.append("title = ?"); vals.append(body.title)
        if body.status is not None:
            updates.append("status = ?"); vals.append(body.status)
        if body.risk_score is not None:
            updates.append("risk_score = ?"); vals.append(body.risk_score)
        if body.synthetic_likelihood is not None:
            updates.append("synthetic_likelihood = ?"); vals.append(body.synthetic_likelihood)
        if body.notes is not None:
            updates.append("notes = ?"); vals.append(body.notes)

        if updates:
            updates.append("updated_at = datetime('now')")
            await db.execute(
                f"UPDATE cases SET {', '.join(updates)} WHERE id = ?",
                vals + [internal_id],
            )
            await db.commit()

        async with db.execute("SELECT * FROM cases WHERE id = ?", (internal_id,)) as cur:
            row = _row_to_dict(await cur.fetchone())
        results = await _fetch_analysis(db, internal_id)
    finally:
        await db.close()

    return _build_response(row, results)


@router.delete("/cases/{case_id}", status_code=204)
async def delete_case(case_id: str):
    """Permanently delete a case and its analysis results."""
    db = await get_db()
    try:
        async with db.execute("SELECT id, file_url FROM cases WHERE case_id = ?", (case_id,)) as cur:
            row = await cur.fetchone()
        if row is None:
            raise HTTPException(404, f"Case '{case_id}' not found.")

        # Remove uploaded file
        if row["file_url"]:
            fname = row["file_url"].split("/uploads/")[-1]
            fpath = UPLOAD_DIR / fname
            if fpath.exists():
                fpath.unlink(missing_ok=True)

        await db.execute("DELETE FROM cases WHERE id = ?", (row["id"],))
        await db.commit()
    finally:
        await db.close()


@router.get("/stats", response_model=DashboardStats)
async def get_stats():
    """Aggregate dashboard statistics."""
    db = await get_db()
    try:
        async with db.execute(
            """
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN status='authentic'   THEN 1 ELSE 0 END) AS authentic,
                SUM(CASE WHEN status='manipulated' THEN 1 ELSE 0 END) AS manipulated,
                SUM(CASE WHEN status='processing'  THEN 1 ELSE 0 END) AS processing,
                AVG(risk_score) AS avg_risk
            FROM cases
            """
        ) as cur:
            r = dict(await cur.fetchone())
    finally:
        await db.close()

    return {
        "totalCases": r["total"] or 0,
        "authentic": r["authentic"] or 0,
        "manipulated": r["manipulated"] or 0,
        "processing": r["processing"] or 0,
        "avgRiskScore": round(r["avg_risk"] or 0, 1),
    }
