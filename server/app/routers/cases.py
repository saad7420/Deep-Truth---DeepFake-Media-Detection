from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit

import aiosqlite
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Response,
    UploadFile,
)
from starlette.concurrency import run_in_threadpool

from app.database import get_db
from app.models import (
    CaseListResponse,
    CaseResponse,
    CaseUpdate,
    DashboardStats,
)
from app.queue import cache, state
from app.queue.redis_client import BrokerUnavailable, require as require_broker
from app.queue.tasks import MAX_ATTEMPTS, analyze_case
from app.security import ratelimit
from app.security.media import validate_signature
from app.security.urlfetch import UrlRejected, validate_url
from app.services.analyser import validate_file
from app.services.report_pdf import build_report

router = APIRouter()

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")


def _row_to_dict(row: aiosqlite.Row) -> dict:
    return dict(row)


def _build_response(row: dict, results: list[dict] | None = None,
                    job: dict | None = None) -> dict:
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
        "job": job,
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


# copy a stored verdict onto a new case with fresh row ids
async def _replay_cached(case_db_id: str, payload: dict) -> None:
    rows = cache.rehydrate_rows(payload, case_db_id)

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
            (payload["status"], payload["risk"], payload["likelihood"], case_db_id),
        )

        source_file = payload.get("sourceFile")
        if source_file and source_file.get("url"):
            await db.execute(
                """
                UPDATE cases
                   SET file_name = COALESCE(file_name, ?),
                       file_url  = ?,
                       file_size = ?
                 WHERE id = ? AND file_url IS NULL
                """,
                (source_file.get("name"), source_file["url"],
                 source_file.get("size"), case_db_id),
            )
        for r in rows:
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
    finally:
        await db.close()

    print(f"[Cache HIT] case={case_db_id} status={payload['status']} "
          f"(originally computed for {payload.get('sourceCaseId')})")


@router.get("/cases", response_model=CaseListResponse,
            dependencies=[Depends(ratelimit.read)])
async def list_cases(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    media_type: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    q: Optional[str] = Query(None, description="Search title"),
):
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

        cases = []
        for r in rows:
            d = _row_to_dict(r)
            job = state.get_job(d["id"]) if d["status"] == "processing" else None
            cases.append(_build_response(d, job=job))
    finally:
        await db.close()

    return {"cases": cases, "total": total, "page": page, "page_size": page_size}


# upload endpoint: validate, hash, check the cache, then queue
@router.post("/cases", response_model=CaseResponse, status_code=201,
             dependencies=[Depends(ratelimit.upload)])
async def create_case(
    title: str = Form(...),
    media_type: str = Form(...),
    user_id: Optional[str] = Form(None),
    notes: Optional[str] = Form(None),
    source_url: Optional[str] = Form(
        None,
        description="Page-media URL this file was captured from. Used as a "
                    "secondary cache key by the browser extension.",
    ),
    media_url: Optional[str] = Form(
        None,
        description="Fetch the media from this URL server-side instead of "
                    "uploading it. Mutually exclusive with `file`.",
    ),
    file: Optional[UploadFile] = File(None),
):
    if media_type not in ("image", "video", "audio"):
        raise HTTPException(400, "media_type must be one of: image, video, audio")

    has_file = file is not None and (file.filename or "") != ""
    if has_file == bool(media_url):
        raise HTTPException(
            400,
            "Provide exactly one of `file` (upload the bytes) or `media_url` "
            "(have the server fetch them).",
        )

    try:
        # refuse early so a rejected request leaves no orphan case
        require_broker()
    except BrokerUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc

    content: bytes | None = None
    sha: str = ""
    file_name: str | None = None
    file_url: str | None = None
    file_size: int | None = None
    file_path: Path | None = None

    if has_file:
        content = await file.read()
        error = validate_file(media_type, file.content_type or "", len(content))
        if error:
            raise HTTPException(422, error)

        # check the real bytes, not the header the client sent
        error = validate_signature(media_type, content)
        if error:
            raise HTTPException(422, error)

        # same bytes give the same hash, so the stored verdict can be reused
        sha = cache.content_hash(content)
        cached = cache.lookup(media_type, sha)

        ext = Path(file.filename or "upload").suffix or ".bin"
        safe_name = f"{uuid.uuid4().hex}{ext}"
        file_path = UPLOAD_DIR / safe_name
        file_path.write_bytes(content)
        file_name = file.filename
        file_url = f"{BASE_URL}/uploads/{safe_name}"
        file_size = len(content)

    else:
        try:
            # cheap safety check now, the real download happens in the worker
            media_url = await run_in_threadpool(validate_url, media_url)
        except UrlRejected as exc:
            raise HTTPException(400, str(exc)) from exc

        sha, cached = cache.lookup_by_url(media_type, media_url)
        sha = sha or ""

        file_name = Path(urlsplit(media_url).path).name or "download"

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
             file_name, file_url, file_size, user_id, notes),
        )
        await db.commit()
    finally:
        await db.close()

    if source_url and sha:
        cache.remember_url(source_url, sha)

    if cached:
        await _replay_cached(db_id, cached)
        state.mark_cached(db_id, case_id=case_id, media_type=media_type,
                          content_hash=sha, source_url=source_url or media_url or "")
    else:
        try:
            # hand the job to a worker
            async_result = analyze_case.delay(
                db_id, media_type,
                str(file_path) if file_path else "",
                sha, source_url or "", media_url or "",
            )
        except Exception as exc:
            await _mark_case_failed(db_id)
            raise HTTPException(
                503,
                f"Could not queue the analysis job: {exc}. The case was "
                f"recorded as failed; resubmit once the queue is healthy.",
            ) from exc

        state.mark_queued(db_id, case_id=case_id, media_type=media_type,
                          celery_task_id=async_result.id,
                          max_attempts=MAX_ATTEMPTS,
                          content_hash=sha,
                          source_url=source_url or media_url or "")

    db = await get_db()
    try:
        async with db.execute("SELECT * FROM cases WHERE id = ?", (db_id,)) as cur:
            row = _row_to_dict(await cur.fetchone())
        results = await _fetch_analysis(db, db_id)
    finally:
        await db.close()

    return _build_response(row, results, state.get_job(db_id))


async def _mark_case_failed(case_db_id: str) -> None:
    db = await get_db()
    try:
        await db.execute(
            "UPDATE cases SET status='failed', updated_at=datetime('now') WHERE id=?",
            (case_db_id,),
        )
        await db.commit()
    finally:
        await db.close()


@router.get("/cases/{case_id}", response_model=CaseResponse,
            dependencies=[Depends(ratelimit.read)])
async def get_case(case_id: str):
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

    return _build_response(row_dict, results, state.get_job(row_dict["id"]))


@router.get("/cases/{case_id}/report.pdf",
            dependencies=[Depends(ratelimit.read)])
async def case_report_pdf(case_id: str):
    db = await get_db()
    try:
        async with db.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)) as cur:
            row = await cur.fetchone()
        if row is None:
            raise HTTPException(404, f"Case '{case_id}' not found.")
        case = _row_to_dict(row)
        results = await _fetch_analysis(db, case["id"])
    finally:
        await db.close()

    # a report now would have no findings
    if case["status"] == "processing":
        raise HTTPException(
            409,
            "This case is still being analysed. A report generated now would "
            "record no findings; wait for it to settle and try again.",
        )

    try:
        pdf = await run_in_threadpool(
            build_report, case, results, state.get_job(case["id"])
        )
    except Exception as exc:
        raise HTTPException(
            500, f"Could not render the report: {exc}"
        ) from exc

    filename = f"deeptruth-{case_id}.pdf"
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Length": str(len(pdf)),
        },
    )


@router.patch("/cases/{case_id}", response_model=CaseResponse,
              dependencies=[Depends(ratelimit.write)])
async def update_case(case_id: str, body: CaseUpdate):
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

    return _build_response(row, results, state.get_job(internal_id))


@router.delete("/cases/{case_id}", status_code=204,
               dependencies=[Depends(ratelimit.write)])
async def delete_case(case_id: str):
    db = await get_db()
    try:
        async with db.execute("SELECT id, file_url FROM cases WHERE case_id = ?", (case_id,)) as cur:
            row = await cur.fetchone()
        if row is None:
            raise HTTPException(404, f"Case '{case_id}' not found.")

        await db.execute("DELETE FROM cases WHERE id = ?", (row["id"],))
        await db.commit()

        if row["file_url"]:
            async with db.execute(
                "SELECT COUNT(*) FROM cases WHERE file_url = ?", (row["file_url"],)
            ) as cur:
                still_referenced = (await cur.fetchone())[0]

            # cache-hit cases share one file, only delete it when nothing uses it
            if still_referenced == 0:
                fname = row["file_url"].split("/uploads/")[-1]
                fpath = UPLOAD_DIR / fname
                if fpath.exists():
                    fpath.unlink(missing_ok=True)
    finally:
        await db.close()


@router.get("/stats", response_model=DashboardStats,
            dependencies=[Depends(ratelimit.read)])
async def get_stats():
    db = await get_db()
    try:
        async with db.execute(
            """
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN status='authentic'   THEN 1 ELSE 0 END) AS authentic,
                SUM(CASE WHEN status='manipulated' THEN 1 ELSE 0 END) AS manipulated,
                SUM(CASE WHEN status='processing'  THEN 1 ELSE 0 END) AS processing,
                SUM(CASE WHEN status='inconclusive' THEN 1 ELSE 0 END) AS inconclusive,
                SUM(CASE WHEN status='failed'      THEN 1 ELSE 0 END) AS failed,
                SUM(CASE WHEN media_type='image'   THEN 1 ELSE 0 END) AS images,
                SUM(CASE WHEN media_type='video'   THEN 1 ELSE 0 END) AS videos,
                SUM(CASE WHEN media_type='audio'   THEN 1 ELSE 0 END) AS audios,
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
        "inconclusive": r["inconclusive"] or 0,
        "failed": r["failed"] or 0,
        "byMediaType": {
            "image": r["images"] or 0,
            "video": r["videos"] or 0,
            "audio": r["audios"] or 0,
        },
        "avgRiskScore": round(r["avg_risk"] or 0, 1),
    }
