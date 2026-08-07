"""
Async SQLite database layer.
Switch to PostgreSQL by replacing the DB_URL and driver.
"""

import os
import aiosqlite

DB_PATH = os.getenv("DB_PATH", "forensics.db")

CREATE_CASES_TABLE = """
CREATE TABLE IF NOT EXISTS cases (
    id              TEXT PRIMARY KEY,
    case_id         TEXT NOT NULL UNIQUE,
    title           TEXT NOT NULL,
    media_type      TEXT NOT NULL CHECK(media_type IN ('image','video','audio')),
    status          TEXT NOT NULL DEFAULT 'processing'
                        CHECK(status IN ('processing','authentic','manipulated','inconclusive','failed')),
    risk_score      REAL NOT NULL DEFAULT 0,
    synthetic_likelihood REAL NOT NULL DEFAULT 0,
    file_name       TEXT,
    file_url        TEXT,
    file_size       INTEGER,
    user_id         TEXT,
    notes           TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

CREATE_ANALYSIS_TABLE = """
CREATE TABLE IF NOT EXISTS analysis_results (
    id              TEXT PRIMARY KEY,
    case_id         TEXT NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
    model_name      TEXT NOT NULL,
    confidence      REAL NOT NULL,
    label           TEXT NOT NULL,
    details         TEXT,          -- JSON blob
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


async def get_db() -> aiosqlite.Connection:
    """Yield a connection; caller must close it."""
    conn = await aiosqlite.connect(DB_PATH)
    conn.row_factory = aiosqlite.Row
    return conn


async def init_db() -> None:
    """Create tables on first run."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA journal_mode=WAL;")
        await db.execute("PRAGMA foreign_keys=ON;")
        await db.execute(CREATE_CASES_TABLE)
        await db.execute(CREATE_ANALYSIS_TABLE)
        await db.commit()
    print(f"[DB] Initialised at {DB_PATH}")
