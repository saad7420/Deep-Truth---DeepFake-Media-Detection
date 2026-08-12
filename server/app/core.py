"""
App factory — wires up CORS, routes, startup/shutdown hooks.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from app.database import init_db
from app.routers import cases, health, queue


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Initialise the SQLite database on startup."""
    await init_db()
    yield


def create_app() -> FastAPI:
    application = FastAPI(
        title="ForensicsHub API",
        description="Deepfake & synthetic media detection backend.",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )

    # ── CORS ─────────────────────────────────────────────────────────────────
    # Two classes of caller reach this API from a browser:
    #   1. the Next.js console, on a normal http origin (ALLOWED_ORIGINS)
    #   2. the Chrome extension's service worker, whose Origin header is
    #      `chrome-extension://<id>` — an id that changes every time the
    #      unpacked extension is reloaded, so it can't be enumerated in .env
    #      and is matched by pattern instead.
    #
    # `allow_credentials=True` alongside `allow_origins=["*"]` is rejected by
    # every browser: the spec forbids echoing the wildcard on a credentialed
    # request. The wildcard therefore drops credentials rather than silently
    # producing a config that fails at runtime. Nothing here uses cookies —
    # the extension sends no credentials — so this costs nothing.
    raw_origins = os.getenv("ALLOWED_ORIGINS", "*")
    wildcard = raw_origins.strip() == "*"
    origins = [] if wildcard else [o.strip() for o in raw_origins.split(",") if o.strip()]

    extension_origin_re = os.getenv(
        "ALLOWED_ORIGIN_REGEX",
        r"^(chrome-extension|moz-extension)://.*$",
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if wildcard else origins,
        allow_origin_regex=None if wildcard else extension_origin_re,
        allow_credentials=not wildcard,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )

    # ── Static file serving (uploaded evidence files) ────────────────────────
    upload_dir = os.getenv("UPLOAD_DIR", "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    application.mount("/uploads", StaticFiles(directory=upload_dir), name="uploads")

    # ── Routers ───────────────────────────────────────────────────────────────
    application.include_router(health.router, prefix="/api", tags=["health"])
    application.include_router(cases.router, prefix="/api", tags=["cases"])
    application.include_router(queue.router, prefix="/api", tags=["queue"])

    return application
