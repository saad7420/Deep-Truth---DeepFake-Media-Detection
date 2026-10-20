from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from app.database import init_db
from app.routers import cases, health, queue


@asynccontextmanager
async def lifespan(application: FastAPI):
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

    # extension origins change on every reload, so they are matched by pattern
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

    upload_dir = os.getenv("UPLOAD_DIR", "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    # evidence files
    application.mount("/uploads", StaticFiles(directory=upload_dir), name="uploads")

    artifact_dir = os.getenv("ARTIFACT_DIR", "artifacts")
    os.makedirs(artifact_dir, exist_ok=True)
    # derived heat maps, safe to delete
    application.mount("/artifacts", StaticFiles(directory=artifact_dir),
                      name="artifacts")

    application.include_router(health.router, prefix="/api", tags=["health"])
    application.include_router(cases.router, prefix="/api", tags=["cases"])
    application.include_router(queue.router, prefix="/api", tags=["queue"])

    return application
