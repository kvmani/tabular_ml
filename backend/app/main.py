"""FastAPI application entry point."""
from __future__ import annotations

import logging
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings

from backend.app.core.security import CSPMiddleware, CSRFMiddleware
from backend.app.api.routes import data, modeling, preprocess, system, visualization
from backend.app.log_stream import LogStreamHandler, log_stream_manager

DEFAULT_DEV_CORS_ORIGINS = (
    "http://127.0.0.1:5173",
    "http://localhost:5173",
    # Vite's preview server defaults to 4173 which mirrors production builds.
    "http://127.0.0.1:4173",
    "http://localhost:4173",
)


logging.basicConfig(
    level=getattr(logging, settings.app.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)


def _cors_origins() -> List[str]:
    """Return the configured CORS origins with sensible local fallbacks."""

    origins = settings.security.cors_origins
    if origins:
        return origins
    # During local development the frontend typically runs on Vite's dev server
    # (port 5173) or preview server (port 4173). When the configuration does not
    # explicitly specify allowed origins we optimistically allow these common
    # localhost URLs so the UI can communicate with the backend without manual
    # configuration tweaks.
    return list(DEFAULT_DEV_CORS_ORIGINS)


def create_app() -> FastAPI:
    app = FastAPI(title=settings.app.name, debug=settings.app.debug)
    log_handler = LogStreamHandler(log_stream_manager)
    logging.getLogger().addHandler(log_handler)
    app.state.log_stream_handler = log_handler

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-CSRF-Token"],
    )
    app.add_middleware(
        CSRFMiddleware,
        enabled=settings.security.csrf_protect,
    )
    app.add_middleware(
        CSPMiddleware,
        enabled=settings.security.csp_enabled,
        policy=settings.security.csp_policy,
    )
    app.include_router(system.router)
    app.include_router(data.router)
    app.include_router(preprocess.router)
    app.include_router(modeling.router)
    app.include_router(visualization.router)

    @app.on_event("shutdown")
    async def _remove_log_handler() -> None:
        logging.getLogger().removeHandler(log_handler)

    @app.get("/")
    def read_root() -> dict:
        return {"app": settings.app.name, "status": "ok"}

    return app


app = create_app()
