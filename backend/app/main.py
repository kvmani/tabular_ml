"""FastAPI application entry point."""
from __future__ import annotations

import logging
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings

from backend.app.api.routes import data, modeling, preprocess, system, visualization


logging.basicConfig(
    level=getattr(logging, settings.app.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)


def _cors_origins() -> List[str]:
    origins = settings.security.cors_origins
    return origins if origins else []


def create_app() -> FastAPI:
    app = FastAPI(title=settings.app.name, debug=settings.app.debug)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(system.router)
    app.include_router(data.router)
    app.include_router(preprocess.router)
    app.include_router(modeling.router)
    app.include_router(visualization.router)

    @app.get("/")
    def read_root() -> dict:
        return {"app": settings.app.name, "status": "ok"}

    return app


app = create_app()
