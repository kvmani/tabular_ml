"""FastAPI application entry point."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes import data, modeling, preprocess, visualization
from backend.app.core.config import settings


app = FastAPI(title=settings.APP_NAME, version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(data.router)
app.include_router(preprocess.router)
app.include_router(modeling.router)
app.include_router(visualization.router)


@app.get("/")
def read_root() -> dict:
    return {"app": settings.APP_NAME, "status": "ok"}
