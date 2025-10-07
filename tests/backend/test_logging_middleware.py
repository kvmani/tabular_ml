"""Unit tests for the error logging middleware."""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from backend.app.core.logging import ErrorLoggingMiddleware, install_exception_logging


def _create_app() -> FastAPI:
    app = FastAPI()
    install_exception_logging(app)
    app.add_middleware(ErrorLoggingMiddleware)

    @app.get("/boom")
    def boom() -> None:
        raise HTTPException(status_code=400, detail="missing things")

    @app.get("/crash")
    def crash() -> None:
        raise RuntimeError("explode")

    @app.get("/manual-response")
    def manual_response():
        return JSONResponse({"detail": "manual"}, status_code=409)

    return app


def test_http_exception_logged(caplog) -> None:
    app = _create_app()
    client = TestClient(app)

    with caplog.at_level("WARNING", logger="backend.api.errors"):
        response = client.get("/boom")

    assert response.status_code == 400
    assert any("/boom" in record.message for record in caplog.records)
    assert any("missing things" in record.message for record in caplog.records)


def test_unhandled_exception_logged(caplog) -> None:
    app = _create_app()
    client = TestClient(app, raise_server_exceptions=False)

    with caplog.at_level("ERROR", logger="backend.api.errors"):
        response = client.get("/crash")

    assert response.status_code == 500
    assert any("/crash" in record.message for record in caplog.records)
    assert any("explode" in record.message for record in caplog.records)


def test_manual_error_response_logged(caplog) -> None:
    app = _create_app()
    client = TestClient(app)

    with caplog.at_level("WARNING", logger="backend.api.errors"):
        response = client.get("/manual-response")

    assert response.status_code == 409
    assert any("/manual-response" in record.message for record in caplog.records)
    assert any("manual" in record.message for record in caplog.records)

