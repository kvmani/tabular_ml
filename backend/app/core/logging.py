"""Logging helpers for streaming backend events to the UI."""

from __future__ import annotations

import json
import logging
from typing import Iterable, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.exception_handlers import (
    http_exception_handler,
    request_validation_exception_handler,
)
from fastapi.exceptions import RequestValidationError
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from starlette.types import ASGIApp

_ERROR_LOGGER_NAME = "backend.api.errors"
_DEFAULT_LOGGER_NAMES: Iterable[str] = ("", "uvicorn", "uvicorn.error", "uvicorn.access")


def configure_stream_logging(handler: logging.Handler) -> List[logging.Logger]:
    """Attach the shared stream handler to core loggers.

    The application emits log events from several logger namespaces. Uvicorn
    configures dedicated ``uvicorn.*`` loggers whose propagation flag is set to
    ``False`` which means attaching the handler only to the root logger would not
    forward those messages to the live log stream. This helper explicitly
    attaches the handler to the key namespaces and returns the logger objects so
    callers can later detach the handler during shutdown.
    """

    attached: List[logging.Logger] = []
    for name in _DEFAULT_LOGGER_NAMES:
        logger = logging.getLogger(name) if name else logging.getLogger()
        if handler not in logger.handlers:
            logger.addHandler(handler)
        attached.append(logger)
    return attached


def format_log_message(
    request: Request,
    *,
    status_code: int,
    detail: object,
    fallback: str | None = None,
) -> str:
    parts = [f"{request.method} {request.url.path} -> {status_code}"]
    if request.url.query:
        parts.append(f"query={request.url.query}")
    if request.client:
        parts.append(f"client={request.client.host}")

    if detail:
        formatted = detail if isinstance(detail, str) else json.dumps(detail)
        parts.append(f"detail={formatted}")
    elif fallback:
        parts.append(fallback)

    return " | ".join(parts)


class ErrorLoggingMiddleware(BaseHTTPMiddleware):
    """Emit structured log messages for error responses and exceptions."""

    def __init__(self, app: ASGIApp, *, logger: Optional[logging.Logger] = None) -> None:
        super().__init__(app)
        self.logger = logger or logging.getLogger(_ERROR_LOGGER_NAME)

    async def dispatch(self, request: Request, call_next) -> Response:
        try:
            response = await call_next(request)
        except HTTPException as exc:
            self._log_http_exception(request, exc)
            raise
        except RequestValidationError as exc:
            self._log_validation_error(request, exc)
            raise
        except Exception as exc:  # pragma: no cover - exercised in integration tests
            self._log_unhandled_exception(request, exc)
            raise

        if response.status_code >= 400 and not getattr(
            request.state, "_error_logged", False
        ):
            await self._log_error_response(request, response)
        return response

    def _log_http_exception(self, request: Request, exc: HTTPException) -> None:
        message = format_log_message(
            request,
            status_code=exc.status_code,
            detail=exc.detail,
        )
        log = self.logger.error if exc.status_code >= 500 else self.logger.warning
        log(message)

    def _log_validation_error(self, request: Request, exc: RequestValidationError) -> None:
        detail = exc.errors()
        message = format_log_message(request, status_code=422, detail=detail)
        self.logger.warning(message)

    def _log_unhandled_exception(self, request: Request, exc: Exception) -> None:
        message = format_log_message(
            request,
            status_code=500,
            detail=str(exc),
            fallback="Unhandled exception while processing request",
        )
        self.logger.exception(message)

    async def _log_error_response(self, request: Request, response: Response) -> None:
        detail = await self._extract_response_detail(response)
        message = format_log_message(
            request,
            status_code=response.status_code,
            detail=detail,
            fallback="Request returned an error response",
        )
        log = self.logger.error if response.status_code >= 500 else self.logger.warning
        log(message)
        request.state._error_logged = True

    async def _extract_response_detail(self, response: Response) -> Optional[str]:
        body_bytes: bytes = b""
        raw_body = getattr(response, "body", b"")
        if isinstance(raw_body, bytes):
            body_bytes = raw_body
        elif isinstance(raw_body, str):
            body_bytes = raw_body.encode("utf-8", errors="ignore")

        if not body_bytes:
            try:
                body_bytes = await response.body()
            except Exception:  # pragma: no cover - defensive fallback
                body_bytes = b""

        if not body_bytes:
            return None

        text = body_bytes.decode("utf-8", errors="ignore")

        # Ensure the response can still be streamed after inspection.
        response.body_iterator = iter([body_bytes])

        if response.media_type and "json" in response.media_type:
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                return text.strip() or None
            detail = payload.get("detail")
            if isinstance(detail, (str, int, float)):
                return str(detail)
            if detail is not None:
                return json.dumps(detail)
            return text.strip() or None

        return text.strip() or None


def install_exception_logging(
    app: FastAPI, *, logger: Optional[logging.Logger] = None
) -> None:
    """Attach exception handlers that mirror middleware logging."""

    error_logger = logger or logging.getLogger(_ERROR_LOGGER_NAME)

    @app.exception_handler(HTTPException)
    async def _http_exception_handler(request: Request, exc: HTTPException):
        message = format_log_message(
            request,
            status_code=exc.status_code,
            detail=exc.detail,
            fallback="HTTP exception",
        )
        log = error_logger.error if exc.status_code >= 500 else error_logger.warning
        log(message)
        request.state._error_logged = True
        return await http_exception_handler(request, exc)

    @app.exception_handler(RequestValidationError)
    async def _validation_exception_handler(
        request: Request, exc: RequestValidationError
    ):
        message = format_log_message(
            request,
            status_code=422,
            detail=exc.errors(),
            fallback="Validation error",
        )
        error_logger.warning(message)
        request.state._error_logged = True
        return await request_validation_exception_handler(request, exc)
