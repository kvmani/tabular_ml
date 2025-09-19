"""Security middleware for CSP and CSRF protection."""
from __future__ import annotations

import secrets
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

SAFE_METHODS = {"GET", "HEAD", "OPTIONS", "TRACE"}


class CSRFMiddleware(BaseHTTPMiddleware):
    """Enforce double-submit CSRF protection using cookie + header."""

    def __init__(
        self,
        app,
        *,
        enabled: bool = True,
        cookie_name: str = "XSRF-TOKEN",
        header_name: str = "X-CSRF-Token",
    ) -> None:
        super().__init__(app)
        self.enabled = enabled
        self.cookie_name = cookie_name
        self.header_name = header_name

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Response]
    ) -> Response:
        if not self.enabled:
            return await call_next(request)

        method = request.method.upper()
        token = request.cookies.get(self.cookie_name)
        if method not in SAFE_METHODS:
            header_token = request.headers.get(self.header_name)
            if not token or not header_token or not secrets.compare_digest(
                header_token, token
            ):
                return JSONResponse(
                    status_code=403,
                    content={"detail": "CSRF token missing or invalid"},
                )

        response = await call_next(request)
        if method in SAFE_METHODS:
            if not token:
                token = secrets.token_urlsafe(32)
            response.headers[self.header_name] = token
            response.set_cookie(
                key=self.cookie_name,
                value=token,
                httponly=False,
                secure=False,
                samesite="strict",
                max_age=3600,
                path="/",
            )
        return response


class CSPMiddleware(BaseHTTPMiddleware):
    """Attach a Content-Security-Policy header to every response."""

    def __init__(self, app, *, enabled: bool = True, policy: str = "") -> None:
        super().__init__(app)
        self.enabled = enabled
        self.policy = policy

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Response]
    ) -> Response:
        response = await call_next(request)
        if self.enabled and self.policy:
            response.headers.setdefault("Content-Security-Policy", self.policy)
            response.headers.setdefault("Referrer-Policy", "no-referrer")
            response.headers.setdefault("X-Content-Type-Options", "nosniff")
        return response
