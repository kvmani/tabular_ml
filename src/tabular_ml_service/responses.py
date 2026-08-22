"""Uniform API response helpers."""

from __future__ import annotations

from typing import Any

from flask import Response, jsonify

from .errors import ApiError


def success(data: Any, *, status: int = 200) -> tuple[Response, int]:
    return jsonify({"success": True, "data": data}), status


def failure(error: ApiError) -> tuple[Response, int]:
    payload: dict[str, Any] = {
        "success": False,
        "error": {"code": error.code, "message": error.message},
    }
    if error.details:
        payload["error"]["details"] = error.details
    return jsonify(payload), error.status
