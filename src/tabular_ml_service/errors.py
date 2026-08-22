"""Public errors that are safe to show in the web application."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ApiError(Exception):
    """A stable API error with a user-facing message and optional details."""

    code: str
    message: str
    status: int = 400
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.message

