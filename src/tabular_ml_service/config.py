"""Runtime settings with conservative, environment-overridable limits."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any


def _env_int(name: str, default: int, minimum: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(value, minimum)


@dataclass(frozen=True, slots=True)
class Settings:
    """Settings shared by standalone and blueprint-mounted deployments."""

    max_upload_mb: int = 25
    max_rows: int = 100_000
    max_columns: int = 300
    max_sessions: int = 24
    session_ttl_seconds: int = 3_600
    preview_rows: int = 100
    random_seed: int = 42

    @classmethod
    def from_environment(cls) -> Settings:
        """Load limits from the ``TABULAR_ML_*`` environment namespace."""
        return cls(
            max_upload_mb=_env_int("TABULAR_ML_MAX_UPLOAD_MB", 25, 1),
            max_rows=_env_int("TABULAR_ML_MAX_ROWS", 100_000, 100),
            max_columns=_env_int("TABULAR_ML_MAX_COLUMNS", 300, 2),
            max_sessions=_env_int("TABULAR_ML_MAX_SESSIONS", 24, 1),
            session_ttl_seconds=_env_int("TABULAR_ML_SESSION_TTL_SECONDS", 3_600, 60),
            preview_rows=_env_int("TABULAR_ML_PREVIEW_ROWS", 100, 5),
            random_seed=_env_int("TABULAR_ML_RANDOM_SEED", 42, 0),
        )

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    def public_dict(self) -> dict[str, Any]:
        """Return non-secret settings suitable for the frontend."""
        return asdict(self) | {
            "accepted_extensions": [".csv", ".xlsx"],
            "runtime_storage": "bounded_process_memory",
        }


DEFAULT_SETTINGS = Settings.from_environment()

