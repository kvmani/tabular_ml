"""Application configuration module."""
from __future__ import annotations

from pathlib import Path


class Settings:
    """Simple settings container for the backend service."""

    APP_NAME: str = "Intranet Tabular ML Studio"
    DATA_DIR: Path = Path("data")
    SAMPLE_DATA_DIR: Path = DATA_DIR / "sample_datasets"
    STORAGE_DIR: Path = Path("backend") / "storage"

    def ensure_directories(self) -> None:
        """Create directories required for runtime artifacts."""
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.SAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.STORAGE_DIR.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_directories()
