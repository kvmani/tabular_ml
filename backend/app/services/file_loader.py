"""File loading utilities for dataset ingestion."""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd


SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


def read_tabular_file(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Read uploaded tabular data into a DataFrame."""

    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file extension '{suffix}'.")
    if suffix == ".csv":
        text_stream = io.StringIO(file_bytes.decode("utf-8", errors="ignore"))
        return pd.read_csv(text_stream)
    return pd.read_excel(io.BytesIO(file_bytes))
