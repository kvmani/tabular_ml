"""File loading utilities for dataset ingestion."""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd


SUPPORTED_EXTENSIONS = {".csv", ".tsv", ".xlsx", ".xls", ".parquet"}


def read_tabular_file(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Read uploaded tabular data into a DataFrame."""

    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file extension '{suffix}'.")
    if suffix in {".csv", ".tsv"}:
        text_stream = io.StringIO(file_bytes.decode("utf-8", errors="ignore"))
        delimiter = "\t" if suffix == ".tsv" else ","
        return pd.read_csv(text_stream, sep=delimiter)
    if suffix == ".parquet":
        return pd.read_parquet(io.BytesIO(file_bytes))
    return pd.read_excel(io.BytesIO(file_bytes))
