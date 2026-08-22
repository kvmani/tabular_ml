"""Safe in-memory parsing and JSON-safe tabular previews."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import Settings
from ..errors import ApiError

ACCEPTED_EXTENSIONS = {".csv", ".xlsx"}


def _read_csv(raw: bytes) -> pd.DataFrame:
    errors: list[str] = []
    for encoding in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            return pd.read_csv(
                BytesIO(raw),
                sep=None,
                engine="python",
                encoding=encoding,
                on_bad_lines="error",
            )
        except (UnicodeDecodeError, pd.errors.ParserError) as exc:
            errors.append(str(exc))
    raise ApiError(
        code="dataset.csv_parse_failed",
        message="The CSV could not be parsed. Check its encoding, delimiter, and row structure.",
        details={"attempts": len(errors)},
    )


def parse_dataset(raw: bytes, filename: str, settings: Settings) -> pd.DataFrame:
    """Parse CSV/XLSX bytes and enforce limits before creating a workspace."""
    suffix = Path(filename).suffix.lower()
    if suffix not in ACCEPTED_EXTENSIONS:
        raise ApiError(
            code="upload.unsupported_type",
            message="Choose a CSV or Excel .xlsx file.",
            details={"accepted_extensions": sorted(ACCEPTED_EXTENSIONS)},
        )
    if not raw:
        raise ApiError(code="upload.empty", message="The selected file is empty.")
    if len(raw) > settings.max_upload_bytes:
        raise ApiError(
            code="upload.too_large",
            message=f"The upload exceeds the {settings.max_upload_mb} MB limit.",
            status=413,
        )
    try:
        frame = _read_csv(raw) if suffix == ".csv" else pd.read_excel(BytesIO(raw))
    except ApiError:
        raise
    except Exception as exc:
        raise ApiError(
            code="dataset.parse_failed",
            message=(
                "The table could not be read. Confirm the file is a valid CSV or XLSX workbook."
            ),
        ) from exc
    if frame.empty or len(frame.columns) == 0:
        raise ApiError(code="dataset.empty", message="The table has no data rows or columns.")
    if len(frame) > settings.max_rows:
        raise ApiError(
            code="dataset.too_many_rows",
            message=f"The table has {len(frame):,} rows; the limit is {settings.max_rows:,}.",
            status=413,
        )
    if len(frame.columns) > settings.max_columns:
        raise ApiError(
            code="dataset.too_many_columns",
            message=(
                f"The table has {len(frame.columns):,} columns; "
                f"the limit is {settings.max_columns:,}."
            ),
            status=413,
        )
    frame.columns = _unique_column_names([str(column).strip() for column in frame.columns])
    frame = frame.replace([np.inf, -np.inf], np.nan)
    return frame


def _unique_column_names(columns: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    output: list[str] = []
    for index, raw_name in enumerate(columns, start=1):
        name = raw_name or f"column_{index}"
        count = seen.get(name, 0)
        seen[name] = count + 1
        output.append(name if count == 0 else f"{name}_{count + 1}")
    return output


def preview_records(frame: pd.DataFrame, rows: int) -> list[dict[str, Any]]:
    """Return strict-JSON records, converting timestamps and missing values."""
    text = frame.head(rows).to_json(orient="records", date_format="iso")
    records = json.loads(text)
    if not isinstance(records, list):
        return []
    return records
