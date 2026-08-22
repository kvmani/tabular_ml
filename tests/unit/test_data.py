from __future__ import annotations

from io import BytesIO

import pandas as pd
import pytest

from tabular_ml_service.config import Settings
from tabular_ml_service.core.data import parse_dataset, preview_records
from tabular_ml_service.errors import ApiError


def test_parse_csv_detects_delimiter_and_normalizes_columns() -> None:
    raw = b"name;score\nAda;10\nLin;12\n"
    frame = parse_dataset(raw, "scores.csv", Settings())
    assert frame.columns.tolist() == ["name", "score"]
    assert frame.to_dict(orient="records")[1] == {"name": "Lin", "score": 12}


def test_parse_xlsx_in_memory() -> None:
    buffer = BytesIO()
    pd.DataFrame({"feature": [1, 2], "label": ["a", "b"]}).to_excel(buffer, index=False)
    frame = parse_dataset(buffer.getvalue(), "demo.xlsx", Settings())
    assert frame.shape == (2, 2)


@pytest.mark.parametrize(
    ("raw", "name", "code"),
    [
        (b"", "demo.csv", "upload.empty"),
        (b"x", "demo.tsv", "upload.unsupported_type"),
        (b"only_header\n", "demo.csv", "dataset.empty"),
    ],
)
def test_parse_rejects_invalid_uploads(raw: bytes, name: str, code: str) -> None:
    with pytest.raises(ApiError) as caught:
        parse_dataset(raw, name, Settings())
    assert caught.value.code == code


def test_parse_enforces_row_and_file_limits() -> None:
    with pytest.raises(ApiError, match="rows"):
        parse_dataset(b"x\n1\n2\n3\n", "demo.csv", Settings(max_rows=2))
    with pytest.raises(ApiError) as caught:
        parse_dataset(b"x\n" + b"1\n" * 600_000, "demo.csv", Settings(max_upload_mb=1))
    assert caught.value.code == "upload.too_large"


def test_preview_records_converts_missing_values_to_null() -> None:
    records = preview_records(pd.DataFrame({"value": [1.0, float("nan")]}), 5)
    assert records == [{"value": 1.0}, {"value": None}]
