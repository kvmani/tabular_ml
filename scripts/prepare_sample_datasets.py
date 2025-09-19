"""Prepare bundled sample datasets from public sources."""
from __future__ import annotations

import csv
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SAMPLE_DIR = DATA_DIR / "sample_datasets"

TITANIC_SOURCE = (
    "https://www.openml.org/data/get_csv/16826755/phpMYEkMl"
)
ADULT_SOURCE = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
)


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response:
        destination.write_bytes(response.read())


def prepare_titanic() -> Path:
    raw_path = DATA_DIR / "titanic_openml.csv"
    if not raw_path.exists():
        print(f"Fetching Titanic dataset from {TITANIC_SOURCE} ...")
        _download(TITANIC_SOURCE, raw_path)
    with raw_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = [row for row in reader if row]
    renamed_header: list[str] = []
    for name in header:
        lowered = name.lower()
        if lowered == "survived":
            renamed_header.append("Survived")
        elif lowered == "pclass":
            renamed_header.append("Pclass")
        elif lowered == "sibsp":
            renamed_header.append("SibSp")
        elif lowered == "parch":
            renamed_header.append("Parch")
        else:
            cleaned = name.replace("_", " ").replace(".", " ")
            renamed_header.append(" ".join(w.capitalize() for w in cleaned.split()))
    target_path = SAMPLE_DIR / "titanic_sample.csv"
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(renamed_header)
        writer.writerows(rows)
    print(f"Wrote {len(rows)} Titanic rows to {target_path.relative_to(ROOT)}")
    return target_path


def prepare_adult(limit: int = 1000) -> Path:
    raw_path = DATA_DIR / "adult_income_raw.csv"
    if not raw_path.exists():
        print(f"Fetching Adult Income dataset from {ADULT_SOURCE} ...")
        _download(ADULT_SOURCE, raw_path)
    columns = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education_num",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital_gain",
        "capital_loss",
        "hours_per_week",
        "native_country",
        "income",
    ]
    rows: list[list[str]] = []
    with raw_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh, skipinitialspace=True)
        for row in reader:
            if not row:
                continue
            if len(row) != len(columns):
                merged = [value.strip() for value in ",".join(row).split(",")]
                if len(merged) != len(columns):
                    continue
                row = merged
            rows.append([value.strip() for value in row])
            if len(rows) >= limit:
                break
    target_path = SAMPLE_DIR / "adult_income_sample.csv"
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(columns)
        writer.writerows(rows)
    print(f"Wrote {len(rows)} Adult Income rows to {target_path.relative_to(ROOT)}")
    return target_path


if __name__ == "__main__":
    prepare_titanic()
    prepare_adult()
