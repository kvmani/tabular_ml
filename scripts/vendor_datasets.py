"""Download, normalize, document, and verify the tracked demonstration datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "src" / "tabular_ml_service" / "datasets"
RETRIEVED = "2026-08-23"
CC_BY_URL = "https://creativecommons.org/licenses/by/4.0/"


@dataclass(frozen=True)
class SourceSpec:
    key: str
    name: str
    task: str
    target: str
    summary: str
    tags: tuple[str, ...]
    source_url: str
    download_url: str
    license_name: str
    license_url: str
    citation: str
    learning_goals: tuple[str, ...]
    recommended_exclusions: tuple[str, ...]
    ethical_notes: str
    builder: Callable[[Path], pd.DataFrame]


def _download(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "tabular-ml-dataset-vendor/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def _zip(url: str, destination: Path) -> Path:
    archive = destination / "source.zip"
    archive.write_bytes(_download(url))
    extracted = destination / "extracted"
    extracted.mkdir()
    with zipfile.ZipFile(archive) as bundle:
        root = extracted.resolve()
        for member in bundle.infolist():
            target = (extracted / member.filename).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(f"Unsafe archive member: {member.filename}")
            bundle.extract(member, extracted)
    return extracted


def _adult(work: Path) -> pd.DataFrame:
    root = _zip("https://archive.ics.uci.edu/static/public/2/adult.zip", work)
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
    options = {
        "names": columns,
        "skipinitialspace": True,
        "na_values": "?",
    }
    train = pd.read_csv(root / "adult.data", **options)
    test = pd.read_csv(root / "adult.test", skiprows=1, **options)
    frame = pd.concat([train, test], ignore_index=True)
    frame["income"] = frame["income"].astype(str).str.removesuffix(".")
    return frame


def _wine(work: Path) -> pd.DataFrame:
    root = _zip(
        "https://archive.ics.uci.edu/static/public/186/wine%2Bquality.zip",
        work,
    )
    red = pd.read_csv(root / "winequality-red.csv", sep=";")
    white = pd.read_csv(root / "winequality-white.csv", sep=";")
    red.insert(0, "wine_type", "red")
    white.insert(0, "wine_type", "white")
    return pd.concat([red, white], ignore_index=True)


def _auto_mpg(work: Path) -> pd.DataFrame:
    root = _zip(
        "https://archive.ics.uci.edu/static/public/9/auto%2Bmpg.zip",
        work,
    )
    columns = [
        "mpg",
        "cylinders",
        "displacement",
        "horsepower",
        "weight",
        "acceleration",
        "model_year",
        "origin",
        "car_name",
    ]
    return pd.read_csv(
        root / "auto-mpg.data",
        sep=r"\s+",
        names=columns,
        na_values="?",
        quotechar='"',
    )


def _breast_cancer(work: Path) -> pd.DataFrame:
    root = _zip(
        "https://archive.ics.uci.edu/static/public/17/breast%2Bcancer%2Bwisconsin%2Bdiagnostic.zip",
        work,
    )
    measurements = [
        "radius",
        "texture",
        "perimeter",
        "area",
        "smoothness",
        "compactness",
        "concavity",
        "concave_points",
        "symmetry",
        "fractal_dimension",
    ]
    columns = ["patient_id", "diagnosis"]
    columns.extend(
        f"{name}_{suffix}" for suffix in ("mean", "se", "worst") for name in measurements
    )
    return pd.read_csv(root / "wdbc.data", names=columns)


def _titanic(_work: Path) -> pd.DataFrame:
    return pd.read_csv(BytesIO(_download("https://hbiostat.org/data/repo/titanic3.csv")))


def _penguins(_work: Path) -> pd.DataFrame:
    url = (
        "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/"
        "refs/heads/main/inst/extdata/penguins.csv"
    )
    return pd.read_csv(BytesIO(_download(url)), na_values="NA")


SOURCES = (
    SourceSpec(
        key="titanic",
        name="Titanic Passenger Survival",
        task="classification",
        target="survived",
        summary=(
            "Mixed passenger attributes, substantial missingness, and a binary survival target."
        ),
        tags=("classification", "missing-values", "mixed-types", "leakage"),
        source_url="https://hbiostat.org/data/",
        download_url="https://hbiostat.org/data/repo/titanic3.csv",
        license_name="Vanderbilt Biostatistics data permission",
        license_url="https://hbiostat.org/data/",
        citation=(
            "Data obtained from http://hbiostat.org/data courtesy of the "
            "Vanderbilt University Department of Biostatistics."
        ),
        learning_goals=(
            "Compare imputation strategies for age, fare, cabin, and embarkation.",
            "Inspect class balance and precision/recall trade-offs.",
            "Recognize post-outcome leakage in boat and body fields.",
        ),
        recommended_exclusions=("name", "ticket", "cabin", "boat", "body", "home.dest"),
        ethical_notes=(
            "Historical passenger records include sex, age, and travel class. "
            "Use for teaching, not to generalize moral worth or modern safety policy."
        ),
        builder=_titanic,
    ),
    SourceSpec(
        key="adult_income",
        name="U.S. Adult Census Income",
        task="classification",
        target="income",
        summary="A larger mixed-type census extract for predicting whether income exceeds $50K.",
        tags=("classification", "fairness", "missing-values", "high-cardinality"),
        source_url="https://archive.ics.uci.edu/dataset/2/adult",
        download_url="https://archive.ics.uci.edu/static/public/2/adult.zip",
        license_name="CC BY 4.0",
        license_url=CC_BY_URL,
        citation=(
            "Becker, B. & Kohavi, R. (1996). Adult [Dataset]. "
            "UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20."
        ),
        learning_goals=(
            "Handle missing categorical values and higher-cardinality one-hot encoding.",
            "Compare balanced metrics on an uneven target.",
            "Audit subgroup performance and proxy variables before interpreting results.",
        ),
        recommended_exclusions=("fnlwgt",),
        ethical_notes=(
            "Contains sensitive demographic attributes including age, race, and sex. "
            "Accuracy is not evidence of fairness; never use this teaching model for decisions."
        ),
        builder=_adult,
    ),
    SourceSpec(
        key="palmer_penguins",
        name="Palmer Penguins",
        task="classification",
        target="species",
        summary=(
            "Three penguin species with body measurements, islands, sex, years, and missing values."
        ),
        tags=("classification", "multiclass", "missing-values", "visualization"),
        source_url="https://allisonhorst.github.io/palmerpenguins/",
        download_url=(
            "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/"
            "refs/heads/main/inst/extdata/penguins.csv"
        ),
        license_name="CC0 1.0",
        license_url="https://allisonhorst.github.io/palmerpenguins/LICENSE.html",
        citation=(
            "Horst AM, Hill AP, Gorman KB (2020). palmerpenguins: Palmer "
            "Archipelago (Antarctica) penguin data. doi:10.5281/zenodo.3960218."
        ),
        learning_goals=(
            "Explore multiclass confusion matrices and weighted metrics.",
            "Visualize morphology by species and island.",
            "Compare simple linear boundaries with nonlinear ensembles.",
        ),
        recommended_exclusions=(),
        ethical_notes=(
            "Ecological observations cover specific species, islands, and study years; "
            "do not extrapolate beyond that sampling context without domain review."
        ),
        builder=_penguins,
    ),
    SourceSpec(
        key="wine_quality",
        name="Vinho Verde Wine Quality",
        task="regression",
        target="quality",
        summary=(
            "Physicochemical measurements and ordinal sensory quality scores "
            "for red and white wines."
        ),
        tags=("regression", "ordinal-target", "nonlinear", "feature-importance"),
        source_url="https://archive.ics.uci.edu/dataset/186/wine+quality",
        download_url=("https://archive.ics.uci.edu/static/public/186/wine%2Bquality.zip"),
        license_name="CC BY 4.0",
        license_url=CC_BY_URL,
        citation=(
            "Cortez, P. et al. (2009). Wine Quality [Dataset]. "
            "UCI Machine Learning Repository. https://doi.org/10.24432/C56S3T."
        ),
        learning_goals=(
            "Compare RMSE, MAE, R², and residual diagnostics.",
            "Inspect nonlinear effects and correlated chemistry measurements.",
            "Discuss regression versus ordinal classification framing.",
        ),
        recommended_exclusions=(),
        ethical_notes=(
            "Sensory quality is subjective and imbalanced; predictions are not a "
            "substitute for qualified tasting, quality assurance, or safety testing."
        ),
        builder=_wine,
    ),
    SourceSpec(
        key="auto_mpg",
        name="Auto MPG",
        task="regression",
        target="mpg",
        summary="Vehicle attributes, missing horsepower, origin, and city-cycle fuel economy.",
        tags=("regression", "missing-values", "mixed-types", "outliers"),
        source_url="https://archive.ics.uci.edu/dataset/9/auto+mpg",
        download_url="https://archive.ics.uci.edu/static/public/9/auto%2Bmpg.zip",
        license_name="CC BY 4.0",
        license_url=CC_BY_URL,
        citation=(
            "Quinlan, R. (1993). Auto MPG [Dataset]. UCI Machine Learning "
            "Repository. https://doi.org/10.24432/C5859H."
        ),
        learning_goals=(
            "Compare imputation for missing horsepower.",
            "Inspect residuals across model years and origins.",
            "Contrast linear, neighbour, and boosted-tree regressors.",
        ),
        recommended_exclusions=("car_name",),
        ethical_notes=(
            "Historical city-cycle estimates are not comparable to current regulatory "
            "tests and should not guide modern purchase or emissions claims."
        ),
        builder=_auto_mpg,
    ),
    SourceSpec(
        key="breast_cancer_diagnostic",
        name="Breast Cancer Wisconsin Diagnostic",
        task="classification",
        target="diagnosis",
        summary="Thirty continuous cell-nucleus measurements with benign or malignant labels.",
        tags=("classification", "health", "continuous-features", "feature-importance"),
        source_url=("https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic"),
        download_url=(
            "https://archive.ics.uci.edu/static/public/17/"
            "breast%2Bcancer%2Bwisconsin%2Bdiagnostic.zip"
        ),
        license_name="CC BY 4.0",
        license_url=CC_BY_URL,
        citation=(
            "Wolberg, W. et al. (1993). Breast Cancer Wisconsin (Diagnostic) "
            "[Dataset]. UCI ML Repository. https://doi.org/10.24432/C5DW2B."
        ),
        learning_goals=(
            "Compare accuracy with recall, ROC AUC, and calibration-sensitive metrics.",
            "Inspect correlated features and importance stability.",
            "Discuss the limits of retrospective diagnostic demonstrations.",
        ),
        recommended_exclusions=("patient_id",),
        ethical_notes=(
            "Teaching data only. A model trained here is not a medical device and must "
            "never be used for diagnosis, triage, or treatment."
        ),
        builder=_breast_cancer,
    ),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metadata(spec: SourceSpec, frame: pd.DataFrame, filename: str) -> dict:
    return {
        "schema_version": 1,
        "key": spec.key,
        "name": spec.name,
        "description": spec.summary,
        "task": spec.task,
        "target": spec.target,
        "file": filename,
        "rows": len(frame),
        "columns": len(frame.columns),
        "column_names": frame.columns.tolist(),
        "missing_cells": int(frame.isna().sum().sum()),
        "source": {
            "page": spec.source_url,
            "download": spec.download_url,
            "retrieved": RETRIEVED,
        },
        "license": {
            "name": spec.license_name,
            "url": spec.license_url,
            "attribution": spec.citation,
        },
        "learning_goals": list(spec.learning_goals),
        "recommended_exclusions": list(spec.recommended_exclusions),
        "ethical_notes": spec.ethical_notes,
        "tags": list(spec.tags),
    }


def build() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    registry = {"schema_version": 1, "datasets": []}
    for spec in SOURCES:
        with tempfile.TemporaryDirectory(prefix=f"tabular-ml-{spec.key}-") as temporary:
            frame = spec.builder(Path(temporary))
        if spec.target not in frame.columns:
            raise RuntimeError(f"{spec.key}: missing target {spec.target}")
        filename = f"{spec.key}.csv"
        csv_path = OUTPUT / filename
        frame.to_csv(csv_path, index=False, lineterminator="\n")
        metadata = _metadata(spec, frame, filename)
        metadata["sha256"] = _sha256(csv_path)
        metadata_file = f"{spec.key}.json"
        (OUTPUT / metadata_file).write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        registry["datasets"].append(
            {
                "key": spec.key,
                "name": spec.name,
                "task": spec.task,
                "target": spec.target,
                "file": filename,
                "metadata_file": metadata_file,
                "rows": len(frame),
                "columns": len(frame.columns),
                "summary": spec.summary,
                "license_name": spec.license_name,
                "tags": list(spec.tags),
            }
        )
        print(f"built {spec.key}: {len(frame):,} rows x {len(frame.columns)} columns")
    (OUTPUT / "registry.json").write_text(
        json.dumps(registry, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    verify()


def verify() -> None:
    registry = json.loads((OUTPUT / "registry.json").read_text(encoding="utf-8"))
    keys = set()
    for record in registry["datasets"]:
        key = record["key"]
        if key in keys:
            raise RuntimeError(f"duplicate dataset key: {key}")
        keys.add(key)
        csv_path = OUTPUT / record["file"]
        metadata = json.loads((OUTPUT / record["metadata_file"]).read_text(encoding="utf-8"))
        frame = pd.read_csv(csv_path)
        checks = {
            "rows": len(frame),
            "columns": len(frame.columns),
            "sha256": _sha256(csv_path),
        }
        for field, actual in checks.items():
            if metadata[field] != actual:
                raise RuntimeError(
                    f"{key}: {field} is {actual!r}, metadata has {metadata[field]!r}"
                )
        if metadata["target"] not in frame.columns:
            raise RuntimeError(f"{key}: target missing from CSV")
    if len(keys) < 5:
        raise RuntimeError("At least five demo datasets are required")
    print(f"verified {len(keys)} tracked datasets")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify tracked files without downloading sources.",
    )
    args = parser.parse_args()
    verify() if args.check else build()


if __name__ == "__main__":
    main()
