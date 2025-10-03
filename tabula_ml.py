"""Compact object-oriented orchestration utilities for the demo notebook."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, StandardScaler


DEFAULT_CONFIG: Dict[str, Any] = {
    "task_type": "auto",
    "target_column": None,
    "test_size": 0.2,
    "random_state": 42,
    "scaling": "standard",
    "categorical_encoding": "onehot",
    "impute_numeric": "median",
    "impute_categorical": "most_frequent",
    "outlier_method": None,
    "outlier_threshold": 1.5,
    "allow_export": False,
}


@dataclass
class ModelResult:
    """Container storing training and evaluation details for a model."""

    name: str
    pipeline: Pipeline
    metrics: Dict[str, float]
    artefact_path: Optional[Path] = None


class TabulaML:
    """Utility class orchestrating the end-to-end tabular ML workflow."""

    def __init__(self, output_dir: Path | str = "tmp", config: Optional[Dict[str, Any]] = None) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config: Dict[str, Any] = {**DEFAULT_CONFIG, **(config or {})}
        self.state: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Data loading & configuration helpers
    # ------------------------------------------------------------------
    def load_data(self, source: Path | str | pd.DataFrame, *, name: Optional[str] = None) -> pd.DataFrame:
        """Load data from disk or use an in-memory frame."""

        if isinstance(source, pd.DataFrame):
            df = source.copy()
            data_name = name or "DataFrame"
        else:
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"Dataset not found: {source}")
            if path.suffix.lower() == ".parquet":
                df = pd.read_parquet(path)
            else:
                df = pd.read_csv(path)
            data_name = name or path.stem

        if df.empty:
            raise ValueError("Loaded dataset is empty.")

        preview_path = self.output_dir / "raw_preview.csv"
        df.head(20).to_csv(preview_path, index=False)

        self.state.update({"df_raw": df, "data_name": data_name})
        return df

    def set_config(self, **updates: Any) -> Dict[str, Any]:
        """Update configuration values and return the merged dictionary."""

        self.config.update(updates)
        return self.config

    # ------------------------------------------------------------------
    # Target handling & preprocessing
    # ------------------------------------------------------------------
    def suggest_target_column(self, df: Optional[pd.DataFrame] = None) -> Optional[str]:
        """Heuristically suggest a target column from a dataframe."""

        frame = df if df is not None else self.state.get("df_raw")
        if frame is None:
            raise RuntimeError("Load data before invoking target suggestion.")

        candidate_names = [
            "target",
            "label",
            "outcome",
            "y",
            "class",
            "survived",
        ]
        lowered = {col.lower(): col for col in frame.columns}
        for candidate in candidate_names:
            if candidate in lowered:
                return lowered[candidate]

        for column in frame.columns:
            series = frame[column]
            unique = series.dropna().unique()
            if series.dtype == object and 2 <= len(unique) <= 10:
                return column
            if pd.api.types.is_integer_dtype(series) and 2 <= len(unique) <= 10:
                return column
        return None

    def resolve_target(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Split raw data into features and target using configuration."""

        df = self.state.get("df_raw")
        if df is None:
            raise RuntimeError("Call load_data before resolving the target column.")

        target_column = self.config.get("target_column")
        if not target_column:
            target_column = self.suggest_target_column(df)
            if not target_column:
                raise ValueError("Unable to infer target column; set config['target_column'] explicitly.")
            self.config["target_column"] = target_column

        if target_column not in df.columns:
            raise KeyError(f"Target column '{target_column}' not found in dataset.")

        X = df.drop(columns=[target_column])
        y = df[target_column]
        X_clean, dropped = self._drop_redundant_columns(X)
        self.state.update({
            "X": X_clean,
            "y": y,
            "dropped_columns": dropped,
            "task_type": self._resolve_task_type(y),
        })
        return X_clean, y

    def _drop_redundant_columns(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
        constant = [col for col in df.columns if df[col].nunique(dropna=False) <= 1]
        duplicates: List[str] = []
        df_clean = df.drop(columns=constant, errors="ignore")
        seen: Dict[str, str] = {}
        for col in list(df_clean.columns):
            values = tuple(df_clean[col].fillna("<NA>").tolist())
            if values in seen:
                duplicates.append(col)
            else:
                seen[values] = col
        df_clean = df_clean.drop(columns=duplicates, errors="ignore")
        return df_clean, {"constant": constant, "duplicates": duplicates}

    def _resolve_task_type(self, target: pd.Series) -> str:
        if pd.api.types.is_numeric_dtype(target):
            unique = target.dropna().unique()
            if len(unique) <= 10 and target.dtype != float:
                return "classification"
            if len(unique) <= 2:
                return "classification"
            return "regression"
        return "classification"

    def build_preprocessing(self) -> ColumnTransformer:
        """Construct a :class:`~sklearn.compose.ColumnTransformer` for features."""

        X = self.state.get("X")
        if X is None:
            raise RuntimeError("Resolve the target before building preprocessing.")

        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = [col for col in X.columns if col not in numeric_cols]

        transformers: List[Tuple[str, Pipeline, Iterable[str]]] = []
        if numeric_cols:
            numeric_steps: List[Tuple[str, Any]] = [("imputer", SimpleImputer(strategy=self.config["impute_numeric"]))]
            scaling = self.config.get("scaling")
            if scaling == "standard":
                numeric_steps.append(("scaler", StandardScaler()))
            elif scaling == "minmax":
                numeric_steps.append(("scaler", MinMaxScaler()))
            transformers.append(("num", Pipeline(numeric_steps), numeric_cols))

        if categorical_cols and self.config.get("categorical_encoding"):
            if self.config["categorical_encoding"] == "onehot":
                encoder = OneHotEncoder(handle_unknown="ignore")
            else:
                encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            cat_steps: List[Tuple[str, Any]] = [
                ("imputer", SimpleImputer(strategy=self.config["impute_categorical"], fill_value="missing")),
                ("encoder", encoder),
            ]
            transformers.append(("cat", Pipeline(cat_steps), categorical_cols))

        if not transformers:
            raise ValueError("No features available for preprocessing.")
        return ColumnTransformer(transformers)

    # ------------------------------------------------------------------
    # Exploratory data analysis
    # ------------------------------------------------------------------
    def perform_eda(self) -> Dict[str, Any]:
        """Generate summary artefacts for the current dataset."""

        X = self.state.get("X")
        y = self.state.get("y")
        if X is None or y is None:
            raise RuntimeError("Resolve target before running EDA.")

        eda_dir = self.output_dir / "eda"
        eda_dir.mkdir(exist_ok=True)

        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = [col for col in X.columns if col not in numeric_cols]

        summaries: Dict[str, Any] = {}
        if numeric_cols:
            numeric_summary = X[numeric_cols].describe().T
            numeric_path = eda_dir / "numeric_summary.csv"
            numeric_summary.to_csv(numeric_path)
            summaries["numeric_summary_path"] = numeric_path
        else:
            summaries["numeric_summary_path"] = None

        categorical_snapshot: Dict[str, pd.Series] = {}
        for col in categorical_cols:
            categorical_snapshot[col] = X[col].value_counts(dropna=False).head(20)
        if categorical_snapshot:
            cat_frame = pd.DataFrame({col: series for col, series in categorical_snapshot.items()})
            cat_path = eda_dir / "categorical_counts.csv"
            cat_frame.to_csv(cat_path)
            summaries["categorical_counts_path"] = cat_path
        else:
            summaries["categorical_counts_path"] = None

        missing = pd.isna(pd.concat([X, y], axis=1)).mean()
        missing_path = eda_dir / "missingness.csv"
        missing.to_csv(missing_path, header=["ratio"])
        summaries["missingness_path"] = missing_path

        self.state["eda_summaries"] = summaries
        return summaries

    # ------------------------------------------------------------------
    # Outlier handling
    # ------------------------------------------------------------------
    def detect_outliers(self) -> pd.Series:
        """Detect potential outliers using the configured strategy."""

        method = self.config.get("outlier_method")
        threshold = self.config.get("outlier_threshold", 1.5)
        X = self.state.get("X")
        if X is None:
            raise RuntimeError("Resolve target before detecting outliers.")

        numeric_df = X.select_dtypes(include=[np.number])
        if numeric_df.empty:
            mask = pd.Series(False, index=X.index)
            self.state["outlier_mask"] = mask
            return mask

        if method == "IQR":
            q1 = numeric_df.quantile(0.25)
            q3 = numeric_df.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - threshold * iqr
            upper = q3 + threshold * iqr
            mask = ((numeric_df < lower) | (numeric_df > upper)).any(axis=1)
        elif method == "ZScore":
            mean = numeric_df.mean()
            std = numeric_df.std(ddof=0).replace(0, np.nan)
            zscores = (numeric_df - mean) / std
            mask = zscores.abs().gt(threshold).any(axis=1).fillna(False)
        else:
            mask = pd.Series(False, index=X.index)

        self.state["outlier_mask"] = mask
        return mask

    # ------------------------------------------------------------------
    # Data splitting & modeling
    # ------------------------------------------------------------------
    def split_data(self, *, stratify: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split features and target into train/test portions."""

        X = self.state.get("X")
        y = self.state.get("y")
        if X is None or y is None:
            raise RuntimeError("Resolve target before splitting data.")

        stratify_y = y if stratify and self.state.get("task_type") == "classification" else None
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config["test_size"],
            random_state=self.config["random_state"],
            stratify=stratify_y,
        )

        self.state.update({
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        })
        return X_train, X_test, y_train, y_test

    def _get_estimators(self) -> List[Tuple[str, ClassifierMixin | RegressorMixin]]:
        task_type = self.state.get("task_type")
        if task_type == "regression":
            return [
                ("LinearRegression", LinearRegression()),
                ("RandomForestRegressor", RandomForestRegressor(n_estimators=50, random_state=self.config["random_state"])),
            ]
        return [
            (
                "LogisticRegression",
                LogisticRegression(max_iter=500, solver="liblinear"),
            ),
            (
                "RandomForestClassifier",
                RandomForestClassifier(n_estimators=100, random_state=self.config["random_state"]),
            ),
        ]

    def train_models(self) -> List[ModelResult]:
        """Train models defined for the current task type."""

        X_train = self.state.get("X_train")
        y_train = self.state.get("y_train")
        if X_train is None or y_train is None:
            raise RuntimeError("Split the data before training models.")

        preprocessing = self.build_preprocessing()
        results: List[ModelResult] = []
        model_dir = self.output_dir / "models"
        model_dir.mkdir(exist_ok=True)

        for name, estimator in self._get_estimators():
            pipeline = Pipeline([("preprocess", preprocessing), ("model", estimator)])
            pipeline.fit(X_train, y_train)
            artefact_path: Optional[Path] = None
            if self.config.get("allow_export"):
                artefact_path = model_dir / f"{name}.joblib"
                joblib.dump(pipeline, artefact_path)
            results.append(ModelResult(name=name, pipeline=pipeline, metrics={}, artefact_path=artefact_path))

        self.state["model_results"] = results
        return results

    def evaluate_models(self) -> List[ModelResult]:
        """Evaluate trained models on the hold-out test set."""

        X_test = self.state.get("X_test")
        y_test = self.state.get("y_test")
        results: List[ModelResult] = self.state.get("model_results", [])
        if X_test is None or y_test is None or not results:
            raise RuntimeError("Train models before evaluation.")

        task_type = self.state.get("task_type")
        metrics_dir = self.output_dir / "metrics"
        metrics_dir.mkdir(exist_ok=True)

        for result in results:
            preds = result.pipeline.predict(X_test)
            metrics: Dict[str, float]
            if task_type == "regression":
                mse = mean_squared_error(y_test, preds)
                metrics = {
                    "rmse": float(np.sqrt(mse)),
                    "r2": float(r2_score(y_test, preds)),
                }
            else:
                accuracy = accuracy_score(y_test, preds)
                metrics = {
                    "accuracy": float(accuracy),
                }
                try:
                    metrics["f1"] = float(f1_score(y_test, preds))
                except ValueError:
                    metrics["f1"] = float("nan")

            metrics_path = metrics_dir / f"{result.name}.json"
            pd.Series(metrics).to_json(metrics_path)
            result.metrics = metrics
            result.artefact_path = result.artefact_path or metrics_path
        self.state["model_results"] = results
        return results

    # ------------------------------------------------------------------
    # Reporting helpers
    # ------------------------------------------------------------------
    def summarise_run(self) -> Dict[str, Any]:
        """Return a high level summary for the completed workflow."""

        if "model_results" not in self.state:
            raise RuntimeError("No models trained yet. Run training and evaluation first.")

        summary = {
            "data_name": self.state.get("data_name"),
            "rows": len(self.state.get("X", [])),
            "columns": len(self.state.get("X", pd.DataFrame()).columns),
            "target_column": self.config.get("target_column"),
            "task_type": self.state.get("task_type"),
            "models": [result.name for result in self.state.get("model_results", [])],
        }
        summary_path = self.output_dir / "run_summary.json"
        pd.Series(summary).to_json(summary_path)
        summary["summary_path"] = summary_path
        return summary

