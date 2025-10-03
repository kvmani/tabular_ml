from pathlib import Path

import pandas as pd
import pytest

from tabula_ml import TabulaML


def _load_titanic_df() -> pd.DataFrame:
    return pd.read_csv(Path("data/sample_datasets/titanic_sample.csv"))


@pytest.fixture()
def tmp_pipeline(tmp_path: Path) -> TabulaML:
    pipeline = TabulaML(output_dir=tmp_path)
    pipeline.load_data(_load_titanic_df(), name="Titanic sample")
    pipeline.set_config(target_column="Survived")
    pipeline.resolve_target()
    return pipeline


def test_load_data_creates_preview(tmp_path: Path) -> None:
    pipeline = TabulaML(output_dir=tmp_path)
    df = pipeline.load_data(_load_titanic_df(), name="Titanic sample")
    assert not df.empty
    assert (tmp_path / "raw_preview.csv").exists()


def test_suggest_target_column_prefers_survived(tmp_pipeline: TabulaML) -> None:
    suggestion = tmp_pipeline.suggest_target_column(tmp_pipeline.state["df_raw"])
    assert suggestion == "Survived"


def test_resolve_target_sets_state(tmp_pipeline: TabulaML) -> None:
    assert "X" in tmp_pipeline.state
    assert "y" in tmp_pipeline.state
    assert tmp_pipeline.state["task_type"] == "classification"


def test_build_preprocessing_handles_numeric_and_categorical(tmp_pipeline: TabulaML) -> None:
    transformer = tmp_pipeline.build_preprocessing()
    numeric_cols = tmp_pipeline.state["X"].select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [col for col in tmp_pipeline.state["X"].columns if col not in numeric_cols]
    # ColumnTransformer stores tuples (name, pipeline, columns)
    transformed_columns = []
    for _, _, cols in transformer.transformers:
        transformed_columns.extend(list(cols))
    for col in numeric_cols + cat_cols:
        assert col in transformed_columns


def test_perform_eda_creates_outputs(tmp_pipeline: TabulaML, tmp_path: Path) -> None:
    outputs = tmp_pipeline.perform_eda()
    for key, path in outputs.items():
        if path is not None:
            assert Path(path).exists(), f"EDA artefact missing: {key}"


def test_detect_outliers_with_default_config(tmp_pipeline: TabulaML) -> None:
    mask = tmp_pipeline.detect_outliers()
    assert mask.dtype == bool
    assert mask.index.equals(tmp_pipeline.state["X"].index)


def test_split_data_returns_expected_shapes(tmp_pipeline: TabulaML) -> None:
    X_train, X_test, y_train, y_test = tmp_pipeline.split_data()
    total_rows = len(tmp_pipeline.state["X"])
    assert len(X_train) + len(X_test) == total_rows
    assert len(y_train) + len(y_test) == total_rows


def test_train_and_evaluate_models(tmp_pipeline: TabulaML) -> None:
    tmp_pipeline.split_data()
    results = tmp_pipeline.train_models()
    assert results, "No models were trained"
    evaluated = tmp_pipeline.evaluate_models()
    for result in evaluated:
        assert result.metrics, f"Metrics missing for {result.name}"


def test_summarise_run_contains_expected_keys(tmp_pipeline: TabulaML) -> None:
    tmp_pipeline.split_data()
    tmp_pipeline.train_models()
    tmp_pipeline.evaluate_models()
    summary = tmp_pipeline.summarise_run()
    assert summary["data_name"] == "Titanic sample"
    assert summary["target_column"] == "Survived"
    assert (Path(tmp_pipeline.output_dir) / "run_summary.json").exists()
