import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from backend.app.services.data_manager import data_manager
from backend.app.services.model_training import model_trainer
from backend.app.services.preprocess import detect_outliers, split_dataset


def test_sample_dataset_loading():
    metadata = data_manager.load_sample_dataset('titanic')
    assert metadata.dataset_id in {entry['dataset_id'] for entry in data_manager.list_datasets()}
    df = data_manager.get_dataset(metadata.dataset_id)
    assert not df.empty


def test_outlier_detection_counts():
    metadata = data_manager.load_sample_dataset('adult_income')
    df = data_manager.get_dataset(metadata.dataset_id)
    report = detect_outliers(df)
    assert report.total_outliers >= 0
    assert isinstance(report.inspected_columns, list)


def test_logistic_regression_training():
    metadata = data_manager.load_sample_dataset('titanic')
    df = data_manager.get_dataset(metadata.dataset_id)
    split = split_dataset(df, target_column='Survived', task_type='classification', test_size=0.2, val_size=0.2)
    split.dataset_id = metadata.dataset_id
    result = model_trainer.train(
        dataset_id=metadata.dataset_id,
        df=df,
        target_column='Survived',
        task_type='classification',
        algorithm='logistic_regression',
        hyperparameters={'max_iter': 200},
        split=split,
    )
    assert 'validation' in result.artifact.metrics
    assert 'test' in result.artifact.metrics
    assert result.artifact.metrics['validation']['accuracy'] >= 0
