from __future__ import annotations

import numpy as np

from backend.app.services.data_manager import data_manager
from backend.app.services.preprocess import split_dataset


def test_split_is_reproducible():
    metadata = data_manager.load_sample_dataset("synthetic_marketing_leads")
    df = data_manager.get_dataset(metadata.dataset_id)
    split_one = split_dataset(
        df,
        target_column="converted",
        task_type="classification",
        test_size=0.2,
        val_size=0.2,
        random_state=42,
        stratify=True,
    )
    split_two = split_dataset(
        df,
        target_column="converted",
        task_type="classification",
        test_size=0.2,
        val_size=0.2,
        random_state=42,
        stratify=True,
    )

    assert np.array_equal(split_one.y_train.values, split_two.y_train.values)
    assert np.array_equal(split_one.y_val.values, split_two.y_val.values)
    assert split_one.target_column == "converted"
    assert split_one.feature_columns == split_two.feature_columns
