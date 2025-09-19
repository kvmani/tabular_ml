"""Synthetic dataset augmentation utilities."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config import settings
from config.schema import SyntheticCfg


def _rng() -> np.random.Generator:
    return np.random.default_rng(settings.app.random_seed)


def smote_like_upsample(
    df: pd.DataFrame,
    target_column: str,
    samples: int,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Generate synthetic classification rows using a SMOTE-inspired heuristic."""

    if samples <= 0:
        return pd.DataFrame(columns=df.columns)
    rng = rng or _rng()
    features = df.drop(columns=[target_column])
    numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [col for col in features.columns if col not in numeric_cols]
    classes = df[target_column].unique()
    new_records: List[Dict[str, object]] = []
    class_indices: Dict[object, np.ndarray] = {
        cls: df.index[df[target_column] == cls].to_numpy() for cls in classes
    }
    for _ in range(samples):
        cls = rng.choice(classes)
        idx_pool = class_indices.get(cls)
        if idx_pool is None or len(idx_pool) == 0:
            continue
        if len(idx_pool) == 1:
            idx1 = idx2 = int(idx_pool[0])
        else:
            idx1, idx2 = rng.choice(idx_pool, size=2, replace=True)
        row1 = df.loc[idx1]
        row2 = df.loc[idx2]
        new_row: Dict[str, object] = {}
        for col in numeric_cols:
            value = row1[col]
            try:
                value = float(value)
            except (TypeError, ValueError):
                value = 0.0
            other = row2[col]
            try:
                other = float(other)
            except (TypeError, ValueError):
                other = value
            interpolated = value + rng.random() * (other - value)
            new_row[col] = interpolated
        for col in categorical_cols:
            new_row[col] = rng.choice([row1[col], row2[col]])
        new_row[target_column] = cls
        new_records.append(new_row)
    return pd.DataFrame(new_records, columns=df.columns)


def gaussian_mixture_noise(
    df: pd.DataFrame,
    target_column: str,
    samples: int,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Create noisy variants of rows using Gaussian perturbations."""

    if samples <= 0:
        return pd.DataFrame(columns=df.columns)
    rng = rng or _rng()
    indices = rng.choice(df.index.to_numpy(), size=samples, replace=True)
    numeric_cols = (
        df.drop(columns=[target_column]).select_dtypes(include=[np.number]).columns
    )
    generated = []
    for idx in indices:
        base_row = df.loc[idx].copy()
        for col in numeric_cols:
            series = df[col].astype(float, errors="ignore")
            std = series.std(ddof=0) if series.std(ddof=0) > 0 else 1.0
            noise = rng.normal(loc=0.0, scale=std * 0.05)
            value = base_row[col]
            try:
                value = float(value) + noise
            except (TypeError, ValueError):
                pass
            base_row[col] = value
        generated.append(base_row.to_dict())
    return pd.DataFrame(generated, columns=df.columns)


def rule_based_noise(
    df: pd.DataFrame,
    target_column: str,
    samples: int,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Generate rows by jittering numeric ranges and permuting categoricals."""

    if samples <= 0:
        return pd.DataFrame(columns=df.columns)
    rng = rng or _rng()
    indices = rng.choice(df.index.to_numpy(), size=samples, replace=True)
    numeric_cols = (
        df.drop(columns=[target_column]).select_dtypes(include=[np.number]).columns
    )
    categorical_cols = [
        col for col in df.columns if col not in numeric_cols and col != target_column
    ]
    generated = []
    for idx in indices:
        base_row = df.loc[idx].copy()
        for col in numeric_cols:
            series = df[col].astype(float, errors="ignore")
            span = series.max() - series.min()
            jitter = 0.02 * span if span else 0.1
            noise = rng.uniform(-jitter, jitter)
            try:
                base_row[col] = float(base_row[col]) + noise
            except (TypeError, ValueError):
                base_row[col] = base_row[col]
        for col in categorical_cols:
            values = df[col].dropna().unique()
            if len(values) > 1 and rng.random() < 0.1:
                base_row[col] = rng.choice(values)
        generated.append(base_row.to_dict())
    return pd.DataFrame(generated, columns=df.columns)


def augment_dataset(
    df: pd.DataFrame,
    target_column: str,
    task_type: str,
    config: SyntheticCfg,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Apply configured generators until the requested multiplier is achieved."""

    rng = _rng()
    base_rows = len(df)
    target_rows = max(base_rows, base_rows * config.per_real_row_multiplier)
    remaining = target_rows - base_rows
    generated_parts: List[pd.DataFrame] = []
    applied: List[str] = []

    for name in config.generators:
        if remaining <= 0:
            break
        if name == "smote" and task_type == "classification":
            chunk = smote_like_upsample(df, target_column, remaining, rng=rng)
        elif name == "gaussian_mixture":
            chunk = gaussian_mixture_noise(
                df,
                target_column,
                min(remaining, base_rows),
                rng=rng,
            )
        elif name == "rule_based_noise":
            chunk = rule_based_noise(
                df,
                target_column,
                min(remaining, base_rows),
                rng=rng,
            )
        else:
            continue
        if chunk.empty:
            continue
        generated_parts.append(chunk)
        applied.append(name)
        remaining = target_rows - (
            base_rows + sum(len(part) for part in generated_parts)
        )

    if generated_parts:
        augmented = pd.concat([df] + generated_parts, ignore_index=True)
    else:
        augmented = df.copy()
    if len(augmented) > target_rows:
        augmented = augmented.sample(
            n=target_rows, random_state=settings.app.random_seed
        ).reset_index(drop=True)
    else:
        augmented = augmented.reset_index(drop=True)

    info = {
        "base_rows": base_rows,
        "target_rows": target_rows,
        "applied_generators": applied,
        "final_rows": len(augmented),
    }
    return augmented, info
