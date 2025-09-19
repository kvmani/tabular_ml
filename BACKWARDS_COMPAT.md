# Backwards Compatibility Notes

The project supports multiple dependency versions that are common in long-lived intranet environments. The switches below keep behaviour consistent across stacks.

## Pydantic v1 vs v2

- The configuration system (`config/schema.py`) uses Pydantic v2 `BaseModel` APIs. If your runtime still ships with Pydantic v1, install `pydantic>=1.10` and enable compatibility mode via:
  ```bash
  export PYDANTIC_V1=true
  ```
  This preserves the old validation semantics while allowing gradual migration.
- All FastAPI response models remain compatible with Pydantic v1/v2 because they avoid v2-only features such as `field_validator` on dataclasses.

## Scikit-learn 1.1 – 1.5

- `settings.ml.sklearn_onehot_sparse` toggles the OneHotEncoder output mode.
  - For scikit-learn <1.2, set it to `false` (default) to use `sparse=False` and avoid the newer `sparse_output` parameter.
  - For scikit-learn ≥1.2, set it to `true` if you prefer sparse matrices to conserve memory on wide categorical datasets.
- RandomForest defaults now honour `settings.ml.n_jobs`, so older releases that only support CPU-bound execution remain usable.

## Optional features

- File uploads rely on `python-multipart`. Keep `app.allow_file_uploads=false` (default) when the dependency is unavailable.
- The optional `neural_network` algorithm relies on PyTorch. When `torch` is absent the algorithm is hidden from the catalog and all flows continue to function with scikit-learn estimators only. Mirror and install PyTorch wheels if you need neural networks in your environment.
- XGBoost remains excluded by default to avoid introducing its GPU/toolchain requirements; add it explicitly in your fork if desired.

Refer to `OFFLINE_OPERATIONS.md` for dependency pinning guidance when preparing bespoke environments.
