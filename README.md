# Tabular ML Workbench

Tabular ML Workbench is a privacy-first, local web application for exploring,
cleaning, visualizing, training, comparing, and applying machine-learning
models to CSV and Excel data. It supports classification and regression,
chooses safe defaults automatically, and keeps every consequential choice
visible and editable.

The repository is designed for two equal run modes:

- Standalone service at `http://127.0.0.1:5070/tabular_ml/`.
- Companion Flask blueprint mounted by `ml_server` at `/tabular_ml/`.

## Intended capabilities

- CSV and `.xlsx` upload, schema inference, target-task recommendation, data
  quality profile, and editable semantic feature roles.
- Configurable missing-value handling, encoding, scaling, date expansion,
  rare-category handling, leakage warnings, and outlier detection/treatment.
- A broad CPU-friendly estimator catalog for classification and regression,
  cross-validation, model comparison, validation curves, and test-set metrics.
- In-session prediction with uncertainty/probability where supported.
- Configurable Plotly visualizations with per-chart gear controls and
  publication-quality SVG/PNG export.
- At least five versioned public demo datasets, each with machine-readable
  metadata, provenance, licensing, learning goals, and recommended workflow.
- Contextual explanations that teach concepts without hiding the underlying
  controls.

## Status

This repository is under active construction. The authoritative, resumable
status is [docs/PROGRESS_LEDGER.md](docs/PROGRESS_LEDGER.md).

## Development

After the first implementation milestone:

~~~powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[test]"
cd frontend
npm ci
npm run build
cd ..
.\.venv\Scripts\tabular-ml
~~~

No data, model, or plot telemetry is sent to external services. Network access
is used only by maintainers to obtain declared public demo datasets during
repository development; runtime is self-contained.

## License

Application source is MIT-licensed. Each vendored dataset keeps its own license
and attribution in the adjacent JSON metadata and aggregate registry.

