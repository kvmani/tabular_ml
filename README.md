# Tabular ML Workbench

Tabular ML Workbench is a privacy-first, local web application for exploring,
cleaning, visualizing, training, comparing, and applying machine-learning
models to CSV and Excel data. It supports classification and regression,
chooses safe defaults automatically, and keeps every consequential choice
visible and editable.

All data processing, training, validation, and inference are CPU-only. The
project has no CUDA, ROCm, PyTorch, TensorFlow, RAPIDS, or other GPU runtime
dependency; the enforceable policy is documented in
[docs/CPU_ONLY.md](docs/CPU_ONLY.md).

The repository is designed for two equal run modes:

- Standalone service at `http://127.0.0.1:5070/tabular_ml/`.
- Companion Flask blueprint mounted by `ml_server` at `/tabular_ml/`.

## What it includes

- CSV and `.xlsx` upload, schema inference, target-task recommendation, data
  quality profile, and editable semantic feature roles.
- Configurable missing-value handling, encoding, scaling, date expansion,
  rare-category handling, leakage warnings, and outlier detection/treatment.
- A broad CPU-friendly estimator catalog for classification and regression,
  cross-validation, model comparison, fold-stability summaries, and test-set
  metrics.
- In-session prediction with uncertainty/probability where supported.
- Configurable Plotly visualizations with per-chart gear controls and
  publication-quality SVG/PNG export.
- Six versioned public demo datasets, each with machine-readable
  metadata, provenance, licensing, learning goals, and recommended workflow.
- Contextual explanations that teach concepts without hiding the underlying
  controls.

## Start the standalone app

Python 3.12 or newer is required. Release checkouts already contain the built
frontend, so Node.js is not needed to run the application.

~~~powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[test]"
.\.venv\Scripts\tabular-ml --port 5070
~~~

Open `http://127.0.0.1:5070/tabular_ml/`. The six demo datasets are available
from the first screen, and uploaded `.csv`/`.xlsx` files remain in the bounded
in-memory session store.

The scientific guide at `http://127.0.0.1:5070/tabular_ml/help` documents the
analysis workflow, cross-validation and metric equations, critical inputs,
leakage controls, and interpretation limits.

## Development

~~~powershell
.\.venv\Scripts\python -m pytest
.\.venv\Scripts\ruff check .
.\.venv\Scripts\mypy src
cd frontend
npm ci
npm run build
npm test
~~~

Run these checks locally before every push. CI (`.github/workflows/quality.yml`) is
manual-only to save GitHub minutes: it never runs on push, pull request or a schedule.
Start it from the Actions tab ("Run workflow") before a release or an upgrade.

## Integrate with `ml_server`

The package exposes the same application as a mountable Flask blueprint:

~~~python
from tabular_ml_service import create_tabular_ml_blueprint

app.register_blueprint(create_tabular_ml_blueprint())
~~~

The sibling `ml_server` repository includes the catalog card, dependency,
content-security policy, setup flow, and integration smoke tests.

## Production deployment

Build and install the release artifact, then run the bundled Waitress entry
point behind the platform reverse proxy:

~~~powershell
python -m build
python -m pip install --no-deps dist\tabular_ml_service-0.2.0-py3-none-any.whl
tabular-ml --host 127.0.0.1 --port 5070
~~~

Use `GET /tabular_ml/api/v1/health` as the readiness probe and preserve the
`/tabular_ml` prefix in the proxy. Roll back by reinstalling the previous wheel
and restarting the service; runtime state is bounded and in-memory, so there is
no database migration or persistent user data to reverse.

## Guides

- [User guide](docs/USER_GUIDE.md)
- [HTTP API](docs/API.md)
- [Architecture](docs/ARCHITECTURE.md)
- [CPU-only policy](docs/CPU_ONLY.md)
- [Resumable progress ledger](docs/PROGRESS_LEDGER.md)

No data, model, or plot telemetry is sent to external services. Network access
is used only by maintainers to obtain declared public demo datasets during
repository development; runtime is self-contained.

## License

Application source is MIT-licensed. Each vendored dataset keeps its own license
and attribution in the adjacent JSON metadata and aggregate registry.
