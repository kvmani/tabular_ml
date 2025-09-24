# Intranet Tabular ML Studio

An offline-first FastAPI + React workspace for exploring and modelling tabular datasets. The project is hardened for air-gapped intranets: dependencies are vendored, datasets ship with the repo, configuration is declarative, and no telemetry or external network calls are performed at runtime.

## Highlights

- **Pydantic v2 + scikit-learn 1.1–1.5 compatibility** with adaptive OneHotEncoder handling.
- **Layered YAML configuration** (defaults, local overrides, env vars) merged via `config.load_settings()` and exposed through `/system/config` and the UI panel.
- **Security by default:** CSP headers, CSRF protection, optional file uploads, in-memory run tracking only.
- **Curated dataset catalog** (1,000+ row Titanic & Adult samples plus synthetics) with a regeneration script.
- **Offline CLI & smoke scripts** mirroring REST workflows and storing artifacts under `artifacts/`.
- **Playwright E2E harness** that blocks non-localhost traffic, captures screenshots, traces, and produces an HTML report.

## Quick start

```bash
# 1. (Optional but recommended) Bootstrap vendored wheels, node modules, and Playwright browsers
make bootstrap          # or make bootstrap-win on Windows

# 2. Create a virtual environment and install from vendor
make setup

# 3. Run the backend (FastAPI + Uvicorn)
make dev                # serves on http://127.0.0.1:8000

# 4. Run the frontend (Vite dev server)
npm --prefix frontend run dev   # serves on http://127.0.0.1:5173
```

For production-style builds, `npm --prefix frontend run build` followed by `npm --prefix frontend run preview -- --host 127.0.0.1 --port 5173` serves the compiled bundle.

## Configuration model

- **Defaults:** `config/config.yaml`
- **Local overrides:** `config/config.local.yaml` (gitignored)
- **Environment:** prefix with `TABULAR_ML__`, separating keys with `__` (e.g. `TABULAR_ML__APP__PORT=9001`).
- **Dataset registry:** `config/datasets.yaml`

`config/schema.py` defines nested Pydantic models (`Settings.app`, `.security`, `.ml`, `.datasets`, `.limits`) and enforces validation. `config/__init__.py` merges all layers, seeds RNGs, and exposes `settings` for import.

The React UI surfaces the effective configuration and dataset registry in the **Configuration** panel, making CSP / CSRF status visible to operators.

## Data catalog

- `data/sample_datasets/titanic_sample.csv` – 1,309 rows from OpenML Titanics.
- `data/sample_datasets/adult_income_sample.csv` – 1,000 rows from the UCI Adult income dataset.
- Synthetic regression/classification datasets with augmentation metadata.

Regenerate or refresh samples with:

```bash
py -3 scripts/prepare_sample_datasets.py
```

## CLI usage

```bash
python cli.py datasets list
python cli.py datasets preview --name titanic --rows 10
python cli.py train --name titanic --algo logreg --task classification
python cli.py evaluate --run-id <model_id>
python cli.py info
```

CLI runs share the same in-memory services as the REST API, and populate `/runs/last` via the `RunTracker` singleton.

## Smoke automation

| Script | Purpose | Output |
|--------|---------|--------|
| `scripts/smoke_api.sh` | Exercises REST endpoints (health, algorithms, sample load, split, train, evaluate) using CSRF headers | `artifacts/api/*.json` |
| `scripts/smoke_backend.sh` | CLI flow (list, preview, train, evaluate, info) | `artifacts/cli/*.json` |
| `npm --prefix frontend run test:e2e` | Playwright UI smoke, network-blocked, captures screenshots and trace | `artifacts/ui/` |

## Testing & linting

```bash
make test              # pytest -q
npm --prefix frontend run test:e2e
make lint              # ruff + black --check
make fmt               # black .
```

Pytest coverage includes configuration layering, dataset registry wiring, preprocessing, algorithm training, CLI commands, API smoke tests, and an integration boot of `run_app.py`. Playwright walks the UI flow end-to-end and stores screenshots in `artifacts/ui/` for auditing.

### UI regression workflow

The full end-to-end browser flow lives in `tests/ui/test_tabular_ml.py`. It boots both servers, fabricates a small CSV, uploads it through the UI, performs preprocessing, trains Random Forest and neural network models, evaluates results, and captures screenshots for each major stage. Run it locally with:

```bash
pytest tests/ui/test_tabular_ml.py -s
```

Screenshots are written to `tests/ui/artifacts/` (gitignored by default). Compress them afterwards for sharing, e.g.:

```bash
cd tests/ui
zip -r artifacts.zip artifacts
```

Attach the resulting `artifacts.zip` out of band (it will not appear in git history) when you need to share visual evidence of the UI run.


## Offline operations

Detailed instructions for preparing and operating the project in air-gapped environments are documented in [RUNNING_OFFLINE.md](RUNNING_OFFLINE.md).

## Project layout

```
backend/                 FastAPI application (routers, services, models)
config/                  YAML config + Pydantic schemas
frontend/                React + Vite single-page app
scripts/                 Operational helpers (bootstrap, smoke tests, dataset prep)
tests/                   Pytest suite + CLI/integration coverage
e2e/                     Playwright config and smoke test
vendor/                  Vendored wheels & node modules (populated via bootstrap)
artifacts/               Smoke-test outputs (ignored by git)
```

## Privacy & security stance

- No telemetry, analytics, or external HTTP calls in application code.
- Logs emit to stdout; user data remains in-memory unless explicitly exported.
- CSP and CSRF protections are enforced when `security.csp_enabled` / `security.csrf_protect` are true (default).
- File uploads are disabled by default (no `python-multipart` import when `allow_file_uploads=false`).

Happy experimenting inside the air gap!
