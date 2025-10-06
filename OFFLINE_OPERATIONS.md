# Offline Operations Guide

This project targets air-gapped deployments. For a detailed walkthrough see [RUNNING_OFFLINE.md](RUNNING_OFFLINE.md). The checklist below summarises the key tasks operators typically perform.

## 1. Bootstrap dependencies

- Run `scripts/bootstrap_offline.sh` (or `scripts/bootstrap_offline.ps1`) while you still have internet access.
- Copy the populated `vendor/` directory, Playwright browsers, and `artifacts/` templates into the secured environment.

## 2. Backend setup

- `make setup` creates a virtualenv and installs from `vendor/python_wheels`.
- Launch the API with `make dev` (default `http://127.0.0.1:8000`).
- Inspect merged configuration via `GET /system/config` or `python cli.py info`.

## 3. Frontend setup

- `npm --prefix frontend run dev` serves the React UI on `http://127.0.0.1:5173`.
- For static hosting run `npm --prefix frontend run build` followed by `npm --prefix frontend run preview -- --host 127.0.0.1 --port 5173`.

## 4. Dataset management

- Sample CSV files live in `data/sample_datasets/`; metadata lives in `config/datasets.yaml`.
- Regenerate the larger samples with `py -3 scripts/prepare_sample_datasets.py` when new upstream data is available.
- Synthetic augmentation is controlled via `datasets.synthetic` in `config/config.yaml`.
- Always ship the Titanic CSV in the bundle—`data/sample_datasets/titanic_sample.csv` must be present or the API will return HTTP 503 for `/data/datasets` and the UI will surface a blocking alert until the file is restored.

## 5. Smoke coverage

| Tool | Command | Notes |
|------|---------|-------|
| REST smoke | `scripts/smoke_api.sh http://127.0.0.1:8000` | Saves JSON payloads to `artifacts/api/` |
| CLI smoke | `scripts/smoke_backend.sh` | Stores CLI outputs under `artifacts/cli/` |
| UI smoke | `npm --prefix frontend run test:e2e` | Requires backend + frontend, writes screenshots to `docs/screenshots/2025-10-06_e2e-smoke/` |

## 6. Routine maintenance

- Run `make lint` and `make test` before promoting changes.
- Keep vendored wheels/node modules up to date by re-running the bootstrap script outside the air gap.
- Document new configuration flags or dataset additions in `README.md` and `RUNNING_OFFLINE.md`.

Following these steps keeps the platform self-contained and auditable inside locked-down intranets.
