# Running Intranet Tabular ML Studio Offline

This guide explains how to prepare, install, and exercise the project in fully offline or air-gapped environments.

## 1. Prerequisites

- Python 3.10 or 3.11
- Node.js 18+
- Git, Bash (or PowerShell on Windows)
- Curl (for the shell smoke script)

> **Tip:** Perform the bootstrap steps while you still have network access. The output in `vendor/` can be copied into the air-gapped environment and reused across machines.

## 2. Bootstrap vendored dependencies

### Bash

```bash
scripts/bootstrap_offline.sh
```

### PowerShell

```powershell
scripts\bootstrap_offline.ps1
```

The bootstrap script downloads:

- Python wheels for everything listed in `REQUIREMENTS.txt`
- Frontend dependencies (React, Plotly, Playwright)
- Playwright browser binaries (Chromium) for headless testing
- Tar/zip archives under `vendor/` for reproducible installs

## 3. Create a virtualenv and install from vendor

```bash
make setup
```

`make setup` upgrades `pip` and then installs using `--no-index --find-links vendor/python_wheels`. If the wheels are missing it automatically falls back to the public index.

## 4. Start backend and frontend

```bash
# Terminal 1
make dev                 # FastAPI backend on 0.0.0.0:8000

# Terminal 2
npm --prefix frontend run dev   # Vite dev server on 127.0.0.1:5173
```

To serve the static bundle instead of the dev server:

```bash
npm --prefix frontend run build
npm --prefix frontend run preview -- --host 127.0.0.1 --port 5173
```

## 5. Run smoke checks

### API smoke (uses REST endpoints and saves results under `artifacts/api/`)

```bash
scripts/smoke_api.sh http://127.0.0.1:8000
```

### CLI smoke

```bash
scripts/smoke_backend.sh
```

### UI end-to-end smoke (Playwright)

Ensure both backend (`:8000`) and frontend (`:5173`) are running, then execute:

```bash
npm --prefix frontend run test:e2e
```

Screenshots are written to `docs/screenshots/2025-10-06_e2e-smoke/`; traces and the HTML report remain in `artifacts/ui/`.

## 6. Full test suite

```bash
make test                 # pytest -q
npm --prefix frontend run test:e2e
```

Pytest exercises configuration layering, data flows, CLI commands, and an integration boot of `run_app.py`. The Playwright suite covers the UX path (load dataset ? split ? train ? evaluate) with network blocking to guarantee offline behaviour.

## 7. Regenerating sample datasets

If you want to refresh the 1,000+ row CSV samples, run:

```bash
py -3 scripts/prepare_sample_datasets.py
```

The script fetches the public sources (OpenML Titanic, UCI Adult) when network is available and rebuilds the curated files under `data/sample_datasets/`.

> **Release gate:** verify `data/sample_datasets/titanic_sample.csv` is included in the artefact bundle. When the file is missing the backend reports HTTP 503 on `/data/datasets` and the UI pins a blocking notification until the bundle is repaired, preventing users from starting runs without the default sample.

## 8. Configuration references

- Default settings: `config/config.yaml`
- Local overrides (gitignored): `config/config.local.yaml`
- Environment variables: prefix `TABULAR_ML__`, e.g. `TABULAR_ML__APP__PORT=9000`
- Dataset registry: `config/datasets.yaml`

At runtime the `/system/config` endpoint and the **Configuration** panel in the UI expose the merged configuration (including CSP/CSRF flags) for auditing.

## 9. Artifacts and logs

- `artifacts/api/` – JSON payloads from the REST smoke script
- `artifacts/cli/` – CLI command outputs
- `docs/screenshots/2025-10-06_e2e-smoke/` – Playwright screenshots regenerated per run
- `artifacts/ui/` – Playwright traces, HTML report
- Logs are emitted to stdout only and never persisted to disk when `persist_user_data=false` (default).

If you need to clear generated files:

```bash
make clean
rm -rf artifacts
```

You're now ready to operate the platform without external network access.
