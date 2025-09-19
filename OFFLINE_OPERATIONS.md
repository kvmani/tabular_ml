# Offline Operations Guide

This project targets air-gapped deployments. The checklist below summarises how to bootstrap, operate, and maintain the stack without internet access.

## 1. Prepare Python dependencies

1. Mirror the packages listed in `REQUIREMENTS.txt` to an internal PyPI proxy **or** download wheels ahead of time.
2. Place cached wheels in a directory such as `vendor/pip/`.
3. Run `make setup` with the environment variable `PIP_FIND_LINKS=vendor/pip` to force pip to install from the local cache:
   ```bash
   PIP_FIND_LINKS=vendor/pip make setup
   ```
4. Verify installation succeeded by launching `python -m pytest -q`.
5. PyTorch is optional. Only mirror and install `torch` wheels if you intend to expose the neural network algorithm; the rest of the system and tests run without it.

## 2. Frontend dependencies

1. Mirror npm packages (`react`, `react-dom`, `vite`, `plotly.js`, `react-plotly.js`).
2. Configure `.npmrc` to point at the mirror or copy `.tgz` tarballs into `frontend/vendor/`.
3. Install dependencies offline:
   ```bash
   cd frontend
   npm install --offline --cache ./vendor
   npm run build
   ```
4. Serve the built assets locally or use `npm run dev` for an interactive session.

## 3. Configuration management

- The default configuration lives in `config/config.yaml`.
- Operators should edit `config/config.local.yaml` (gitignored) to add local overrides for host, port, or feature flags.
- Environment overrides use the `TABULAR_ML__...` prefix. Example:
  ```bash
  TABULAR_ML__APP__LOG_LEVEL=DEBUG python run_app.py
  ```
- Use `GET /system/config` (or `python cli.py info`) to inspect the effective configuration at runtime.

## 4. Dataset handling

- All sample datasets are stored under `data/sample_datasets/`.
- To add new datasets, update `config/datasets.yaml` and place the CSV/XLSX file in the same directory.
- Synthetic augmentation is controlled via `datasets.synthetic` in `config/config.yaml`. Toggle generators or multipliers there.
- Use `python scripts/gen_synthetic_all.py` to dry-run augmentation without persisting files.

## 5. Operational smoke tests

- API health: `curl http://<host>:<port>/health`
- CLI dataset list: `python cli.py datasets list`
- CLI training smoke: `python cli.py train --name titanic --algo logreg --task classification`
- REST smoke: `scripts/smoke_backend.sh http://<host>:<port>`

## 6. Routine maintenance

- Run `make lint` and `make test` before promoting changes.
- Periodically refresh cached wheels/tarballs to pick up security updates.
- Rotate datasets by updating `config/datasets.yaml` and documenting provenance in `README.md`.

By following these steps the platform remains self-contained and auditable within a locked-down intranet.
