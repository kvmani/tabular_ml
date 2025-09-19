# Intranet Tabular ML Studio

Intranet Tabular ML Studio is an offline-first workflow for exploring, preparing, and modelling tabular datasets. The project is designed for air-gapped deployments: every dependency is vendored locally, configuration lives in YAML, and all state remains in-memory unless explicitly exported.

## ✨ Highlights

- **Layered configuration** – defaults live in `config/config.yaml`, operators override via `config.local.yaml`, and environment variables win last.
- **Expanded dataset catalog** – 22 curated and synthetic CSV files ship in `data/sample_datasets/` with registry metadata in `config/datasets.yaml`.
- **Synthetic augmentation** – configurable generators (SMOTE-like, Gaussian noise, and rule-based perturbations) expand each dataset ≥10× at runtime.
- **FastAPI backend + React frontend** – runs without external APIs, telemetry, or CDN assets.
- **Offline CLI & smoke scripts** – `cli.py` mirrors core REST flows for shell automation and `scripts/smoke_backend.sh` validates a running backend.
- **Deterministic training** – seeds propagate from configuration so pytest, CLI, and API produce reproducible splits and metrics.

## 🏗 Repository map

```
├── backend/
│   └── app/
│       ├── api/               # FastAPI routers and Pydantic schemas
│       ├── core/              # Runtime config helpers
│       ├── models/            # Dataclasses for in-memory storage
│       ├── services/          # Data loading, preprocessing, augmentation, training
│       └── main.py            # FastAPI application factory
├── config/
│   ├── config.yaml            # Safe defaults committed to source
│   ├── config.local.yaml      # Operator overrides (gitignored)
│   └── datasets.yaml          # Dataset registry consumed by backend + CLI
├── data/sample_datasets/      # Curated CSV files ready for air-gapped use
├── frontend/                  # React + Vite single page application
├── scripts/                   # Operational helpers (smoke tests, synthetic generation)
├── tests/                     # Pytest suite covering config, data, API, and training flows
├── cli.py                     # Offline orchestration CLI
├── run_app.py                 # uvicorn bootstrap respecting configuration
├── Makefile                   # Common developer tasks
└── REQUIREMENTS.txt           # Python runtime dependencies
```

## ⚙️ Running the backend

```bash
make setup        # create virtualenv and install offline-ready wheels
make dev          # run FastAPI with auto-reload (uses config host/port)
```

For a one-shot invocation without reload:

```bash
python run_app.py
```

Health is available at `http://<host>:<port>/health`. The `/system/config` endpoint exposes the merged configuration and dataset registry for auditing.

## 🧪 Testing & linting

```bash
make test         # pytest -q
make lint         # ruff + black --check
make fmt          # black .
```

Pytest exercises configuration layering, dataset registry wiring, preprocessing determinism, algorithm training, and REST smoke flows with uploads disabled.

## 🧩 Optional dependencies

The default install path keeps dependencies lightweight for CPU-only environments. Core workflows use scikit-learn models exclusively; installing [PyTorch](https://pytorch.org/) is optional. When `torch` is present the `neural_network` algorithm (CLI alias `nn`) is exposed in the API and CLI. The pytest suite automatically detects the library and runs neural-network coverage only when it is available, so environments without PyTorch remain fully supported.

## 🗂 Dataset registry & synthetic expansion

Dataset metadata is stored in `config/datasets.yaml`. Each entry defines a key, friendly name, file, target column, description, and task type. Example excerpt:

```yaml
titanic:
  name: "Titanic Survival"
  description: "Passenger data with survival labels"
  file: "titanic_sample.csv"
  target: "Survived"
  task: "classification"
synthetic_sales_forecast:
  name: "Sales Forecast"
  description: "Synthetic quarterly revenue regression"
  file: "synthetic_sales_forecast.csv"
  target: "quarterly_revenue_k"
  task: "regression"
```

When a dataset is loaded, the backend automatically materialises an augmented sibling dataset (respecting `datasets.synthetic` settings) and tracks provenance via metadata extras. `scripts/gen_synthetic_all.py` shows how to invoke the augmentation pipeline without persisting data.

## 🛠 CLI usage

```bash
python cli.py datasets list
python cli.py datasets preview --name titanic --rows 20
python cli.py train --name titanic --algo logreg --task classification
python cli.py evaluate --run-id <model_id>
python cli.py info
```

The CLI operates entirely offline using the same services as the API. Runs triggered through the CLI populate the in-memory `run_tracker`, which also backs the `/runs/last` endpoint.

## 🔐 Air-gapped considerations

- No CDN links or telemetry endpoints are referenced in the frontend; all assets are bundled locally.
- File uploads are feature-flagged (`app.allow_file_uploads`). When disabled (the default), the backend never imports `python-multipart` and the upload route is omitted entirely.
- Logging defaults to stdout with operator-configurable level; no files are written unless explicitly configured.
- Limits (`limits.max_rows_preview`, `limits.max_rows_train`, `limits.max_cols`) are enforced on preview and training routes to guard resources.

## 📚 Additional docs

- [`OFFLINE_OPERATIONS.md`](OFFLINE_OPERATIONS.md) – end-to-end bootstrap guidance for air-gapped environments.
- [`BACKWARDS_COMPAT.md`](BACKWARDS_COMPAT.md) – notes on Pydantic v1/v2 and scikit-learn 1.1–1.5 compatibility flags.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) – coding standards, testing expectations, and commit guidance.

Happy hacking inside the air-gap! 👩‍💻
