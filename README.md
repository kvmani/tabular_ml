# Intranet Tabular ML Studio

Intranet Tabular ML Studio is an offline-first machine learning platform for tabular datasets. It pairs a FastAPI backend with a React web client to deliver an end-to-end workflow for analysts and data scientists operating inside air-gapped networks.

## ✨ Highlights

- **Air-gapped friendly** – all dependencies are local Python/Node packages, no external APIs or telemetry.
- **End-to-end workflow** – upload data, profile, clean, split, train, and evaluate in a minimal-click UI.
- **Multiple algorithms** – Random Forest, Logistic Regression, XGBoost, Feedforward Neural Networks, and Linear Regression out of the box.
- **Rich evaluation visuals** – confusion matrices, ROC curves, regression diagnostics, and training history plots rendered with Plotly.
- **Sample datasets included** – Titanic survival and US Census Income data ship with the platform for instant exploration.

## 🏗 Architecture

```
├── backend/                # FastAPI application
│   ├── app/
│   │   ├── api/            # Routers and Pydantic schemas
│   │   ├── core/           # Configuration
│   │   ├── models/         # Runtime storage models
│   │   ├── services/       # Data, preprocessing, training, evaluation helpers
│   │   └── main.py         # FastAPI entrypoint
├── frontend/               # React + Vite single-page application
│   └── src/
│       ├── api/            # REST client helpers
│       ├── components/     # UI components per workflow stage
│       ├── App.jsx         # Application shell
│       └── styles.css      # Tailored styling
├── data/sample_datasets/   # Curated offline datasets
├── REQUIREMENTS.txt        # Python dependencies
└── README.md               # You are here
```

The backend keeps datasets, preprocessing splits, and trained models in-memory for fast experimentation. Plotly figures are generated server-side and transferred to the UI as JSON.

## 📦 Prerequisites

- Python **3.10+**
- Node.js **18+** (for the front-end tooling)
- npm or yarn
- (Optional) A Python virtual environment for isolation

## ⚙️ Backend setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r REQUIREMENTS.txt
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

The FastAPI docs become available at `http://localhost:8000/docs` for quick inspection. All endpoints are designed to operate without external services.

> **Note:** PyTorch and XGBoost wheels are large. For CPU-only deployments, download the appropriate offline wheels from the official repositories (e.g. `https://download.pytorch.org/whl/cpu`) and cache them inside your package mirror before running the installation command above.

## 💡 Frontend setup

```bash
cd frontend
npm install
npm run dev -- --host 0.0.0.0 --port 5173
```

Open `http://localhost:5173` in a browser within the intranet environment. The SPA discovers the backend at `http://localhost:8000` by default; override via the `VITE_API_BASE_URL` environment variable if needed.

## 🚀 Workflow overview

1. **Upload or load data** – drag in `.csv` or `.xlsx` files or pick one of the bundled datasets. Preview, schema, and descriptive stats appear immediately.
2. **Preprocess** – detect/remove outliers, impute missing values, craft filters, and produce train/validation/test splits with stratification.
3. **Explore** – build publication-ready histograms and scatter plots with Plotly.
4. **Train** – select an algorithm, tune hyperparameters (JSON input for full control), and launch training. Validation/test metrics and per-epoch logs are captured.
5. **Evaluate** – generate confusion matrices, ROC curves, regression diagnostics, and review metrics for presentation or auditing.

Contextual messaging and help text guide non-expert users throughout the workflow.

## 📊 Sample datasets

| Key            | File                               | Target column | Description                                             |
|----------------|------------------------------------|---------------|---------------------------------------------------------|
| `titanic`      | `data/sample_datasets/titanic_sample.csv`       | `Survived`    | Titanic passenger manifest with survival labels.        |
| `adult_income` | `data/sample_datasets/adult_income_sample.csv`  | `income`      | US Census features to predict high/low income brackets. |

The datasets originate from public Kaggle/UCI sources and were subsampled for rapid iteration.

## 🧪 Testing

The backend ships with pytest-based sanity checks. Execute them after installing requirements:

```bash
pytest
```

Front-end unit tests are not included; rely on manual QA in the dev server.

## 🔒 Air-gapped operations

- No CDN references or telemetry are included; Plotly renders entirely client-side.
- All datasets reside on disk and are never transmitted outside the network.
- Dependencies can be pre-fetched into an internal package registry for offline installation.

## 🤝 Contributing

See [`HOWTOCONTRIBUTE.md`](HOWTOCONTRIBUTE.md) and [`CONTRIBUTE.md`](CONTRIBUTE.md) for development guidelines, branching strategy, and coding standards tailored to multi-agent collaboration.

## 📄 License

Distributed under the terms of the included `LICENSE` file.
