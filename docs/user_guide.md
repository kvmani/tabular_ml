# User Guide

This guide documents the end-to-end flow validated for the default Titanic dataset. The interface is optimised for air-gapped browsers and works out-of-the-box without uploading new files.

## Launching the stack

1. Start the FastAPI backend: `make dev` (serves on port 8000 by default).
2. Start the React frontend: `npm --prefix frontend run dev` (serves on port 5173 by default).
3. Open the frontend URL in your browser. The SPA automatically detects the backend origin, swapping dev-server ports such as `5173` for `8000` (and translating GitHub Codespaces host suffixes) when `VITE_API_BASE_URL` is not provided, so no additional environment variables are required when both services are exposed through the same host (Codespaces, Gitpod, localhost, etc.).

## Initial dataset load

- On first load the UI checks the dataset registry via `/system/config` and `/data/datasets`.
- If the in-memory store is empty it issues `POST /data/samples/titanic` to hydrate the catalogue.
- The dataset preview, descriptive summary, and column metadata are fetched during the initial dataset bootstrap so the Titanic table renders as soon as the app loads—no extra clicks required.
- If the backend reports the default dataset is unavailable (HTTP 503 with the recorded reason) the UI automatically invokes the Titanic loader and surfaces a blocking notification when the bundle is missing so operators can restore it before proceeding.

## Exploration workflow

1. **Preview & summary** – use the Dataset panel to confirm the rows/columns that auto-loaded.
2. **Preprocessing** – create filtered or imputed variants and persist splits for model training. Newly created datasets are added to the selector automatically.
3. **Visualisation** – generate histograms and scatter plots by selecting the relevant columns; the hook reloads preview + schema data whenever the dataset changes to ensure the options are in sync.
4. **Training** – choose an algorithm (Logistic Regression, Random Forest, etc.), select the target column (pre-populated from the dataset metadata), and kick off a training run.
5. **Evaluation** – the Evaluate button posts to `/model/evaluate` and renders metrics, confusion matrices, ROC curves, or regression diagnostics depending on the trained model.

## Upload policy banner

The System Configuration API exposes `settings.app.allow_file_uploads`. The dataset hook mirrors this flag into the UI banner and upload form, so administrators immediately see when uploads are disabled (as in air-gapped environments) and can rely on the bundled datasets instead.

## Live Logs console

- The bottom drawer labelled **“Show Live Logs”** reveals an accessible log console that streams backend log events via `/system/logs/stream`.
- Each entry includes the ISO-8601 timestamp, severity icon (ℹ️ info, ⚠️ warning, ❌ error, 🐞 debug), and logger name so operators can correlate events.
- Use the level checkboxes to filter what is rendered. Debug messages require enabling the **Debug** checkbox, which re-subscribes with `?level=DEBUG`.
- Controls along the header allow you to pause/resume the feed, clear the buffer, and toggle auto-scroll. When paused, events are buffered and replayed on resume.
- The console stores up to 500 entries in-memory (oldest dropped first) and mirrors toast errors so that transient notifications leave an audit trail. Only high-level diagnostics are streamed—no dataset contents or PII leave the server.

## Troubleshooting

- **Backend reachable?** Use the Configuration panel’s dataset registry list to confirm the frontend is connected to the backend.
- **No datasets listed?** Refresh the list from the Dataset panel; the hook will reload the Titanic sample if the store was cleared and will pin a blocking alert if the bundled CSV was not shipped with the release.
- **Different hostname/port?** Set `VITE_API_BASE_URL` in the frontend environment to the desired API origin. The hook first honours this variable, then applies the dev heuristics (5173 → 8000 / Codespaces suffix swap), and finally defaults to the current origin, so overrides are only required for true cross-domain deployments.
