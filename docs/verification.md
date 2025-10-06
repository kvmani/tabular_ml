# Verification Playbook

This document explains how to reproduce the Playwright smoke coverage manually and how to run the automated suite that now backs CI.

## Automated coverage (`npm --prefix frontend run test:e2e`)

1. Install backend dependencies: `python -m pip install -r REQUIREMENTS.txt`.
2. Install frontend dependencies: `npm --prefix frontend ci` or `npm --prefix frontend install` for local development.
3. Install Playwright browsers once per machine: `npx --prefix frontend playwright install --with-deps`.
4. Execute the suite: `npm --prefix frontend run test:e2e`.

The script boots both the FastAPI backend and the Vite dev server, then performs the following assertions:

- Titanic preview, summary, and column metadata render on first paint without user input.
- Uploading `e2e/fixtures/iris_small.csv` refreshes the preview table and column list.
- Histogram and scatter plots are generated from the visualization panel and expose Plotly canvases.
- Model training runs to completion while observing streaming (SSE) readiness and exposes metrics.
- Evaluation triggers Plotly plots for the confusion matrix and training history.
- Full-page screenshots are saved to `docs/screenshots/2025-10-06_e2e-smoke/` for each phase.

CI publishes Playwright traces under `artifacts/ui/` for debugging failures.

## Manual browser walkthrough

Follow these steps in a regular browser session if you need to audit behaviour without the automation harness.

1. Start the backend (`uvicorn backend.app.main:app --host 127.0.0.1 --port 8000`) and frontend (`npm --prefix frontend run dev -- --host 127.0.0.1 --port 5173`).
2. Load `http://127.0.0.1:5173`.
   - Confirm the Titanic preview is visible immediately with familiar columns such as `PassengerId`, `Survived`, and `Fare`.
   - Confirm the column list is populated without additional clicks.
3. Upload `e2e/fixtures/iris_small.csv` via the Dataset panel.
   - Set the display name to “Iris fixture (Playwright)” so the notification matches the automated check.
   - Wait for the notification banner and confirm the preview updates with `sepal_length`, `sepal_width`, etc.
4. In the visualization panel:
   - Generate a histogram for `sepal_length`.
   - Generate a scatter plot for `sepal_length` vs. `petal_length` coloured by `species`.
   - Ensure both Plotly canvases render.
5. Train a model:
   - Pick `species` as the target column.
   - Start training and observe notification updates until the success toast appears.
   - Verify validation/test metrics populate immediately after training.
6. Evaluate the trained model:
   - Click **Evaluate model**.
   - Confirm the metrics grid, training history plot, and confusion matrix render.
7. Capture the screenshots listed in `docs/screenshots/2025-10-06_e2e-smoke/README.md`.

If any step fails, capture console/network logs and attach them to the CI failure or issue report.
