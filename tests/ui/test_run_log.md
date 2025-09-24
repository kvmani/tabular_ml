# UI Test Run Log

## Environment Preparation
- Created a Python virtual environment and installed backend dependencies from `REQUIREMENTS.txt`.
- Installed Playwright for Python and downloaded the Chromium browser runtime.
- Installed frontend dependencies using `npm install` inside `frontend/`.

## Backend Adjustments
- Updated the model training service to provide an sklearn-based neural network fallback when PyTorch is unavailable. This keeps the "Neural Network" option functional without heavy dependencies.
- Ensured algorithm listings reflect the active backend implementation.

## Test Flow Summary
1. Started FastAPI backend with uploads enabled and CORS configured for the Vite dev server.
2. Started the Vite frontend pointing to the local API.
3. Playwright automated the UI to:
   - Load the homepage and upload a generated CSV dataset.
   - Inspect dataset preview and statistics.
   - Detect and remove outliers, filter rows, and create a train/val/test split.
   - Train Random Forest and Neural Network models with custom hyperparameters.
   - Evaluate the trained model and capture metrics/plots.
4. Captured screenshots for each major milestone in `tests/ui/artifacts/` (excluded from git).

## Observations
- With the sklearn fallback in place, neural network training completes within seconds on the small dataset.
- CSRF protection works seamlessly once the browser reuses the token emitted by the backend.
- Plotly visualisations render correctly in the dev server environment during automated runs.

## Latest Session Notes
- Exposed the `X-CSRF-Token` header via CORS so the frontend can read and reuse CSRF tokens during uploads.
- Relaxed dataset summary generation to avoid using the removed `datetime_is_numeric` argument on newer pandas releases.
- Hooked Model Trainer labels to their controls, enabling Playwright's label selectors and improving accessibility.
- Completed an end-to-end run of `pytest tests/ui/test_tabular_ml.py -s`, generating fresh screenshots in `tests/ui/artifacts/`.

