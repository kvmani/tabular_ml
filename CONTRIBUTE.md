# Developer Notes

This document complements `HOWTOCONTRIBUTE.md` with deeper architectural context and conventions for contributors.

## Backend conventions

- **Routing layout**: group endpoints by domain (`data`, `preprocess`, `model`, `visualization`). Every router should expose a FastAPI `APIRouter` instance and be registered in `backend/app/main.py`.
- **Service layer**: heavy lifting lives in `backend/app/services/`. Services should be pure functions or small classes that can be unit tested without HTTP context.
- **DataManager**: acts as an in-memory registry for datasets, splits, and model artifacts. Treat it as a singleton; if you need persistence, add adapters under `backend/app/services/storage_*.py` and do not modify `DataManager` state outside service functions.
- **ModelTrainer**: extend `ModelTrainer.algorithm_catalog` when adding new algorithms. Ensure new algorithms:
  - Accept hyperparameters via dictionaries.
  - Return validation metrics and history lists with primitive types.
  - Store any preprocessing artefacts (encoders, scalers) in the `model_object` so evaluation stays consistent.

## Frontend conventions

- **State orchestration**: keep global workflow state in `App.jsx`; components should remain as stateless as possible and communicate through callbacks.
- **Plot rendering**: use `react-plotly.js` for all charting. When you add a new Plotly figure on the backend, return `{ "figure": figure.to_dict() }` so the UI can consume it without transformations.
- **API helpers**: add new endpoints to `frontend/src/api/client.js` and keep the naming consistent (`verbNoun`). Client functions should raise on non-2xx responses to simplify error handling in components.

## Testing strategy

- **Unit tests**: place backend tests under `tests/`. Mock heavy ML training loops when possible to keep test runs fast.
- **Data fixtures**: prefer small CSV snippets under `tests/fixtures/`. Never commit binary blobs (`.png`, `.pkl`, etc.).
- **Manual QA**: record noteworthy manual test steps in the PR description, especially when altering the training pipeline or UI flows.

## Documentation

- Update `README.md` whenever setup commands or high-level capabilities change.
- Provide inline docstrings for complex functions (e.g., non-trivial preprocessing or model orchestration logic).
- When adding datasets, document their provenance and intended target column.

## Security & offline guarantees

- No outbound network calls are allowed at runtime. If a new dependency performs telemetry by default, disable it explicitly.
- Use environment variables for configuration overrides; never hard-code secrets.
- Keep third-party licences compatible with the existing `LICENSE` file.

Stay mindful of these guidelines to ensure the platform remains predictable and maintainable in constrained environments.
