# Agent Guidelines

This repository is developed iteratively by autonomous agents. Please follow these principles:

1. Prefer clear, maintainable implementations with comprehensive inline comments when logic may be non-obvious.
2. Keep modules small and cohesive; avoid monolithic files.
3. Always provide FastAPI routers in `backend/app/api/routes` and keep business logic in `backend/app/services`.
4. For React components, colocate hook logic inside `frontend/src/hooks` when it is reused across components.
5. Tests belong in the `tests/` directory and should avoid network calls.
6. Documentation updates must remain in sync with implemented features.
7. When introducing datasets, store them under `data/` and document their provenance in the README.
