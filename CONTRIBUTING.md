# Contributing Guide

Thanks for improving Intranet Tabular ML Studio! This guide complements `HOWTOCONTRIBUTE.md` and `CONTRIBUTE.md` with the specifics introduced by the latest offline hardening.

## Workflow summary

1. Create focused commits on top of `main`. Branching is optional inside secure CI/CD environments.
2. Run `make lint` and `make test` before submitting patches.
3. When touching the frontend, follow the smoke checklist in `frontend/e2e-smoke.md`.
4. Document new configuration knobs or datasets in `README.md`.

## Coding practices

- Keep FastAPI routes thin and delegate work to services under `backend/app/services/`.
- When adding configuration options, update `config/schema.py`, `config/config.yaml`, and document the change in `OFFLINE_OPERATIONS.md` if operators need to act.
- Respect `settings.limits` when introducing new data processing endpoints.
- Prefer deterministic behaviour—propagate the configured seed when using random generators.

## Tests

- Add or extend pytest modules in `tests/` for new functionality. Avoid network calls and file writes.
- Use the fixtures in `tests/conftest.py` to keep the in-memory store clean between tests.
- Keep test runtime below 60 seconds; leverage the synthetic datasets for reproducible inputs.

## Offline & privacy requirements

- Never introduce outbound network requests or CDN links. Vendored assets live inside the repository.
- Do not persist user data to disk unless a new configuration option explicitly requests it.
- Guard optional dependencies (e.g., file uploads) behind feature flags so the backend runs with the minimum install set.

Following these guidelines keeps the platform predictable and secure for operators working inside air-gapped networks.
