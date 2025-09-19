# Contributing Guide

Thanks for improving Intranet Tabular ML Studio! This guide complements the historical `HOWTOCONTRIBUTE.md` and `CONTRIBUTE.md` files with the specifics introduced by the offline hardening effort.

## Workflow summary

1. Create focused commits on top of `main`. Branching is optional inside secure CI/CD environments.
2. Run `make lint` and `make test` before submitting patches.
3. When touching the frontend, also execute `npm --prefix frontend run test:e2e`.
4. Document new configuration knobs or datasets in `README.md` and `RUNNING_OFFLINE.md`.

## Coding practices

- Keep FastAPI routes thin and delegate business logic to services under `backend/app/services/`.
- When adding configuration options, update `config/schema.py`, `config/config.yaml`, and mention operator impacts in the docs.
- Respect `settings.limits` (row counts, timeout) when introducing new compute-heavy routines.
- Prefer deterministic behaviour: propagate the configured seed whenever random generators are involved.

## Tests

- Add or extend pytest modules in `tests/` for new functionality. Avoid network calls and file writes.
- Use the fixtures in `tests/conftest.py` to keep the in-memory store clean between tests.
- Keep test runtime reasonable (under a minute); rely on the bundled datasets for reproducible inputs.

## Offline & privacy requirements

- Never introduce outbound network requests or CDN links. Vendored assets must live inside the repository (see `scripts/bootstrap_offline.*`).
- Do not persist user data to disk unless a new configuration option explicitly requests it.
- Guard optional dependencies (for example file uploads) behind feature flags so the backend runs with the minimal install set.

Following these guidelines keeps the platform predictable and secure for operators working inside air-gapped networks.
