# Vendored Dependencies

This directory stores offline copies of runtime dependencies so the project can be installed in air-gapped environments.

- `python_wheels/` &mdash; Python package wheels downloaded with `pip download`.
- `node_modules/` &mdash; npm packages mirrored from `frontend/node_modules`.

Use `scripts/bootstrap_offline.sh` (or the PowerShell equivalent) before entering the air gap to refresh the contents. The `Makefile` expects these directories to exist and will install from them when available.
