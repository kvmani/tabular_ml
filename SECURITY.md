# Security Policy

## Supported versions

Security fixes are applied to the latest release on `main`.

## Reporting

Please report a vulnerability privately to the repository owner rather than
opening a public issue. Include reproduction steps, affected routes, and the
expected impact, but do not include real confidential datasets.

## Security model

- Uploaded tables and trained runs are held in a bounded, expiring in-memory
  store; the application has no telemetry or automatic cloud upload.
- File types, upload size, table dimensions, and spreadsheet contents are
  validated before use.
- User-supplied pickle/joblib artifacts are never loaded. Downloaded artifacts
  must be treated as trusted-code files and opened only in trusted environments.
- Responses apply restrictive security headers and use same-origin frontend
  assets. Plotly requires the documented inline-style exception.
- Algorithms and visualization traces are CPU-only; accelerator runtimes are
  forbidden by automated dependency and source-policy tests.

Do not expose the development server directly to an untrusted network. Deploy
behind an authenticated, maintained WSGI/reverse-proxy boundary when access is
shared.
