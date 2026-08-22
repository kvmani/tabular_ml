# Progress Ledger

This file is the authoritative restart point for the `tabular_ml` build goal.
Update it before every milestone commit and whenever scope, validation, or a
blocker changes.

## Goal

Deliver a professional standalone and `ml_server`-integrated tabular machine
learning workbench, with broad CPU-friendly classification/regression support,
guided preprocessing, configurable publication-quality visualization, safe
inference, rich built-in datasets, comprehensive tests, and GitHub publication.

## Current status

- Phase: 7 — `ml_server` integration.
- State: in progress.
- Resume point: commit the completed frontend/dataset milestone, then install
  and mount the companion blueprint in `ml_server`, add catalog/navigation,
  smoke tests, setup scripts, and host documentation.
- Target repository: `C:\Users\kvman\PycharmProjects\tabular_ml`.
- Host repository: `C:\Users\kvman\PycharmProjects\ml_server`.
- Branch: `main` in both repositories.
- Remote: not yet created for `tabular_ml`; `ml_server` uses
  `https://github.com/kvmani/ml_server.git`.

## Decisions

| Date | Decision | Reason |
|---|---|---|
| 2026-08-23 | Package as `tabular-ml-service`, import as `tabular_ml_service`. | Clear PyPI/package ownership while retaining the requested repository name. |
| 2026-08-23 | Flask blueprint + React/Vite/TypeScript frontend. | Matches the current host companion pattern and supports the required rich UI. |
| 2026-08-23 | One route contract for standalone and integrated modes. | Avoids feature drift and duplicate tests. |
| 2026-08-23 | Uploaded data and fitted runs stay in a bounded in-memory store. | Privacy-first behavior without unbounded worker memory. |
| 2026-08-23 | Do not load user-supplied pickle/joblib files. | Python deserialization is unsafe for untrusted uploads. |
| 2026-08-23 | Bundle Plotly locally and expose settings per chart. | Interactive, configurable, publication-quality exports without CDNs. |
| 2026-08-23 | Keep Codex plugin scaffold under `codex/plugins/tabular-ml`. | Preserves exact requested repo path while satisfying normalized plugin naming. |

## Upstream audit

- Public source:
  `https://github.com/kvmani/ml_server_all_in_one/tree/main/plugins/tabular_ml`
- Matching local reference:
  `C:\Users\kvman\PycharmProjects\_legacy_ml_server_reference`
- Baseline capabilities and gaps are recorded in
  [ARCHITECTURE.md](ARCHITECTURE.md).
- No baseline code will be copied blindly; useful behavior will be
  reimplemented behind the cleaner core/API boundary.

## Milestones

- [x] M1 — application/package foundation, registry contract, memory store,
  security headers, CI skeleton, and first tests.
- [x] M2 — data upload, Excel/CSV parsing, semantic schema inference, data
  quality report, missingness and leakage analysis.
- [x] M3 — preprocessing, feature roles, outlier detection/treatment, and
  reproducible train/test preparation.
- [x] M4 — broad algorithms, training/comparison, metrics, diagnostics,
  importance, inference, and model report downloads.
- [x] M5 — polished responsive guided frontend, configurable Plotly panels,
  accessibility, contextual help, and error/progress states.
- [x] M6 — at least five tracked demo CSV datasets with adjacent metadata,
  checksums, provenance, license, learning goals, and registry tests.
- [ ] M7 — `ml_server` install, blueprint/catalog integration, setup/docs,
  smoke tests, and compatible CSP.
- [ ] M8 — complete QA: lint/type/test/build/coverage/E2E/package/security,
  visual inspection screenshots, docs, and release readiness.
- [ ] M9 — GitHub repository creation, push, remote verification, and final
  host repository push.

## Dataset plan

The final registry will contain a deliberate mix rather than five redundant
toy classifiers:

| Dataset | Task | Demonstrates | Planned source/license verification |
|---|---|---|---|
| Titanic | Classification | mixed types, missingness, imbalance | public source; metadata will record exact upstream and terms |
| Adult Census Income | Classification | larger mixed table, fairness-sensitive fields | UCI; CC BY 4.0 |
| Palmer Penguins | Classification | compact multiclass, missingness | Allison Horst; CC0 |
| Wine Quality | Regression/classification | ordinal target, nonlinear relationships | UCI; CC BY 4.0 |
| Auto MPG | Regression | missing values, categorical/numeric mix | UCI; CC BY 4.0 |
| Breast Cancer Wisconsin | Classification | many continuous features | UCI; CC BY 4.0 |
| Diabetes progression | Regression | standardized numeric baseline | scikit-learn source metadata to be verified |

Only datasets with verifiable redistribution terms will ship. Metadata will
distinguish sensitive/ethical teaching considerations from model endorsement.

## Validation log

| Date/time | Command/check | Result |
|---|---|---|
| 2026-08-23 | Inspected `ml_server` status, server factory, catalog, package configuration, and PDF Tools companion package. | Host clean; integration contract confirmed. |
| 2026-08-23 | Opened public upstream repository and audited matching local source/tests/frontend. | Baseline recorded; two datasets and limited model/visualization workflow confirmed. |
| 2026-08-23 | Initialized `tabular_ml` Git repository on `main`. | Passed. |
| 2026-08-23 | Ran plugin-creator scaffold for `codex/plugins/tabular-ml`. | Scaffold created; final validation pending. |
| 2026-08-23 | Installed the project under Python 3.13 with pandas 2.3.3, scikit-learn 1.9.0, SciPy 1.18.1, and OpenPyXL 3.1.5. | Passed; compatible wheels available. |
| 2026-08-23 | `pytest --cov --cov-report=term-missing` | Passed: 9 tests, 85.04% branch coverage before final host-mount test. |
| 2026-08-23 | `ruff check .` | Passed. |
| 2026-08-23 | `mypy src` with strict mode | Passed. |
| 2026-08-23 | `python -m build` | Passed: sdist and wheel contain templates, static shell, and registry. |
| 2026-08-23 | plugin-creator `validate_plugin.py codex/plugins/tabular-ml` | Passed. |
| 2026-08-23 | Mixed CSV/XLSX ingestion, semantic schema, target recommendations, JSON-safe previews, bounded session flow. | Passed unit/integration fixtures including invalid type/size/shape cases. |
| 2026-08-23 | Preprocessing plan/builder across simple/KNN imputation, one-hot/ordinal encoding, four scaling choices, date components, exclusions, and rare-level handling. | Passed; transformer remains unfitted until training/validation. |
| 2026-08-23 | IQR, z-score, modified z-score, Isolation Forest, and LOF detection plus drop/mask/winsorize/reset. | Passed deterministic unit and API workflow tests. |
| 2026-08-23 | `pytest --cov --cov-report=term-missing` after M2/M3 | Passed: 41 tests, 90.46% branch coverage. |
| 2026-08-23 | `ruff check .` and strict `mypy src` after M2/M3 | Passed. |
| 2026-08-23 | Algorithm registry across 14 linear, neighbour, tree, bagging, boosting, kernel, SGD, MLP, probabilistic, and sparse-linear choices. | Passed registry construction and parameter/task validation tests. |
| 2026-08-23 | Leakage-safe holdout and cross-validation, classification/regression metrics, ROC/PR/confusion/residual diagnostics, native/permutation importance. | Passed synthetic signal fixtures for both task types. |
| 2026-08-23 | Model comparison, in-session inference, report retrieval, and generated joblib artifact APIs. | Passed full HTTP workflow tests; no user-supplied pickle loading. |
| 2026-08-23 | `pytest --cov --cov-report=term-missing` after M4 | Passed: 54 tests, 87.76% branch coverage. |
| 2026-08-23 | `ruff check .` and strict `mypy src` after M4 | Passed. |
| 2026-08-23 | Vendored Titanic, Adult Income, Palmer Penguins, Wine Quality, Auto MPG, and Breast Cancer Diagnostic with adjacent metadata, checksums, provenance, licensing, teaching goals, and cautions. | Passed offline registry verification for all 6 datasets. |
| 2026-08-23 | Visualization computation routes for distribution/KDE, correlation, scatter/trend, grouped box, and missingness. | Passed unit and HTTP integration tests with bounded points/columns. |
| 2026-08-23 | `npm run build` and `npm test` for the React/TypeScript workspace. | Passed production type/build and 4 component/layout tests; Plotly is a lazy same-origin chunk. |
| 2026-08-23 | `pytest --cov`, `ruff check .`, strict `mypy src`, and `vendor_datasets.py --check` after M5/M6. | Passed: 61 tests, 88.16% branch coverage, 6 tracked datasets verified. |
| 2026-08-23 | Headed Playwright visual QA at desktop and mobile widths. | Passed landing, workspace, chart/settings, train/evaluate, and predict flows; corrected scroll restoration, desktop nav cascade, intrinsic overflow, CSP, and audit sizing findings. |

## Commit policy

Use conventional commits after coherent vertical slices. Before each commit:

1. Update this ledger.
2. Run the tests/builds relevant to the slice.
3. Review `git diff --check` and `git status`.
4. Commit only owned, in-scope changes.

Expected commits are approximately M1, M2–M3, M4, M5–M6, M7, and M8–M9,
splitting only when a milestone becomes too large to review safely.

## Open risks

- Minimum Python is now 3.12; Python 3.13 compatibility is verified locally.
- Plotly is a 4.6 MB minified lazy chunk (1.38 MB gzip); acceptable for the
  offline scientific surface, but recorded as an explicit build budget.
- Multiclass ROC/PR and explainability payloads can become large; bound points
  and feature counts.
- Plotly requires inline style attributes for its SVG/canvas layout. The
  companion keeps scripts same-origin while allowing inline styles; host
  integration must preserve that explicit exception.
- GitHub CLI is not currently visible on PATH; repository creation may require
  authenticated GitHub API access or browser/credential-manager workflow.
