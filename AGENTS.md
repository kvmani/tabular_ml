# AGENTS.md — Engineering & Agentic Workflow Guidelines

These guidelines define how contributors (including automated “agents”) design, implement, test, document, and **verify end-to-end functionality** across this repo. The goal is a clean, testable, and reproducible pipeline with a UI that actually works.

---

## 1) Core Engineering Principles

- **Clarity first:** Prefer clear, maintainable implementations; add **comprehensive inline comments** wherever logic isn’t obvious.
- **Small, cohesive modules:** Avoid monoliths. Split code by responsibility and keep functions/classes short with single responsibility.
- **Separation of concerns:**  
  - **FastAPI routers** live in `backend/app/api/routes/`  
  - **Business logic/services** live in `backend/app/services/`  
  - **Shared models/schemas** live in `backend/app/models/` and `backend/app/schemas/`
- **React architecture:**  
  - Reusable hooks in `frontend/src/hooks/`  
  - Feature-scoped components under `frontend/src/features/<feature>/`  
  - Avoid cross-feature coupling.
- **Tests:** All tests live in `tests/`. Unit/integration tests **must not perform network calls**; mock external I/O.
- **Documentation parity:** Docs **must remain in sync** with implemented features and APIs—no exceptions.
- **Datasets:** Place under `data/`, keep **provenance in README**, and ensure licensing allows distribution.

---
## 2) Repos objective
1. ** To provide no code platform for beginners and senior practitioners for running tabular ML jobs over web.
2. ** Provide simple, intitive and advanced ML processing routines for common data anayslis workflows.

## 3) Definition of Done (DoD)

A change is **not** complete until **all** of the following are true:

1. **Code quality:** Adheres to style/lint/typing:
   - Python: `ruff`, `black`, `mypy` (or pyright for TS).
   - TS/JS: `eslint`, `prettier`, type-safe code.
2. **Separation respected:** Routers in `backend/app/api/routes/`, services in `backend/app/services/`, hooks in `frontend/src/hooks/`.
3. **End-to-end validation:**  
   - **Run the entire pipeline**: data load → exploration → outlier detection → training loop → metrics evaluation.  
   - **UI verified in a real browser** (not headless only). Use **Titanic** as the default dataset and confirm it is selectable and works out-of-the-box.
4. **Tests pass:**  
   - Unit + integration tests (no network).  
   - E2E tests against a **local** dev server using local fixtures.  
   - Add/extend tests to cover the new/changed behavior.
5. **Docs updated:**  
   - Update `docs/api.md`, `docs/user_guide.md`, `README.md`, and any relevant feature docs to reflect new/changed APIs and UI behavior.
6. **Artifacts captured:**  
   - Capture UI screenshots showing the flow (see §8) and save to `docs/screenshots/<YYYY-MM-DD>_<task-name>/`.
7. **Task Report updated:**  
   - Append an entry to `docs/task_report.md` (see §9).

---

## 4) Backend Standards (FastAPI)

- **Routers** expose only request/response translation, validation, and orchestration—not business logic.
- **Services** contain domain logic; they return domain objects or DTOs the router serializes.
- **Schemas**: Every route must have `pydantic` request/response models.
- **Config & logging** live under `backend/app/core/`.
- **I/O boundaries** (file read/write, model load/save) are **isolated in utilities** to enable mocking in tests.
- **Error handling:** Raise typed exceptions in services; map to proper HTTP codes in routers.

---

## 5) Frontend Standards (React)

- **Hooks** in `frontend/src/hooks/` encapsulate fetch/state/business logic reused across components.
- **Components** are presentational when possible; keep data-fetching inside hooks.
- **UI Testing:** Unit test components; stub hook returns. E2E uses local fixtures and dev server.
- **Accessibility & UX:** Provide labels, landmarks, keyboard navigation, and visible focus styles.

---

## 6) Data & Default Dataset Policy

- **Default dataset:** **Titanic** ships in `data/titanic/` and is surfaced in the UI by default, even in testing mode.
- **UI requirement:** On first load, users can see Titanic data pre-loaded **without uploads** and run the whole flow.
- **Provenance:** Document dataset source, license, and any preprocessing in `README.md` (Data section).
- **Versioning:** If dataset changes (columns, schema), bump a minor version, migrate tests/fixtures, and update docs.

---

## 7) Testing Policy

- **No network calls** in unit/integration tests—mock all HTTP/file/network boundaries.
- **Fixtures** for deterministic data and model outputs.
- **Coverage targets:** Aim for ≥80% logical coverage for changed files; critical modules ≥90%.
- **E2E (local only):** Spin up backend + frontend locally with the Titanic fixture and run user flows.
- **Performance sanity:** Add quick smoke checks to keep regressions visible.

---

## 8) Manual UI Verification Script (Agent Runs in Own Browser)

Agents must **manually** verify the UI in a real browser and **capture screenshots** for each step below:

1. **Launch** local backend and frontend. Confirm both health checks pass.
2. **Load default dataset (Titanic)** from UI:
   - Confirm sample rows preview renders.
   - Confirm column summary is populated.
3. **Exploration:**
   - Generate distribution plots; verify axes labels and legends are legible.
4. **Outlier detection:**
   - Run detection; confirm flagged items and summary metrics render.
5. **Train loop:**
   - Start training for ≥20 epochs (for Titanic).  
   - Verify live/recorded metric plots (loss/accuracy) are visible and update.
6. **Evaluation:**
   - Confirm accuracy metrics and confusion/regression plots show.
7. **Export/Download (if applicable):**
   - Verify artifacts download without errors.

Save screenshots in:  
`docs/screenshots/<YYYY-MM-DD>_<task-name>/`  
Include: `01_home.png`, `02_data_preview.png`, `03_explore.png`, `04_outliers.png`, `05_training.png`, `06_metrics.png`, etc.

> Note: These screenshots are **verification artifacts** and should not be used as golden tests. Keep them small and anonymized.

---

## 9) Task Report (Append-Only Log)

For **every** completed task, append to `docs/task_report.md`:

```
## <Task Name>
**Task date:** YYYY-MM-DD
**Task name:** <short slug>
**Details of issues resolved or features added:**
- <bullet 1>
- <bullet 2>
- <links to PRs or commit SHAs if applicable>

**Verification artifacts:**
- Screenshots: docs/screenshots/<YYYY-MM-DD>_<task-name>/
- Test summary: <paste or link to local run output>

**Notes:**
- <edge cases, follow-ups, TODOs>
```

This creates a clear “fingerprint” of activities over time.

---

## 10) Commit Messages & PR Template

**Commit format:**
```
<type>(<scope>): <short summary>

[body: what/why, any migration notes]
[footer: Related: #issue, BREAKING-CHANGE: ...]
```
Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `build`, `perf`.

**PR Checklist (must check all):**
- [ ] Code follows structure (routers/services/hooks).
- [ ] New/changed code has tests; all tests pass locally.
- [ ] Entire pipeline manually verified in a real browser with **Titanic**.
- [ ] Screenshots added under `docs/screenshots/…`.
- [ ] Docs updated (`README.md`, `docs/api.md`, `docs/user_guide.md`).
- [ ] `docs/task_report.md` entry appended.
- [ ] No network calls in unit/integration tests.

---

## 11) Tooling & Automation

- **Pre-commit hooks:** run `ruff`, `black`, `mypy`, `eslint`, `prettier`, and basic test subset.
- **Make/Task scripts:** Provide `make dev`, `make test`, `make e2e`, `make verify-ui`, `make lint`, `make format` for consistent local workflows.
- **Local E2E harness:** A script in `tools/scripts/` seeds the Titanic dataset, starts services, and drives a minimal flow for demo/verification.

---

## 12) API & Contract Documentation

- Maintain `docs/api.md` with request/response schemas (examples included).
- Each router must reference the corresponding section in `docs/api.md`.
- Breaking API changes require a **minor/major** version bump and a migration note in `docs/user_guide.md`.

---

## 13) Handling Non-Obvious Logic

- Add **explanatory comments** near complex control flow, math, or heuristics.
- For algorithms (e.g., outlier detection), briefly document the rationale, parameters, and limitations at the module top.

---

## 14) Releases

- **Versioning:** Semantic Versioning (`MAJOR.MINOR.PATCH`).
- **Changelog:** Summarize notable changes and verification status; link to `task_report.md` entries.

---

## 15) Governance

- **Single source of truth:** This **AGENTS.md** governs agent and human contributions.  
- Changes to this file require approval by maintainers and should be accompanied by updates to `docs/dev_guide.md` where relevant.

---

### Quick Start — Agent Checklist

- [ ] Implement feature with clear structure and comments.  
- [ ] Write/update tests (no network).  
- [ ] Run full pipeline locally.  
- [ ] Verify UI in real browser using **Titanic**; capture screenshots.  
- [ ] Update all relevant docs.  
- [ ] Append to `docs/task_report.md`.  
- [ ] Open PR using template and pass all checks.

---

**Remember:** Every code change must prove, via **tests + manual UI run**, that the **entire pipeline** still works end-to-end.