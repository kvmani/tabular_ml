# How to Contribute

Thanks for helping evolve the Intranet Tabular ML Studio. Follow the guidelines below to keep the codebase consistent and easy to reason about.

## 🔁 Workflow

1. **Fork or clone** the repository inside your secure environment.
2. **Create an issue** (or reference an existing ticket) describing the enhancement or bug fix.
3. **Work on a feature branch** named `feature/<short-description>` or `fix/<short-description>`.
4. **Keep commits focused** and atomic. Each commit should represent one logical change.
5. **Open a pull request** against `main` with:
   - A descriptive title.
   - A summary of the changes and any important caveats.
   - Evidence of manual or automated testing (command logs, screenshots when UI is touched).

## 🧪 Testing expectations

- **Backend**: run `pytest` before opening a PR. Add tests under `tests/` for new behaviours.
- **Frontend**: run `npm run dev` locally and smoke-test the affected screens. Attach screenshots for notable UI changes when possible.
- **Documentation**: ensure README snippets stay in sync with actual commands.

## 🧱 Coding standards

### Backend (FastAPI)

- Place API routes under `backend/app/api/routes/` and keep request/response schemas in `backend/app/api/schemas.py`.
- Use service modules (`backend/app/services/`) for business logic. Routes should orchestrate, not implement heavy logic.
- Prefer Pydantic models or dataclasses for structured payloads.
- Handle validation errors by raising `HTTPException` with clear messages.
- Keep functions pure and side-effect free when feasible—especially in `services/` modules.

### Frontend (React)

- Place reusable UI in `frontend/src/components/` and API helpers in `frontend/src/api/`.
- Use functional components and React hooks. Avoid class components.
- Keep styling within `styles.css`; for new reusable styles consider using CSS variables defined at the top.
- Access the backend only through the helper functions in `api/client.js`.

### General

- Follow the guidance in [`AGENTS.md`](AGENTS.md) when introducing agents or automation.
- Avoid adding binary assets to git; prefer JSON/CSV fixtures for datasets.
- Keep documentation in Markdown with clear headings and code blocks.

## ✅ Definition of done

A change is considered complete when:

- Code compiles without errors.
- Tests pass locally.
- Documentation (README, inline docstrings) reflects the new behaviour.
- The platform remains fully functional in an offline environment.

Happy building!
