"""Playwright end-to-end flow for the Tabular ML UI."""
from __future__ import annotations

import csv
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Generator
import threading

import httpx
import pytest
from playwright.sync_api import BrowserContext, Page, expect, sync_playwright

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def _wait_for_http(url: str, *, timeout: float = 60.0) -> None:
    """Poll an HTTP endpoint until it becomes responsive or times out."""

    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            response = httpx.get(url, timeout=5.0)
        except httpx.HTTPError as exc:  # pragma: no cover - network hiccup during poll
            last_error = exc
        else:
            if response.status_code < 500:
                return
            last_error = RuntimeError(
                f"Received status {response.status_code} from {url}"
            )
        time.sleep(0.5)
    if last_error is None:
        last_error = TimeoutError(f"Timed out waiting for {url}")
    raise RuntimeError(f"Service at {url} unavailable: {last_error}")


def _stream_process_output(label: str, stream: subprocess.Popen[str].stdout) -> list[str]:
    """Stream process logs to stdout while retaining them for debugging."""

    lines: list[str] = []

    def _reader() -> None:
        for line in iter(stream.readline, ""):
            lines.append(line)
            print(f"[{label}] {line}", end="")

    thread = threading.Thread(target=_reader, daemon=True)
    thread.start()
    return lines


@pytest.fixture(scope="session")
def dataset_csv(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Create a small classification dataset for upload tests."""

    tmp_dir = tmp_path_factory.mktemp("ui_dataset")
    path = tmp_dir / "playwright_dataset.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["feature_one", "feature_two", "category", "target"])
        for idx in range(120):
            feature_one = idx % 15
            feature_two = (idx * 7) % 11
            category = "A" if idx % 3 == 0 else ("B" if idx % 3 == 1 else "C")
            target = "yes" if (feature_one + feature_two) % 4 < 2 else "no"
            writer.writerow([feature_one, feature_two, category, target])
    return str(path)


@pytest.fixture(scope="session")
def server_urls() -> Generator[Dict[str, str], None, None]:
    """Launch backend and frontend servers for the duration of the test session."""

    env = os.environ.copy()
    project_path = str(PROJECT_ROOT)
    existing_path = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{project_path}{os.pathsep}{existing_path}" if existing_path else project_path
    )
    env["TABULAR_ML__APP__ALLOW_FILE_UPLOADS"] = "true"
    env["TABULAR_ML__SECURITY__CORS_ORIGINS"] = json.dumps(
        ["http://127.0.0.1:5173", "http://localhost:5173"]
    )

    backend_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "backend.app.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
    ]
    print(f"[ui-test] PYTHONPATH={env['PYTHONPATH']}")
    print("[ui-test] Starting backend server...")
    backend = subprocess.Popen(  # noqa: S603 - controlled command
        backend_cmd,
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
        text=True,
    )
    backend_logs: list[str] = []
    if backend.stdout is not None:
        backend_logs = _stream_process_output("backend", backend.stdout)

    try:
        _wait_for_http("http://127.0.0.1:8000/health", timeout=90.0)
        print("[ui-test] Backend ready.")
    except Exception:
        backend_output = "".join(backend_logs[-40:]) if backend_logs else ""
        backend.terminate()
        backend.wait(timeout=10)
        raise RuntimeError(
            f"Backend failed to start. Output:\n{backend_output}"
        )

    frontend_env = env.copy()
    frontend_env["VITE_API_BASE_URL"] = "http://127.0.0.1:8000"
    frontend_cmd = [
        "npm",
        "run",
        "dev",
        "--",
        "--host",
        "127.0.0.1",
        "--port",
        "5173",
    ]
    print("[ui-test] Starting frontend dev server...")
    frontend = subprocess.Popen(  # noqa: S603 - controlled command
        frontend_cmd,
        cwd=PROJECT_ROOT / "frontend",
        env=frontend_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
        text=True,
    )
    frontend_logs: list[str] = []
    if frontend.stdout is not None:
        frontend_logs = _stream_process_output("frontend", frontend.stdout)

    try:
        _wait_for_http("http://127.0.0.1:5173", timeout=90.0)
        print("[ui-test] Frontend ready.")
    except Exception:
        frontend_output = "".join(frontend_logs[-40:]) if frontend_logs else ""
        backend_output = "".join(backend_logs[-40:]) if backend_logs else ""
        _terminate_process(frontend)
        _terminate_process(backend)
        raise RuntimeError(
            "Frontend failed to start."
            f"\nFrontend output:\n{frontend_output}"
            f"\nBackend output:\n{backend_output}"
        )

    urls = {"frontend": "http://127.0.0.1:5173", "backend": "http://127.0.0.1:8000"}
    try:
        yield urls
    finally:
        _terminate_process(frontend)
        _terminate_process(backend)


def _terminate_process(process: subprocess.Popen[bytes | str | None]) -> None:
    """Terminate a subprocess and its children safely."""

    if process.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except ProcessLookupError:  # pragma: no cover - already terminated
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:  # pragma: no cover - rare slow shutdown
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        process.wait(timeout=5)


@pytest.fixture(scope="session")
def browser_context(server_urls: Dict[str, str]) -> Generator[BrowserContext, None, None]:
    """Provide a Playwright browser context bound to the frontend URL."""

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(base_url=server_urls["frontend"])
        try:
            yield context
        finally:
            context.close()
            browser.close()


@pytest.fixture()
def page(browser_context: BrowserContext) -> Generator[Page, None, None]:
    page = browser_context.new_page()
    try:
        yield page
    finally:
        page.close()


def _capture(page: Page, filename: str) -> None:
    """Capture a full-page screenshot into the artifacts directory."""

    path = ARTIFACT_DIR / filename
    page.screenshot(path=path, full_page=True)


def test_tabular_ml_end_to_end(
    page: Page,
    server_urls: Dict[str, str],
    dataset_csv: str,
) -> None:
    """Walk through the major UI workflow and assert key states."""

    page.goto(server_urls["frontend"])
    expect(page.get_by_role("heading", name="Intranet Tabular ML Studio")).to_be_visible()
    _capture(page, "01_homepage.png")

    dataset_selector = page.locator('[data-testid="dataset-selector"]')
    dataset_selector.wait_for()

    upload_form = page.locator("form.upload-form")
    file_input = upload_form.locator('input[type="file"]')
    file_input.set_input_files(dataset_csv)
    upload_form.locator('input[placeholder="Display name"]').fill(
        "Playwright Test Dataset"
    )
    upload_form.locator('textarea[placeholder="Description"]').fill(
        "Generated automatically for UI validation."
    )
    upload_form.get_by_role("button", name="Upload dataset").click()
    expect(page.locator(".notification")).to_contain_text("Uploaded Playwright Test Dataset")
    preview_rows = page.locator('[data-testid="dataset-preview"] tbody tr')
    expect(preview_rows.first).to_be_visible()
    summary_rows = page.locator('[data-testid="summary-table"] tbody tr')
    expect(summary_rows.first).to_be_visible()
    _capture(page, "02_dataset_upload.png")

    page.locator('[data-testid="dataset-preview"]').scroll_into_view_if_needed()
    _capture(page, "03_data_exploration.png")

    preprocess_header = page.get_by_role("heading", name="2. Preprocess")
    preprocess_card = preprocess_header.locator("xpath=..").locator("xpath=..").first
    preprocess_card.scroll_into_view_if_needed()
    page.locator('[data-testid="detect-outliers"]').click()
    expect(page.locator(".notification")).to_contain_text("Detected")
    page.locator('[data-testid="remove-outliers"]').click()
    expect(page.locator(".notification")).to_contain_text("outliers removed")

    filter_section = page.locator('div:has(> h3:has-text("Filter"))')
    filter_section.locator("select").first.select_option("category")
    filter_section.locator("input[type='text']").fill("A")
    filter_section.locator('[data-testid="filter-apply"]').click()
    expect(page.locator(".notification")).to_contain_text("filtered")

    page.locator('[data-testid="split-target"]').select_option("target")
    page.locator('[data-testid="create-split"]').click()
    expect(page.locator(".notification")).to_contain_text("Created split")
    _capture(page, "04_preprocess.png")

    model_header = page.get_by_role("heading", name="4. Train models")
    model_card = model_header.locator("xpath=..").locator("xpath=..").first
    model_card.scroll_into_view_if_needed()
    page.get_by_label("Algorithm").select_option("random_forest")
    page.get_by_test_id("train-target").select_option("target")
    page.get_by_role("button", name="Start training").click()
    expect(page.locator(".notification")).to_contain_text("Model trained successfully")
    expect(page.locator('[data-testid="metrics-summary"]')).to_be_visible()
    _capture(page, "05_training_random_forest.png")

    page.get_by_label("Algorithm").select_option("neural_network")
    hyperparams = (
        '{"hidden_layer_sizes": [32, 16], "max_iter": 200, "early_stopping": true, "random_state": 42}'
    )
    model_card.locator("textarea").fill(hyperparams)
    page.get_by_role("button", name="Start training").click()
    expect(page.locator(".notification")).to_contain_text("Model trained successfully")
    expect(page.locator('[data-testid="metrics-summary"]')).to_be_visible()
    _capture(page, "06_training_neural_network.png")

    evaluation_header = page.get_by_role("heading", name="5. Evaluate & share")
    evaluation_card = evaluation_header.locator("xpath=..").locator("xpath=..").first
    evaluation_card.scroll_into_view_if_needed()
    page.get_by_test_id("evaluate-button").click()
    expect(page.locator(".notification")).to_contain_text("Evaluation completed")
    expect(page.locator('[data-testid="metrics-summary"]')).to_be_visible()
    expect(page.locator('[data-testid="confusion-matrix"]')).to_be_visible()
    _capture(page, "07_evaluation_results.png")
