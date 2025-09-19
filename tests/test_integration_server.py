from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]


def _wait_for_health(url: str, timeout: float = 15.0) -> httpx.Response:
    start = time.time()
    while time.time() - start < timeout:
        try:
            response = httpx.get(url, timeout=1.0)
            if response.status_code == 200:
                return response
        except httpx.RequestError:
            time.sleep(0.5)
    raise AssertionError("Server did not become ready in time")


def test_run_app_startup():
    env = os.environ.copy()
    env["TABULAR_ML__APP__PORT"] = "8055"
    proc = subprocess.Popen(
        [sys.executable, "run_app.py"],
        cwd=ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        health_url = "http://127.0.0.1:8055/health"
        health_resp = _wait_for_health(health_url)
        assert health_resp.json()["status"] == "ok"

        alg_resp = httpx.get("http://127.0.0.1:8055/model/algorithms", timeout=2.0)
        assert alg_resp.status_code == 200
        assert alg_resp.json()["algorithms"]
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
