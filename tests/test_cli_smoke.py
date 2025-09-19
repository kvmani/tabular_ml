from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = ROOT / "cli.py"


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CLI_PATH), *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_cli_training_flow(tmp_path):
    list_proc = run_cli("datasets", "list")
    assert "titanic" in list_proc.stdout

    preview_proc = run_cli("datasets", "preview", "--name", "titanic", "--rows", "3")
    assert "Survived" in preview_proc.stdout

    train_proc = run_cli("train", "--name", "titanic", "--algo", "logreg", "--task", "classification")
    train_payload = json.loads(train_proc.stdout)
    assert "model_id" in train_payload
    model_id = train_payload["model_id"]

    eval_proc = run_cli("evaluate", "--run-id", model_id)
    eval_payload = json.loads(eval_proc.stdout)
    assert "metrics" in eval_payload

    info_proc = run_cli("info")
    info_payload = json.loads(info_proc.stdout)
    assert "settings" in info_payload
    assert "datasets" in info_payload
