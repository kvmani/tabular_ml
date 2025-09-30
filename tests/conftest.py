from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.services.data_manager import data_manager  # noqa: E402
from backend.app.services.run_tracker import run_tracker  # noqa: E402


@pytest.fixture(autouse=True)
def reset_state():
    data_manager._datasets.clear()
    data_manager._metadata.clear()
    data_manager._splits.clear()
    data_manager._models.clear()
    run_tracker.reset()
    data_manager.ensure_default_dataset()
    yield
    data_manager._datasets.clear()
    data_manager._metadata.clear()
    data_manager._splits.clear()
    data_manager._models.clear()
    run_tracker.reset()
