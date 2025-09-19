"""In-memory tracker for recent model runs."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from threading import Lock
from typing import Any, Dict, Optional


@dataclass
class RunSummary:
    run_id: str
    dataset_id: str
    dataset_name: str
    algorithm: str
    task_type: str
    metrics: Dict[str, Any]
    created_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        return data


class RunTracker:
    """Store the latest run summary in-memory only."""

    def __init__(self) -> None:
        self._last: Optional[RunSummary] = None
        self._lock = Lock()

    def record(self, summary: RunSummary) -> RunSummary:
        with self._lock:
            self._last = summary
        return summary

    def get_last(self) -> Optional[RunSummary]:
        with self._lock:
            return self._last

    def reset(self) -> None:
        with self._lock:
            self._last = None


run_tracker = RunTracker()
