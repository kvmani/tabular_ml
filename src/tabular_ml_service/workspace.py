"""In-memory user workspace state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(slots=True)
class DatasetWorkspace:
    name: str
    source: str
    original: pd.DataFrame
    working: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        *,
        name: str,
        source: str,
        frame: pd.DataFrame,
        metadata: dict[str, Any] | None = None,
    ) -> DatasetWorkspace:
        return cls(
            name=name,
            source=source,
            original=frame.copy(deep=True),
            working=frame.copy(deep=True),
            metadata=metadata or {},
        )

    def reset(self) -> None:
        self.working = self.original.copy(deep=True)
        self.history.clear()

