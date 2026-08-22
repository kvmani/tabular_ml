"""Thread-safe, TTL- and capacity-bounded in-memory artifact storage."""

from __future__ import annotations

import secrets
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

from .errors import ApiError


@dataclass(slots=True)
class _Entry[T]:
    value: T
    touched_at: float


class BoundedTTLStore[T]:
    """Store opaque workflow objects without persisting user data."""

    def __init__(
        self,
        *,
        max_items: int,
        ttl_seconds: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_items < 1 or ttl_seconds < 1:
            raise ValueError("Store limits must be positive")
        self._max_items = max_items
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._entries: OrderedDict[str, _Entry[T]] = OrderedDict()
        self._lock = threading.RLock()

    def put(self, value: T, *, key: str | None = None) -> str:
        with self._lock:
            self._purge_expired()
            item_key = key or secrets.token_urlsafe(18)
            self._entries[item_key] = _Entry(value=value, touched_at=self._clock())
            self._entries.move_to_end(item_key)
            while len(self._entries) > self._max_items:
                self._entries.popitem(last=False)
            return item_key

    def get(self, key: str) -> T:
        with self._lock:
            self._purge_expired()
            entry = self._entries.get(key)
            if entry is None:
                raise ApiError(
                    code="session.not_found",
                    message="This workspace expired or does not exist. Load the dataset again.",
                    status=404,
                )
            entry.touched_at = self._clock()
            self._entries.move_to_end(key)
            return entry.value

    def delete(self, key: str) -> bool:
        with self._lock:
            return self._entries.pop(key, None) is not None

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            self._purge_expired()
            return len(self._entries)

    def _purge_expired(self) -> None:
        cutoff = self._clock() - self._ttl_seconds
        expired = [key for key, entry in self._entries.items() if entry.touched_at < cutoff]
        for key in expired:
            del self._entries[key]
