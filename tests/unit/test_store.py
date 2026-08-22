from __future__ import annotations

import pytest

from tabular_ml_service.errors import ApiError
from tabular_ml_service.store import BoundedTTLStore


def test_store_evicts_least_recently_used_item() -> None:
    store = BoundedTTLStore[str](max_items=2, ttl_seconds=30)
    first = store.put("first")
    second = store.put("second")
    assert store.get(first) == "first"

    third = store.put("third")

    with pytest.raises(ApiError, match="expired"):
        store.get(second)
    assert store.get(first) == "first"
    assert store.get(third) == "third"


def test_store_expires_idle_items() -> None:
    now = [10.0]
    store = BoundedTTLStore[str](max_items=2, ttl_seconds=5, clock=lambda: now[0])
    key = store.put("temporary")
    now[0] = 16.0

    with pytest.raises(ApiError) as caught:
        store.get(key)

    assert caught.value.status == 404
    assert len(store) == 0


def test_store_delete_and_clear() -> None:
    store = BoundedTTLStore[str](max_items=2, ttl_seconds=30)
    first = store.put("first")
    store.put("second")
    assert store.delete(first) is True
    assert store.delete(first) is False
    store.clear()
    assert len(store) == 0

