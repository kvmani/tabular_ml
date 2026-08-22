from __future__ import annotations

import pytest

from tabular_ml_service.app import create_app
from tabular_ml_service.config import Settings


@pytest.fixture()
def app():
    return create_app(
        {"TESTING": True},
        settings=Settings(max_sessions=3, session_ttl_seconds=60),
    )


@pytest.fixture()
def client(app):
    return app.test_client()

