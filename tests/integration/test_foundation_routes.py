from __future__ import annotations

from flask import Flask

from tabular_ml_service.app import tabular_ml_bp


def test_index_is_served_at_companion_prefix(client) -> None:
    response = client.get("/tabular_ml/")
    assert response.status_code == 200
    assert b"Tabular ML Workbench" in response.data
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert "default-src 'self'" in response.headers["Content-Security-Policy"]


def test_health_uses_response_envelope(client) -> None:
    response = client.get("/tabular_ml/api/v1/health")
    payload = response.get_json()
    assert response.status_code == 200
    assert payload["success"] is True
    assert payload["data"]["service"] == "tabular_ml"
    assert payload["data"]["status"] == "ok"
    assert payload["data"]["cpu_only"] is True
    assert response.headers["Cache-Control"] == "no-store"


def test_config_documents_privacy_and_limits(client) -> None:
    response = client.get("/tabular_ml/api/v1/config")
    payload = response.get_json()["data"]
    assert payload["privacy"]["telemetry"] is False
    assert payload["privacy"]["persistence"] == "none"
    assert payload["limits"]["accepted_extensions"] == [".csv", ".xlsx"]


def test_dataset_list_has_stable_shape(client) -> None:
    response = client.get("/tabular_ml/api/v1/datasets")
    payload = response.get_json()
    assert payload["success"] is True
    assert len(payload["data"]["datasets"]) == 6
    assert payload["data"]["datasets"][0]["key"] == "titanic"


def test_blueprint_initializes_extensions_when_mounted_by_host() -> None:
    host = Flask("host")
    host.register_blueprint(tabular_ml_bp)

    response = host.test_client().get("/tabular_ml/api/v1/config")

    assert response.status_code == 200
    assert "tabular_ml_settings" in host.extensions
    assert "tabular_ml_store" in host.extensions
