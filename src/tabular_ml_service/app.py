"""Flask application factory and reusable companion blueprint."""

from __future__ import annotations

from typing import Any

from flask import Blueprint, Flask, Response, render_template, request
from flask.blueprints import BlueprintSetupState
from werkzeug.exceptions import RequestEntityTooLarge

from .api import api_bp
from .config import DEFAULT_SETTINGS, Settings
from .errors import ApiError
from .responses import failure
from .store import BoundedTTLStore

tabular_ml_bp = Blueprint(
    "tabular_ml",
    __name__,
    url_prefix="/tabular_ml",
    template_folder="templates",
    static_folder="static",
    static_url_path="/assets",
)
tabular_ml_bp.register_blueprint(api_bp)


@tabular_ml_bp.record_once
def initialize_host(state: BlueprintSetupState) -> None:
    """Install blueprint-owned extensions when mounted in another Flask app."""
    app = state.app
    resolved = app.config.get("TABULAR_ML_SETTINGS", DEFAULT_SETTINGS)
    if not isinstance(resolved, Settings):
        raise TypeError("TABULAR_ML_SETTINGS must be a tabular_ml_service.config.Settings")
    app.extensions.setdefault("tabular_ml_settings", resolved)
    app.extensions.setdefault(
        "tabular_ml_store",
        BoundedTTLStore[Any](
            max_items=resolved.max_sessions,
            ttl_seconds=resolved.session_ttl_seconds,
        ),
    )
    current_limit = app.config.get("MAX_CONTENT_LENGTH")
    if current_limit is None:
        app.config["MAX_CONTENT_LENGTH"] = resolved.max_upload_bytes


@tabular_ml_bp.get("/")
def index() -> str:
    return render_template("index.html")


@tabular_ml_bp.after_request
def security_headers(response: Response) -> Response:
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    response.headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
    response.headers.setdefault(
        "Content-Security-Policy",
        "default-src 'self'; script-src 'self'; style-src 'self'; "
        "img-src 'self' data: blob:; font-src 'self'; connect-src 'self'; "
        "worker-src 'self' blob:; object-src 'none'; base-uri 'self'; frame-ancestors 'self'",
    )
    if request.path.startswith("/tabular_ml/api/"):
        response.headers.setdefault("Cache-Control", "no-store")
    return response


@tabular_ml_bp.errorhandler(RequestEntityTooLarge)
def upload_too_large(_error: RequestEntityTooLarge) -> tuple[Response, int]:
    return failure(
        ApiError(
            code="upload.too_large",
            message="The upload exceeds the configured size limit.",
            status=413,
        )
    )


def create_app(
    test_config: dict[str, Any] | None = None,
    *,
    settings: Settings | None = None,
) -> Flask:
    """Create a standalone Flask application using the integration blueprint."""
    resolved = settings or DEFAULT_SETTINGS
    app = Flask(__name__)
    app.config.update(
        MAX_CONTENT_LENGTH=resolved.max_upload_bytes,
        JSON_SORT_KEYS=False,
    )
    if test_config:
        app.config.update(test_config)
    app.config["TABULAR_ML_SETTINGS"] = resolved
    app.register_blueprint(tabular_ml_bp)
    return app
