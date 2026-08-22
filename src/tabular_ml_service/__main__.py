"""Standalone production entry point."""

from __future__ import annotations

import argparse
import os

from waitress import serve

from .app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Tabular ML Workbench")
    parser.add_argument("--host", default=os.getenv("TABULAR_ML_HOST", "127.0.0.1"))
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("TABULAR_ML_PORT", "5070")),
    )
    parser.add_argument("--dev", action="store_true", help="Use Flask's development server")
    args = parser.parse_args()
    app = create_app()
    if args.dev:
        app.run(host=args.host, port=args.port, debug=False)
    else:
        serve(app, host=args.host, port=args.port, threads=4)


if __name__ == "__main__":
    main()

