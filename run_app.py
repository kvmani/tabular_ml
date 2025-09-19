"""Start the FastAPI backend using uvicorn."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import uvicorn

from config import settings


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("PYTHONPATH", str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the tabular ML backend")
    parser.add_argument("--reload", action="store_true", help="Enable auto reload")
    args = parser.parse_args()

    uvicorn.run(
        "backend.app.main:app",
        host=settings.app.host,
        port=settings.app.port,
        reload=args.reload,
        log_level=settings.app.log_level.lower(),
    )


if __name__ == "__main__":
    main()
