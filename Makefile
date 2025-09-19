.PHONY: setup dev run back test lint fmt clean

PYTHON ?= python3
VENV ?= .venv
PIP := $(VENV)/bin/pip
PYTHON_BIN := $(VENV)/bin/python

setup: ## create venv and install dependencies
@test -d $(VENV) || $(PYTHON) -m venv $(VENV)
$(PIP) install --upgrade pip
$(PIP) install -r REQUIREMENTS.txt

back: dev

dev: ## run backend in development mode
$(PYTHON) run_app.py --reload

run: ## run backend (and optional frontend if available)
$(PYTHON) run_app.py

test: ## run pytest suite
$(PYTHON) -m pytest -q

lint: ## run static analysis
$(PYTHON) -m ruff check .
$(PYTHON) -m black --check .

fmt: ## format codebase
$(PYTHON) -m black .

clean: ## remove caches and build artifacts
rm -rf __pycache__ .pytest_cache frontend/node_modules backend/storage/*.pkl
