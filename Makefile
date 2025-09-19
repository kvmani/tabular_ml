SHELL := /bin/sh

.PHONY: setup bootstrap bootstrap-win dev back run frontend test lint fmt clean

PYTHON ?= python3
VENV ?= .venv
WHEEL_DIR := vendor/python_wheels

ifeq ($(OS),Windows_NT)
PYTHON := py -3
PIP := $(VENV)/Scripts/pip.exe
PYTHON_BIN := $(VENV)/Scripts/python.exe
else
PIP := $(VENV)/bin/pip
PYTHON_BIN := $(VENV)/bin/python
endif

setup: ## create venv and install dependencies from vendor when available
	@echo ">>> Preparing virtual environment at $(VENV)"
	@if [ ! -d "$(VENV)" ]; then $(PYTHON) -m venv $(VENV); fi
	@echo ">>> Upgrading pip"
	@$(PIP) install --upgrade pip
	@echo ">>> Installing Python dependencies"
	@set -e; \
	if [ -d "$(WHEEL_DIR)" ] && ls "$(WHEEL_DIR)"/*.whl >/dev/null 2>&1; then \
	    if $(PIP) install --no-index --find-links "$(WHEEL_DIR)" -r REQUIREMENTS.txt; then \
	        echo "Installed from vendored wheels"; \
	    else \
	        echo "Vendored wheels incomplete; falling back to online installation"; \
	        $(PIP) install -r REQUIREMENTS.txt; \
	    fi; \
	else \
	    $(PIP) install -r REQUIREMENTS.txt; \
	fi

bootstrap: ## refresh vendored dependencies via bash script
	@bash scripts/bootstrap_offline.sh

bootstrap-win: ## refresh vendored dependencies via PowerShell script
	@pwsh -File scripts/bootstrap_offline.ps1

back: dev

dev: ## run backend in development mode
	$(PYTHON_BIN) run_app.py --reload

run: ## run backend with production settings
	$(PYTHON_BIN) run_app.py

frontend: ## run the frontend dev server
	cd frontend && npm install && npm run dev

test: ## run pytest suite
	$(PYTHON_BIN) -m pytest -q

lint: ## run static analysis
	$(PYTHON_BIN) -m ruff check .
	$(PYTHON_BIN) -m black --check .

fmt: ## format codebase
	$(PYTHON_BIN) -m black .

clean: ## remove caches and build artifacts
	rm -rf __pycache__ .pytest_cache frontend/node_modules backend/storage/*.pkl $(VENV)
	rm -f vendor/node_modules.tar.gz vendor/node_modules.zip
