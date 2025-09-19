#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
NPM_BIN="${NPM_BIN:-npm}"
WHEEL_DIR="$ROOT_DIR/vendor/python_wheels"
NODE_VENDOR_DIR="$ROOT_DIR/vendor/node_modules"
PACKAGE_LOCK="$ROOT_DIR/vendor/package-lock.json"
FRONTEND_DIR="$ROOT_DIR/frontend"

mkdir -p "$WHEEL_DIR" "$NODE_VENDOR_DIR"

echo "[bootstrap] Cleaning previous wheel cache"
rm -f "$WHEEL_DIR"/* || true

echo "[bootstrap] Downloading Python wheels into $WHEEL_DIR"
"$PYTHON_BIN" -m pip download --dest "$WHEEL_DIR" --requirement "$ROOT_DIR/REQUIREMENTS.txt"

echo "[bootstrap] Installing frontend dependencies with $NPM_BIN"
cd "$FRONTEND_DIR"
"$NPM_BIN" install --no-audit --no-fund
if [ -f "./node_modules/.bin/playwright" ]; then
  echo "[bootstrap] Installing Playwright browsers"
  ./node_modules/.bin/playwright install --with-deps
else
  echo "[bootstrap] Playwright binary not found; skipping browser installation" >&2
fi

cd "$ROOT_DIR"

echo "[bootstrap] Syncing node modules into vendor directory"
rm -rf "$NODE_VENDOR_DIR"
mkdir -p "$NODE_VENDOR_DIR"
cp -R "$FRONTEND_DIR/node_modules/." "$NODE_VENDOR_DIR/"
cp "$FRONTEND_DIR/package-lock.json" "$PACKAGE_LOCK"

echo "[bootstrap] Creating compressed node_modules archive"
cd "$ROOT_DIR/vendor"
rm -f node_modules.tar.gz
if command -v tar >/dev/null 2>&1; then
  tar -czf node_modules.tar.gz node_modules
else
  echo "tar not available; skipping archive creation" >&2
fi

cd "$ROOT_DIR"
echo "[bootstrap] Done. Vendored artifacts are ready for offline use."
