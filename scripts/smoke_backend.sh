#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACT_DIR="${ARTIFACT_DIR:-$ROOT_DIR/artifacts/cli}"
mkdir -p "$ARTIFACT_DIR"

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  elif command -v py >/dev/null 2>&1; then
    PYTHON_BIN="py -3"
  else
    echo "Python interpreter not found" >&2
    exit 1
  fi
else
  CMD_NAME="${PYTHON_BIN%% *}"
  if ! command -v "$CMD_NAME" >/dev/null 2>&1; then
    echo "Configured PYTHON_BIN '$PYTHON_BIN' is not executable" >&2
    exit 1
  fi
fi

CLI="$PYTHON_BIN $ROOT_DIR/cli.py"

echo "[smoke-cli] Listing registry datasets"
$CLI datasets list > "$ARTIFACT_DIR/datasets_list.txt"

echo "[smoke-cli] Previewing Titanic dataset"
$CLI datasets preview --name titanic --rows 5 > "$ARTIFACT_DIR/preview_titanic.txt"

echo "[smoke-cli] Training logistic regression model"
TRAIN_OUTPUT="$ARTIFACT_DIR/train.json"
$CLI train --name titanic --algo logreg --task classification > "$TRAIN_OUTPUT"

MODEL_ID=$($PYTHON_BIN - <<PY
import json, pathlib
with open(r"$TRAIN_OUTPUT", "r", encoding="utf-8") as fh:
    data = json.load(fh)
print(data["model_id"])
PY
)

echo "[smoke-cli] Evaluating trained model"
$CLI evaluate --run-id "$MODEL_ID" > "$ARTIFACT_DIR/evaluate.json"

echo "[smoke-cli] Dumping active configuration"
$CLI info > "$ARTIFACT_DIR/info.json"

echo "[smoke-cli] Smoke checks complete"
