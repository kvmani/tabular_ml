#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACT_DIR="${ARTIFACT_DIR:-$ROOT_DIR/artifacts/api}"
mkdir -p "$ARTIFACT_DIR"

BASE_URL="${1:-http://127.0.0.1:8000}"
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

COOKIE_FILE="$(mktemp)"
HEADER_FILE="$(mktemp)"
trap 'rm -f "$COOKIE_FILE" "$HEADER_FILE"' EXIT

update_token() {
  local token_line
  token_line=$(grep -i 'X-CSRF-Token' "$HEADER_FILE" | tail -n1 || true)
  if [ -n "$token_line" ]; then
    CSRF_TOKEN=$(echo "$token_line" | awk '{print $2}' | tr -d '\r')
  fi
}

request_get() {
  local path=$1
  local dest=$2
  curl -sSf -D "$HEADER_FILE" -b "$COOKIE_FILE" -c "$COOKIE_FILE" "$BASE_URL$path" > "$dest"
  update_token
}

request_post() {
  local path=$1
  local body=$2
  local dest=$3
  curl -sSf -D "$HEADER_FILE" -b "$COOKIE_FILE" -c "$COOKIE_FILE" \
    -H "Content-Type: application/json" \
    -H "X-CSRF-Token: $CSRF_TOKEN" \
    -X POST "$BASE_URL$path" \
    -d "$body" > "$dest"
  update_token
}

echo "[smoke] Checking backend health at $BASE_URL"
request_get "/health" "$ARTIFACT_DIR/health.json"

if [ -z "${CSRF_TOKEN:-}" ]; then
  echo "[smoke] CSRF token not found in response headers" >&2
  exit 1
fi

echo "[smoke] Listing available algorithms"
request_get "/model/algorithms" "$ARTIFACT_DIR/algorithms.json"

echo "[smoke] Loading Titanic sample dataset"
request_post "/data/samples/titanic" '{}' "$ARTIFACT_DIR/load_sample.json"

DATASET_ID=$($PYTHON_BIN - <<PY
import json, pathlib
with open(r"$ARTIFACT_DIR/load_sample.json", "r", encoding="utf-8") as fh:
    data = json.load(fh)
print(data["dataset"]["dataset_id"])
PY
)

echo "[smoke] Requesting dataset split"
SPLIT_PAYLOAD="{\"target_column\": \"Survived\", \"task_type\": \"classification\", \"test_size\": 0.2, \"val_size\": 0.2, \"random_state\": 42, \"stratify\": true}"
request_post "/preprocess/$DATASET_ID/split" "$SPLIT_PAYLOAD" "$ARTIFACT_DIR/split.json"

SPLIT_ID=$($PYTHON_BIN - <<PY
import json, pathlib
with open(r"$ARTIFACT_DIR/split.json", "r", encoding="utf-8") as fh:
    data = json.load(fh)
print(data["split_id"])
PY
)

echo "[smoke] Training logistic regression model"
TRAIN_PAYLOAD="{\"dataset_id\": \"$DATASET_ID\", \"target_column\": \"Survived\", \"task_type\": \"classification\", \"algorithm\": \"logistic_regression\", \"split_id\": \"$SPLIT_ID\"}"
request_post "/model/train" "$TRAIN_PAYLOAD" "$ARTIFACT_DIR/train.json"

MODEL_ID=$($PYTHON_BIN - <<PY
import json, pathlib
with open(r"$ARTIFACT_DIR/train.json", "r", encoding="utf-8") as fh:
    data = json.load(fh)
print(data["model_id"])
PY
)

echo "[smoke] Evaluating trained model"
EVAL_PAYLOAD="{\"model_id\": \"$MODEL_ID\"}"
request_post "/model/evaluate" "$EVAL_PAYLOAD" "$ARTIFACT_DIR/evaluate.json"

echo "[smoke] Capturing last run metadata"
request_get "/runs/last" "$ARTIFACT_DIR/last_run.json"

echo "[smoke] API smoke test completed successfully"
