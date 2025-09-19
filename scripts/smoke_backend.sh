#!/usr/bin/env bash
set -euo pipefail

BASE_URL=${1:-http://127.0.0.1:8000}

printf "Checking backend health at %s...\n" "$BASE_URL"
curl -sf "${BASE_URL}/health" | python -m json.tool

printf "Listing available algorithms...\n"
curl -sf "${BASE_URL}/model/algorithms" | python -m json.tool

printf "Running CLI training smoke test...\n"
python cli.py train --name titanic --algo logreg --task classification
