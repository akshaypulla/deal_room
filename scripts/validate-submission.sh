#!/usr/bin/env bash
#
# validate-submission.sh — local OpenEnv submission validator for DealRoom
#
# Usage:
#   bash scripts/validate-submission.sh [base_url]
#

set -euo pipefail

VALIDATE_PORT="${VALIDATE_PORT:-7861}"
BASE_URL="${1:-http://127.0.0.1:${VALIDATE_PORT}}"
IMAGE_NAME="${IMAGE_NAME:-deal-room-env:validate}"
CONTAINER_NAME="${CONTAINER_NAME:-deal-room-validate}"

echo "== OpenEnv validate =="
openenv validate

echo
echo "== Pytest =="
pytest -q

echo
echo "== Docker build =="
docker build -t "$IMAGE_NAME" -f Dockerfile .

echo
echo "== Docker run =="
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
docker run -d --rm --name "$CONTAINER_NAME" -p "${VALIDATE_PORT}:7860" "$IMAGE_NAME" >/dev/null

cleanup() {
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo
echo "== Waiting for server =="
for _ in $(seq 1 30); do
  if curl -fsS "${BASE_URL}/health" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
curl -fsS "${BASE_URL}/health" >/dev/null

echo
echo "== Route smoke =="
python scripts/container_route_smoke.py "$BASE_URL"

echo
echo "== Inference =="
docker run --rm "$IMAGE_NAME" python inference.py

echo
echo "Submission validation completed successfully."
