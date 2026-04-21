#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

IMAGE_NAME="${DEALROOM_IMAGE_NAME:-dealroom-v3-test:latest}"
CONTAINER_NAME="${DEALROOM_CONTAINER_NAME:-dealroom-v3-test}"
BASE_URL="${DEALROOM_BASE_URL:-http://127.0.0.1:7860}"

echo "=== DealRoom v3 Reproduction ==="
echo "This runs repo-local validation and may take several minutes."

python -m pip install -r requirements.txt -q

docker build -t "$IMAGE_NAME" .
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
docker run --rm -d \
  -p 7860:7860 \
  ${MINIMAX_API_KEY:+-e MINIMAX_API_KEY="$MINIMAX_API_KEY"} \
  ${OPENAI_API_KEY:+-e OPENAI_API_KEY="$OPENAI_API_KEY"} \
  --name "$CONTAINER_NAME" \
  "$IMAGE_NAME" >/dev/null

echo "Waiting for container health..."
python - <<'PY'
import os
import time
import requests

base = os.environ.get("DEALROOM_BASE_URL", "http://127.0.0.1:7860")
for _ in range(60):
    try:
        response = requests.get(f"{base}/health", timeout=2)
        if response.status_code == 200:
            print("Container is healthy.")
            break
    except Exception:
        pass
    time.sleep(1)
else:
    raise SystemExit("Container did not become healthy in 60 seconds")
PY

export DEALROOM_BASE_URL="$BASE_URL"
pytest -q tests/v3/test_04_cvar_veto.py::test_4_3_veto_deterministic
pytest -q tests/v3/test_10_training_integration.py::test_training_actually_improves
pytest -q tests/v3/test_assertion_hygiene.py
python tests/container_test_api.py
bash tests/v3/run_all_tests.sh
python -m deal_room.training.run_benchmark --episodes-per-task 2 --max-steps 6

echo "=== Reproduction successful ==="
