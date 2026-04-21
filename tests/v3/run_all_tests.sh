#!/bin/bash
# =============================================================================
# DealRoom v3 — Complete Test Suite Runner
# =============================================================================
# Usage: ./run_all_tests.sh
#
# Prerequisites:
#   1. cp .env.example .env  →  fill in your API keys
#   2. Docker container running: docker run --rm -d -p 7860:7860 \
#        -e MINIMAX_API_KEY -e OPENAI_API_KEY \
#        --name dealroom-v3-test dealroom-v3-test:latest
#
# Environment variables (from .env):
#   DEALROOM_BASE_URL, DEALROOM_CONTAINER_NAME
#   MINIMAX_API_KEY, OPENAI_API_KEY
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH="$(cd "$SCRIPT_DIR/../.." && pwd):${PYTHONPATH}"

# ── Load .env if present ────────────────────────────────────────────────────────
if [ -f "$SCRIPT_DIR/.env" ]; then
    echo "Loading environment from .env..."
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
fi

# ── Container check ────────────────────────────────────────────────────────────
CONTAINER_NAME="${DEALROOM_CONTAINER_NAME:-dealroom-v3-test}"

if ! docker ps --filter "name=$CONTAINER_NAME" -q | grep -q .; then
    echo "Container '$CONTAINER_NAME' not running. Starting..."
    docker run --rm -d \
        -p 7860:7860 \
        ${MINIMAX_API_KEY:+-e MINIMAX_API_KEY="$MINIMAX_API_KEY"} \
        ${OPENAI_API_KEY:+-e OPENAI_API_KEY="$OPENAI_API_KEY"} \
        --name "$CONTAINER_NAME" \
        dealroom-v3-test:latest
    echo "Waiting 15s for container startup..."
    sleep 15
else
    echo "Container '$CONTAINER_NAME' is running."
fi

BASE_URL="${DEALROOM_BASE_URL:-http://127.0.0.1:7860}"

# ── Helper ─────────────────────────────────────────────────────────────────────
run_test() {
    local section="$1"
    local name="$2"
    local cmd="$3"
    local rc=0
    echo ""
    echo "━━━ $section $name ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    if [ -n "$cmd" ]; then
        eval "$cmd"
        rc=$?
    else
        python3 "${SCRIPT_DIR}/test_${section}_${name}.py"
        rc=$?
    fi
    if [ $rc -ne 0 ]; then
        return $rc
    fi
    echo "✓ $section $name PASSED"
}

pass_count=0
fail_count=0

run_section() {
    local num="$1"
    local name="$2"
    local cmd="$3"

    if run_test "$num" "$name" "$cmd" 2>&1; then
        ((pass_count++))
    else
        echo "✗ $num $name FAILED"
        ((fail_count++))
        # Continue running other tests but exit with error at end
    fi
    echo ""
}

# ── SECTION 0: Environment Setup ───────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  DealRoom v3 — Comprehensive Test Suite                 ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
if [ -n "$MINIMAX_API_KEY" ]; then
    MM_KEY="***${MINIMAX_API_KEY: -4}"
else
    MM_KEY="(not set)"
fi
if [ -n "$OPENAI_API_KEY" ]; then
    OA_KEY="***${OPENAI_API_KEY: -4}"
else
    OA_KEY="(not set)"
fi
echo "API Keys:  MINIMAX_API_KEY=${MM_KEY}  OPENAI_API_KEY=${OA_KEY}"
echo "Container: $CONTAINER_NAME"
echo "Base URL:  $BASE_URL"
echo ""

# ── SECTION 0: Environment + API Key Validation ────────────────────────────────
run_section "00" "environment_setup" \
    "python3 test_00_environment_setup.py"

# ── SECTION 1: Schema Validation ────────────────────────────────────────────────
run_section "01" "schema_validation" \
    "python3 test_01_schema_validation.py"

# ── SECTION 2: Reward Integrity & Unhackability ──────────────────────────────────
run_section "02" "reward_integrity" \
    "python3 test_02_reward_integrity.py"

# ── SECTION 3: Causal Inference Signal ─────────────────────────────────────────
run_section "03" "causal_inference" \
    "python3 test_03_causal_inference.py"

# ── SECTION 4: CVaR Veto Mechanism ────────────────────────────────────────────
run_section "04" "cvar_veto" \
    "python3 test_04_cvar_veto.py"

# ── SECTION 5: Episode Isolation ────────────────────────────────────────────────
run_section "05" "episode_isolation" \
    "python3 test_05_episode_isolation.py"

# ── SECTION 6: Probabilistic Signals (container) ───────────────────────────────
run_section "06" "probabilistic_signals" \
    "docker cp test_06_probabilistic_signals.py $CONTAINER_NAME:/app/ && docker exec $CONTAINER_NAME python3 /app/test_06_probabilistic_signals.py"

# ── SECTION 7: Causal Graph Unit Tests (container) ─────────────────────────────
run_section "07" "causal_graph" \
    "docker cp test_07_causal_graph.py $CONTAINER_NAME:/app/ && docker exec $CONTAINER_NAME python3 /app/test_07_causal_graph.py"

# ── SECTION 8: CVaR Preferences Unit Tests (container) ─────────────────────────
run_section "08" "cvar_preferences" \
    "docker cp test_08_cvar_preferences.py $CONTAINER_NAME:/app/ && docker exec $CONTAINER_NAME python3 /app/test_08_cvar_preferences.py"

# ── SECTION 9: Full Episode End-to-End ───────────────────────────────────────
run_section "09" "full_episode_e2e" \
    "python3 test_09_full_episode_e2e.py"

# ── SECTION 10: Training Infrastructure (container) ────────────────────────────
run_section "10" "training_infrastructure" \
    "python3 test_assertion_hygiene.py && docker cp test_10_training_infrastructure.py $CONTAINER_NAME:/app/ && docker cp test_10_training_integration.py $CONTAINER_NAME:/app/ && docker exec $CONTAINER_NAME python3 /app/test_10_training_infrastructure.py && docker exec $CONTAINER_NAME python3 /app/test_10_training_integration.py"

# ── SECTION 11: All 12 Research Properties (container) ─────────────────────────
run_section "11" "research_properties" \
    "docker cp test_11_research_properties.py $CONTAINER_NAME:/app/ && docker exec $CONTAINER_NAME python3 /app/test_11_research_properties.py"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  TEST SUMMARY                                              ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "  Passed:  $pass_count"
echo "  Failed:  $fail_count"
echo "╚══════════════════════════════════════════════════════════╝"

if [ $fail_count -gt 0 ]; then
    echo ""
    echo "Some tests failed. Fix issues before using the environment for training."
    exit 1
else
    echo ""
    echo "ALL TESTS PASSED — DealRoom v3 is implementation-correct and ready for training."
    exit 0
fi
