#!/usr/bin/env python3
"""
test_00_environment_setup.py
DealRoom v3 — Environment Setup & API Key Validation

Validates before any test runs:
- Required API keys are set (MINIMAX_API_KEY + OPENAI_API_KEY)
- Docker container is running and responsive
- All server endpoints respond correctly
"""

import os
import sys
import json
import subprocess
from pathlib import Path

_dotenv = Path(__file__).parent / ".env"
if _dotenv.exists():
    try:
        from dotenv import load_dotenv

        load_dotenv(_dotenv)
    except ImportError:
        pass

BASE_URL = os.getenv("DEALROOM_BASE_URL", "http://127.0.0.1:7860")
CONTAINER_NAME = os.getenv("DEALROOM_CONTAINER_NAME", "dealroom-v3-test")
REQUIRED_KEYS = ["MINIMAX_API_KEY", "OPENAI_API_KEY"]


def test_0_1_python_deps():
    print("\n[0.1] Python dependencies...")
    for dep in ["requests", "numpy"]:
        try:
            __import__(dep)
            print(f"  ✓ {dep}")
        except ImportError:
            print(f"  ✗ {dep} — NOT INSTALLED")
            sys.exit(1)
    try:
        from dotenv import load_dotenv

        print(f"  ✓ python-dotenv")
    except ImportError:
        print(f"  ⚠ python-dotenv (optional — exported vars still work)")
    print("  [OK] All critical dependencies present")


def test_0_2_api_keys_configured():
    print("\n[0.2] API keys...")
    all_ok = True
    for key in REQUIRED_KEYS:
        val = os.getenv(key)
        if val and val != f"your_{key.lower()}_key_here":
            display = f"{val[:4]}...{val[-4:]}" if len(val) > 8 else "(set)"
            print(f"  ✓ {key}: {display}")
        else:
            print(f"  ✗ {key}: NOT SET")
            all_ok = False
    if not all_ok:
        print("\n  Fix: cp .env.example .env  →  edit .env with your real keys")
        print("  Or:  export MINIMAX_API_KEY=...  &&  export OPENAI_API_KEY=...")
        sys.exit(1)
    print("  [OK] Both API keys configured")


def test_0_3_docker_container_running():
    print("\n[0.3] Docker container...")
    result = subprocess.run(
        ["docker", "ps", "--filter", f"name={CONTAINER_NAME}", "-q"],
        capture_output=True,
        text=True,
    )
    if not result.stdout.strip():
        print(f"  ✗ Container '{CONTAINER_NAME}' NOT running")
        print(f"  Start: docker run --rm -d -p 7860:7860 \\")
        print(f"      -e MINIMAX_API_KEY -e OPENAI_API_KEY \\")
        print(f"      --name {CONTAINER_NAME} dealroom-v3-test:latest")
        sys.exit(1)
    print(f"  ✓ Container '{CONTAINER_NAME}' running")


def test_0_4_server_endpoints_responsive():
    print("\n[0.4] Server endpoints...")
    import requests

    session = requests.Session()
    session_id = None

    # /health
    r = session.get(f"{BASE_URL}/health", timeout=10)
    assert r.status_code == 200, f"/health failed: {r.status_code}"
    print("  ✓ GET /health → 200")

    # /metadata
    r = session.get(f"{BASE_URL}/metadata", timeout=10)
    assert r.status_code == 200, f"/metadata failed: {r.status_code}"
    print("  ✓ GET /metadata → 200")

    # /reset
    r = session.post(f"{BASE_URL}/reset", json={"task": "aligned"}, timeout=30)
    assert r.status_code == 200, f"/reset failed: {r.status_code}"
    obs = r.json()
    session_id = obs.get("metadata", {}).get("session_id") or obs.get("session_id")
    assert session_id, "No session_id in reset response"
    print("  ✓ POST /reset → 200")

    # /step
    r = session.post(
        f"{BASE_URL}/step",
        json={
            "metadata": {"session_id": session_id},
            "action_type": "direct_message",
            "target_ids": ["Finance"],
            "message": "Health check.",
            "documents": [],
            "lookahead": None,
        },
        timeout=60,
    )
    assert r.status_code == 200, f"/step failed: {r.status_code}"
    print("  ✓ POST /step → 200")

    print("  [OK] All endpoints responsive")


def test_0_5_llm_client_imports():
    print("\n[0.5] llm_client module...")
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from deal_room.environment.llm_client import (
            validate_api_keys,
            MAX_TOKENS,
        )

        print(f"  ✓ llm_client imported (MAX_TOKENS={MAX_TOKENS})")
    except ImportError as e:
        print(f"  ⚠ llm_client not in local deal_room/ — fine if running in container")


def run_all():
    print("=" * 60)
    print("  DealRoom v3 — Environment Setup Validation")
    print("=" * 60)
    test_0_1_python_deps()
    test_0_2_api_keys_configured()
    test_0_3_docker_container_running()
    test_0_4_server_endpoints_responsive()
    test_0_5_llm_client_imports()
    print("\n" + "=" * 60)
    print("  ✓ SECTION 0 PASSED — Environment ready")
    print("=" * 60)


if __name__ == "__main__":
    run_all()
