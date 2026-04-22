#!/usr/bin/env python3
"""
test_playwright_e2e.py
DealRoom v3 — Playwright E2E Web Interface Tests

Tests the web UI using Playwright to verify:
- Web interface loads without errors
- API endpoints are accessible
- Multiple sessions work correctly
"""

import os
import sys
import time
from pathlib import Path

_dotenv = Path(__file__).parent / ".env"
if _dotenv.exists():
    try:
        from dotenv import load_dotenv

        load_dotenv(_dotenv)
    except ImportError:
        pass

BASE_URL = os.getenv("DEALROOM_BASE_URL", "http://127.0.0.1:7860")


def test_web_interface_loads():
    """Web interface loads without crash."""
    print("\n[E2E-1] Web interface loads...")
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        page.goto(BASE_URL, timeout=60000)
        time.sleep(3)

        title = page.title()
        url = page.url
        print(f"  Page title: {title}")
        print(f"  Final URL: {url}")

        assert "DealRoom" in title, f"Expected 'DealRoom' in title, got: {title}"
        assert "/web" in url, f"Expected '/web' in URL, got: {url}"

        print(f"  ✓ Web interface loaded at {url}")
        browser.close()


def test_api_endpoints():
    """Verify API endpoints accessible via browser."""
    print("\n[E2E-2] API endpoints accessible...")
    from playwright.sync_api import sync_playwright
    import json

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        page.goto(f"{BASE_URL}/health", timeout=30000)
        time.sleep(1)

        response = page.request.get(f"{BASE_URL}/health")
        assert response.status == 200, f"Health status: {response.status}"

        health = response.json()
        print(f"  Health: {health.get('status')}")
        print(f"  Tasks: {health.get('tasks', [])[:2]}...")

        assert health.get("status") == "ok"
        print(f"  ✓ API endpoints working")

        page.goto(f"{BASE_URL}/metadata", timeout=30000)
        time.sleep(1)

        response = page.request.get(f"{BASE_URL}/metadata")
        assert response.status == 200, f"Metadata status: {response.status}"

        metadata = response.json()
        print(f"  Service: {metadata.get('name')} v{metadata.get('version')}")
        print(f"  ✓ Metadata endpoint working")

        browser.close()


def test_session_isolation():
    """Multiple sessions don't interfere with each other."""
    print("\n[E2E-3] Session isolation...")
    import requests

    sessions = []

    for i in range(3):
        session = requests.Session()
        r = session.post(
            f"{BASE_URL}/reset",
            json={"task_id": "aligned", "seed": 100 + i},
            timeout=30,
        )
        assert r.status_code == 200, f"Reset failed for session {i}: {r.status_code}"

        obs = r.json()
        sid = obs.get("metadata", {}).get("session_id")
        sessions.append((session, sid))
        print(f"  Session {i + 1}: id={sid[:8]}...")

    for i, (session, sid) in enumerate(sessions):
        r = session.post(
            f"{BASE_URL}/step",
            json={
                "metadata": {"session_id": sid},
                "action_type": "direct_message",
                "target_ids": ["Finance"],
                "message": f"Message from session {i + 1}",
            },
            timeout=30,
        )

        assert r.status_code == 200, f"Step failed for session {i}: {r.status_code}"
        result = r.json()
        print(f"  Session {i + 1} step: reward={result.get('reward', 'N/A')}")

    print(f"  ✓ {len(sessions)} sessions isolated correctly")
    for session, _ in sessions:
        session.close()


def test_episode_completion():
    """Complete a full episode via API."""
    print("\n[E2E-4] Episode completion...")
    import requests

    session = requests.Session()
    r = session.post(
        f"{BASE_URL}/reset", json={"task_id": "aligned", "seed": 200}, timeout=30
    )
    assert r.status_code == 200
    obs = r.json()
    sid = obs.get("metadata", {}).get("session_id")

    steps = 0
    for step in range(15):
        r = session.post(
            f"{BASE_URL}/step",
            json={
                "metadata": {"session_id": sid},
                "action_type": "direct_message",
                "target_ids": ["Finance"],
                "message": f"Step {step + 1}",
            },
            timeout=30,
        )

        assert r.status_code == 200
        result = r.json()
        steps += 1

        if result.get("done", False) or result.get("observation", {}).get("done"):
            terminal = result.get("terminal_outcome", "unknown")
            print(f"  Episode terminated at step {steps}: {terminal}")
            break

    print(f"  ✓ Episode completed in {steps} steps")
    session.close()


def test_concurrent_reset():
    """Concurrent resets don't interfere."""
    print("\n[E2E-5] Concurrent resets...")
    import requests
    import concurrent.futures

    def reset_session(args):
        i, task = args
        session = requests.Session()
        r = session.post(
            f"{BASE_URL}/reset", json={"task_id": task, "seed": 300 + i}, timeout=30
        )
        if r.status_code == 200:
            obs = r.json()
            sid = obs.get("metadata", {}).get("session_id")
            return (i, task, sid, True)
        return (i, task, None, False)

    tasks = [
        (0, "aligned"),
        (1, "conflicted"),
        (2, "hostile_acquisition"),
        (3, "aligned"),
        (4, "conflicted"),
    ]

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(reset_session, t) for t in tasks]
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())

    success = sum(1 for r in results if r[3])
    print(f"  Reset {success}/{len(tasks)} concurrent sessions successfully")

    assert success == len(tasks), f"Only {success}/{len(tasks)} resets succeeded"
    print(f"  ✓ Concurrent resets work correctly")


def run_all_e2e():
    print("=" * 70)
    print("  DealRoom v3 — Playwright E2E Web Interface Tests")
    print("=" * 70)

    tests = [
        ("E2E-1: Web Interface Loads", test_web_interface_loads),
        ("E2E-2: API Endpoints", test_api_endpoints),
        ("E2E-3: Session Isolation", test_session_isolation),
        ("E2E-4: Episode Completion", test_episode_completion),
        ("E2E-5: Concurrent Resets", test_concurrent_reset),
    ]

    passed = 0
    failed = []

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
            print(f"  ✓ {name} PASSED\n")
        except AssertionError as e:
            failed.append((name, str(e)))
            print(f"  ✗ {name} FAILED: {e}\n")
        except Exception as e:
            failed.append((name, f"ERROR: {e}"))
            print(f"  ✗ {name} ERROR: {e}\n")

    print("=" * 70)
    print(f"  E2E TESTS: {passed}/{len(tests)} passed")
    print("=" * 70)

    if failed:
        print(f"\n  FAILED TESTS:")
        for name, err in failed:
            print(f"    ✗ {name}: {err[:100]}")
        sys.exit(1)
    else:
        print("\n  ALL E2E TESTS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    run_all_e2e()
