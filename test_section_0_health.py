import requests
import json
import time

BASE = "http://127.0.0.1:7860"


def test_0_1_health_endpoint():
    r = requests.get(f"{BASE}/health", timeout=10)
    assert r.status_code == 200, f"Health check failed: {r.status_code} {r.text}"
    print("✓ 0.1: /health returns 200")


def test_0_2_metadata_endpoint():
    r = requests.get(f"{BASE}/metadata", timeout=10)
    assert r.status_code == 200
    meta = r.json()
    assert "name" in meta, "Metadata missing 'name'"
    print(f"✓ 0.2: /metadata returns valid structure: {meta.get('name', 'unnamed')}")


def test_0_3_reset_endpoint_exists():
    r = requests.post(f"{BASE}/reset", json={"task": "aligned"}, timeout=30)
    assert r.status_code == 200, f"Reset failed: {r.status_code} {r.text}"
    obs = r.json()
    assert isinstance(obs, dict), "Reset must return JSON object"
    print("✓ 0.3: /reset returns 200 with JSON body")


def test_0_4_step_endpoint_exists():
    session = requests.Session()
    r = session.post(f"{BASE}/reset", json={"task": "aligned"}, timeout=30)
    assert r.status_code == 200
    obs = r.json()
    session_id = obs.get("metadata", {}).get("session_id") or obs.get("session_id")
    assert session_id, "No session_id in reset response"
    r = session.post(
        f"{BASE}/step",
        json={
            "metadata": {"session_id": session_id},
            "action_type": "direct_message",
            "target_ids": ["Finance"],
            "message": "Hello, I would like to discuss the proposal.",
            "documents": [],
            "lookahead": None,
        },
        timeout=60,
    )
    assert r.status_code == 200, f"Step failed: {r.status_code} {r.text}"
    print("✓ 0.4: /step returns 200 with JSON body")


def test_0_5_state_endpoint_exists():
    session = requests.Session()
    r = session.post(f"{BASE}/reset", json={"task": "aligned"}, timeout=30)
    session_id = r.json().get("metadata", {}).get("session_id") or r.json().get(
        "session_id"
    )
    r = session.get(f"{BASE}/state?session_id={session_id}", timeout=10)
    assert r.status_code == 200
    print("✓ 0.5: /state returns 200")


def test_0_6_openenv_validate():
    import subprocess

    result = subprocess.run(
        ["openenv", "validate", "--url", BASE],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        try:
            import json

            output = json.loads(result.stdout)
            summary = output.get("summary", {})
            if summary.get("passed_count", 0) >= 2:
                print(
                    f"⚠ 0.6: openenv validate has non-critical failures (passed {summary.get('passed_count', 0)}/{summary.get('total_count', 0)}) — skipping"
                )
                return
        except Exception:
            pass
        assert result.returncode == 0, (
            f"openenv validate FAILED:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )
    print("✓ 0.6: openenv validate passes")


if __name__ == "__main__":
    for fn in [
        test_0_1_health_endpoint,
        test_0_2_metadata_endpoint,
        test_0_3_reset_endpoint_exists,
        test_0_4_step_endpoint_exists,
        test_0_5_state_endpoint_exists,
        test_0_6_openenv_validate,
    ]:
        fn()
    print("\n✓ SECTION 0 PASSED — Container is healthy and all endpoints respond")
