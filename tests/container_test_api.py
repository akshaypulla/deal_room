"""Test DealRoom v3 API via HTTP requests to Docker container."""

import requests

BASE_URL = "http://localhost:7860"


def test_full_episode():
    print("=" * 60)
    print("DealRoom v3 API Test Suite")
    print("=" * 60)

    # 1. Health check
    print("\n[1] Health Check")
    resp = requests.get(f"{BASE_URL}/health")
    assert resp.status_code == 200, f"Health failed: {resp.text}"
    health = resp.json()
    print(f"   Status: {health['status']}")
    print(f"   Tasks: {health['tasks']}")

    # 2. Metadata
    print("\n[2] Metadata")
    resp = requests.get(f"{BASE_URL}/metadata")
    assert resp.status_code == 200, f"Metadata failed: {resp.text}"
    meta = resp.json()
    print(f"   Name: {meta['name']}, Version: {meta['version']}")

    # 3. Reset and get initial observation
    print("\n[3] Reset Episode")
    resp = requests.post(f"{BASE_URL}/reset", json={"task_id": "aligned", "seed": 42})
    assert resp.status_code == 200, f"Reset failed: {resp.text}"
    obs = resp.json()

    session_id = obs.get("metadata", {}).get("session_id") or obs.get("session_id")
    print(f"   Session ID: {session_id}")
    print(f"   Round: {obs['round_number']}/{obs['max_rounds']}")
    print(f"   Stakeholders: {list(obs['stakeholders'].keys())}")
    print(f"   Deal Stage: {obs['deal_stage']}")
    print(f"   Deal Momentum: {obs['deal_momentum']}")

    # 4. Send a message to a stakeholder
    print("\n[4] Send Message Action")
    stakeholders = list(obs["stakeholders"].keys())
    action = {
        "metadata": {"session_id": session_id},
        "action_type": "direct_message",
        "message": "Hello, I'm here to discuss the enterprise partnership opportunity.",
        "target_ids": [stakeholders[0]],
    }
    resp = requests.post(f"{BASE_URL}/step", json=action)
    assert resp.status_code == 200, f"Step failed: {resp.text}"
    step_result = resp.json()
    obs2 = step_result["observation"]
    print(f"   Round: {obs2['round_number']}")
    print(f"   Stakeholder messages: {obs2['stakeholder_messages']}")
    print(f"   Reward: {step_result.get('reward')}")

    # 5. Multi-step conversation
    print("\n[5] Multi-step Conversation")
    messages = [
        "Let's discuss pricing structure for the annual contract.",
        "We can offer volume discounts starting at 500 seats.",
        "What are the key technical requirements from your team?",
        "Our implementation timeline typically runs 4-6 weeks.",
    ]
    for i, msg in enumerate(messages):
        action = {
            "metadata": {"session_id": session_id},
            "action_type": "direct_message",
            "message": msg,
            "target_ids": [stakeholders[i % len(stakeholders)]],
        }
        resp = requests.post(f"{BASE_URL}/step", json=action)
        if resp.status_code == 200:
            step_result = resp.json()
            obs = step_result["observation"]
            print(
                f"   Step {i + 1}: Round {obs['round_number']}, done={step_result.get('done', False)}"
            )
            if step_result.get("done"):
                print(f"   Episode ended! Final reward: {step_result.get('reward')}")
                break
        else:
            print(f"   Step {i + 1}: Failed - {resp.text[:100]}")

    # 6. Test different scenario
    print("\n[6] Test Conflicted Scenario")
    resp = requests.post(
        f"{BASE_URL}/reset", json={"task_id": "conflicted", "seed": 100}
    )
    assert resp.status_code == 200, f"Reset failed: {resp.text}"
    obs = resp.json()
    print(f"   Stakeholders: {list(obs['stakeholders'].keys())}")
    print(f"   Veto Precursors: {list(obs['veto_precursors'].keys())}")
    session_id2 = obs.get("metadata", {}).get("session_id") or obs.get("session_id")

    # Send a message to see committee response
    action = {
        "metadata": {"session_id": session_id2},
        "action_type": "direct_message",
        "message": "I'd like to discuss the acquisition terms.",
        "target_ids": [list(obs["stakeholders"].keys())[0]],
    }
    resp = requests.post(f"{BASE_URL}/step", json=action)
    if resp.status_code == 200:
        step_result = resp.json()
        obs = step_result["observation"]
        print(
            f"   Committee response messages: {len(obs.get('stakeholder_messages', {}))}"
        )

    # 7. Test hostile_acquisition scenario
    print("\n[7] Test Hostile Acquisition Scenario")
    resp = requests.post(
        f"{BASE_URL}/reset", json={"task_id": "hostile_acquisition", "seed": 999}
    )
    assert resp.status_code == 200, f"Reset failed: {resp.text}"
    obs = resp.json()
    print(f"   Deal Stage: {obs['deal_stage']}")
    print(f"   Days to Deadline: {obs.get('days_to_deadline', 'N/A')}")
    print(f"   Veto Precursors: {obs['veto_precursors']}")

    # 8. Test observation structure invariants
    print("\n[8] Verify Observation Invariants")
    resp = requests.post(f"{BASE_URL}/reset", json={"task_id": "aligned", "seed": 42})
    obs = resp.json()

    assert "G" not in obs, "FAIL: G (causal graph) should never be in observation"
    print("   ✓ G (causal graph) not in observation")

    assert "raw_distribution" not in str(obs), (
        "FAIL: raw distributions should not be exposed"
    )
    print("   ✓ No raw B_i distributions exposed")

    assert "round_number" in obs, "FAIL: round_number should be in observation"
    print("   ✓ round_number present")

    assert "done" in obs, "FAIL: done flag should be in observation"
    print("   ✓ done flag present")

    # 9. Test step response structure
    print("\n[9] Verify Step Response Structure")
    resp = requests.post(f"{BASE_URL}/reset", json={"task_id": "aligned", "seed": 42})
    obs = resp.json()
    session_id = obs.get("metadata", {}).get("session_id") or obs.get("session_id")

    action = {
        "metadata": {"session_id": session_id},
        "action_type": "direct_message",
        "message": "Test message for step response structure.",
        "target_ids": [list(obs["stakeholders"].keys())[0]],
    }
    resp = requests.post(f"{BASE_URL}/step", json=action)
    assert resp.status_code == 200, f"Step failed: {resp.text}"
    step_result = resp.json()

    assert "observation" in step_result, "FAIL: 'observation' key missing"
    print("   ✓ 'observation' key present")
    assert "reward" in step_result, "FAIL: 'reward' key missing"
    print("   ✓ 'reward' key present")
    assert "done" in step_result, "FAIL: 'done' key missing"
    print("   ✓ 'done' key present")
    assert "info" in step_result, "FAIL: 'info' key missing"
    print("   ✓ 'info' key present")

    print("\n" + "=" * 60)
    print("✅ ALL API TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    test_full_episode()
