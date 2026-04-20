import requests

BASE = "http://127.0.0.1:7860"


def test_5_1_reset_produces_different_engagement_levels():
    """Two resets with different seeds produce different engagement levels."""
    r1 = requests.post(f"{BASE}/reset", json={"task": "conflicted", "seed": 42})
    r2 = requests.post(f"{BASE}/reset", json={"task": "conflicted", "seed": 43})

    obs1, obs2 = r1.json(), r2.json()
    eng1 = obs1.get("engagement_level", {})
    eng2 = obs2.get("engagement_level", {})

    all_same = all(abs(eng1.get(sid, 0) - eng2.get(sid, 0)) < 0.001 for sid in eng1)
    assert not all_same, (
        "Two resets with different seeds produced identical engagement levels. "
        "G and B_i(0) must be resampled on every reset."
    )
    print("✓ 5.1: Two resets with different seeds produce different initial states")


def test_5_2_round_number_resets_to_zero():
    session = requests.Session()
    r = session.post(f"{BASE}/reset", json={"task": "aligned", "seed": 10})
    session_id = r.json().get("metadata", {}).get("session_id")
    for _ in range(3):
        session.post(
            f"{BASE}/step",
            json={
                "metadata": {"session_id": session_id},
                "action_type": "direct_message",
                "target_ids": ["Finance"],
                "message": "Step.",
                "documents": [],
                "lookahead": None,
            },
        )
    obs_after_steps = session.post(
        f"{BASE}/reset", json={"task": "aligned", "seed": 11}
    ).json()
    assert obs_after_steps.get("round_number", 999) == 0, (
        f"round_number after reset = {obs_after_steps.get('round_number')}. Must be 0."
    )
    print("✓ 5.2: round_number resets to 0")


def test_5_3_done_false_after_reset():
    r = requests.post(f"{BASE}/reset", json={"task": "hostile_acquisition", "seed": 20})
    obs = r.json()
    assert obs.get("done") is False, (
        f"done={obs.get('done')} after reset. Must be False."
    )
    print("✓ 5.3: done=False immediately after reset")


def test_5_4_engagement_history_initialized():
    r = requests.post(f"{BASE}/reset", json={"task": "aligned", "seed": 30})
    obs = r.json()
    history = obs.get("engagement_history", [])
    assert len(history) >= 5, f"engagement_history has only {len(history)} entries"
    print("✓ 5.4: engagement_history initialized")


def test_5_5_step_counter_increments():
    session = requests.Session()
    r = session.post(f"{BASE}/reset", json={"task": "aligned", "seed": 40})
    session_id = r.json().get("metadata", {}).get("session_id")
    assert r.json().get("round_number") == 0

    for expected_round in range(1, 5):
        r = session.post(
            f"{BASE}/step",
            json={
                "metadata": {"session_id": session_id},
                "action_type": "direct_message",
                "target_ids": ["Finance"],
                "message": f"Round {expected_round}.",
                "documents": [],
                "lookahead": None,
            },
        )
        obs = r.json().get("observation", r.json())
        actual = obs.get("round_number")
        assert actual == expected_round, (
            f"round_number={actual} after {expected_round} steps. Expected {expected_round}."
        )
    print("✓ 5.5: round_number increments correctly")


def test_5_6_three_scenario_types_all_work():
    for i, scenario in enumerate(["aligned", "conflicted", "hostile_acquisition"]):
        session = requests.Session()
        r = session.post(f"{BASE}/reset", json={"task": scenario, "seed": 50 + i})
        obs = r.json()
        assert obs.get("done") is False, (
            f"{scenario}: done=True immediately after reset"
        )

        stakeholders = list(obs.get("stakeholders", {}).keys())
        r = session.post(
            f"{BASE}/step",
            json={
                "metadata": {"session_id": obs.get("metadata", {}).get("session_id")},
                "action_type": "direct_message",
                "target_ids": [stakeholders[0]] if stakeholders else ["Finance"],
                "message": "Opening communication.",
                "documents": [],
                "lookahead": None,
            },
        )
        assert r.status_code == 200, f"{scenario}: step failed with {r.status_code}"
        print(f"  ✓ Scenario '{scenario}' works")
    print("✓ 5.6: All three scenario types functional")


if __name__ == "__main__":
    for fn in [
        test_5_1_reset_produces_different_engagement_levels,
        test_5_2_round_number_resets_to_zero,
        test_5_3_done_false_after_reset,
        test_5_4_engagement_history_initialized,
        test_5_5_step_counter_increments,
        test_5_6_three_scenario_types_all_work,
    ]:
        fn()
    print("\n✓ SECTION 5 PASSED — Episode isolation and reset are correct")
