#!/usr/bin/env python3
"""
test_05_episode_isolation.py
DealRoom v3 — Episode Isolation & Determinism Tests

Validates:
- Two resets with different seeds produce different engagement levels
- round_number resets to 0 on reset
- done=False immediately after reset
- engagement_history is initialized correctly
- round_number increments correctly with each step
- All three scenario types work without crash
- Session state does not leak between episodes
"""

import os
import sys
from pathlib import Path

_dotenv = Path(__file__).parent / ".env"
if _dotenv.exists():
    try:
        from dotenv import load_dotenv

        load_dotenv(_dotenv)
    except ImportError:
        pass

import requests

BASE_URL = os.getenv("DEALROOM_BASE_URL", "http://127.0.0.1:7860")


def make_action(
    session_id, action_type, target_ids, message="", documents=None, lookahead=None
):
    return {
        "metadata": {"session_id": session_id},
        "action_type": action_type,
        "target_ids": target_ids,
        "message": message,
        "documents": documents or [],
        "lookahead": lookahead,
    }


def test_5_1_different_seeds_different_initial_state():
    print("\n[5.1] Two resets with different seeds → different initial states...")
    r1 = requests.post(
        f"{BASE_URL}/reset", json={"task_id": "conflicted", "seed": 42}, timeout=30
    )
    r2 = requests.post(
        f"{BASE_URL}/reset", json={"task_id": "conflicted", "seed": 43}, timeout=30
    )

    obs1, obs2 = r1.json(), r2.json()
    eng1 = obs1.get("engagement_level", {})
    eng2 = obs2.get("engagement_level", {})

    diff = sum(abs(eng1.get(sid, 0) - eng2.get(sid, 0)) for sid in eng1)
    assert diff > 0.001, (
        f"Two resets with different seeds produced identical states (diff={diff:.6f}). G not resampled."
    )

    print(f"  ✓ Different seeds → different initial states (total diff={diff:.4f})")


def test_5_2_round_number_resets_to_zero():
    print("\n[5.2] round_number resets to 0 on reset...")
    session = requests.Session()
    r = session.post(
        f"{BASE_URL}/reset", json={"task_id": "aligned", "seed": 10}, timeout=30
    )
    session_id = r.json().get("metadata", {}).get("session_id")

    for _ in range(3):
        session.post(
            f"{BASE_URL}/step",
            json=make_action(
                session_id,
                "direct_message",
                ["Finance"],
                "Step.",
                [],
                None,
            ),
            timeout=60,
        )

    # Reset
    obs_after = session.post(
        f"{BASE_URL}/reset", json={"task_id": "aligned", "seed": 11}, timeout=30
    ).json()
    assert obs_after.get("round_number", 999) == 0, (
        f"round_number after reset = {obs_after.get('round_number')} (expected 0)"
    )

    print("  ✓ round_number = 0 after reset")


def test_5_3_done_false_after_reset():
    print("\n[5.3] done=False immediately after reset...")
    for scenario in ["aligned", "conflicted", "hostile_acquisition"]:
        r = requests.post(
            f"{BASE_URL}/reset", json={"task_id": scenario, "seed": 20}, timeout=30
        )
        obs = r.json()
        done = obs.get("done")
        assert done is False, f"{scenario}: done={done} after reset (expected False)"
        print(f"  ✓ {scenario}: done=False")


def test_5_4_engagement_history_initialized():
    print("\n[5.4] engagement_history initialized correctly...")
    r = requests.post(
        f"{BASE_URL}/reset", json={"task_id": "aligned", "seed": 30}, timeout=30
    )
    obs = r.json()
    history = obs.get("engagement_history", [])

    assert len(history) >= 5, f"History has {len(history)} entries (need ≥5)"
    print(f"  ✓ History initialized with {len(history)} entries")


def test_5_5_round_number_increments_correctly():
    print("\n[5.5] round_number increments correctly...")
    session = requests.Session()
    r = session.post(
        f"{BASE_URL}/reset", json={"task_id": "aligned", "seed": 40}, timeout=30
    )
    session_id = r.json().get("metadata", {}).get("session_id")

    assert r.json().get("round_number", 999) == 0

    for expected_round in range(1, 6):
        r = session.post(
            f"{BASE_URL}/step",
            json=make_action(
                session_id,
                "direct_message",
                ["Finance"],
                f"Round {expected_round}.",
                [],
                None,
            ),
            timeout=60,
        )
        obs = r.json().get("observation", r.json())
        actual = obs.get("round_number")
        assert actual == expected_round, (
            f"round_number={actual} after {expected_round} steps (expected {expected_round})"
        )

    print("  ✓ round_number increments correctly: 0 → 1 → 2 → ...")


def test_5_6_all_three_scenarios_work():
    print("\n[5.6] All three scenario types work without crash...")
    for i, scenario in enumerate(["aligned", "conflicted", "hostile_acquisition"]):
        session = requests.Session()
        r = session.post(
            f"{BASE_URL}/reset", json={"task_id": scenario, "seed": 50 + i}, timeout=30
        )
        obs = r.json()
        assert obs.get("done") is False, f"{scenario}: done=True after reset"

        stakeholders = list(obs.get("stakeholders", {}).keys())
        target = stakeholders[0] if stakeholders else "Finance"

        r = session.post(
            f"{BASE_URL}/step",
            json=make_action(
                obs.get("metadata", {}).get("session_id"),
                "direct_message",
                [target],
                "Opening communication.",
                [],
                None,
            ),
            timeout=60,
        )

        assert r.status_code == 200, f"{scenario}: step failed with {r.status_code}"
        print(f"  ✓ {scenario} works (stakeholders={stakeholders})")


def test_5_7_same_session_same_state_across_steps():
    print("\n[5.7] Same session maintains consistent state across steps...")
    session = requests.Session()
    r = session.post(
        f"{BASE_URL}/reset", json={"task_id": "aligned", "seed": 60}, timeout=30
    )
    obs0 = r.json()
    session_id = obs0.get("metadata", {}).get("session_id")

    r1 = session.post(
        f"{BASE_URL}/step",
        json=make_action(
            session_id,
            "direct_message",
            ["Finance"],
            "Step 1.",
            [],
            None,
        ),
        timeout=60,
    )
    obs1 = r1.json().get("observation", r1.json())

    r2 = session.post(
        f"{BASE_URL}/step",
        json=make_action(
            session_id,
            "direct_message",
            ["Finance"],
            "Step 2.",
            [],
            None,
        ),
        timeout=60,
    )
    obs2 = r2.json().get("observation", r2.json())

    assert obs2.get("round_number") == obs1.get("round_number") + 1, (
        "Round number should increment consistently within a session"
    )

    print("  ✓ Session state consistent across steps")


def test_5_8_reset_clears_all_session_state():
    print("\n[5.8] Reset clears all session state...")
    session = requests.Session()
    r = session.post(
        f"{BASE_URL}/reset", json={"task_id": "aligned", "seed": 70}, timeout=30
    )
    sid = r.json().get("metadata", {}).get("session_id")

    # Take 3 steps
    for _ in range(3):
        session.post(
            f"{BASE_URL}/step",
            json=make_action(
                sid,
                "direct_message",
                ["Finance"],
                "Step.",
                [],
                None,
            ),
            timeout=60,
        )

    # Reset
    obs_after = session.post(
        f"{BASE_URL}/reset", json={"task_id": "aligned", "seed": 71}, timeout=30
    ).json()

    assert obs_after.get("round_number") == 0, "Round number not reset"
    assert obs_after.get("done") is False, "done not reset to False"
    history = obs_after.get("engagement_history", [])
    assert len(history) >= 5, "History not reinitialized"

    print("  ✓ Reset clears round_number, done, and engagement history")


def run_all():
    print("=" * 60)
    print("  DealRoom v3 — Episode Isolation & Determinism")
    print("=" * 60)

    tests = [
        test_5_1_different_seeds_different_initial_state,
        test_5_2_round_number_resets_to_zero,
        test_5_3_done_false_after_reset,
        test_5_4_engagement_history_initialized,
        test_5_5_round_number_increments_correctly,
        test_5_6_all_three_scenarios_work,
        test_5_7_same_session_same_state_across_steps,
        test_5_8_reset_clears_all_session_state,
    ]

    failed = []
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            failed.append(t.__name__)

    print("\n" + "=" * 60)
    passed = len(tests) - len(failed)
    print(f"  ✓ SECTION 5 — {passed}/{len(tests)} checks passed")
    if failed:
        print(f"  ✗ FAILED: {failed}")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    run_all()
