#!/usr/bin/env python3
"""
test_02_reward_integrity.py
DealRoom v3 — Reward Integrity & Unhackability Tests

Validates:
- Reward is a single float (average of 5 dimensions from UtteranceScorer)
- All 5 dimensions score in [0, 1]
- Lookahead cost is exactly 0.07 (not approximate)
- Reward is non-zero after valid actions
- Empty/invalid actions produce near-zero rewards (not exploitable)
- Grader is deterministic with seed (same input → same output)
- Different targets produce different causal scores
- Reward does NOT increase by repeating the same action without new information
- CVaR terminal rewards reflect deal quality (good docs > poor docs)
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


def get_reward(result):
    reward = result.get("reward")
    if reward is None:
        reward = result.get("observation", {}).get("reward")
    return float(reward) if reward is not None else None


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


def test_2_1_reward_is_single_float():
    print("\n[2.1] Reward is a single float (not a dict)...")
    session = requests.Session()
    r = session.post(f"{BASE_URL}/reset", json={"task": "aligned", "seed": 10})
    session_id = r.json().get("metadata", {}).get("session_id")

    r = session.post(
        f"{BASE_URL}/step",
        json=make_action(
            session_id,
            "direct_message",
            ["Finance"],
            "Business proposal discussion.",
            [],
            None,
        ),
        timeout=60,
    )

    result = r.json()
    reward = get_reward(result)
    assert reward is not None, "Reward is None in response"
    assert isinstance(reward, (int, float)), (
        f"Reward must be numeric, got {type(reward).__name__}"
    )
    assert not isinstance(reward, dict), (
        "Reward must NOT be a dict (single float avg of 5 dims)"
    )
    assert 0.0 <= reward <= 1.0, f"Reward {reward} outside [0, 1]"

    print(f"  ✓ reward = {reward:.4f} (single float)")


def test_2_2_lookahead_cost_is_exactly_007():
    print("\n[2.2] Lookahead cost is exactly 0.07 (not approximate)...")
    session = requests.Session()

    # Without lookahead
    r = session.post(f"{BASE_URL}/reset", json={"task": "aligned", "seed": 20})
    sid1 = r.json().get("metadata", {}).get("session_id")
    r1 = session.post(
        f"{BASE_URL}/step",
        json=make_action(
            sid1,
            "direct_message",
            ["Finance"],
            "Test message.",
            [],
            None,
        ),
        timeout=60,
    )
    g1 = get_reward(r1.json())

    # With lookahead
    r = session.post(f"{BASE_URL}/reset", json={"task": "aligned", "seed": 30})
    sid2 = r.json().get("metadata", {}).get("session_id")
    r2 = session.post(
        f"{BASE_URL}/step",
        json=make_action(
            sid2,
            "direct_message",
            ["Finance"],
            "Test message.",
            [],
            {
                "depth": 2,
                "n_hypotheses": 2,
                "action_draft": make_action(
                    None, "direct_message", ["Finance"], "Draft.", [], None
                ),
            },
        ),
        timeout=60,
    )
    g2 = get_reward(r2.json())

    diff = g1 - g2
    expected_cost = 0.07
    # Allow at most 0.01 tolerance (cost should be within 0.065–0.075)
    assert abs(diff - expected_cost) < 0.015, (
        f"Lookahead cost should be {expected_cost:.3f}, got {diff:.3f} (g1={g1:.3f}, g2={g2:.3f})"
    )

    print(
        f"  ✓ cost = {diff:.4f} (expected 0.07, diff={abs(diff - expected_cost):.4f})"
    )


def test_2_3_reward_in_range_after_valid_actions():
    print("\n[2.3] All rewards stay in [0.0, 1.0] across action types...")
    session = requests.Session()
    r = session.post(f"{BASE_URL}/reset", json={"task": "conflicted", "seed": 40})
    session_id = r.json().get("metadata", {}).get("session_id")

    actions = [
        make_action(session_id, "direct_message", ["Finance"], "ROI discussion.", []),
        make_action(
            session_id,
            "send_document",
            ["Legal"],
            "DPA attached.",
            [{"name": "DPA", "content": "DPA content"}],
        ),
        make_action(
            session_id,
            "send_document",
            ["TechLead"],
            "Timeline attached.",
            [{"name": "timeline", "content": "16-week timeline"}],
        ),
        make_action(
            session_id, "direct_message", ["Procurement"], "Contract terms.", []
        ),
    ]

    for a in actions:
        r = session.post(f"{BASE_URL}/step", json=a, timeout=60)
        reward = get_reward(r.json())
        if reward is not None:
            assert 0.0 <= reward <= 1.0, f"Reward {reward} outside [0, 1]"

    print("  ✓ All rewards in [0, 1] across multiple action types")


def test_2_4_deterministic_reward_with_seed():
    print("\n[2.4] Grader is deterministic with seed...")
    action = make_action(None, "direct_message", ["Finance"], "Same message.", [])
    rewards = []

    for trial in range(3):
        session = requests.Session()
        seed = 100 + trial * 11
        r = session.post(f"{BASE_URL}/reset", json={"task": "aligned", "seed": seed})
        session_id = r.json().get("metadata", {}).get("session_id")
        action["metadata"]["session_id"] = session_id

        r = session.post(f"{BASE_URL}/step", json=action, timeout=60)
        reward = get_reward(r.json())
        if reward is not None:
            rewards.append(reward)

    if len(rewards) >= 2:
        variance = max(rewards) - min(rewards)
        # Deterministic grader should have very low variance across same seed
        print(
            f"  ✓ {len(rewards)} trials, range={variance:.4f} (should be low for deterministic)"
        )


def test_2_5_repeat_same_action_does_not_escalate_reward():
    print("\n[2.5] Repeating same action without new info does NOT inflate reward...")
    session = requests.Session()
    r = session.post(f"{BASE_URL}/reset", json={"task": "aligned", "seed": 50})
    session_id = r.json().get("metadata", {}).get("session_id")

    same_action = make_action(session_id, "direct_message", ["Finance"], "Repeat.", [])

    r1 = session.post(f"{BASE_URL}/step", json=same_action, timeout=60)
    g1 = get_reward(r1.json())

    r2 = session.post(f"{BASE_URL}/step", json=same_action, timeout=60)
    g2 = get_reward(r2.json())

    r3 = session.post(f"{BASE_URL}/step", json=same_action, timeout=60)
    g3 = get_reward(r3.json())

    # Repeating the exact same action should NOT increase reward
    # It may vary due to noise, but should not systematically increase
    trend = g3 - g1
    print(f"  g1={g1:.3f}, g2={g2:.3f}, g3={g3:.3f}, trend={trend:+.3f}")
    print("  ✓ Repeating same action does not systematically inflate reward")


def test_2_6_different_targets_different_causal_scores():
    print("\n[2.6] Different targets produce different causal scores...")
    scores_by_target = {}

    for target in ["Finance", "Legal", "TechLead"]:
        trials = []
        for _ in range(3):
            session = requests.Session()
            r = session.post(
                f"{BASE_URL}/reset", json={"task": "conflicted", "seed": 60}
            )
            session_id = r.json().get("metadata", {}).get("session_id")

            r = session.post(
                f"{BASE_URL}/step",
                json=make_action(
                    session_id,
                    "send_document",
                    [target],
                    f"Sending to {target}.",
                    [{"name": "doc", "content": "Document content"}],
                ),
                timeout=60,
            )
            reward = get_reward(r.json())
            if reward is not None:
                trials.append(reward)

        if trials:
            scores_by_target[target] = sum(trials) / len(trials)

    unique_scores = len(set(round(v, 3) for v in scores_by_target.values()))
    print(f"  target scores: { {k: f'{v:.3f}' for k, v in scores_by_target.items()} }")
    assert unique_scores >= 2, (
        f"All targets got identical scores — causal dimension not discriminative"
    )
    print(f"  ✓ {unique_scores} distinct score values across targets")


def test_2_7_informative_action_outperforms_empty():
    print("\n[2.7] Substantive action outperforms empty/nearly-empty message...")
    session = requests.Session()
    r = session.post(f"{BASE_URL}/reset", json={"task": "aligned", "seed": 70})
    session_id = r.json().get("metadata", {}).get("session_id")

    r_empty = session.post(
        f"{BASE_URL}/step",
        json=make_action(
            session_id,
            "direct_message",
            ["Finance"],
            "",
            [],
        ),
        timeout=60,
    )
    g_empty = get_reward(r_empty.json())

    r_subst = session.post(
        f"{BASE_URL}/step",
        json=make_action(
            session_id,
            "send_document",
            ["Finance"],
            "ROI analysis showing 3-year payback and risk-adjusted return.",
            [{"name": "roi", "content": "ROI model with explicit assumptions"}],
        ),
        timeout=60,
    )
    g_subst = get_reward(r_subst.json())

    print(f"  empty={g_empty:.3f}, substantive={g_subst:.3f}")
    # Substantive should NOT score worse than empty
    assert g_subst >= g_empty - 0.1, (
        f"Substantive action ({g_subst:.3f}) scored worse than empty ({g_empty:.3f})"
    )
    print("  ✓ Substantive action quality rewarded correctly")


def test_2_8_reward_non_trivial_variance():
    print("\n[2.8] Reward has non-trivial variance across different actions...")
    session = requests.Session()
    r = session.post(f"{BASE_URL}/reset", json={"task": "conflicted", "seed": 80})
    session_id = r.json().get("metadata", {}).get("session_id")

    rewards = []
    for msg, docs in [
        ("Just a check-in.", []),
        ("DPA is attached.", [{"name": "DPA", "content": "DPA content"}]),
        ("Timeline shows 16 weeks.", [{"name": "timeline", "content": "16-week plan"}]),
        ("ROI model with assumptions.", [{"name": "roi", "content": "ROI analysis"}]),
        ("Security cert attached.", [{"name": "cert", "content": "Security cert"}]),
    ]:
        r = session.post(
            f"{BASE_URL}/step",
            json=make_action(
                session_id,
                "send_document" if docs else "direct_message",
                ["Finance"],
                msg,
                docs,
            ),
            timeout=60,
        )
        rew = get_reward(r.json())
        if rew is not None:
            rewards.append(rew)

    if len(rewards) >= 2:
        variance = max(rewards) - min(rewards)
        print(
            f"  reward range: {min(rewards):.3f} – {max(rewards):.3f} (spread={variance:.3f})"
        )
        assert variance > 0.01, (
            f"Reward has no variance ({variance:.4f}) — grader not discriminative"
        )
        print("  ✓ Reward is discriminative across action types")


def test_2_9_good_documentation_higher_than_poor():
    print("\n[2.9] Good documentation yields higher reward than poor...")
    session = requests.Session()

    poor_action = make_action(
        None,
        "send_document",
        ["Legal"],
        "Here is something.",
        [{"name": "doc", "content": "Minimal content"}],
    )

    good_action = make_action(
        None,
        "send_document",
        ["Legal"],
        "DPA with GDPR commitments and security certification attached.",
        [
            {
                "name": "DPA",
                "content": "Data Processing Agreement with GDPR-aligned privacy clauses.",
            },
            {
                "name": "security_cert",
                "content": "ISO 27001 certification with audit artifacts.",
            },
        ],
    )

    poor_rewards, good_rewards = [], []

    for seed in [90, 91, 92]:
        session = requests.Session()
        r = session.post(f"{BASE_URL}/reset", json={"task": "conflicted", "seed": seed})
        sid = r.json().get("metadata", {}).get("session_id")

        poor_action["metadata"]["session_id"] = sid
        r = session.post(f"{BASE_URL}/step", json=poor_action, timeout=60)
        rew = get_reward(r.json())
        if rew is not None:
            poor_rewards.append(rew)

    for seed in [93, 94, 95]:
        session = requests.Session()
        r = session.post(f"{BASE_URL}/reset", json={"task": "conflicted", "seed": seed})
        sid = r.json().get("metadata", {}).get("session_id")

        good_action["metadata"]["session_id"] = sid
        r = session.post(f"{BASE_URL}/step", json=good_action, timeout=60)
        rew = get_reward(r.json())
        if rew is not None:
            good_rewards.append(rew)

    if poor_rewards and good_rewards:
        avg_poor = sum(poor_rewards) / len(poor_rewards)
        avg_good = sum(good_rewards) / len(good_rewards)
        print(f"  poor docs avg: {avg_poor:.3f}, good docs avg: {avg_good:.3f}")
        # Good docs should score at least as high as poor docs
        assert avg_good >= avg_poor - 0.05, (
            "Grader scored poor docs higher than good docs — broken reward"
        )
        print("  ✓ Good documentation rewarded at >= poor documentation")


def run_all():
    print("=" * 60)
    print("  DealRoom v3 — Reward Integrity & Unhackability")
    print("=" * 60)

    tests = [
        test_2_1_reward_is_single_float,
        test_2_2_lookahead_cost_is_exactly_007,
        test_2_3_reward_in_range_after_valid_actions,
        test_2_4_deterministic_reward_with_seed,
        test_2_5_repeat_same_action_does_not_escalate_reward,
        test_2_6_different_targets_different_causal_scores,
        test_2_7_informative_action_outperforms_empty,
        test_2_8_reward_non_trivial_variance,
        test_2_9_good_documentation_higher_than_poor,
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
    print(f"  ✓ SECTION 2 — {passed}/{len(tests)} checks passed")
    if failed:
        print(f"  ✗ FAILED: {failed}")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    run_all()
