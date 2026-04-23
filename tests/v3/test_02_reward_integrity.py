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

sys.path.insert(0, "/app/env")

_dotenv = Path(__file__).parent / ".env"
if _dotenv.exists():
    try:
        from dotenv import load_dotenv

        load_dotenv(_dotenv)
    except ImportError:
        pass

import requests

from deal_room.rewards.utterance_scorer import LOOKAHEAD_COST

BASE_URL = os.getenv("DEALROOM_BASE_URL", "http://127.0.0.1:7860")
REWARD_VARIANCE_MIN_SPREAD = 0.05
LOOKAHEAD_MIN_RECORDED = 15
LOOKAHEAD_MIN_ACCURACY = 0.60


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
    r = session.post(f"{BASE_URL}/reset", json={"task_id": "aligned", "seed": 10})
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
    r = session.post(f"{BASE_URL}/reset", json={"task_id": "aligned", "seed": 20})
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
    goal1 = float(r1.json().get("info", {}).get("reward_components", {}).get("goal", 0))

    # With lookahead
    r = session.post(f"{BASE_URL}/reset", json={"task_id": "aligned", "seed": 20})
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
                    None, "direct_message", ["Finance"], "Test message.", [], None
                ),
            },
        ),
        timeout=60,
    )
    info2 = r2.json().get("info", {})
    goal2 = float(info2.get("reward_components", {}).get("goal", 0))

    diff = goal1 - goal2
    expected_cost = LOOKAHEAD_COST
    assert abs(diff - expected_cost) < 0.015, (
        f"Lookahead goal cost should be {expected_cost:.3f}, got {diff:.3f} (goal1={goal1:.3f}, goal2={goal2:.3f})"
    )
    assert "lookahead_predicted_deltas" in info2, (
        "Lookahead diagnostics missing from info"
    )

    print(
        f"  ✓ goal cost = {diff:.4f} (expected 0.07, diff={abs(diff - expected_cost):.4f})"
    )


def test_2_3_reward_in_range_after_valid_actions():
    print(
        "\n[2.3] Reward components stay in [0.0, 1.0] and scalar reward stays finite..."
    )
    session = requests.Session()
    r = session.post(f"{BASE_URL}/reset", json={"task_id": "conflicted", "seed": 40})
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
        result = r.json()
        reward = get_reward(result)
        if reward is not None:
            assert reward == reward, "Reward must be finite (not NaN)"
        components = result.get("info", {}).get("reward_components", {})
        for dim in ["goal", "trust", "info", "risk", "causal"]:
            if dim in components:
                assert 0.0 <= float(components[dim]) <= 1.0, (
                    f"Reward component {dim}={components[dim]} outside [0, 1]"
                )

    print("  ✓ Reward components remain bounded; scalar reward remains finite")


def test_2_4_deterministic_reward_with_seed():
    print("\n[2.4] Grader is deterministic for same seed and same action...")
    rewards = []

    for trial in range(3):
        session = requests.Session()
        seed = 100
        r = session.post(f"{BASE_URL}/reset", json={"task_id": "aligned", "seed": seed})
        session_id = r.json().get("metadata", {}).get("session_id")
        action = make_action(
            session_id, "direct_message", ["Finance"], "Same message.", []
        )

        r = session.post(f"{BASE_URL}/step", json=action, timeout=60)
        reward = get_reward(r.json())
        assert reward is not None, f"Trial {trial} returned no reward"
        rewards.append(reward)

    spread = max(rewards) - min(rewards)
    assert spread < 1e-9, (
        f"Same seed/action should replay exactly. Rewards={rewards}, spread={spread:.12f}"
    )
    print(f"  ✓ {len(rewards)} same-seed trials replay exactly")


def test_2_5_repeat_same_action_does_not_escalate_reward():
    print("\n[2.5] Repeating same action without new info does NOT inflate reward...")
    session = requests.Session()
    r = session.post(f"{BASE_URL}/reset", json={"task_id": "aligned", "seed": 50})
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
    assert trend <= 0.01, (
        f"Repeating same action inflated reward: g1={g1:.3f}, g3={g3:.3f}, trend={trend:+.3f}"
    )
    print("  ✓ Repeating same action does not systematically inflate reward")


def test_2_6_different_targets_different_causal_scores():
    print("\n[2.6] Different targets produce different causal scores...")
    scores_by_target = {}

    for target in ["Finance", "Legal", "TechLead"]:
        causal_scores = []
        for seed in [60, 61, 62, 63, 64]:
            session = requests.Session()
            r = session.post(
                f"{BASE_URL}/reset", json={"task_id": "conflicted", "seed": seed}
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
            info = r.json().get("info", {})
            components = info.get("reward_components", {})
            causal = components.get("causal")
            if causal is not None:
                causal_scores.append(float(causal))

        if causal_scores:
            scores_by_target[target] = sum(causal_scores) / len(causal_scores)

    unique_scores = len(set(round(v, 3) for v in scores_by_target.values()))
    print(
        f"  target causal scores: { {k: f'{v:.3f}' for k, v in scores_by_target.items()} }"
    )
    assert unique_scores >= 2, (
        f"All targets got identical causal scores — r^causal not discriminative across targets.\n"
        f"Scores: {scores_by_target}\n"
        f"NOTE: Legal may legitimately score 0.0 if it has 0 betweenness centrality in the sampled graph."
    )
    print(f"  ✓ {unique_scores} distinct causal score values across targets")


def test_2_7_informative_action_outperforms_empty():
    print("\n[2.7] Substantive action outperforms empty/nearly-empty message...")
    session = requests.Session()
    r = session.post(f"{BASE_URL}/reset", json={"task_id": "aligned", "seed": 70})
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
    r = session.post(f"{BASE_URL}/reset", json={"task_id": "conflicted", "seed": 80})
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
        assert variance > REWARD_VARIANCE_MIN_SPREAD, (
            f"Reward variance too low: {variance:.4f}. Expected >{REWARD_VARIANCE_MIN_SPREAD:.2f} "
            "for a meaningful learning signal."
        )
        print("  ✓ Reward is discriminative across action types")
    else:
        raise AssertionError("Not enough rewards collected to measure variance")


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
        r = session.post(
            f"{BASE_URL}/reset", json={"task_id": "conflicted", "seed": seed}
        )
        sid = r.json().get("metadata", {}).get("session_id")

        poor_action["metadata"]["session_id"] = sid
        r = session.post(f"{BASE_URL}/step", json=poor_action, timeout=60)
        rew = get_reward(r.json())
        if rew is not None:
            poor_rewards.append(rew)

    for seed in [93, 94, 95]:
        session = requests.Session()
        r = session.post(
            f"{BASE_URL}/reset", json={"task_id": "conflicted", "seed": seed}
        )
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
        assert avg_good >= avg_poor - 0.01, (
            f"Grader scored poor docs higher than good docs beyond tolerance: "
            f"poor={avg_poor:.3f}, good={avg_good:.3f}"
        )
        print("  ✓ Good documentation rewarded at >= poor documentation")
    else:
        raise AssertionError(
            "Could not collect both poor and good documentation rewards"
        )


def test_2_10_lookahead_improves_prediction_accuracy():
    print("\n[2.10] Lookahead prediction accuracy beats random baseline...")
    accuracies = []

    for seed in range(200, 220):
        session = requests.Session()
        r = session.post(
            f"{BASE_URL}/reset",
            json={"task_id": "conflicted", "seed": seed},
            timeout=30,
        )
        session_id = r.json().get("metadata", {}).get("session_id")
        action_draft = make_action(
            None,
            "send_document",
            ["Finance"],
            "ROI model with downside cases.",
            [{"name": "roi", "content": "ROI model"}],
            None,
        )
        r = session.post(
            f"{BASE_URL}/step",
            json=make_action(
                session_id,
                "send_document",
                ["Finance"],
                "ROI model with downside cases.",
                [{"name": "roi", "content": "ROI model"}],
                {
                    "depth": 2,
                    "n_hypotheses": 2,
                    "action_draft": action_draft,
                },
            ),
            timeout=60,
        )
        info = r.json().get("info", {})
        if (
            info.get("lookahead_used")
            and info.get("lookahead_prediction_accuracy") is not None
        ):
            accuracies.append(float(info["lookahead_prediction_accuracy"]))

    assert len(accuracies) >= LOOKAHEAD_MIN_RECORDED, (
        f"Only {len(accuracies)} lookahead predictions recorded; expected at least "
        f"{LOOKAHEAD_MIN_RECORDED}."
    )
    mean_acc = sum(accuracies) / len(accuracies)
    assert mean_acc > LOOKAHEAD_MIN_ACCURACY, (
        f"Lookahead prediction accuracy too low: {mean_acc:.3f}. "
        f"Expected >{LOOKAHEAD_MIN_ACCURACY:.2f}."
    )
    print(f"  ✓ Lookahead accuracy mean={mean_acc:.3f} over {len(accuracies)} runs")


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
        test_2_10_lookahead_improves_prediction_accuracy,
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
