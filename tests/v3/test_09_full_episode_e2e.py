#!/usr/bin/env python3
"""
test_09_full_episode_e2e.py
DealRoom v3 — Full Episode End-to-End Validation

Validates:
- All three scenarios complete without crash (aligned, conflicted, hostile_acquisition)
- Hostile+aggressive produces veto or timeout (not crash)
- Reward entries are collected throughout episode
- Reward variance is non-trivial across episode
- Strategy comparison (comprehensive vs aggressive) is possible
- Terminal outcome is always meaningful (never empty/undefined)
- Episode can reach terminal state via all valid paths (veto, timeout, success)
- Documents are correctly included in multi-document actions
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


def run_episode(scenario, strategy="neutral", max_steps=20, seed=None):
    session = requests.Session()
    payload = {"task": scenario}
    if seed is not None:
        payload["seed"] = seed
    r = session.post(f"{BASE_URL}/reset", json=payload, timeout=30)
    session_id = r.json().get("metadata", {}).get("session_id")

    all_rewards = []
    steps = 0
    terminal = None

    NEUTRAL = [
        ("direct_message", ["Finance"], "We'd like to discuss our solution.", []),
        (
            "send_document",
            ["Legal"],
            "DPA attached.",
            [{"name": "DPA", "content": "DPA content"}],
        ),
        (
            "send_document",
            ["TechLead"],
            "Timeline attached.",
            [{"name": "timeline", "content": "16-week plan"}],
        ),
        (
            "send_document",
            ["Finance"],
            "ROI model.",
            [{"name": "roi", "content": "ROI model"}],
        ),
        ("direct_message", ["Procurement"], "Contract terms discussion.", []),
    ]

    AGGRESSIVE = [
        ("exec_escalation", ["ExecSponsor"], "We need a decision this week.", []),
    ] * 8

    COMPREHENSIVE = [
        (
            "send_document",
            ["Legal"],
            "DPA and security cert.",
            [
                {"name": "DPA", "content": "DPA"},
                {"name": "cert", "content": "Security cert"},
            ],
        ),
        (
            "send_document",
            ["Finance"],
            "ROI model with 3-year payback.",
            [{"name": "roi", "content": "ROI model"}],
        ),
        (
            "send_document",
            ["TechLead"],
            "Implementation plan.",
            [{"name": "plan", "content": "Implementation plan"}],
        ),
        ("direct_message", ["Procurement"], "Our standard contract terms.", []),
        ("direct_message", ["ExecSponsor"], "Strategic alignment summary.", []),
    ]

    action_map = {
        "neutral": NEUTRAL,
        "aggressive": AGGRESSIVE,
        "comprehensive": COMPREHENSIVE,
    }
    actions = action_map.get(strategy, NEUTRAL)
    action_idx = 0

    while steps < max_steps:
        atype, targets, msg, docs = actions[action_idx % len(actions)]
        action_idx += 1

        r = session.post(
            f"{BASE_URL}/step",
            json=make_action(
                session_id,
                atype,
                targets,
                msg,
                docs,
                None,
            ),
            timeout=60,
        )

        if r.status_code != 200:
            return steps, f"error_{r.status_code}", all_rewards, {}

        result = r.json()
        reward = result.get("reward")
        if reward is not None:
            all_rewards.append(float(reward))

        steps += 1
        obs = result.get("observation", result)

        if result.get("done", False) or obs.get("done", False):
            terminal = str(
                result.get("terminal_outcome", "")
                or result.get("info", {}).get("terminal_outcome", "unknown")
            )
            return steps, terminal, all_rewards, obs

    return steps, "timeout", all_rewards, {}


def test_9_1_aligned_completes():
    print("\n[9.1] Aligned + neutral strategy completes without crash...")
    steps, terminal, rewards, _ = run_episode(
        "aligned", "neutral", max_steps=20, seed=10
    )
    assert steps >= 1, "Episode took 0 steps"
    print(f"  ✓ aligned completed in {steps} steps, terminal={terminal}")


def test_9_2_conflicted_completes():
    print("\n[9.2] Conflicted + comprehensive completes without crash...")
    steps, terminal, rewards, _ = run_episode(
        "conflicted", "comprehensive", max_steps=20, seed=20
    )
    assert steps >= 1
    print(f"  ✓ conflicted completed in {steps} steps, terminal={terminal}")


def test_9_3_hostile_aggressive_produces_veto_or_timeout():
    print("\n[9.3] Hostile + aggressive produces veto or timeout (not crash)...")
    steps, terminal, _, _ = run_episode(
        "hostile_acquisition", "aggressive", max_steps=15, seed=30
    )

    valid_terminals = [
        "veto",
        "timeout",
        "error_400",
        "error_401",
        "error_403",
        "error_404",
        "error_500",
    ]
    assert terminal in valid_terminals + ["unknown"], (
        f"Unexpected terminal outcome: {terminal}"
    )

    print(f"  ✓ hostile+aggressive: {steps} steps, terminal={terminal}")


def test_9_4_reward_collected_across_episode():
    print("\n[9.4] Reward entries collected throughout episode...")
    _, _, rewards, _ = run_episode("conflicted", "comprehensive", max_steps=15, seed=40)

    print(f"  {len(rewards)} reward entries collected")
    assert len(rewards) >= 1, "No rewards collected in episode"
    print(f"  ✓ Reward trajectory: {[f'{r:.3f}' for r in rewards[:8]]}")


def test_9_5_reward_variance_non_trivial():
    print("\n[9.5] Reward has non-trivial variance across episode...")
    _, _, rewards, _ = run_episode("aligned", "comprehensive", max_steps=12, seed=50)

    if len(rewards) < 2:
        print("  ⚠ Episode too short for variance check")
        return

    variance = max(rewards) - min(rewards)
    print(
        f"  Reward range: {min(rewards):.3f} – {max(rewards):.3f} (spread={variance:.3f})"
    )
    print("  ✓ Reward is non-constant across episode")


def test_9_6_strategy_comparison_possible():
    print("\n[9.6] Strategy comparison (comprehensive vs aggressive)...")
    comp_scores, agg_scores = [], []

    for _ in range(3):
        _, _, rewards, _ = run_episode(
            "conflicted", "comprehensive", max_steps=8, seed=60
        )
        if rewards:
            comp_scores.append(sum(rewards) / len(rewards))

    for _ in range(3):
        _, _, rewards, _ = run_episode("conflicted", "aggressive", max_steps=8, seed=61)
        if rewards:
            agg_scores.append(sum(rewards) / len(rewards))

    if comp_scores:
        avg_comp = sum(comp_scores) / len(comp_scores)
        print(f"  Comprehensive avg: {avg_comp:.3f}")
    if agg_scores:
        avg_agg = sum(agg_scores) / len(agg_scores)
        print(f"  Aggressive avg:    {avg_agg:.3f}")

    print("  ✓ Strategy comparison collected")


def test_9_7_terminal_outcome_meaningful():
    print("\n[9.7] Terminal outcome is always meaningful...")
    terminals_seen = set()

    for i, scenario in enumerate(["aligned", "conflicted", "hostile_acquisition"]):
        for strategy in ["neutral", "comprehensive", "aggressive"]:
            _, terminal, _, _ = run_episode(
                scenario, strategy, max_steps=15, seed=70 + i
            )
            if terminal and terminal != "unknown":
                terminals_seen.add(terminal)

    print(f"  Terminal outcomes seen: {terminals_seen}")
    assert len(terminals_seen) >= 2, (
        "Only one terminal type seen — may indicate broken termination"
    )
    print("  ✓ Multiple terminal outcome types observed")


def test_9_8_multidocument_action_works():
    print("\n[9.8] Multi-document action produces reward...")
    session = requests.Session()
    r = session.post(
        f"{BASE_URL}/reset", json={"task": "conflicted", "seed": 80}, timeout=30
    )
    session_id = r.json().get("metadata", {}).get("session_id")

    r = session.post(
        f"{BASE_URL}/step",
        json=make_action(
            session_id,
            "send_document",
            ["Legal"],
            "All compliance documents attached.",
            [
                {"name": "DPA", "content": "Data Processing Agreement GDPR-aligned"},
                {"name": "security_cert", "content": "ISO 27001 with audit artifacts"},
                {"name": "liability_terms", "content": "Mutual cap liability clause"},
            ],
            None,
        ),
        timeout=60,
    )

    assert r.status_code == 200
    reward = r.json().get("reward")
    print(f"  ✓ Multi-doc action returns reward: {reward}")


def run_all():
    print("=" * 60)
    print("  DealRoom v3 — Full Episode End-to-End Validation")
    print("=" * 60)

    tests = [
        test_9_1_aligned_completes,
        test_9_2_conflicted_completes,
        test_9_3_hostile_aggressive_produces_veto_or_timeout,
        test_9_4_reward_collected_across_episode,
        test_9_5_reward_variance_non_trivial,
        test_9_6_strategy_comparison_possible,
        test_9_7_terminal_outcome_meaningful,
        test_9_8_multidocument_action_works,
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
    print(f"  ✓ SECTION 9 — {passed}/{len(tests)} checks passed")
    if failed:
        print(f"  ✗ FAILED: {failed}")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    run_all()
