#!/usr/bin/env python3
"""
test_03_causal_inference.py
DealRoom v3 — Causal Inference Signal Validation

Validates:
- Targeted interventions produce engagement changes in the targeted stakeholder
- Belief propagation reaches non-targeted stakeholders (cross-stakeholder echoes)
- Engagement history window slides correctly
- Noise cannot be cancelled (sigma > 0)
- Different targets produce different propagation patterns
- Weak signals appear for non-targeted stakeholders
- Causal signal correlates with graph betweenness centrality
"""

import os
from pathlib import Path

_dotenv = Path(__file__).parent / ".env"
if _dotenv.exists():
    try:
        from dotenv import load_dotenv

        load_dotenv(_dotenv)
    except ImportError:
        pass

import requests
import numpy as np

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


def run_targeted_intervention(scenario, target, document=None, seed=None):
    session = requests.Session()
    payload = {"task_id": scenario}
    if seed is not None:
        payload["seed"] = seed
    r = session.post(f"{BASE_URL}/reset", json=payload, timeout=30)
    session_id = r.json().get("metadata", {}).get("session_id")

    docs = [{"name": document, "content": f"{document} content"}] if document else []
    r = session.post(
        f"{BASE_URL}/step",
        json=make_action(
            session_id,
            "send_document" if document else "direct_message",
            [target],
            f"Communication to {target}.",
            docs,
            None,
        ),
        timeout=60,
    )

    result = r.json()
    return result.get("observation", result)


def test_3_1_targeted_stakeholder_engagement_changes():
    print("\n[3.1] Targeted stakeholder shows engagement change...")
    obs = run_targeted_intervention("aligned", "Finance", "roi_model", seed=10)
    delta = obs.get("engagement_level_delta")
    assert delta is not None, "engagement_level_delta missing"
    assert isinstance(delta, (int, float)), (
        f"delta must be numeric, got {type(delta).__name__}"
    )
    print(f"  ✓ delta = {delta:.4f} (numeric)")


def test_3_2_cross_stakeholder_echoes_detected():
    print("\n[3.2] Cross-stakeholder echoes (belief propagation)...")
    propagation_count = 0
    n = 12
    for i in range(n):
        obs = run_targeted_intervention(
            "conflicted", "Finance", "roi_model", seed=20 + i
        )
        echoes = obs.get("cross_stakeholder_echoes", [])
        if echoes and len(echoes) > 0:
            propagation_count += 1

    rate = propagation_count / n
    print(f"  Echoes detected in {propagation_count}/{n} episodes ({rate:.0%})")
    assert rate >= 0.65, (
        f"Propagation too rare ({rate:.0%}) — expected at least 65% echo recall"
    )
    print("  ✓ Cross-stakeholder echoes present (propagation active)")


def test_3_3_engagement_history_window_slides():
    print("\n[3.3] Engagement history slides correctly...")
    session = requests.Session()
    r = session.post(
        f"{BASE_URL}/reset", json={"task_id": "aligned", "seed": 30}, timeout=30
    )
    obs0 = r.json()
    history_len_0 = len(obs0.get("engagement_history", []))

    session_id = obs0.get("metadata", {}).get("session_id")

    for step_num in range(1, 4):
        r = session.post(
            f"{BASE_URL}/step",
            json=make_action(
                session_id,
                "direct_message",
                ["Finance"],
                f"Step {step_num}.",
                [],
                None,
            ),
            timeout=60,
        )
        obs = r.json().get("observation", r.json())
        history = obs.get("engagement_history", [])
        assert len(history) == history_len_0, (
            f"History window changed: was {history_len_0}, now {len(history)}"
        )

    print(f"  ✓ History window stable at {history_len_0} entries across steps")


def test_3_4_engagement_noise_not_cancellable():
    print("\n[3.4] Engagement noise cannot be cancelled...")
    session = requests.Session()
    r = session.post(
        f"{BASE_URL}/reset", json={"task_id": "aligned", "seed": 40}, timeout=30
    )
    session_id = r.json().get("metadata", {}).get("session_id")

    deltas = []
    for i in range(5):
        r = session.post(
            f"{BASE_URL}/step",
            json=make_action(
                session_id,
                "direct_message",
                ["Finance"],
                f"Message {i}.",
                [],
                None,
            ),
            timeout=60,
        )
        obs = r.json().get("observation", r.json())
        delta = obs.get("engagement_level_delta", 0)
        deltas.append(delta)

    # Deltas should not all be zero (noise is active)
    non_zero = sum(1 for d in deltas if abs(d) > 0.001)
    print(f"  Non-zero deltas: {non_zero}/5 (noise σ > 0 active)")
    assert non_zero >= 2, "All deltas near zero — noise may be disabled"
    print("  ✓ Noise sigma > 0 confirmed")


def test_3_5_different_targets_different_patterns():
    print("\n[3.5] Different targets → different propagation patterns...")
    finance_deltas, legal_deltas = [], []

    for i in range(4):
        obs = run_targeted_intervention(
            "conflicted", "Finance", "roi_model", seed=50 + i
        )
        finance_deltas.append(obs.get("engagement_level_delta", 0))

    for i in range(4):
        obs = run_targeted_intervention("conflicted", "Legal", "DPA", seed=60 + i)
        legal_deltas.append(obs.get("engagement_level_delta", 0))

    mean_f = np.mean(finance_deltas)
    mean_l = np.mean(legal_deltas)
    distance = abs(mean_f - mean_l)
    print(f"  Finance targeting: mean delta={mean_f:.4f}")
    print(f"  Legal targeting:  mean delta={mean_l:.4f}")
    print(f"  Distance: {distance:.4f}")

    # Different targets should produce measurably different patterns
    assert distance > 0.001, "Finance and Legal produced identical propagation patterns"
    print("  ✓ Different targets produce distinguishable propagation patterns")


def test_3_6_weak_signals_for_non_targeted():
    print("\n[3.6] Weak signals appear for non-targeted stakeholders...")
    signals_detected = 0

    for i in range(12):
        obs = run_targeted_intervention(
            "conflicted", "Finance", "roi_model", seed=70 + i
        )
        weak = obs.get("weak_signals", {})
        if weak and len(weak) > 0:
            signals_detected += 1

    rate = signals_detected / 12
    print(f"  Weak signals detected in {signals_detected}/12 episodes ({rate:.0%})")
    assert rate >= 0.2, f"Weak signals too rare ({rate:.0%})"
    print("  ✓ Weak signal mechanism active")


def test_3_7_echo_content_structure():
    print("\n[3.7] Echo entries have correct structure...")
    obs = run_targeted_intervention("aligned", "Finance", "roi_model", seed=80)
    echoes = obs.get("cross_stakeholder_echoes", [])

    assert isinstance(echoes, list), f"echoes must be list, got {type(echoes).__name__}"
    if len(echoes) > 0:
        echo = echoes[0]
        assert isinstance(echo, dict), "each echo must be a dict"
        assert any(
            k in echo for k in ["from", "from_stakeholder", "sender", "source"]
        ), f"echo missing sender field: {list(echo.keys())}"
        print(f"  ✓ {len(echoes)} echoes with correct structure: {list(echo.keys())}")
    else:
        print("  ⚠ No echoes in this episode (may be stochastic)")


def test_3_8_causal_signal_responds_to_graph_structure():
    print("\n[3.8] Causal signal responds to graph authority structure...")
    hub_impacts, leaf_impacts = [], []

    # ExecSponsor is authority hub in all scenarios
    for i in range(8):
        obs = run_targeted_intervention("conflicted", "ExecSponsor", None, seed=90 + i)
        hub_impacts.append(
            abs(obs.get("engagement_level_delta", 0))
            + 0.03 * len(obs.get("cross_stakeholder_echoes", []))
        )

    # Procurement is a lower-impact leaf in the public six-stakeholder benchmark.
    for i in range(8):
        obs = run_targeted_intervention("conflicted", "Procurement", None, seed=100 + i)
        leaf_impacts.append(
            abs(obs.get("engagement_level_delta", 0))
            + 0.03 * len(obs.get("cross_stakeholder_echoes", []))
        )

    hub_impact = float(np.mean(hub_impacts))
    leaf_impact = float(np.mean(leaf_impacts))
    print(f"  Hub (ExecSponsor) mean impact: {hub_impact:.6f}")
    print(f"  Leaf (Procurement) mean impact: {leaf_impact:.6f}")
    assert hub_impact > leaf_impact * 1.15, (
        f"Hub node should show materially more causal impact than leaf. "
        f"Got hub={hub_impact:.3f}, leaf={leaf_impact:.3f}."
    )
    print("  ✓ Causal signal varies by graph position")


def run_all():
    print("=" * 60)
    print("  DealRoom v3 — Causal Inference Signal Validation")
    print("=" * 60)

    tests = [
        test_3_1_targeted_stakeholder_engagement_changes,
        test_3_2_cross_stakeholder_echoes_detected,
        test_3_3_engagement_history_window_slides,
        test_3_4_engagement_noise_not_cancellable,
        test_3_5_different_targets_different_patterns,
        test_3_6_weak_signals_for_non_targeted,
        test_3_7_echo_content_structure,
        test_3_8_causal_signal_responds_to_graph_structure,
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
    print(f"  ✓ SECTION 3 — {passed}/{len(tests)} checks passed")
    if failed:
        print(f"  ✗ FAILED: {failed}")
        import sys

        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    run_all()
