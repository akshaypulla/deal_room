#!/usr/bin/env python3
"""
test_07_causal_graph.py
DealRoom v3 — Causal Graph Unit Tests (runs inside container)

Validates:
- Propagation direction follows graph edges
- Signal carries from A to B through edge
- Damping prevents runaway amplification
- All beliefs stay normalized after propagation
- No self-loops in any scenario
- ExecSponsor has outgoing edges (authority invariant)
- Hub node has highest betweenness centrality
- Graph identifiability: all sampled graphs produce unique behavioral signatures
"""

import sys

sys.path.insert(0, "/app/env")

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    _dotenv = Path("/app/.env")
    if _dotenv.exists():
        load_dotenv(_dotenv)
except Exception:
    pass

import numpy as np

STANDARD_5 = ["Finance", "Legal", "TechLead", "Procurement", "ExecSponsor"]
STANDARD_H = {
    "ExecSponsor": 5,
    "Finance": 3,
    "Legal": 3,
    "TechLead": 2,
    "Procurement": 2,
}


def test_7_1_propagation_direction():
    print("\n[7.1] Propagation follows graph edge direction...")
    from deal_room.committee.causal_graph import (
        CausalGraph,
        propagate_beliefs,
        create_neutral_beliefs,
        apply_positive_delta,
    )

    g = CausalGraph(
        nodes=["A", "B", "C"],
        edges={("A", "B"): 0.7},
        authority_weights={},
        scenario_type="aligned",
        seed=0,
    )
    before = create_neutral_beliefs(["A", "B", "C"])
    after = {**before, "A": apply_positive_delta(before["A"], 0.4)}
    updated = propagate_beliefs(g, before, after, n_steps=3)

    assert updated["B"].positive_mass() > before["B"].positive_mass() + 0.05, (
        "B's belief should increase (signal flows A→B)"
    )
    assert abs(updated["C"].positive_mass() - before["C"].positive_mass()) < 0.03, (
        "C should be unaffected (no edge C←A)"
    )

    print("  ✓ Signal propagates correctly along edges")


def test_7_2_signal_carries_through_edge():
    print("\n[7.2] Signal carries from A to B through edge...")
    from deal_room.committee.causal_graph import (
        CausalGraph,
        propagate_beliefs,
        create_neutral_beliefs,
        apply_positive_delta,
    )

    g = CausalGraph(
        nodes=["A", "B"],
        edges={("A", "B"): 0.8},
        authority_weights={},
        scenario_type="aligned",
        seed=0,
    )
    before = create_neutral_beliefs(["A", "B"])
    before["A"] = apply_positive_delta(before["A"], 0.5)
    before["B"] = apply_positive_delta(before["B"], 0.3)

    after_A = apply_positive_delta(before["A"], 0.2)
    after = {"A": after_A, "B": before["B"]}
    updated = propagate_beliefs(g, before, after, n_steps=3)

    pm_B_after = updated["B"].positive_mass()
    print(f"  A: {before['A'].positive_mass():.3f}→{after_A.positive_mass():.3f}")
    print(f"  B: {before['B'].positive_mass():.3f}→{pm_B_after:.3f}")

    assert pm_B_after != before["B"].positive_mass(), (
        "B's belief must change when A changes"
    )
    print("  ✓ Signal carries through graph edge")


def test_7_3_damping_prevents_runaway():
    print("\n[7.3] Damping prevents runaway amplification...")
    from deal_room.committee.causal_graph import (
        CausalGraph,
        propagate_beliefs,
        create_neutral_beliefs,
        apply_positive_delta,
    )

    nodes = list("ABCDE")
    edges = {(s, d): 0.8 for s in nodes for d in nodes if s != d}
    g = CausalGraph(
        nodes=nodes,
        edges=edges,
        authority_weights={},
        scenario_type="hostile_acquisition",
        seed=0,
    )
    before = create_neutral_beliefs(nodes)
    after = {**before, "A": apply_positive_delta(before["A"], 0.5)}
    updated = propagate_beliefs(g, before, after, n_steps=5)

    for sid in "BCDE":
        pm = updated[sid].positive_mass()
        assert 0.0 < pm < 1.0, f"{sid} has runaway belief: {pm}"

    print("  ✓ All beliefs bounded [0,1] after dense propagation")


def test_7_4_beliefs_normalized_after_propagation():
    print("\n[7.4] All beliefs normalized after propagation...")
    from deal_room.committee.causal_graph import (
        sample_graph,
        propagate_beliefs,
        create_neutral_beliefs,
        apply_positive_delta,
    )

    rng = np.random.default_rng(42)
    g = sample_graph(STANDARD_5, STANDARD_H, "conflicted", rng)
    before = create_neutral_beliefs(STANDARD_5)
    after = {**before, "Finance": apply_positive_delta(before["Finance"], 0.4)}
    updated = propagate_beliefs(g, before, after, n_steps=3)

    for sid, b in updated.items():
        total = sum(b.distribution.values())
        assert abs(total - 1.0) < 1e-6, f"{sid} not normalized: sum={total}"

    print("  ✓ All beliefs sum to 1.0 after propagation")


def test_7_5_no_self_loops():
    print("\n[7.5] No self-loops in any scenario type...")
    from deal_room.committee.causal_graph import sample_graph

    for scenario in ["aligned", "conflicted", "hostile_acquisition"]:
        g = sample_graph(STANDARD_5, STANDARD_H, scenario, np.random.default_rng())
        for sid in STANDARD_5:
            w = g.get_weight(sid, sid)
            assert w == 0.0, f"Self-loop detected: {sid}→{sid} weight={w}"

    print("  ✓ No self-loops in any scenario")


def test_7_6_exec_sponsor_outgoing_authority():
    print("\n[7.6] ExecSponsor has outgoing edges (authority invariant)...")
    from deal_room.committee.causal_graph import sample_graph

    for scenario in ["aligned", "conflicted", "hostile_acquisition"]:
        for seed in range(10):
            g = sample_graph(
                STANDARD_5, STANDARD_H, scenario, np.random.default_rng(seed)
            )
            outgoing = [w for w in g.get_outgoing("ExecSponsor").values() if w > 0.1]
            assert len(outgoing) >= 2, (
                f"Scenario={scenario} seed={seed}: ExecSponsor has only {len(outgoing)} outgoing edges"
            )

    print("  ✓ ExecSponsor has >=2 outgoing edges in all scenarios/seeds")


def test_7_7_hub_centrality_beats_leaf():
    print("\n[7.7] Hub node has highest betweenness centrality...")
    from deal_room.committee.causal_graph import CausalGraph, get_betweenness_centrality

    edges = {("Hub", l): 0.8 for l in ["A", "B", "C", "D"]}
    g = CausalGraph(
        nodes=["Hub", "A", "B", "C", "D"],
        edges=edges,
        authority_weights={},
        scenario_type="aligned",
        seed=0,
    )

    hub_c = get_betweenness_centrality(g, "Hub")
    leaf_c = max(get_betweenness_centrality(g, leaf) for leaf in ["A", "B", "C", "D"])

    print(f"  Hub (betweenness): {hub_c:.3f}")
    print(f"  Max leaf (betweenness): {leaf_c:.3f}")

    assert hub_c >= leaf_c, (
        f"Hub centrality {hub_c:.3f} should >= max leaf {leaf_c:.3f}"
    )
    print("  ✓ Hub node has highest betweenness centrality")


def test_7_8_graph_identifiability():
    print("\n[7.8] Graph identifiability — all graphs produce unique signatures...")
    from deal_room.committee.causal_graph import (
        sample_graph,
        compute_behavioral_signature,
    )

    print("  Sampling 20 graphs (may take ~60 seconds)...")
    n_graphs = 20
    signatures = []

    for i in range(n_graphs):
        rng = np.random.default_rng(i * 100 + 42)
        g = sample_graph(STANDARD_5, STANDARD_H, "conflicted", rng)
        sig = compute_behavioral_signature(g, "Finance", 0.4, n_steps=3)
        signatures.append((i, sig))

    distinguishable = 0
    for i in range(n_graphs):
        for j in range(i + 1, n_graphs):
            _, sig_i = signatures[i]
            _, sig_j = signatures[j]
            if sig_i != sig_j:
                distinguishable += 1

    total_pairs = n_graphs * (n_graphs - 1) // 2
    ratio = distinguishable / total_pairs if total_pairs > 0 else 0
    print(
        f"  {distinguishable}/{total_pairs} graph pairs distinguishable ({ratio:.1%})"
    )

    assert distinguishable == total_pairs, (
        f"CRITICAL: Only {distinguishable}/{total_pairs} graph pairs distinguishable"
    )

    print("  ✓ GRAPH IDENTIFIABILITY CONFIRMED — all 20 graphs pairwise unique")


def run_all():
    print("=" * 60)
    print("  DealRoom v3 — Causal Graph Unit Tests (Container)")
    print("=" * 60)

    tests = [
        test_7_1_propagation_direction,
        test_7_2_signal_carries_through_edge,
        test_7_3_damping_prevents_runaway,
        test_7_4_beliefs_normalized_after_propagation,
        test_7_5_no_self_loops,
        test_7_6_exec_sponsor_outgoing_authority,
        test_7_7_hub_centrality_beats_leaf,
        test_7_8_graph_identifiability,
    ]

    failed = []
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            failed.append(t.__name__)
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            failed.append(t.__name__)

    print("\n" + "=" * 60)
    passed = len(tests) - len(failed)
    print(f"  ✓ SECTION 7 — {passed}/{len(tests)} checks passed")
    if failed:
        print(f"  ✗ FAILED: {failed}")
        import sys

        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    run_all()
