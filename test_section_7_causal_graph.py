import sys

sys.path.insert(0, "/app/env")
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
    assert updated["B"].positive_mass() > before["B"].positive_mass() + 0.05
    assert abs(updated["C"].positive_mass() - before["C"].positive_mass()) < 0.03
    print("✓ 7.1: Propagation direction correct")


def test_7_2_propagation_carries_signal():
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
    after_A_changed = apply_positive_delta(before["A"], 0.2)
    after = {"A": after_A_changed, "B": before["B"]}
    updated = propagate_beliefs(g, before, after, n_steps=3)
    pm_B = updated["B"].positive_mass()
    print(
        f"  A: {before['A'].positive_mass():.3f}→{after_A_changed.positive_mass():.3f}, B: {before['B'].positive_mass():.3f}→{pm_B:.3f}"
    )
    assert pm_B != before["B"].positive_mass(), "B's belief must change after A changes"
    print("✓ 7.2: Propagation carries signal from A to B")


def test_7_3_damping_prevents_runaway():
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
        assert 0.0 < pm < 1.0, f"{sid} runaway: {pm}"
    print("✓ 7.3: Damping prevents runaway in dense graph")


def test_7_4_all_beliefs_normalized():
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
    print("✓ 7.4: All beliefs normalized after propagation")


def test_7_5_no_self_loops():
    from deal_room.committee.causal_graph import sample_graph

    for scenario in ["aligned", "conflicted", "hostile_acquisition"]:
        g = sample_graph(STANDARD_5, STANDARD_H, scenario, np.random.default_rng())
        for sid in STANDARD_5:
            assert g.get_weight(sid, sid) == 0.0, f"Self-loop: {sid}→{sid}"
    print("✓ 7.5: No self-loops in any scenario type")


def test_7_6_exec_sponsor_authority_invariant():
    from deal_room.committee.causal_graph import sample_graph

    for scenario in ["aligned", "conflicted", "hostile_acquisition"]:
        for seed in range(10):
            g = sample_graph(
                STANDARD_5, STANDARD_H, scenario, np.random.default_rng(seed)
            )
            outgoing = [w for w in g.get_outgoing("ExecSponsor").values() if w > 0.1]
            assert len(outgoing) >= 2, (
                f"Scenario={scenario}, seed={seed}: ExecSponsor has only {len(outgoing)} edges"
            )
    print("✓ 7.6: ExecSponsor authority invariant holds across all scenarios")


def test_7_7_hub_centrality_beats_leaf():
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
    print(f"  Hub centrality: {hub_c:.3f}, max leaf: {leaf_c:.3f}")
    assert hub_c >= leaf_c, f"Hub ({hub_c:.3f}) not >= max leaf ({leaf_c:.3f})"
    print("✓ 7.7: Hub node has highest betweenness centrality")


def test_7_8_graph_identifiability():
    from deal_room.committee.causal_graph import (
        sample_graph,
        compute_behavioral_signature,
    )

    print("  Running identifiability test (may take 30-60 seconds)...")
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
        f"CRITICAL: Only {distinguishable}/{total_pairs} graph pairs distinguishable."
    )
    print(
        "✓ 7.8: GRAPH IDENTIFIABILITY CONFIRMED — all 20 graphs pairwise distinguishable"
    )


if __name__ == "__main__":
    for fn in [
        test_7_1_propagation_direction,
        test_7_2_propagation_carries_signal,
        test_7_3_damping_prevents_runaway,
        test_7_4_all_beliefs_normalized,
        test_7_5_no_self_loops,
        test_7_6_exec_sponsor_authority_invariant,
        test_7_7_hub_centrality_beats_leaf,
        test_7_8_graph_identifiability,
    ]:
        fn()
    print("\n✓ SECTION 7 PASSED — Causal graph implementation is correct")
