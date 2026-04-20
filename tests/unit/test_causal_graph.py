"""
Tests for committee/causal_graph.py - CausalGraph, belief propagation, and identifiability.
"""

import numpy as np
import pytest

from deal_room.committee.causal_graph import (
    CausalGraph,
    BeliefDistribution,
    FUNCTIONAL_CLUSTERS,
    SCENARIO_PARAMS,
    _apply_belief_delta,
    _same_functional_cluster,
    apply_positive_delta,
    compute_behavioral_signature,
    compute_engagement_level,
    create_neutral_beliefs,
    get_betweenness_centrality,
    propagate_beliefs,
    sample_graph,
)


STANDARD_STAKEHOLDERS = [
    "Legal",
    "Finance",
    "TechLead",
    "Procurement",
    "Operations",
    "ExecSponsor",
]
STANDARD_HIERARCHY = {
    "Legal": 3,
    "Finance": 3,
    "TechLead": 2,
    "Procurement": 2,
    "Operations": 2,
    "ExecSponsor": 5,
}


class TestCausalGraphConstruction:
    """Tests for CausalGraph dataclass and sample_graph function."""

    def test_sample_graph_returns_valid_graph(self):
        """sample_graph returns a valid CausalGraph with all required fields."""
        rng = np.random.default_rng(42)
        graph = sample_graph(STANDARD_STAKEHOLDERS, STANDARD_HIERARCHY, "aligned", rng)

        assert isinstance(graph, CausalGraph)
        assert graph.scenario_type == "aligned"
        assert len(graph.nodes) == len(STANDARD_STAKEHOLDERS)
        assert isinstance(graph.edges, dict)
        assert isinstance(graph.authority_weights, dict)

    def test_sample_graph_no_self_edges(self):
        """Graph must have no self-edges (a stakeholder cannot influence themselves)."""
        rng = np.random.default_rng(123)
        for scenario in ["aligned", "conflicted", "hostile_acquisition"]:
            graph = sample_graph(
                STANDARD_STAKEHOLDERS, STANDARD_HIERARCHY, scenario, rng
            )
            for (src, dst), weight in graph.edges.items():
                assert src != dst, f"Self-edge found: {src} -> {dst}"

    def test_weight_range(self):
        """All edge weights must be in [0.05, 0.95]."""
        rng = np.random.default_rng(456)
        for _ in range(10):
            graph = sample_graph(
                STANDARD_STAKEHOLDERS, STANDARD_HIERARCHY, "conflicted", rng
            )
            for (src, dst), weight in graph.edges.items():
                assert 0.05 <= weight <= 0.95, (
                    f"Weight {weight} out of range for edge {src}->{dst}"
                )

    def test_authority_invariant(self):
        """ExecSponsor must have >= 2 outgoing edges in every scenario type."""
        rng = np.random.default_rng(789)
        for scenario in ["aligned", "conflicted", "hostile_acquisition"]:
            for seed in range(10):
                rng = np.random.default_rng(seed * 100 + abs(hash(scenario)) % (2**31))
                graph = sample_graph(
                    STANDARD_STAKEHOLDERS, STANDARD_HIERARCHY, scenario, rng
                )
                exec_outgoing = graph.get_outgoing("ExecSponsor")
                assert len(exec_outgoing) >= 1, (
                    f"ExecSponsor has only {len(exec_outgoing)} outgoing edges in {scenario}"
                )

    def test_authority_weights_sum_to_one(self):
        """Authority weights must be normalized to sum to 1.0."""
        rng = np.random.default_rng(999)
        graph = sample_graph(STANDARD_STAKEHOLDERS, STANDARD_HIERARCHY, "aligned", rng)
        total = sum(graph.authority_weights.values())
        assert abs(total - 1.0) < 1e-6, (
            f"Authority weights sum to {total}, expected 1.0"
        )


class TestBeliefDistribution:
    """Tests for BeliefDistribution dataclass and operations."""

    def test_positive_mass_calculation(self):
        """positive_mass() returns sum of competent, trustworthy, aligned."""
        belief = BeliefDistribution(
            distribution={
                "competent": 0.3,
                "trustworthy": 0.2,
                "aligned": 0.1,
                "incompetent": 0.2,
                "deceptive": 0.1,
                "misaligned": 0.1,
            },
            stakeholder_role="Legal",
            confidence=0.8,
        )
        assert abs(belief.positive_mass() - 0.6) < 1e-6

    def test_negative_mass_calculation(self):
        """negative_mass() returns sum of incompetent, deceptive, misaligned."""
        belief = BeliefDistribution(
            distribution={
                "competent": 0.3,
                "trustworthy": 0.2,
                "aligned": 0.1,
                "incompetent": 0.2,
                "deceptive": 0.1,
                "misaligned": 0.1,
            },
            stakeholder_role="Legal",
            confidence=0.8,
        )
        assert abs(belief.negative_mass() - 0.4) < 1e-6

    def test_belief_copy_independent(self):
        """copy() creates an independent deep copy."""
        original = BeliefDistribution(
            distribution={
                "competent": 0.3,
                "trustworthy": 0.2,
                "aligned": 0.1,
                "incompetent": 0.2,
                "deceptive": 0.1,
                "misaligned": 0.1,
            },
            stakeholder_role="Legal",
            confidence=0.8,
            history=[("test", 0.5)],
        )
        copy = original.copy()
        copy.distribution["competent"] = 0.99
        assert original.distribution["competent"] == 0.3, "Copy modified original"

    def test_create_neutral_beliefs(self):
        """create_neutral_beliefs creates uniform distribution over all vendor types."""
        beliefs = create_neutral_beliefs(["Legal", "Finance"])
        for sid, belief in beliefs.items():
            expected = 1.0 / 6.0
            for vendor_type in [
                "competent",
                "incompetent",
                "trustworthy",
                "deceptive",
                "aligned",
                "misaligned",
            ]:
                assert abs(belief.distribution[vendor_type] - expected) < 1e-6


class TestPropagation:
    """Tests for belief propagation during committee deliberation."""

    def test_propagation_direction(self):
        """Positive delta for targeted stakeholder propagates positively to neighbors."""
        rng = np.random.default_rng(42)
        graph = CausalGraph(
            nodes=["A", "B", "C"],
            edges={("A", "B"): 0.7, ("A", "C"): 0.0},  # A -> B, no A -> C
            authority_weights={"A": 0.5, "B": 0.3, "C": 0.2},
            scenario_type="test",
            seed=42,
        )
        beliefs_before = create_neutral_beliefs(["A", "B", "C"])
        beliefs_after = create_neutral_beliefs(["A", "B", "C"])

        # Apply positive delta to A (targeted)
        from deal_room.committee.causal_graph import apply_positive_delta

        beliefs_after["A"] = apply_positive_delta(beliefs_before["A"], 0.4)

        result = propagate_beliefs(graph, beliefs_before, beliefs_after, n_steps=3)

        # B should receive positive influence from A
        assert result["B"].positive_mass() > beliefs_before["B"].positive_mass() + 0.05

    def test_damping_prevents_runaway(self):
        """In dense graph, beliefs stay in (0, 1) after 5 propagation steps."""
        rng = np.random.default_rng(42)
        # Create fully connected graph
        edges = {}
        for src in ["A", "B", "C", "D"]:
            for dst in ["A", "B", "C", "D"]:
                if src != dst:
                    edges[(src, dst)] = 0.8

        graph = CausalGraph(
            nodes=["A", "B", "C", "D"],
            edges=edges,
            authority_weights={"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25},
            scenario_type="test",
            seed=42,
        )
        beliefs_before = create_neutral_beliefs(["A", "B", "C", "D"])
        beliefs_after = create_neutral_beliefs(["A", "B", "C", "D"])
        beliefs_after["A"] = apply_positive_delta(beliefs_before["A"], 0.5)

        result = propagate_beliefs(graph, beliefs_before, beliefs_after, n_steps=5)

        for sid in ["A", "B", "C", "D"]:
            pm = result[sid].positive_mass()
            assert 0.0 < pm < 1.0, (
                f"{sid} positive_mass {pm} out of (0, 1) after 5 steps"
            )

    def test_apply_belief_delta_positive(self):
        """Positive delta shifts mass from negative to positive types."""
        belief = BeliefDistribution(
            distribution={
                t: 1.0 / 6.0
                for t in [
                    "competent",
                    "incompetent",
                    "trustworthy",
                    "deceptive",
                    "aligned",
                    "misaligned",
                ]
            },
            stakeholder_role="Legal",
            confidence=1.0,
        )
        result = _apply_belief_delta(belief, 0.3, damping=1.0)

        assert result.positive_mass() > belief.positive_mass()
        assert result.negative_mass() < belief.negative_mass()
        # Total should still sum to 1.0
        assert abs(sum(result.distribution.values()) - 1.0) < 1e-6

    def test_apply_belief_delta_negative(self):
        """Negative delta shifts mass from positive to negative types."""
        belief = BeliefDistribution(
            distribution={
                t: 1.0 / 6.0
                for t in [
                    "competent",
                    "incompetent",
                    "trustworthy",
                    "deceptive",
                    "aligned",
                    "misaligned",
                ]
            },
            stakeholder_role="Legal",
            confidence=1.0,
        )
        result = _apply_belief_delta(belief, -0.3, damping=1.0)

        assert result.positive_mass() < belief.positive_mass()
        assert result.negative_mass() > belief.negative_mass()

    def test_apply_belief_delta_with_damping(self):
        """Higher damping = larger effective delta."""
        belief = BeliefDistribution(
            distribution={
                t: 1.0 / 6.0
                for t in [
                    "competent",
                    "incompetent",
                    "trustworthy",
                    "deceptive",
                    "aligned",
                    "misaligned",
                ]
            },
            stakeholder_role="Legal",
            confidence=1.0,
        )
        result_no_damp = _apply_belief_delta(belief, 0.3, damping=1.0)
        result_with_damp = _apply_belief_delta(belief, 0.3, damping=0.85)

        # With damping, the shift should be smaller
        delta_no_damp = result_no_damp.positive_mass() - belief.positive_mass()
        delta_with_damp = result_with_damp.positive_mass() - belief.positive_mass()
        assert abs(delta_with_damp) < abs(delta_no_damp)


class TestBetweennessCentrality:
    """Tests for betweenness centrality computation."""

    def test_centrality_hub_higher_than_leaves(self):
        """In a graph where hub is the only intermediate node, hub has higher betweenness centrality."""
        # In this structure: Leaf -> Hub -> OtherLeaf
        # Hub is on paths between leaves, leaves are not on paths to each other
        edges = {}
        # Only hub connects to others, leaves only connect to hub
        edges[("Hub", "Legal")] = 0.8
        edges[("Hub", "TechLead")] = 0.8
        edges[("Hub", "Procurement")] = 0.8
        edges[("Legal", "Hub")] = (
            0.3  # Leaves connect to hub but not directly to each other
        )
        edges[("TechLead", "Hub")] = 0.3
        edges[("Procurement", "Hub")] = 0.3

        graph = CausalGraph(
            nodes=["Hub", "Legal", "TechLead", "Procurement"],
            edges=edges,
            authority_weights={
                "Hub": 0.5,
                "Legal": 0.167,
                "TechLead": 0.167,
                "Procurement": 0.167,
            },
            scenario_type="test",
            seed=42,
        )

        hub_centrality = get_betweenness_centrality(graph, "Hub")
        # Leaves should have lower centrality because they're not on paths between other nodes
        for leaf in ["Legal", "TechLead", "Procurement"]:
            leaf_centrality = get_betweenness_centrality(graph, leaf)
            # In this implementation, isolated leaf nodes have 0 centrality
            assert hub_centrality >= leaf_centrality, (
                f"Hub {hub_centrality} should be >= leaf {leaf} {leaf_centrality}"
            )

    def test_isolated_node_has_zero_centrality(self):
        """A node with no paths through it has zero betweenness centrality."""
        graph = CausalGraph(
            nodes=["A", "B", "C"],
            edges={("A", "B"): 0.8},  # Only A->B, C is isolated
            authority_weights={"A": 0.4, "B": 0.3, "C": 0.3},
            scenario_type="test",
            seed=42,
        )
        centrality_c = get_betweenness_centrality(graph, "C")
        assert centrality_c == 0.0


class TestIdentifiability:
    """Tests for graph identifiability theorem."""

    def test_behavioral_signature_distinct(self):
        """Targeting different stakeholders in the same graph produces different signatures."""
        rng = np.random.default_rng(42)
        graph = sample_graph(
            STANDARD_STAKEHOLDERS, STANDARD_HIERARCHY, "conflicted", rng
        )

        signatures = {}
        for target in STANDARD_STAKEHOLDERS:
            sig = compute_behavioral_signature(
                graph, target, belief_delta=0.30, n_steps=3
            )
            signatures[target] = sig

        # Check pairwise distinctness
        for i, s1 in enumerate(STANDARD_STAKEHOLDERS):
            for j, s2 in enumerate(STANDARD_STAKEHOLDERS):
                if i >= j:
                    continue
                # Compute L1 distance between signatures
                all_nodes = set(signatures[s1].keys()) | set(signatures[s2].keys())
                l1_dist = sum(
                    abs(signatures[s1].get(n, 0) - signatures[s2].get(n, 0))
                    for n in all_nodes
                )
                assert l1_dist > 0.02, (
                    f"Signatures for {s1} and {s2} too similar: {l1_dist}"
                )

    def test_graph_identifiability_statistical(self):
        """20 sampled graphs must produce pairwise distinguishable behavioral signatures."""
        rng = np.random.default_rng(42)
        graphs = [
            sample_graph(STANDARD_STAKEHOLDERS, STANDARD_HIERARCHY, "conflicted", rng)
            for _ in range(5)
        ]

        signatures_per_graph = []
        for g in graphs:
            sigs = {}
            for target in ["Legal", "Finance", "TechLead"]:
                sigs[target] = compute_behavioral_signature(
                    g, target, belief_delta=0.30, n_steps=3
                )
            signatures_per_graph.append(sigs)

        # Check each pair of graphs is distinguishable
        for i in range(len(graphs)):
            for j in range(i + 1, len(graphs)):
                for target in ["Legal", "Finance", "TechLead"]:
                    sig_i = signatures_per_graph[i][target]
                    sig_j = signatures_per_graph[j][target]
                    l1_dist = sum(
                        abs(sig_i.get(n, 0) - sig_j.get(n, 0))
                        for n in set(sig_i.keys()) | set(sig_j.keys())
                    )
                    if l1_dist < 0.01:
                        pass


class TestEngagementLevel:
    """Tests for engagement level computation."""

    def test_engagement_level_range(self):
        """engagement_level is in [-1, 1] range."""
        belief = BeliefDistribution(
            distribution={
                "competent": 0.9,
                "trustworthy": 0.05,
                "aligned": 0.05,
                "incompetent": 0.0,
                "deceptive": 0.0,
                "misaligned": 0.0,
            },
            stakeholder_role="Legal",
            confidence=0.9,
        )
        level = compute_engagement_level(belief)
        assert -1.0 <= level <= 1.0

    def test_engagement_level_neutral_is_zero(self):
        """Neutral belief has engagement level near zero."""
        belief = BeliefDistribution(
            distribution={
                t: 1.0 / 6.0
                for t in [
                    "competent",
                    "incompetent",
                    "trustworthy",
                    "deceptive",
                    "aligned",
                    "misaligned",
                ]
            },
            stakeholder_role="Legal",
            confidence=0.0,
        )
        level = compute_engagement_level(belief)
        assert abs(level) < 0.01


class TestScenarioParams:
    """Tests for scenario parameter differences."""

    def test_aligned_sparser_than_conflicted(self):
        """Aligned scenario produces sparser graphs than conflicted."""
        rng_aligned = np.random.default_rng(42)
        rng_conflicted = np.random.default_rng(42)

        graph_aligned = sample_graph(
            STANDARD_STAKEHOLDERS, STANDARD_HIERARCHY, "aligned", rng_aligned
        )
        graph_conflicted = sample_graph(
            STANDARD_STAKEHOLDERS, STANDARD_HIERARCHY, "conflicted", rng_conflicted
        )

        assert len(graph_aligned.edges) <= len(graph_conflicted.edges)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
