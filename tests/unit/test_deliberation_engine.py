"""
Tests for committee/deliberation_engine.py - Two-layer deliberation engine.
"""

import numpy as np
import pytest

from deal_room.committee.causal_graph import (
    CausalGraph,
    BeliefDistribution,
    create_neutral_beliefs,
    propagate_beliefs,
)
from deal_room.committee.deliberation_engine import (
    DELIBERATION_STEPS,
    CommitteeDeliberationEngine,
    DeliberationResult,
)
from models import DealRoomAction


STANDARD_STAKEHOLDERS = [
    "Legal",
    "Finance",
    "TechLead",
    "Procurement",
    "Operations",
    "ExecSponsor",
]


class TestDeliberationEngine:
    """Tests for CommitteeDeliberationEngine."""

    def test_deliberation_result_structure(self):
        """DeliberationResult has all required fields."""
        result = DeliberationResult(
            updated_beliefs={},
            summary_dialogue="test",
            propagation_deltas={},
        )

        assert hasattr(result, "updated_beliefs")
        assert hasattr(result, "summary_dialogue")
        assert hasattr(result, "propagation_deltas")

    def test_deliberation_steps_per_scenario(self):
        """Each scenario type has correct number of deliberation steps."""
        assert DELIBERATION_STEPS["aligned"] == 3
        assert DELIBERATION_STEPS["conflicted"] == 3
        assert DELIBERATION_STEPS["hostile_acquisition"] == 4

    def test_deliberation_updates_beliefs(self):
        """Deliberation engine updates beliefs through propagation."""
        rng = np.random.default_rng(42)

        graph = CausalGraph(
            nodes=["A", "B", "C"],
            edges={("A", "B"): 0.7, ("B", "C"): 0.5},
            authority_weights={"A": 0.4, "B": 0.3, "C": 0.3},
            scenario_type="aligned",
            seed=42,
        )

        beliefs_before = create_neutral_beliefs(["A", "B", "C"])
        beliefs_after = create_neutral_beliefs(["A", "B", "C"])

        # Apply positive delta to A
        from deal_room.committee.causal_graph import apply_positive_delta

        beliefs_after["A"] = apply_positive_delta(beliefs_after["A"], 0.4)

        engine = CommitteeDeliberationEngine(graph=graph, n_deliberation_steps=3)

        action = DealRoomAction(
            action_type="direct_message",
            target_ids=["A"],
            message="Test",
        )

        result = engine.run(
            vendor_action=action,
            beliefs_before_action=beliefs_before,
            beliefs_after_vendor_action=beliefs_after,
            render_summary=False,
        )

        assert isinstance(result, DeliberationResult)
        assert len(result.updated_beliefs) == 3

    def test_deliberation_propagation_deltas_recorded(self):
        """propagation_deltas records the belief change for each stakeholder."""
        rng = np.random.default_rng(42)

        graph = CausalGraph(
            nodes=["A", "B"],
            edges={("A", "B"): 0.8},
            authority_weights={"A": 0.5, "B": 0.5},
            scenario_type="aligned",
            seed=42,
        )

        beliefs_before = create_neutral_beliefs(["A", "B"])
        beliefs_after = create_neutral_beliefs(["A", "B"])

        from deal_room.committee.causal_graph import apply_positive_delta

        beliefs_after["A"] = apply_positive_delta(beliefs_after["A"], 0.3)

        engine = CommitteeDeliberationEngine(graph=graph, n_deliberation_steps=3)

        action = DealRoomAction(
            action_type="direct_message", target_ids=["A"], message=""
        )

        result = engine.run(
            vendor_action=action,
            beliefs_before_action=beliefs_before,
            beliefs_after_vendor_action=beliefs_after,
            render_summary=False,
        )

        assert "A" in result.propagation_deltas
        assert "B" in result.propagation_deltas

    def test_deliberation_no_summary_when_no_targets(self):
        """No summary generated when vendor action has no targets."""
        graph = CausalGraph(
            nodes=["A", "B"],
            edges={("A", "B"): 0.5},
            authority_weights={"A": 0.5, "B": 0.5},
            scenario_type="aligned",
            seed=42,
        )

        beliefs_before = create_neutral_beliefs(["A", "B"])
        beliefs_after = create_neutral_beliefs(["A", "B"])

        engine = CommitteeDeliberationEngine(graph=graph)

        action = DealRoomAction(action_type="direct_message", target_ids=[], message="")

        result = engine.run(
            vendor_action=action,
            beliefs_before_action=beliefs_before,
            beliefs_after_vendor_action=beliefs_after,
            render_summary=True,
        )

        # Summary should be None or empty when no targets
        assert result.summary_dialogue is None or result.summary_dialogue == ""

    def test_deliberation_pure_python_layer1(self):
        """Layer 1 (propagate_beliefs) runs without LLM calls."""
        graph = CausalGraph(
            nodes=["A", "B", "C"],
            edges={("A", "B"): 0.6, ("A", "C"): 0.4},
            authority_weights={"A": 0.4, "B": 0.3, "C": 0.3},
            scenario_type="aligned",
            seed=42,
        )

        beliefs_before = create_neutral_beliefs(["A", "B", "C"])
        beliefs_after = create_neutral_beliefs(["A", "B", "C"])

        from deal_room.committee.causal_graph import apply_positive_delta

        beliefs_after["A"] = apply_positive_delta(beliefs_after["A"], 0.3)

        engine = CommitteeDeliberationEngine(graph=graph, n_deliberation_steps=3)

        # Just call propagate_beliefs directly (Layer 1)
        result = propagate_beliefs(graph, beliefs_before, beliefs_after, n_steps=3)

        assert len(result) == 3
        # Verify it worked (B and C should be affected by A)
        assert result["B"].positive_mass() != beliefs_before["B"].positive_mass()


class TestLayer2Summary:
    """Tests for Layer 2 summary generation (MiniMax calls)."""

    def test_layer2_returns_string_or_empty(self):
        """Layer 2 returns string (summary) or empty string on failure."""
        graph = CausalGraph(
            nodes=["A", "B", "C"],
            edges={("A", "B"): 0.7, ("A", "C"): 0.5, ("B", "C"): 0.3},
            authority_weights={"A": 0.5, "B": 0.3, "C": 0.2},
            scenario_type="aligned",
            seed=42,
        )

        beliefs_before = create_neutral_beliefs(["A", "B", "C"])
        beliefs_after = create_neutral_beliefs(["A", "B", "C"])

        from deal_room.committee.causal_graph import apply_positive_delta

        beliefs_after["A"] = apply_positive_delta(beliefs_after["A"], 0.4)

        engine = CommitteeDeliberationEngine(graph=graph, n_deliberation_steps=3)

        action = DealRoomAction(
            action_type="direct_message", target_ids=["A"], message=""
        )

        # Layer 2 is called with render_summary=True
        # It may return empty string if LLM call fails
        result = engine.run(
            vendor_action=action,
            beliefs_before_action=beliefs_before,
            beliefs_after_vendor_action=beliefs_after,
            render_summary=True,
        )

        # Should return a DeliberationResult
        assert isinstance(result, DeliberationResult)
        # Summary should be string (possibly empty)
        assert isinstance(result.summary_dialogue, (str, type(None)))


class TestDeliberationSteps:
    """Tests for configurable deliberation steps."""

    def test_single_step_deliberation(self):
        """Deliberation with 1 step still produces valid results."""
        graph = CausalGraph(
            nodes=["A", "B"],
            edges={("A", "B"): 0.8},
            authority_weights={"A": 0.5, "B": 0.5},
            scenario_type="aligned",
            seed=42,
        )

        beliefs_before = create_neutral_beliefs(["A", "B"])
        beliefs_after = create_neutral_beliefs(["A", "B"])

        from deal_room.committee.causal_graph import apply_positive_delta

        beliefs_after["A"] = apply_positive_delta(beliefs_after["A"], 0.3)

        engine = CommitteeDeliberationEngine(graph=graph, n_deliberation_steps=1)

        action = DealRoomAction(
            action_type="direct_message", target_ids=["A"], message=""
        )

        result = engine.run(
            vendor_action=action,
            beliefs_before_action=beliefs_before,
            beliefs_after_vendor_action=beliefs_after,
            render_summary=False,
        )

        assert len(result.updated_beliefs) == 2

    def test_many_steps_damping(self):
        """With many steps, damping prevents runaway propagation."""
        graph = CausalGraph(
            nodes=["A", "B", "C"],
            edges={("A", "B"): 0.9, ("B", "C"): 0.9},
            authority_weights={"A": 0.5, "B": 0.3, "C": 0.2},
            scenario_type="aligned",
            seed=42,
        )

        beliefs_before = create_neutral_beliefs(["A", "B", "C"])
        beliefs_after = create_neutral_beliefs(["A", "B", "C"])

        from deal_room.committee.causal_graph import apply_positive_delta

        beliefs_after["A"] = apply_positive_delta(beliefs_after["A"], 0.5)

        # 10 steps with heavy damping
        engine = CommitteeDeliberationEngine(graph=graph, n_deliberation_steps=10)

        action = DealRoomAction(
            action_type="direct_message", target_ids=["A"], message=""
        )

        result = engine.run(
            vendor_action=action,
            beliefs_before_action=beliefs_before,
            beliefs_after_vendor_action=beliefs_after,
            render_summary=False,
        )

        # Even with 10 steps, beliefs should stay in valid range
        for sid in ["A", "B", "C"]:
            pm = result.updated_beliefs[sid].positive_mass()
            assert 0.0 < pm < 1.0


class TestBeliefPropagationIntegration:
    """Integration tests for deliberation + propagation."""

    def test_propagation_follows_graph_structure(self):
        """Belief changes follow the graph edge structure."""
        # A->B->C chain, no A->C
        graph = CausalGraph(
            nodes=["A", "B", "C"],
            edges={("A", "B"): 0.8, ("B", "C"): 0.8},
            authority_weights={"A": 0.4, "B": 0.4, "C": 0.2},
            scenario_type="aligned",
            seed=42,
        )

        beliefs_before = create_neutral_beliefs(["A", "B", "C"])
        beliefs_after = create_neutral_beliefs(["A", "B", "C"])

        from deal_room.committee.causal_graph import apply_positive_delta

        beliefs_after["A"] = apply_positive_delta(beliefs_after["A"], 0.5)

        # B is directly connected to A, C is not
        result = propagate_beliefs(graph, beliefs_before, beliefs_after, n_steps=3)

        # B should have more change than C (closer to A in graph)
        b_change = abs(
            result["B"].positive_mass() - beliefs_before["B"].positive_mass()
        )
        c_change = abs(
            result["C"].positive_mass() - beliefs_before["C"].positive_mass()
        )

        assert b_change > c_change, (
            "B should change more than C (directly connected to A)"
        )

    def test_no_edges_no_propagation(self):
        """With no edges, only targeted stakeholder changes."""
        graph = CausalGraph(
            nodes=["A", "B"],
            edges={},  # No edges
            authority_weights={"A": 0.5, "B": 0.5},
            scenario_type="aligned",
            seed=42,
        )

        beliefs_before = create_neutral_beliefs(["A", "B"])
        beliefs_after = create_neutral_beliefs(["A", "B"])

        from deal_room.committee.causal_graph import apply_positive_delta

        beliefs_after["A"] = apply_positive_delta(beliefs_after["A"], 0.4)

        result = propagate_beliefs(graph, beliefs_before, beliefs_after, n_steps=3)

        # B should be unchanged (no edges)
        assert (
            abs(result["B"].positive_mass() - beliefs_before["B"].positive_mass())
            < 0.001
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
