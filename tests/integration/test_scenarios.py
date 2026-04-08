"""
Integration tests for scenario configurations and scenarios module.
Tests all three scenarios and their parameters.
"""

import pytest
from server.scenarios import SCENARIOS, STAKEHOLDER_IDS, expand_targets


class TestScenariosExist:
    """Test that all required scenarios exist."""

    def test_aligned_exists(self):
        assert "aligned" in SCENARIOS

    def test_conflicted_exists(self):
        assert "conflicted" in SCENARIOS

    def test_hostile_acquisition_exists(self):
        assert "hostile_acquisition" in SCENARIOS


class TestScenarioStructure:
    """Test that each scenario has all required keys."""

    def test_aligned_keys(self):
        scenario = SCENARIOS["aligned"]
        required_keys = [
            "max_rounds",
            "veto_threshold",
            "block_threshold",
            "shock_prob",
            "round_3_hint",
            "days_to_deadline",
            "initial_beliefs",
            "initial_satisfaction",
            "coalition_tension",
            "description",
        ]
        for key in required_keys:
            assert key in scenario, f"aligned missing {key}"

    def test_conflicted_keys(self):
        scenario = SCENARIOS["conflicted"]
        required_keys = [
            "max_rounds",
            "veto_threshold",
            "block_threshold",
            "shock_prob",
            "round_3_hint",
            "days_to_deadline",
            "initial_beliefs",
            "initial_satisfaction",
            "coalition_tension",
            "description",
        ]
        for key in required_keys:
            assert key in scenario, f"conflicted missing {key}"

    def test_hostile_acquisition_keys(self):
        scenario = SCENARIOS["hostile_acquisition"]
        required_keys = [
            "max_rounds",
            "veto_threshold",
            "block_threshold",
            "shock_prob",
            "round_3_hint",
            "days_to_deadline",
            "initial_beliefs",
            "initial_satisfaction",
            "coalition_tension",
            "description",
        ]
        for key in required_keys:
            assert key in scenario, f"hostile_acquisition missing {key}"


class TestScenarioBeliefs:
    """Test initial beliefs for each scenario."""

    def test_all_stakeholders_have_beliefs(self):
        for task_id, scenario in SCENARIOS.items():
            for sid in STAKEHOLDER_IDS:
                assert sid in scenario["initial_beliefs"], (
                    f"{task_id} missing {sid} beliefs"
                )

    def test_beliefs_have_required_dimensions(self):
        required_dims = {"competence", "risk_tolerance", "pricing_rigor"}
        for task_id, scenario in SCENARIOS.items():
            for sid, beliefs in scenario["initial_beliefs"].items():
                assert required_dims.issubset(beliefs.keys()), (
                    f"{task_id}/{sid} missing dimensions"
                )

    def test_belief_values_in_range(self):
        for task_id, scenario in SCENARIOS.items():
            for sid, beliefs in scenario["initial_beliefs"].items():
                for dim, value in beliefs.items():
                    assert 0.0 <= value <= 1.0, (
                        f"{task_id}/{sid}.{dim} = {value} out of range"
                    )


class TestScenarioSatisfaction:
    """Test initial satisfaction for each scenario."""

    def test_all_stakeholders_have_satisfaction(self):
        for task_id, scenario in SCENARIOS.items():
            for sid in STAKEHOLDER_IDS:
                assert sid in scenario["initial_satisfaction"], (
                    f"{task_id} missing {sid} satisfaction"
                )

    def test_satisfaction_values_in_range(self):
        for task_id, scenario in SCENARIOS.items():
            for sid, sat in scenario["initial_satisfaction"].items():
                assert 0.0 <= sat <= 1.0, (
                    f"{task_id}/{sid} satisfaction = {sat} out of range"
                )


class TestScenarioVetoThresholds:
    """Test veto thresholds are properly set."""

    def test_thresholds_decrease_with_difficulty(self):
        aligned_threshold = SCENARIOS["aligned"]["veto_threshold"]
        conflicted_threshold = SCENARIOS["conflicted"]["veto_threshold"]
        hostile_threshold = SCENARIOS["hostile_acquisition"]["veto_threshold"]

        # More difficult scenarios should have lower thresholds (easier to veto)
        assert aligned_threshold > conflicted_threshold, (
            "aligned should have higher threshold than conflicted"
        )
        assert conflicted_threshold > hostile_threshold, (
            "conflicted should have higher threshold than hostile"
        )

    def test_thresholds_in_valid_range(self):
        for task_id, scenario in SCENARIOS.items():
            threshold = scenario["veto_threshold"]
            assert 0.0 < threshold < 1.0, (
                f"{task_id} veto threshold {threshold} invalid"
            )


class TestScenarioBlockThresholds:
    """Test block thresholds."""

    def test_block_thresholds_decrease_with_difficulty(self):
        aligned = SCENARIOS["aligned"]["block_threshold"]
        conflicted = SCENARIOS["conflicted"]["block_threshold"]
        hostile = SCENARIOS["hostile_acquisition"]["block_threshold"]

        # Aligned has lowest threshold (easier to create blockers)
        # Conflicted and hostile have higher thresholds
        assert aligned < conflicted
        assert aligned < hostile

    def test_block_thresholds_in_valid_range(self):
        for task_id, scenario in SCENARIOS.items():
            threshold = scenario["block_threshold"]
            assert 0.0 < threshold < 1.0


class TestScenarioShockProbability:
    """Test shock probabilities."""

    def test_shock_prob_decreases_with_difficulty(self):
        aligned = SCENARIOS["aligned"]["shock_prob"]
        conflicted = SCENARIOS["conflicted"]["shock_prob"]
        hostile = SCENARIOS["hostile_acquisition"]["shock_prob"]

        assert aligned < conflicted < hostile

    def test_shock_probs_in_valid_range(self):
        for task_id, scenario in SCENARIOS.items():
            prob = scenario["shock_prob"]
            assert 0.0 <= prob <= 1.0


class TestScenarioDeadlines:
    """Test deadline configurations."""

    def test_days_to_deadline_decreases_with_difficulty(self):
        aligned = SCENARIOS["aligned"]["days_to_deadline"]
        conflicted = SCENARIOS["conflicted"]["days_to_deadline"]
        hostile = SCENARIOS["hostile_acquisition"]["days_to_deadline"]

        assert aligned > conflicted > hostile

    def test_deadlines_reasonable(self):
        for task_id, scenario in SCENARIOS.items():
            days = scenario["days_to_deadline"]
            assert 5 < days < 60, f"{task_id} deadline {days} unreasonable"


class TestScenarioMaxRounds:
    """Test max rounds configuration."""

    def test_aligned_has_fewer_rounds(self):
        assert SCENARIOS["aligned"]["max_rounds"] == 8
        assert SCENARIOS["conflicted"]["max_rounds"] == 10
        assert SCENARIOS["hostile_acquisition"]["max_rounds"] == 10


class TestScenarioCoalitionTensions:
    """Test coalition tension configurations."""

    def test_aligned_no_tension(self):
        assert SCENARIOS["aligned"]["coalition_tension"] is None

    def test_conflicted_has_tension(self):
        tension = SCENARIOS["conflicted"]["coalition_tension"]
        assert tension is not None
        assert "cto_cfo" in tension
        assert tension["cto_cfo"] == "conflict"

    def test_hostile_acquisition_has_tension(self):
        tension = SCENARIOS["hostile_acquisition"]["coalition_tension"]
        assert tension is not None
        assert "cto_cfo" in tension


class TestScenarioDescriptions:
    """Test that descriptions exist and are meaningful."""

    def test_all_have_descriptions(self):
        for task_id, scenario in SCENARIOS.items():
            assert "description" in scenario
            assert len(scenario["description"]) > 20


class TestScenarioHints:
    """Test scenario hints."""

    def test_aligned_has_no_hint(self):
        assert SCENARIOS["aligned"]["round_3_hint"] is None

    def test_hostile_acquisition_has_hint(self):
        hint = SCENARIOS["hostile_acquisition"]["round_3_hint"]
        assert hint is not None
        assert len(hint) > 10
        assert "GDPR" in hint or "compliance" in hint.lower()


class TestStakeholderIdsConstant:
    """Test STAKEHOLDER_IDS constant."""

    def test_has_five_ids(self):
        assert len(STAKEHOLDER_IDS) == 5

    def test_has_expected_ids(self):
        expected = ["CFO", "CTO", "Legal", "Procurement", "Ops"]
        assert set(STAKEHOLDER_IDS) == set(expected)


class TestExpandTargetsInScenarios:
    """Test expand_targets function with scenario data."""

    def test_expand_targets_from_scenario(self):
        from server.claims import expand_targets

        assert expand_targets("all") == STAKEHOLDER_IDS

    def test_expand_subgroups(self):
        from server.claims import expand_targets

        assert "CTO" in expand_targets("cto_cfo")
        assert "CFO" in expand_targets("cto_cfo")
        assert "Legal" in expand_targets("legal_procurement")
        assert "Procurement" in expand_targets("legal_procurement")
