"""
Unit tests for StakeholderEngine and stakeholder behavior.
Tests response generation, belief updates, and stance transitions.
"""

import pytest
import numpy as np
from copy import deepcopy
from models import DealRoomState
from server.stakeholders import StakeholderEngine, STAKEHOLDER_TEMPLATES


class TestStakeholderEngineInit:
    """Test StakeholderEngine initialization."""

    def test_initialization(self):
        engine = StakeholderEngine()
        assert engine.state is None
        assert engine.rng is None
        assert len(engine.STAKEHOLDER_IDS) == 5

    def test_stakeholder_ids(self, stakeholder_ids):
        engine = StakeholderEngine()
        assert set(engine.STAKEHOLDER_IDS) == set(stakeholder_ids)


class TestStakeholderEngineReset:
    """Test StakeholderEngine reset and initialization."""

    def test_reset_with_scenario(self, rng):
        engine = StakeholderEngine()
        state = DealRoomState(
            beliefs={
                "CFO": {
                    "competence": 0.55,
                    "risk_tolerance": 0.52,
                    "pricing_rigor": 0.50,
                },
                "CTO": {
                    "competence": 0.58,
                    "risk_tolerance": 0.55,
                    "pricing_rigor": 0.48,
                },
                "Legal": {
                    "competence": 0.50,
                    "risk_tolerance": 0.45,
                    "pricing_rigor": 0.52,
                },
                "Procurement": {
                    "competence": 0.53,
                    "risk_tolerance": 0.50,
                    "pricing_rigor": 0.55,
                },
                "Ops": {
                    "competence": 0.60,
                    "risk_tolerance": 0.58,
                    "pricing_rigor": 0.45,
                },
            },
            satisfaction={
                "CFO": 0.54,
                "CTO": 0.56,
                "Legal": 0.48,
                "Procurement": 0.52,
                "Ops": 0.60,
            },
        )
        scenario = {"description": "test"}
        engine.reset(state, rng, scenario)
        assert engine.state == state
        assert engine.rng == rng


class TestStakeholderEngineGenerateOpening:
    """Test opening message generation."""

    def test_generate_opening_returns_dict(self):
        engine = StakeholderEngine()
        state = DealRoomState(
            beliefs={
                sid: {"competence": 0.5, "risk_tolerance": 0.5, "pricing_rigor": 0.5}
                for sid in ["CFO", "CTO", "Legal", "Procurement", "Ops"]
            },
            satisfaction={
                sid: 0.5 for sid in ["CFO", "CTO", "Legal", "Procurement", "Ops"]
            },
        )
        rng = np.random.default_rng(42)
        engine.reset(state, rng, {"description": "test"})
        opening = engine.generate_opening()
        assert isinstance(opening, dict)
        assert len(opening) == 5

    def test_opening_has_all_stakeholders(self):
        engine = StakeholderEngine()
        state = DealRoomState(
            beliefs={
                sid: {"competence": 0.5, "risk_tolerance": 0.5, "pricing_rigor": 0.5}
                for sid in ["CFO", "CTO", "Legal", "Procurement", "Ops"]
            },
            satisfaction={
                sid: 0.5 for sid in ["CFO", "CTO", "Legal", "Procurement", "Ops"]
            },
        )
        rng = np.random.default_rng(42)
        engine.reset(state, rng, {"description": "test"})
        opening = engine.generate_opening()
        for sid in ["CFO", "CTO", "Legal", "Procurement", "Ops"]:
            assert sid in opening
            assert len(opening[sid]) > 0

    def test_opening_messages_are_strings(self):
        engine = StakeholderEngine()
        state = DealRoomState(
            beliefs={
                sid: {"competence": 0.5, "risk_tolerance": 0.5, "pricing_rigor": 0.5}
                for sid in ["CFO", "CTO", "Legal", "Procurement", "Ops"]
            },
            satisfaction={
                sid: 0.5 for sid in ["CFO", "CTO", "Legal", "Procurement", "Ops"]
            },
        )
        rng = np.random.default_rng(42)
        engine.reset(state, rng, {"description": "test"})
        opening = engine.generate_opening()
        for msg in opening.values():
            assert isinstance(msg, str)


class TestStakeholderEngineApplyAction:
    """Test action application and belief updates."""

    def test_apply_direct_message_updates_contact(self):
        engine = StakeholderEngine()
        state = DealRoomState(
            beliefs={
                sid: {"competence": 0.5, "risk_tolerance": 0.5, "pricing_rigor": 0.5}
                for sid in ["CFO", "CTO", "Legal", "Procurement", "Ops"]
            },
            satisfaction={
                sid: 0.5 for sid in ["CFO", "CTO", "Legal", "Procurement", "Ops"]
            },
            trust_floors={
                sid: 0.2 for sid in ["CFO", "CTO", "Legal", "Procurement", "Ops"]
            },
            permanent_marks={
                sid: [] for sid in ["CFO", "CTO", "Legal", "Procurement", "Ops"]
            },
            rounds_since_last_contact={
                sid: 1 for sid in ["CFO", "CTO", "Legal", "Procurement", "Ops"]
            },
        )
        rng = np.random.default_rng(42)
        engine.reset(state, rng, {"description": "test"})

        action = {
            "action_type": "direct_message",
            "target": "CFO",
            "message": "Thank you for your time. We appreciate your partnership.",
        }
        engine.apply_action(action, rng)

        # Contacted stakeholder should have rounds_since_last_contact reset to 0
        assert state.rounds_since_last_contact["CFO"] == 0
        # Non-contacted stakeholders should have incremented
        assert state.rounds_since_last_contact["CTO"] == 2

    def test_apply_document_effects(self):
        engine = StakeholderEngine()
        state = DealRoomState(
            beliefs={
                sid: {"competence": 0.5, "risk_tolerance": 0.5, "pricing_rigor": 0.5}
                for sid in ["CFO", "CTO", "Legal", "Procurement", "Ops"]
            },
            satisfaction={
                sid: 0.5 for sid in ["CFO", "CTO", "Legal", "Procurement", "Ops"]
            },
            trust_floors={
                sid: 0.2 for sid in ["CFO", "CTO", "Legal", "Procurement", "Ops"]
            },
            permanent_marks={
                sid: [] for sid in ["CFO", "CTO", "Legal", "Procurement", "Ops"]
            },
            rounds_since_last_contact={
                sid: 0 for sid in ["CFO", "CTO", "Legal", "Procurement", "Ops"]
            },
        )
        rng = np.random.default_rng(42)
        engine.reset(state, rng, {"description": "test"})

        initial_sat = state.satisfaction["CFO"]
        action = {
            "action_type": "send_document",
            "target": "CFO",
            "message": "Here is our ROI model.",
            "documents": [{"type": "roi_model", "specificity": "high"}],
        }
        engine.apply_action(action, rng)

        # High specificity roi_model to CFO should boost satisfaction
        assert state.satisfaction["CFO"] >= initial_sat

    def test_apply_rapport_collaborative(self):
        engine = StakeholderEngine()
        state = DealRoomState(
            beliefs={
                sid: {"competence": 0.5, "risk_tolerance": 0.5, "pricing_rigor": 0.5}
                for sid in ["CFO", "CTO", "Legal", "Procurement", "Ops"]
            },
            satisfaction={
                sid: 0.5 for sid in ["CFO", "CTO", "Legal", "Procurement", "Ops"]
            },
            trust_floors={
                sid: 0.2 for sid in ["CFO", "CTO", "Legal", "Procurement", "Ops"]
            },
            permanent_marks={
                sid: [] for sid in ["CFO", "CTO", "Legal", "Procurement", "Ops"]
            },
            rounds_since_last_contact={
                sid: 0 for sid in ["CFO", "CTO", "Legal", "Procurement", "Ops"]
            },
        )
        rng = np.random.default_rng(42)
        engine.reset(state, rng, {"description": "test"})

        initial_competence = state.beliefs["CFO"]["competence"]
        action = {
            "action_type": "direct_message",
            "target": "CFO",
            "message": "We value our partnership and want to work together for mutual success.",
        }
        engine.apply_action(action, rng)

        # Collaborative signals should increase competence
        assert state.beliefs["CFO"]["competence"] >= initial_competence

    def test_apply_rapport_aggressive(self):
        engine = StakeholderEngine()
        state = DealRoomState(
            beliefs={
                sid: {"competence": 0.5, "risk_tolerance": 0.5, "pricing_rigor": 0.5}
                for sid in ["CFO", "CTO", "Legal", "Procurement", "Ops"]
            },
            satisfaction={
                sid: 0.5 for sid in ["CFO", "CTO", "Legal", "Procurement", "Ops"]
            },
            trust_floors={
                sid: 0.2 for sid in ["CFO", "CTO", "Legal", "Procurement", "Ops"]
            },
            permanent_marks={
                sid: [] for sid in ["CFO", "CTO", "Legal", "Procurement", "Ops"]
            },
            rounds_since_last_contact={
                sid: 0 for sid in ["CFO", "CTO", "Legal", "Procurement", "Ops"]
            },
        )
        rng = np.random.default_rng(42)
        engine.reset(state, rng, {"description": "test"})

        initial_competence = state.beliefs["CFO"]["competence"]
        action = {
            "action_type": "direct_message",
            "target": "CFO",
            "message": "This is our final offer. Take it or leave it. Deadline is non-negotiable.",
        }
        engine.apply_action(action, rng)

        # Aggressive signals should decrease competence
        assert state.beliefs["CFO"]["competence"] <= initial_competence

    def test_scrutiny_mode_penalizes_satisfaction(self):
        engine = StakeholderEngine()
        state = DealRoomState(
            beliefs={
                sid: {"competence": 0.5, "risk_tolerance": 0.5, "pricing_rigor": 0.5}
                for sid in ["CFO", "CTO", "Legal", "Procurement", "Ops"]
            },
            satisfaction={
                sid: 0.5 for sid in ["CFO", "CTO", "Legal", "Procurement", "Ops"]
            },
            trust_floors={
                sid: 0.2 for sid in ["CFO", "CTO", "Legal", "Procurement", "Ops"]
            },
            permanent_marks={
                sid: [] for sid in ["CFO", "CTO", "Legal", "Procurement", "Ops"]
            },
            rounds_since_last_contact={
                sid: 0 for sid in ["CFO", "CTO", "Legal", "Procurement", "Ops"]
            },
            scrutiny_mode=True,
        )
        rng = np.random.default_rng(42)
        engine.reset(state, rng, {"description": "test"})

        initial_sat = state.satisfaction["CFO"]
        action = {
            "action_type": "direct_message",
            "target": "CFO",
            "message": "Thank you for your time.",
        }
        engine.apply_action(action, rng)

        # Scrutiny mode should penalize satisfaction
        assert state.satisfaction["CFO"] <= initial_sat


class TestStakeholderEngineBeliefDeltas:
    """Test belief delta computation."""

    def test_belief_deltas_after_action(self):
        engine = StakeholderEngine()
        state = DealRoomState(
            beliefs={
                sid: {"competence": 0.5, "risk_tolerance": 0.5, "pricing_rigor": 0.5}
                for sid in ["CFO", "CTO", "Legal", "Procurement", "Ops"]
            },
            satisfaction={
                sid: 0.5 for sid in ["CFO", "CTO", "Legal", "Procurement", "Ops"]
            },
            trust_floors={
                sid: 0.2 for sid in ["CFO", "CTO", "Legal", "Procurement", "Ops"]
            },
            permanent_marks={
                sid: [] for sid in ["CFO", "CTO", "Legal", "Procurement", "Ops"]
            },
            rounds_since_last_contact={
                sid: 0 for sid in ["CFO", "CTO", "Legal", "Procurement", "Ops"]
            },
        )
        rng = np.random.default_rng(42)
        engine.reset(state, rng, {"description": "test"})

        action = {
            "action_type": "direct_message",
            "target": "CFO",
            "message": "We appreciate your partnership.",
        }
        engine.apply_action(action, rng)
        deltas = engine.get_belief_deltas()

        assert isinstance(deltas, dict)
        assert "CFO" in deltas
        assert isinstance(deltas["CFO"], float)

    def test_belief_deltas_zero_for_no_change(self):
        engine = StakeholderEngine()
        state = DealRoomState(
            beliefs={
                sid: {"competence": 0.5, "risk_tolerance": 0.5, "pricing_rigor": 0.5}
                for sid in ["CFO", "CTO", "Legal", "Procurement", "Ops"]
            },
            satisfaction={
                sid: 0.5 for sid in ["CFO", "CTO", "Legal", "Procurement", "Ops"]
            },
            trust_floors={
                sid: 0.2 for sid in ["CFO", "CTO", "Legal", "Procurement", "Ops"]
            },
            permanent_marks={
                sid: [] for sid in ["CFO", "CTO", "Legal", "Procurement", "Ops"]
            },
            rounds_since_last_contact={
                sid: 0 for sid in ["CFO", "CTO", "Legal", "Procurement", "Ops"]
            },
        )
        rng = np.random.default_rng(42)
        engine.reset(state, rng, {"description": "test"})

        action = {
            "action_type": "direct_message",
            "target": "CFO",
            "message": "Hello.",  # Neutral message
        }
        engine.apply_action(action, rng)
        deltas = engine.get_belief_deltas()

        # With neutral message, belief deltas should be small or zero
        assert deltas["CFO"] < 0.01


class TestStakeholderEngineStanceTransitions:
    """Test stance determination and transitions."""

    def test_stance_cooperative_when_sat_high(self):
        engine = StakeholderEngine()
        state = DealRoomState(
            beliefs={
                sid: {"competence": 0.5, "risk_tolerance": 0.5, "pricing_rigor": 0.5}
                for sid in ["CFO", "CTO", "Legal", "Procurement", "Ops"]
            },
            satisfaction={
                sid: 0.7 for sid in ["CFO", "CTO", "Legal", "Procurement", "Ops"]
            },
            trust_floors={
                sid: 0.2 for sid in ["CFO", "CTO", "Legal", "Procurement", "Ops"]
            },
            permanent_marks={
                sid: [] for sid in ["CFO", "CTO", "Legal", "Procurement", "Ops"]
            },
            rounds_since_last_contact={
                sid: 0 for sid in ["CFO", "CTO", "Legal", "Procurement", "Ops"]
            },
        )
        rng = np.random.default_rng(42)
        engine.reset(state, rng, {"description": "test"})

        assert engine._stances["CFO"] == "cooperative"

    def test_stance_delaying_when_sat_low(self):
        engine = StakeholderEngine()
        state = DealRoomState(
            beliefs={
                sid: {"competence": 0.5, "risk_tolerance": 0.5, "pricing_rigor": 0.5}
                for sid in ["CFO", "CTO", "Legal", "Procurement", "Ops"]
            },
            satisfaction={
                sid: 0.3 for sid in ["CFO", "CTO", "Legal", "Procurement", "Ops"]
            },
            trust_floors={
                sid: 0.2 for sid in ["CFO", "CTO", "Legal", "Procurement", "Ops"]
            },
            permanent_marks={
                sid: [] for sid in ["CFO", "CTO", "Legal", "Procurement", "Ops"]
            },
            rounds_since_last_contact={
                sid: 0 for sid in ["CFO", "CTO", "Legal", "Procurement", "Ops"]
            },
        )
        rng = np.random.default_rng(42)
        engine.reset(state, rng, {"description": "test"})

        assert engine._stances["CFO"] == "delaying"


class TestStakeholderTemplates:
    """Test that stakeholder templates are properly structured."""

    def test_all_stakeholders_have_templates(self, stakeholder_ids):
        for sid in stakeholder_ids:
            assert sid in STAKEHOLDER_TEMPLATES

    def test_all_stances_exist(self, stakeholder_ids):
        stances = ["cooperative", "testing", "delaying", "obfuscating"]
        for sid in stakeholder_ids:
            for stance in stances:
                assert stance in STAKEHOLDER_TEMPLATES[sid]

    def test_all_buckets_exist(self, stakeholder_ids):
        buckets = ["high", "mid", "low"]
        for sid in stakeholder_ids:
            for stance in STAKEHOLDER_TEMPLATES[sid]:
                for bucket in buckets:
                    assert bucket in STAKEHOLDER_TEMPLATES[sid][stance]

    def test_templates_have_variety(self, stakeholder_ids):
        """Each bucket should have 2+ options to prevent repetition."""
        for sid in stakeholder_ids:
            for stance in STAKEHOLDER_TEMPLATES[sid]:
                for bucket in STAKEHOLDER_TEMPLATES[sid][stance]:
                    assert len(STAKEHOLDER_TEMPLATES[sid][stance][bucket]) >= 2


class TestDocumentEffects:
    """Test document effect constants."""

    def test_document_types_exist(self):
        from server.stakeholders import DOCUMENT_EFFECTS

        expected = [
            "roi_model",
            "security_cert",
            "implementation_timeline",
            "dpa",
            "reference_case",
        ]
        for doc_type in expected:
            assert doc_type in DOCUMENT_EFFECTS

    def test_specificity_levels_exist(self):
        from server.stakeholders import DOCUMENT_EFFECTS

        for doc_type, effects in DOCUMENT_EFFECTS.items():
            assert "high" in effects
            assert "med" in effects
            assert "low" in effects

    def test_effects_target_stakeholders(self):
        from server.stakeholders import DOCUMENT_EFFECTS

        for doc_type, effects in DOCUMENT_EFFECTS.items():
            for spec_level, target_effects in effects.items():
                assert (
                    len(target_effects) > 0
                )  # Each specificity should affect at least one stakeholder
