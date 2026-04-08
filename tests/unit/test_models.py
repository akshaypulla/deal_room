"""
Unit tests for Pydantic models (DealRoomAction, DealRoomObservation, DealRoomState).
Tests validation, serialization, and field constraints.
"""

import pytest
from pydantic import ValidationError
from models import DealRoomAction, DealRoomObservation, DealRoomState


class TestDealRoomAction:
    """Test DealRoomAction model validation and defaults."""

    def test_default_values(self):
        action = DealRoomAction()
        assert action.action_type == "direct_message"
        assert action.target == "all"
        assert action.message == ""
        assert action.channel == "formal"
        assert action.mode == "async_email"
        assert action.documents == []
        assert action.proposed_terms is None

    def test_valid_action_types(self, action_types):
        for at in action_types:
            action = DealRoomAction(action_type=at, target="CFO", message="Test")
            assert action.action_type == at

    def test_valid_targets(self, valid_targets):
        for target in valid_targets:
            action = DealRoomAction(target=target, message="Test")
            assert action.target == target

    def test_documents_field(self):
        action = DealRoomAction(
            action_type="send_document",
            target="CFO",
            documents=[
                {"type": "roi_model", "specificity": "high"},
                {"type": "security_cert", "specificity": "med"},
            ],
        )
        assert len(action.documents) == 2
        assert action.documents[0]["type"] == "roi_model"
        assert action.documents[0]["specificity"] == "high"

    def test_proposed_terms(self):
        terms = {"payment": "net 30", "delivery": "Q3 2025"}
        action = DealRoomAction(proposed_terms=terms)
        assert action.proposed_terms == terms

    def test_serialization_roundtrip(self):
        action = DealRoomAction(
            action_type="direct_message",
            target="CFO",
            message="Hello",
            channel="formal",
            mode="async_email",
        )
        data = action.dict()
        restored = DealRoomAction(**data)
        assert restored.action_type == action.action_type
        assert restored.target == action.target
        assert restored.message == action.message


class TestDealRoomObservation:
    """Test DealRoomObservation model."""

    def test_default_values(self):
        obs = DealRoomObservation()
        assert obs.round_number == 0
        assert obs.max_rounds == 10
        assert obs.stakeholder_messages == {}
        assert obs.engagement_level == {}
        assert obs.deal_momentum == "stalling"
        assert obs.deal_stage == "evaluation"
        assert obs.competitor_events == []
        assert obs.veto_precursors == {}
        assert obs.scenario_hint is None
        assert obs.active_blockers == []
        assert obs.days_to_deadline == 30
        assert obs.done is False
        assert obs.info == {}

    def test_stakeholder_messages(self):
        messages = {
            "CFO": "Thanks for reaching out.",
            "CTO": "Happy to evaluate this.",
            "Legal": "We'll require documentation.",
        }
        obs = DealRoomObservation(stakeholder_messages=messages)
        assert obs.stakeholder_messages["CFO"] == "Thanks for reaching out."
        assert obs.stakeholder_messages["CTO"] == "Happy to evaluate this."

    def test_engagement_level(self):
        engagement = {
            "CFO": 0.54,
            "CTO": 0.56,
            "Legal": 0.48,
            "Procurement": 0.52,
            "Ops": 0.60,
        }
        obs = DealRoomObservation(engagement_level=engagement)
        assert obs.engagement_level["CFO"] == 0.54

    def test_deal_stage_transitions(self):
        for stage in [
            "evaluation",
            "negotiation",
            "legal_review",
            "final_approval",
            "closed",
        ]:
            obs = DealRoomObservation(deal_stage=stage)
            assert obs.deal_stage == stage

    def test_momentum_values(self):
        for momentum in ["stalling", "progressing", "critical"]:
            obs = DealRoomObservation(deal_momentum=momentum)
            assert obs.deal_momentum == momentum

    def test_competitor_events(self):
        events = ["competitor_demo_scheduled", "competitor_pricing_leak"]
        obs = DealRoomObservation(competitor_events=events)
        assert len(obs.competitor_events) == 2

    def test_veto_precursors(self):
        precursors = {
            "CFO": "CFO has been unusually brief in recent replies.",
            "CTO": "CTO delegated follow-up coordination to their assistant.",
        }
        obs = DealRoomObservation(veto_precursors=precursors)
        assert (
            obs.veto_precursors["CFO"]
            == "CFO has been unusually brief in recent replies."
        )

    def test_info_dict(self):
        info = {"validation_confidence": 0.8, "momentum_direction": 1}
        obs = DealRoomObservation(info=info)
        assert obs.info["validation_confidence"] == 0.8


class TestDealRoomState:
    """Test DealRoomState model validation and constraints."""

    def test_default_values(self):
        state = DealRoomState()
        assert state.episode_id == ""
        assert state.task_id == ""
        assert state.round_number == 0
        assert state.max_rounds == 10
        assert state.deal_stage == "evaluation"
        assert state.deal_closed is False
        assert state.deal_failed is False

    def test_beliefs_validation_valid(self, stakeholder_ids):
        beliefs = {
            sid: {"competence": 0.5, "risk_tolerance": 0.5, "pricing_rigor": 0.5}
            for sid in stakeholder_ids
        }
        state = DealRoomState(beliefs=beliefs)
        assert len(state.beliefs) == 5

    def test_beliefs_validation_missing_dim(self, stakeholder_ids):
        beliefs = {
            sid: {"competence": 0.5, "risk_tolerance": 0.5} for sid in stakeholder_ids
        }  # missing pricing_rigor
        with pytest.raises(ValidationError):
            DealRoomState(beliefs=beliefs)

    def test_beliefs_validation_extra_dim(self, stakeholder_ids):
        beliefs = {
            sid: {
                "competence": 0.5,
                "risk_tolerance": 0.5,
                "pricing_rigor": 0.5,
                "extra_dim": 0.5,
            }
            for sid in stakeholder_ids
        }
        state = DealRoomState(beliefs=beliefs)  # Should work - extra dims allowed

    def test_satisfaction_dict(self, stakeholder_ids):
        satisfaction = {sid: 0.5 for sid in stakeholder_ids}
        state = DealRoomState(satisfaction=satisfaction)
        assert len(state.satisfaction) == 5

    def test_veto_risk(self, stakeholder_ids):
        veto_risk = {sid: 0.0 for sid in stakeholder_ids}
        state = DealRoomState(veto_risk=veto_risk)
        assert all(v == 0.0 for v in state.veto_risk.values())

    def test_active_blockers(self):
        state = DealRoomState(active_blockers=["CFO", "Legal"])
        assert len(state.active_blockers) == 2

    def test_permanent_marks(self):
        marks = {"CFO": ["contradiction_penalty"], "CTO": []}
        state = DealRoomState(permanent_marks=marks)
        assert state.permanent_marks["CFO"] == ["contradiction_penalty"]

    def test_failure_reason(self):
        state = DealRoomState(failure_reason="silent_veto:CFO")
        assert state.failure_reason == "silent_veto:CFO"

    def test_final_terms(self):
        terms = {"price": 2000000, "payment": "net 30"}
        state = DealRoomState(final_terms=terms)
        assert state.final_terms["price"] == 2000000

    def test_serialization_roundtrip(self):
        state = DealRoomState(
            episode_id="test-123",
            task_id="aligned",
            beliefs={
                "CFO": {
                    "competence": 0.55,
                    "risk_tolerance": 0.52,
                    "pricing_rigor": 0.50,
                },
            },
        )
        data = state.dict()
        restored = DealRoomState(**data)
        assert restored.episode_id == state.episode_id
        assert restored.task_id == state.task_id

    def test_stage_regressions(self):
        state = DealRoomState(stage_regressions=2)
        assert state.stage_regressions == 2

    def test_validation_failures(self):
        state = DealRoomState(validation_failures=5)
        assert state.validation_failures == 5

    def test_scrutiny_mode(self):
        state = DealRoomState(scrutiny_mode=True)
        assert state.scrutiny_mode is True

    def test_exec_escalation_used(self):
        state = DealRoomState(exec_escalation_used=True)
        assert state.exec_escalation_used is True

    def test_rounds_since_last_contact(self, stakeholder_ids):
        contact = {sid: 0 for sid in stakeholder_ids}
        state = DealRoomState(rounds_since_last_contact=contact)
        assert state.rounds_since_last_contact["CFO"] == 0


class TestModelEdgeCases:
    """Test edge cases and boundary conditions for models."""

    def test_empty_message(self):
        action = DealRoomAction(message="")
        assert action.message == ""

    def test_very_long_message(self):
        long_msg = "A" * 1000
        action = DealRoomAction(message=long_msg)
        assert len(action.message) == 1000

    def test_unicode_in_message(self):
        action = DealRoomAction(message="Hello! 🌍 Café negotiations €1000000")
        assert "🌍" in action.message

    def test_zero_values_in_satisfaction(self):
        state = DealRoomState(satisfaction={"CFO": 0.0, "CTO": 0.0})
        assert state.satisfaction["CFO"] == 0.0

    def test_max_values_in_beliefs(self):
        beliefs = {
            "CFO": {"competence": 1.0, "risk_tolerance": 1.0, "pricing_rigor": 1.0}
        }
        state = DealRoomState(beliefs=beliefs)
        assert state.beliefs["CFO"]["competence"] == 1.0

    def test_none_values_in_optional_fields(self):
        state = DealRoomState(final_terms=None, failure_reason="")
        assert state.final_terms is None
        assert state.failure_reason == ""
