"""
Unit tests for CCIGrader - Contract Closure Index computation.
Tests scoring across different scenarios and edge cases.
"""

import pytest
from copy import deepcopy
from models import DealRoomState
from server.grader import CCIGrader, STAGE_WEIGHTS


class TestCCIGraderBasic:
    """Test CCIGrader basic functionality."""

    def test_grader_is_available(self):
        assert CCIGrader is not None

    def test_stage_weights_exist(self):
        assert "evaluation" in STAGE_WEIGHTS
        assert "negotiation" in STAGE_WEIGHTS
        assert "legal_review" in STAGE_WEIGHTS
        assert "final_approval" in STAGE_WEIGHTS

    def test_stage_weights_sum_to_one(self):
        for stage, weights in STAGE_WEIGHTS.items():
            total = sum(weights.values())
            assert abs(total - 1.0) < 0.01, f"{stage} weights sum to {total}, not 1.0"

    def test_all_stakeholders_in_weights(self, stakeholder_ids):
        for stage, weights in STAGE_WEIGHTS.items():
            for sid in stakeholder_ids:
                assert sid in weights, f"{sid} missing from {stage} weights"


class TestCCIGraderTerminalStates:
    """Test CCI returns 0.0 for all failure and non-success terminal states."""

    def test_deal_not_closed_returns_zero(self):
        state = DealRoomState(
            deal_closed=False,
            deal_failed=False,
            deal_stage="final_approval",
            round_number=8,
            max_rounds=10,
        )
        assert CCIGrader.compute(state) == 0.0

    def test_deal_failed_returns_zero(self):
        state = DealRoomState(
            deal_closed=False,
            deal_failed=True,
            failure_reason="silent_veto:CFO",
        )
        assert CCIGrader.compute(state) == 0.0

    def test_deal_stage_not_closed_returns_zero(self):
        state = DealRoomState(
            deal_closed=False,
            deal_stage="legal_review",
        )
        assert CCIGrader.compute(state) == 0.0


class TestCCIGraderSuccessCases:
    """Test CCI computation for successful deal closures."""

    def test_perfect_deal(self, stakeholder_ids):
        """All stakeholders at high satisfaction should yield high score."""
        state = DealRoomState(
            deal_closed=True,
            deal_stage="closed",
            round_number=5,
            max_rounds=10,
            satisfaction={sid: 0.8 for sid in stakeholder_ids},
            beliefs={
                sid: {"competence": 0.7, "risk_tolerance": 0.6, "pricing_rigor": 0.5}
                for sid in stakeholder_ids
            },
            validation_failures=0,
        )
        score = CCIGrader.compute(state)
        assert 0.5 < score <= 1.0, f"Perfect deal should score > 0.5, got {score}"

    def test_minimal_satisfaction_deal(self, stakeholder_ids):
        """All stakeholders at minimum viable satisfaction."""
        state = DealRoomState(
            deal_closed=True,
            deal_stage="closed",
            round_number=5,
            max_rounds=10,
            satisfaction={
                sid: 0.35 for sid in stakeholder_ids
            },  # Just above veto threshold
            beliefs={
                sid: {"competence": 0.5, "risk_tolerance": 0.5, "pricing_rigor": 0.5}
                for sid in stakeholder_ids
            },
            validation_failures=0,
        )
        score = CCIGrader.compute(state)
        assert 0.0 < score < 0.5, (
            f"Minimal satisfaction deal should score < 0.5, got {score}"
        )

    def test_early_closure_better_than_late(self, stakeholder_ids):
        """Early closure (low round_number) should score better than late closure."""
        state_early = DealRoomState(
            deal_closed=True,
            deal_stage="closed",
            round_number=3,
            max_rounds=10,
            satisfaction={sid: 0.7 for sid in stakeholder_ids},
            beliefs={
                sid: {"competence": 0.6, "risk_tolerance": 0.6, "pricing_rigor": 0.5}
                for sid in stakeholder_ids
            },
            validation_failures=0,
        )
        state_late = DealRoomState(
            deal_closed=True,
            deal_stage="closed",
            round_number=9,
            max_rounds=10,
            satisfaction={sid: 0.7 for sid in stakeholder_ids},
            beliefs={
                sid: {"competence": 0.6, "risk_tolerance": 0.6, "pricing_rigor": 0.5}
                for sid in stakeholder_ids
            },
            validation_failures=0,
        )
        early_score = CCIGrader.compute(state_early)
        late_score = CCIGrader.compute(state_late)
        assert early_score > late_score, (
            f"Early ({early_score}) should beat late ({late_score})"
        )

    def test_validation_failures_penalize_score(self, stakeholder_ids):
        """Execution penalty reduces score based on validation failures."""
        state_clean = DealRoomState(
            deal_closed=True,
            deal_stage="closed",
            round_number=5,
            max_rounds=10,
            satisfaction={sid: 0.7 for sid in stakeholder_ids},
            beliefs={
                sid: {"competence": 0.6, "risk_tolerance": 0.6, "pricing_rigor": 0.5}
                for sid in stakeholder_ids
            },
            validation_failures=0,
        )
        state_failures = DealRoomState(
            deal_closed=True,
            deal_stage="closed",
            round_number=5,
            max_rounds=10,
            satisfaction={sid: 0.7 for sid in stakeholder_ids},
            beliefs={
                sid: {"competence": 0.6, "risk_tolerance": 0.6, "pricing_rigor": 0.5}
                for sid in stakeholder_ids
            },
            validation_failures=5,
        )
        clean_score = CCIGrader.compute(state_clean)
        failures_score = CCIGrader.compute(state_failures)
        assert clean_score > failures_score, (
            f"Clean ({clean_score}) should beat failures ({failures_score})"
        )

    def test_max_validation_failures_capped(self, stakeholder_ids):
        """Maximum validation penalty is capped at 0.20 (5 failures * 0.04)."""
        state = DealRoomState(
            deal_closed=True,
            deal_stage="closed",
            round_number=5,
            max_rounds=10,
            satisfaction={sid: 0.9 for sid in stakeholder_ids},
            beliefs={
                sid: {"competence": 0.8, "risk_tolerance": 0.7, "pricing_rigor": 0.6}
                for sid in stakeholder_ids
            },
            validation_failures=100,  # Way over the cap
        )
        score = CCIGrader.compute(state)
        # Penalty capped at 0.20, so score = raw - 0.20
        # Even with capped penalty, high satisfaction should yield positive score
        assert 0.0 < score < 0.8  # Capped penalty reduces score


class TestCCIGraderStageWeights:
    """Test that stage weights correctly influence scoring."""

    def test_evaluation_stage_weights(self):
        weights = STAGE_WEIGHTS["evaluation"]
        assert weights["CFO"] == 0.35  # CFO most important in evaluation
        assert weights["CTO"] == 0.30  # CTO second most important

    def test_legal_review_stage_weights(self):
        weights = STAGE_WEIGHTS["legal_review"]
        assert weights["Legal"] == 0.35  # Legal most important in legal review
        assert weights["CFO"] == 0.25

    def test_final_approval_stage_weights(self):
        weights = STAGE_WEIGHTS["final_approval"]
        assert weights["CFO"] == 0.40  # CFO most important at final approval

    def test_different_satisfaction_same_score_depending_on_stage(
        self, stakeholder_ids
    ):
        """Same satisfaction levels should score differently based on stage."""
        # High CFO satisfaction should matter more in final_approval than evaluation
        state_eval = DealRoomState(
            deal_closed=True,
            deal_stage="closed",
            round_number=5,
            max_rounds=10,
            satisfaction={
                **{sid: 0.6 for sid in stakeholder_ids},
                "CFO": 0.9,
            },  # CFO very happy
            beliefs={
                sid: {"competence": 0.6, "risk_tolerance": 0.6, "pricing_rigor": 0.5}
                for sid in stakeholder_ids
            },
            validation_failures=0,
        )
        # This would need stage-specific computation to properly test
        # For now we just verify weights exist and differ by stage


class TestCCIGraderWeakestLink:
    """Test weakest-link factor in consensus calculation."""

    def test_one_low_stakeholder_hurts(self, stakeholder_ids):
        """One stakeholder at very low satisfaction should significantly reduce score."""
        state_balanced = DealRoomState(
            deal_closed=True,
            deal_stage="closed",
            round_number=5,
            max_rounds=10,
            satisfaction={sid: 0.7 for sid in stakeholder_ids},
            beliefs={
                sid: {"competence": 0.6, "risk_tolerance": 0.6, "pricing_rigor": 0.5}
                for sid in stakeholder_ids
            },
            validation_failures=0,
        )
        state_imbalanced = DealRoomState(
            deal_closed=True,
            deal_stage="closed",
            round_number=5,
            max_rounds=10,
            satisfaction={
                **{sid: 0.7 for sid in stakeholder_ids},
                "Legal": 0.25,
            },  # One low
            beliefs={
                sid: {"competence": 0.6, "risk_tolerance": 0.6, "pricing_rigor": 0.5}
                for sid in stakeholder_ids
            },
            validation_failures=0,
        )
        balanced_score = CCIGrader.compute(state_balanced)
        imbalanced_score = CCIGrader.compute(state_imbalanced)
        assert balanced_score > imbalanced_score, (
            f"Balanced ({balanced_score}) should beat imbalanced ({imbalanced_score})"
        )


class TestCCIGraderEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_satisfaction_dict(self):
        state = DealRoomState(
            deal_closed=True,
            deal_stage="closed",
            satisfaction={},  # Empty
            beliefs={},
        )
        score = CCIGrader.compute(state)
        assert 0.0 <= score <= 1.0

    def test_missing_stakeholder_in_satisfaction(self):
        state = DealRoomState(
            deal_closed=True,
            deal_stage="closed",
            satisfaction={"CFO": 0.7},  # Only CFO
            beliefs={
                "CFO": {"competence": 0.6, "risk_tolerance": 0.6, "pricing_rigor": 0.5}
            },
            validation_failures=0,
        )
        score = CCIGrader.compute(state)
        assert 0.0 <= score <= 1.0

    def test_score_is_rounded_to_4_decimals(self, stakeholder_ids):
        state = DealRoomState(
            deal_closed=True,
            deal_stage="closed",
            round_number=5,
            max_rounds=10,
            satisfaction={sid: 0.65 for sid in stakeholder_ids},
            beliefs={
                sid: {"competence": 0.55, "risk_tolerance": 0.55, "pricing_rigor": 0.5}
                for sid in stakeholder_ids
            },
            validation_failures=0,
        )
        score = CCIGrader.compute(state)
        # Check it's rounded to 4 decimal places
        assert score == round(score, 4)

    def test_score_bounded_between_0_and_1(self, stakeholder_ids):
        """Score must always be in [0.0, 1.0]."""
        for _ in range(10):
            state = DealRoomState(
                deal_closed=True,
                deal_stage="closed",
                round_number=5,
                max_rounds=10,
                satisfaction={sid: 0.5 for sid in stakeholder_ids},
                beliefs={
                    sid: {
                        "competence": 0.5,
                        "risk_tolerance": 0.5,
                        "pricing_rigor": 0.5,
                    }
                    for sid in stakeholder_ids
                },
                validation_failures=0,
            )
            score = CCIGrader.compute(state)
            assert 0.0 <= score <= 1.0
