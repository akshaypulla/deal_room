"""
Tests for rewards/utterance_scorer.py - Five-dimensional world-state scoring.
"""

import numpy as np
import pytest

from deal_room.committee.causal_graph import (
    BeliefDistribution,
    CausalGraph,
)
from models import DealRoomAction
from deal_room.environment.dealroom_v3 import StateSnapshot, REWARD_WEIGHTS
from deal_room.rewards.utterance_scorer import (
    LOOKAHEAD_COST,
    UtteranceScore,
    UtteranceScorer,
)


class TestLookaheadCost:
    """Tests for lookahead cost application."""

    def test_lookahead_cost_value(self):
        assert LOOKAHEAD_COST == 0.07

    def test_lookahead_cost_subtracted_from_goal(self):
        """Action with lookahead has r^goal exactly LOOKAHEAD_COST lower than without."""
        scorer = UtteranceScorer()

        action = DealRoomAction(
            action_type="direct_message",
            target_ids=["Legal"],
            message="Test message",
        )

        beliefs = {
            sid: BeliefDistribution(
                distribution={
                    "aligned": 0.4,
                    "misaligned": 0.3,
                    "competent": 0.3,
                    "deceptive": 0.3,
                },
                stakeholder_role=sid,
                confidence=0.7,
                history=[],
            )
            for sid in ["Legal", "Finance"]
        }

        state_before = StateSnapshot(
            beliefs=dict(beliefs),
            active_blockers=[],
            risk_profiles={},
            authority_weights={"Legal": 0.5, "Finance": 0.5},
            current_terms={},
            round_number=1,
            deal_stage="evaluation",
            deal_momentum="progressing",
        )

        beliefs_after = {
            sid: BeliefDistribution(
                distribution={
                    "aligned": 0.5,
                    "misaligned": 0.25,
                    "competent": 0.35,
                    "deceptive": 0.25,
                },
                stakeholder_role=sid,
                confidence=0.75,
                history=[],
            )
            for sid in ["Legal", "Finance"]
        }

        state_after = StateSnapshot(
            beliefs=beliefs_after,
            active_blockers=[],
            risk_profiles={},
            authority_weights={"Legal": 0.5, "Finance": 0.5},
            current_terms={},
            round_number=2,
            deal_stage="evaluation",
            deal_momentum="progressing",
        )

        graph = CausalGraph(
            nodes=["Legal", "Finance"],
            edges={("Legal", "Finance"): 0.5},
            authority_weights={"Legal": 0.5, "Finance": 0.5},
            scenario_type="aligned",
            seed=42,
        )

        score_without = scorer.score(
            action=action,
            state_before=state_before,
            state_after=state_after,
            true_graph=graph,
            lookahead_used=False,
        )

        score_with = scorer.score(
            action=action,
            state_before=state_before,
            state_after=state_after,
            true_graph=graph,
            lookahead_used=True,
        )

        assert abs(score_without.goal - score_with.goal - LOOKAHEAD_COST) < 1e-6, (
            f"Goal diff {score_without.goal - score_with.goal} != {LOOKAHEAD_COST}"
        )


class TestScoringDimensions:
    """Tests for all five scoring dimensions."""

    def test_all_dimensions_in_range(self):
        """All five dimensions must be in [0.0, 1.0] for random inputs."""
        scorer = UtteranceScorer()

        graph = CausalGraph(
            nodes=["Legal", "Finance"],
            edges={("Legal", "Finance"): 0.5},
            authority_weights={"Legal": 0.5, "Finance": 0.5},
            scenario_type="aligned",
            seed=42,
        )

        for i in range(20):
            action = DealRoomAction(
                action_type="direct_message",
                target_ids=["Legal"],
                message=f"Test message {i}",
            )

            beliefs = {
                sid: BeliefDistribution(
                    distribution={
                        "aligned": 0.3 + np.random.rand() * 0.2,
                        "misaligned": 0.2,
                        "competent": 0.3,
                        "deceptive": 0.2,
                    },
                    stakeholder_role=sid,
                    confidence=0.6 + np.random.rand() * 0.3,
                    history=[],
                )
                for sid in ["Legal", "Finance"]
            }

            state_before = StateSnapshot(
                beliefs=dict(beliefs),
                active_blockers=[],
                risk_profiles={},
                authority_weights={"Legal": 0.5, "Finance": 0.5},
                current_terms={},
                round_number=i % 10 + 1,
                deal_stage="evaluation",
                deal_momentum="progressing",
            )

            beliefs_after = {
                sid: BeliefDistribution(
                    distribution={
                        "aligned": 0.35 + np.random.rand() * 0.2,
                        "misaligned": 0.18,
                        "competent": 0.33,
                        "deceptive": 0.18,
                    },
                    stakeholder_role=sid,
                    confidence=0.65 + np.random.rand() * 0.3,
                    history=[],
                )
                for sid in ["Legal", "Finance"]
            }

            state_after = StateSnapshot(
                beliefs=beliefs_after,
                active_blockers=[],
                risk_profiles={},
                authority_weights={"Legal": 0.5, "Finance": 0.5},
                current_terms={},
                round_number=i % 10 + 2,
                deal_stage="evaluation",
                deal_momentum="progressing",
            )

            score = scorer.score(
                action=action,
                state_before=state_before,
                state_after=state_after,
                true_graph=graph,
                lookahead_used=False,
            )

            assert 0.0 <= score.goal <= 1.0, f"goal {score.goal} out of [0, 1]"
            assert 0.0 <= score.trust <= 1.0, f"trust {score.trust} out of [0, 1]"
            assert 0.0 <= score.info <= 1.0, f"info {score.info} out of [0, 1]"
            assert 0.0 <= score.risk <= 1.0, f"risk {score.risk} out of [0, 1]"
            assert 0.0 <= score.causal <= 1.0, f"causal {score.causal} out of [0, 1]"


class TestCausalScoring:
    """Tests for r^causal deterministic scoring."""

    def test_causal_score_deterministic(self):
        """Same graph + same target must return identical r^causal."""
        scorer = UtteranceScorer()

        graph = CausalGraph(
            nodes=["Finance", "Legal", "TechLead"],
            edges={("Finance", "Legal"): 0.8, ("Finance", "TechLead"): 0.6},
            authority_weights={"Finance": 0.5, "Legal": 0.25, "TechLead": 0.25},
            scenario_type="aligned",
            seed=42,
        )

        beliefs = {
            sid: BeliefDistribution(
                distribution={"aligned": 0.5},
                stakeholder_role=sid,
                confidence=0.7,
                history=[],
            )
            for sid in graph.nodes
        }

        state = StateSnapshot(
            beliefs=beliefs,
            active_blockers=[],
            risk_profiles={},
            authority_weights={n: 1.0 for n in graph.nodes},
            current_terms={},
            round_number=1,
            deal_stage="evaluation",
            deal_momentum="progressing",
        )

        action1 = DealRoomAction(
            action_type="direct_message", target_ids=["Finance"], message="Test"
        )
        action2 = DealRoomAction(
            action_type="direct_message", target_ids=["Finance"], message="Test"
        )

        score1 = scorer.score(action1, state, state, graph, False)
        score2 = scorer.score(action2, state, state, graph, False)

        assert score1.causal == score2.causal


class TestUtteranceScore:
    """Tests for UtteranceScore dataclass."""

    def test_utterance_score_defaults(self):
        """UtteranceScore has correct default values."""
        score = UtteranceScore()

        assert score.goal == 0.0
        assert score.trust == 0.0
        assert score.info == 0.0
        assert score.risk == 0.0
        assert score.causal == 0.0

    def test_weighted_sum(self):
        """weighted_sum returns correct scalar reward."""
        score = UtteranceScore(goal=0.8, trust=0.6, info=0.7, risk=0.5, causal=0.4)
        result = score.weighted_sum(REWARD_WEIGHTS)
        expected = 0.30 * 0.8 + 0.18 * 0.6 + 0.18 * 0.7 + 0.17 * 0.5 + 0.17 * 0.4
        assert abs(result - expected) < 1e-9

    def test_to_dict(self):
        """to_dict returns correct dict."""
        score = UtteranceScore(goal=0.8, trust=0.6, info=0.7, risk=0.5, causal=0.4)
        d = score.to_dict()
        assert d["goal"] == 0.8
        assert d["trust"] == 0.6
        assert d["info"] == 0.7
        assert d["risk"] == 0.5
        assert d["causal"] == 0.4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
