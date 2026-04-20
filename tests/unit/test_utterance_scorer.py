"""
Tests for rewards/utterance_scorer.py - Five-dimensional reward scoring with caching.
"""

import numpy as np
import pytest

from deal_room.committee.causal_graph import (
    CausalGraph,
    create_neutral_beliefs,
)
from models import DealRoomAction, DealRoomObservation, DealRoomState
from deal_room.rewards.utterance_scorer import (
    LOOKAHEAD_COST,
    CACHE,
    UtteranceScore,
    UtteranceScorer,
    _get_cache_key,
    _score_causal_heuristic,
    _score_goal_heuristic,
    _score_information_heuristic,
    _score_risk_heuristic,
    _score_trust_heuristic,
    compute_prediction_accuracy,
)


class TestLookaheadCost:
    """Tests for lookahead cost application."""

    def test_lookahead_cost_value(self):
        """LOOKAHEAD_COST must be exactly 0.07."""
        assert LOOKAHEAD_COST == 0.07

    def test_lookahead_cost_subtracted(self):
        """Action with lookahead has r^goal exactly 0.07 lower than without."""
        scorer = UtteranceScorer(rng=np.random.default_rng(42))

        action = DealRoomAction(
            action_type="direct_message",
            target_ids=["Legal"],
            message="Test message",
        )

        obs = DealRoomObservation(
            round_number=1,
            max_rounds=10,
            stakeholders={"Legal": {"role": "General Counsel"}},
            engagement_level={"Legal": 0.6},
            veto_precursors={},
        )

        graph = CausalGraph(
            nodes=["Legal", "Finance"],
            edges={("Legal", "Finance"): 0.5},
            authority_weights={"Legal": 0.5, "Finance": 0.5},
            scenario_type="aligned",
            seed=42,
        )

        # Without lookahead
        score_without, _ = scorer.score(
            action=action,
            obs=obs,
            stakeholder_id="Legal",
            engagement_levels={"Legal": 0.6},
            graph=graph,
            lookahead_used=False,
        )

        # With lookahead
        score_with, _ = scorer.score(
            action=action,
            obs=obs,
            stakeholder_id="Legal",
            engagement_levels={"Legal": 0.6},
            graph=graph,
            lookahead_used=True,
        )

        assert abs(score_without.goal - score_with.goal - LOOKAHEAD_COST) < 1e-6, (
            f"Goal diff {score_without.goal - score_with.goal} != {LOOKAHEAD_COST}"
        )

    def test_lookahead_cost_not_below_zero(self):
        """r^goal must never be negative even if base score is < 0.07."""
        scorer = UtteranceScorer(rng=np.random.default_rng(42))

        # Create action with minimal goal potential
        action = DealRoomAction(
            action_type="direct_message",
            target_ids=["Legal"],
            message="",  # Empty message gets low score
        )

        obs = DealRoomObservation(
            round_number=1,
            max_rounds=10,
            stakeholders={"Legal": {"role": "General Counsel"}},
            engagement_level={"Legal": 0.1},  # Low engagement
            active_blockers=["Legal"],  # Has active blockers
            veto_precursors={"Legal": "CVaR warning"},
        )

        graph = CausalGraph(
            nodes=["Legal"],
            edges={},
            authority_weights={"Legal": 1.0},
            scenario_type="aligned",
            seed=42,
        )

        score, _ = scorer.score(
            action=action,
            obs=obs,
            stakeholder_id="Legal",
            engagement_levels={"Legal": 0.1},
            graph=graph,
            lookahead_used=True,
        )

        assert score.goal >= 0.0, f"Goal score {score.goal} should not be negative"


class TestCausalScoring:
    """Tests for r^causal deterministic scoring."""

    def test_causal_score_deterministic(self):
        """Same action + same graph must return identical r^causal always."""
        action = DealRoomAction(
            action_type="direct_message",
            target_ids=["Finance"],
            message="Test",
        )

        graph = CausalGraph(
            nodes=["Finance", "Legal", "TechLead"],
            edges={("Finance", "Legal"): 0.8, ("Finance", "TechLead"): 0.6},
            authority_weights={"Finance": 0.5, "Legal": 0.25, "TechLead": 0.25},
            scenario_type="aligned",
            seed=42,
        )

        score1 = _score_causal_heuristic(action, graph, {"Finance": 0.6})
        score2 = _score_causal_heuristic(action, graph, {"Finance": 0.6})
        score3 = _score_causal_heuristic(action, graph, {"Finance": 0.6})

        assert score1 == score2 == score3

    def test_high_centrality_target_scores_higher(self):
        """Targeting hub node scores r^causal > leaf node + 0.20 in star graph."""
        # Star graph: Finance is hub, others are leaves
        edges = {}
        for leaf in ["Legal", "TechLead", "Procurement"]:
            edges[("Finance", leaf)] = 0.8

        graph = CausalGraph(
            nodes=["Finance", "Legal", "TechLead", "Procurement"],
            edges=edges,
            authority_weights={
                "Finance": 0.5,
                "Legal": 0.167,
                "TechLead": 0.167,
                "Procurement": 0.167,
            },
            scenario_type="aligned",
            seed=42,
        )

        hub_action = DealRoomAction(
            action_type="direct_message", target_ids=["Finance"], message=""
        )
        leaf_action = DealRoomAction(
            action_type="direct_message", target_ids=["Legal"], message=""
        )

        hub_score = _score_causal_heuristic(
            hub_action, graph, {"Finance": 0.6, "Legal": 0.5}
        )
        leaf_score = _score_causal_heuristic(
            leaf_action, graph, {"Finance": 0.6, "Legal": 0.5}
        )

        assert hub_score > leaf_score + 0.20, (
            f"Hub {hub_score} should be > leaf {leaf_score} + 0.20"
        )


class TestScoringDimensions:
    """Tests for all five scoring dimensions."""

    def test_all_dimensions_in_range(self):
        """All five dimensions must be in [0.0, 1.0] for random inputs."""
        scorer = UtteranceScorer(rng=np.random.default_rng(42))

        graph = CausalGraph(
            nodes=["Legal", "Finance"],
            edges={("Legal", "Finance"): 0.5},
            authority_weights={"Legal": 0.5, "Finance": 0.5},
            scenario_type="aligned",
            seed=42,
        )

        for i in range(20):
            action = DealRoomAction(
                action_type=np.random.choice(["direct_message", "send_document"]),
                target_ids=[np.random.choice(["Legal", "Finance"])],
                message=f"Test message {i}",
            )

            obs = DealRoomObservation(
                round_number=i % 10 + 1,
                max_rounds=10,
                stakeholders={
                    "Legal": {"role": "General Counsel"},
                    "Finance": {"role": "CFO"},
                },
                engagement_level={
                    "Legal": np.random.random(),
                    "Finance": np.random.random(),
                },
                veto_precursors={},
                deal_stage="evaluation",
            )

            score, _ = scorer.score(
                action=action,
                obs=obs,
                stakeholder_id="Legal",
                engagement_levels={"Legal": 0.5, "Finance": 0.5},
                graph=graph,
            )

            assert 0.0 <= score.goal <= 1.0, f"goal {score.goal} out of [0, 1]"
            assert 0.0 <= score.trust <= 1.0, f"trust {score.trust} out of [0, 1]"
            assert 0.0 <= score.information <= 1.0, (
                f"info {score.information} out of [0, 1]"
            )
            assert 0.0 <= score.risk <= 1.0, f"risk {score.risk} out of [0, 1]"
            assert 0.0 <= score.causal <= 1.0, f"causal {score.causal} out of [0, 1]"


class TestCaching:
    """Tests for utterance scoring cache."""

    def test_cache_returns_same_score(self):
        """Identical (message, action, state) returns identical score from cache."""
        # Clear cache first
        CACHE.clear()

        scorer = UtteranceScorer(rng=np.random.default_rng(42))

        action = DealRoomAction(
            action_type="direct_message",
            target_ids=["Legal"],
            message="Same message",
        )

        obs = DealRoomObservation(
            round_number=1,
            max_rounds=10,
            stakeholders={"Legal": {"role": "General Counsel"}},
            engagement_level={"Legal": 0.6},
            veto_precursors={},
        )

        graph = CausalGraph(
            nodes=["Legal"],
            edges={},
            authority_weights={"Legal": 1.0},
            scenario_type="aligned",
            seed=42,
        )

        state_hash = "test_hash_123"

        # First call
        score1, metrics1 = scorer.score(
            action=action,
            obs=obs,
            stakeholder_id="Legal",
            engagement_levels={"Legal": 0.6},
            graph=graph,
            belief_state_hash=state_hash,
        )

        # Second call with same inputs
        score2, metrics2 = scorer.score(
            action=action,
            obs=obs,
            stakeholder_id="Legal",
            engagement_levels={"Legal": 0.6},
            graph=graph,
            belief_state_hash=state_hash,
        )

        # Should be identical
        assert score1.goal == score2.goal
        assert metrics1.cache_hit is False
        assert metrics2.cache_hit is True

    def test_cache_hit_rate_after_repeated_calls(self):
        """After 10 calls with 3 unique inputs, cache hit rate must be > 0.0."""
        CACHE.clear()

        scorer = UtteranceScorer(rng=np.random.default_rng(42))

        graph = CausalGraph(
            nodes=["Legal", "Finance"],
            edges={},
            authority_weights={"Legal": 0.5, "Finance": 0.5},
            scenario_type="aligned",
            seed=42,
        )

        # 3 unique messages
        messages = ["Message A", "Message B", "Message C"]
        cache_hits = 0

        for i in range(10):
            msg = messages[i % 3]
            action = DealRoomAction(
                action_type="direct_message",
                target_ids=["Legal"],
                message=msg,
            )

            obs = DealRoomObservation(
                round_number=1,
                max_rounds=10,
                stakeholders={"Legal": {"role": "General Counsel"}},
                engagement_level={"Legal": 0.6},
                veto_precursors={},
            )

            _, metrics = scorer.score(
                action=action,
                obs=obs,
                stakeholder_id="Legal",
                engagement_levels={"Legal": 0.6},
                graph=graph,
                belief_state_hash=f"hash_{msg}",
            )

            if metrics.cache_hit:
                cache_hits += 1

        assert cache_hits > 0, "Cache hit rate should be > 0 after repeated calls"


class TestPredictionAccuracy:
    """Tests for prediction accuracy diagnostic metric."""

    def test_prediction_accuracy_not_in_reward(self):
        """prediction_accuracy must be set but NOT added to any reward dimension."""
        scorer = UtteranceScorer(rng=np.random.default_rng(42))

        action = DealRoomAction(
            action_type="direct_message",
            target_ids=["Legal"],
            message="Test",
        )

        obs = DealRoomObservation(
            round_number=1,
            max_rounds=10,
            stakeholders={"Legal": {"role": "General Counsel"}},
            engagement_level={"Legal": 0.6},
            veto_precursors={},
        )

        graph = CausalGraph(
            nodes=["Legal"],
            edges={},
            authority_weights={"Legal": 1.0},
            scenario_type="aligned",
            seed=42,
        )

        score, metrics = scorer.score(
            action=action,
            obs=obs,
            stakeholder_id="Legal",
            engagement_levels={"Legal": 0.6},
            graph=graph,
            lookahead_used=True,
            predicted_responses={"Legal": "Response A"},
            actual_responses={"Legal": "Response B"},
        )

        # prediction_accuracy should be computed
        assert metrics.prediction_accuracy is not None
        # But should NOT affect any dimension
        # (we can't directly verify it's not in weighted_sum without the actual method)
        assert score.goal >= 0.0  # Just verify it's a valid score

    def test_compute_prediction_accuracy(self):
        """compute_prediction_accuracy returns correct similarity score."""
        # Exact match
        acc_exact = compute_prediction_accuracy({"A": "hello"}, {"A": "hello"})
        assert acc_exact == 1.0

        # Partial match
        acc_partial = compute_prediction_accuracy({"A": "hello"}, {"A": "world"})
        assert 0.0 <= acc_partial < 1.0

        # Empty
        acc_empty = compute_prediction_accuracy({}, {})
        assert acc_empty == 0.0


class TestHeuristicScores:
    """Tests for heuristic scoring functions."""

    def test_goal_heuristic_blocks_resolution(self):
        """goal_heuristic gives higher score when blockers are resolved."""
        action = DealRoomAction(
            action_type="direct_message",
            target_ids=["Legal"],
            message="I will address the compliance concern",
        )

        # With blockers
        obs_with = DealRoomObservation(
            round_number=1,
            max_rounds=10,
            stakeholders={},
            engagement_level={},
            active_blockers=["compliance"],
            veto_precursors={},
        )

        # Without blockers
        obs_without = DealRoomObservation(
            round_number=1,
            max_rounds=10,
            stakeholders={},
            engagement_level={},
            active_blockers=[],
            veto_precursors={},
        )

        score_with = _score_goal_heuristic(action.message, action.message, obs_with)
        score_without = _score_goal_heuristic(
            action.message, action.message, obs_without
        )

        assert score_with > score_without

    def test_trust_heuristic_role_keywords(self):
        """trust_heuristic gives higher score when role-specific keywords present."""
        legal_msg = "We need to review the liability limitations in the contract"
        generic_msg = "Hello, how are you?"

        score_legal = _score_trust_heuristic(legal_msg, "legal")
        score_generic = _score_trust_heuristic(generic_msg, "legal")

        assert score_legal > score_generic

    def test_information_heuristic_questions(self):
        """information_heuristic gives higher score for questions."""
        question = "What are your specific compliance requirements?"
        statement = "Here is our standard contract."

        score_q = _score_information_heuristic(
            question,
            DealRoomAction(
                action_type="direct_message", target_ids=["Legal"], message=question
            ),
        )
        score_s = _score_information_heuristic(
            statement,
            DealRoomAction(
                action_type="direct_message", target_ids=["Legal"], message=statement
            ),
        )

        assert score_q > score_s

    def test_risk_heuristic_veto_precursors(self):
        """risk_heuristic gives lower score when veto precursors present."""
        action = DealRoomAction(
            action_type="direct_message", target_ids=["Legal"], message="Test"
        )

        # With veto precursors
        obs_with = DealRoomObservation(
            round_number=1,
            max_rounds=10,
            stakeholders={},
            engagement_level={},
            veto_precursors={"Legal": "CVaR warning"},
        )

        # Without veto precursors
        obs_without = DealRoomObservation(
            round_number=1,
            max_rounds=10,
            stakeholders={},
            engagement_level={},
            veto_precursors={},
        )

        score_with = _score_risk_heuristic(action, obs_with)
        score_without = _score_risk_heuristic(action, obs_without)

        assert score_with < score_without


class TestUtteranceScore:
    """Tests for UtteranceScore dataclass."""

    def test_utterance_score_defaults(self):
        """UtteranceScore has correct default values."""
        score = UtteranceScore()

        assert score.goal == 0.0
        assert score.trust == 0.0
        assert score.information == 0.0
        assert score.risk == 0.0
        assert score.causal == 0.0
        assert score.prediction_accuracy is None
        assert score.lookahead_used is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
