"""
Lookahead simulator for DealRoom v3 - mental state hypothesis generation and minimax robustness.
"""

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from deal_room.rewards.utterance_scorer import LOOKAHEAD_COST


@dataclass
class SimulationResult:
    predicted_responses: Dict[str, str] = field(default_factory=dict)
    predicted_belief_deltas: Dict[str, float] = field(default_factory=dict)
    cvar_impact: Dict[str, float] = field(default_factory=dict)
    graph_information_gain: float = 0.0
    cost: float = LOOKAHEAD_COST


@dataclass
class BeliefDistribution:
    distribution: Dict[str, float]
    stakeholder_role: str
    confidence: float = 1.0
    history: List = field(default_factory=list)

    def positive_mass(self) -> float:
        return sum(
            self.distribution.get(t, 0) for t in ["competent", "trustworthy", "aligned"]
        )

    def negative_mass(self) -> float:
        return sum(
            self.distribution.get(t, 0)
            for t in ["incompetent", "deceptive", "misaligned"]
        )


class LookaheadSimulator:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def simulate(
        self,
        action_draft,  # DealRoomAction
        current_beliefs: Dict[str, BeliefDistribution],
        n_hypotheses: int = 2,
        depth: int = 2,
    ) -> SimulationResult:
        if not action_draft.target_ids:
            return self._empty_result()

        target_id = action_draft.target_ids[0]
        if target_id not in current_beliefs:
            return self._empty_result()

        hypotheses = self._generate_hypotheses(target_id, current_beliefs)

        simulation_results = []
        for hypothesis in hypotheses:
            sim_result = self._simulate_one_hypothesis(
                action_draft, target_id, hypothesis, depth
            )
            simulation_results.append(sim_result)

        worst_case = min(simulation_results, key=lambda x: x.predicted_goal_delta)

        return SimulationResult(
            predicted_responses=worst_case.responses,
            predicted_belief_deltas=worst_case.belief_deltas,
            cvar_impact=worst_case.cvar_impact,
            graph_information_gain=worst_case.graph_info_gain,
            cost=LOOKAHEAD_COST,
        )

    def _generate_hypotheses(
        self, target_stakeholder: str, current_beliefs: Dict[str, BeliefDistribution]
    ) -> List[BeliefDistribution]:
        current = current_beliefs.get(target_stakeholder)
        if not current:
            return []

        pos_mass = current.positive_mass()

        optimistic_dist = dict(current.distribution)
        optimistic_dist["competent"] = min(
            0.95, optimistic_dist.get("competent", 0) + 0.15
        )
        optimistic_dist["trustworthy"] = min(
            0.95, optimistic_dist.get("trustworthy", 0) + 0.10
        )
        optimistic_dist["aligned"] = min(0.95, optimistic_dist.get("aligned", 0) + 0.10)
        total = sum(optimistic_dist.values())
        optimistic_dist = {k: v / total for k, v in optimistic_dist.items()}

        pessimistic_dist = dict(current.distribution)
        pessimistic_dist["incompetent"] = min(
            0.95, pessimistic_dist.get("incompetent", 0) + 0.15
        )
        pessimistic_dist["deceptive"] = min(
            0.95, pessimistic_dist.get("deceptive", 0) + 0.10
        )
        pessimistic_dist["misaligned"] = min(
            0.95, pessimistic_dist.get("misaligned", 0) + 0.10
        )
        total = sum(pessimistic_dist.values())
        pessimistic_dist = {k: v / total for k, v in pessimistic_dist.items()}

        return [
            BeliefDistribution(
                distribution=optimistic_dist,
                stakeholder_role=target_stakeholder,
                confidence=current.confidence,
            ),
            BeliefDistribution(
                distribution=pessimistic_dist,
                stakeholder_role=target_stakeholder,
                confidence=current.confidence,
            ),
        ]

    def _simulate_one_hypothesis(
        self, action_draft, target_id: str, hypothesis: BeliefDistribution, depth: int
    ) -> "SimOutput":
        base_response_score = hypothesis.positive_mass()

        if action_draft.action_type == "send_document":
            response_score = min(1.0, base_response_score + 0.15)
        elif action_draft.action_type == "direct_message":
            response_score = base_response_score + (
                0.05 if action_draft.message else -0.05
            )
        else:
            response_score = base_response_score

        belief_delta = (response_score - base_response_score) * 0.5

        return SimOutput(
            responses={target_id: self._predict_response_text(response_score, action_draft)},
            belief_deltas={target_id: belief_delta},
            cvar_impact={target_id: -abs(belief_delta) * 0.3},
            graph_info_gain=0.1 if action_draft.documents else 0.05,
            predicted_goal_delta=response_score,
        )

    def _predict_response_text(self, response_score: float, action_draft) -> str:
        supportive_threshold = 0.50 if action_draft.documents else 0.60
        if response_score > supportive_threshold:
            return (
                f"Thank you for the {'document' if action_draft.documents else 'message'}. "
                "I can see the merit in this approach and will review accordingly."
            )
        if response_score > 0.4:
            return (
                "I appreciate the information. Let me consider the implications for "
                "our evaluation before committing."
            )
        return (
            "I have concerns about this direction. We need more detail before I can "
            "support this proposal."
        )

    def _empty_result(self) -> SimulationResult:
        return SimulationResult(
            predicted_responses={},
            predicted_belief_deltas={},
            cvar_impact={},
            graph_information_gain=0.0,
            cost=LOOKAHEAD_COST,
        )


@dataclass
class SimOutput:
    responses: Dict[str, str]
    belief_deltas: Dict[str, float]
    cvar_impact: Dict[str, float]
    graph_info_gain: float
    predicted_goal_delta: float
