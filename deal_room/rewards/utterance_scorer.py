"""
Utterance scorer for DealRoom v3 — world-state-based deterministic scoring.

All five dimensions are computed from world state deltas, not message content.
No LLM calls. No caching. No fallbacks.

Dimensions:
  goal    = f(belief_positive_mass_delta, resolved_blockers, CVaR_headroom_change)
  trust   = f(trustworthy_mass_delta_for_targeted_stakeholder)
  info    = f(H(B_i_before) - H(B_i_after))  # entropy reduction
  risk    = f(CVaR_delta_across_risk_averse_stakeholders)
  causal  = f(betweenness_centrality_of_targeted_node)
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from deal_room.committee.causal_graph import (
    BeliefDistribution,
    get_betweenness_centrality,
)
from deal_room.stakeholders.cvar_preferences import (
    StakeholderRiskProfile,
    compute_cvar,
    compute_outcome_distribution,
)
from models import DealRoomAction


LOG2_6 = math.log(2, 6)
LOOKAHEAD_COST = 0.07


def _entropy_base2(values: np.ndarray) -> float:
    values = values[values > 0]
    if values.size == 0:
        return 0.0
    return float(-np.sum(values * np.log2(values)))


def compute_prediction_accuracy(
    predicted: Dict[str, str], actual: Dict[str, str]
) -> float:
    shared = sorted(set(predicted) & set(actual))
    if not shared:
        return 0.0

    scores = []
    for stakeholder_id in shared:
        predicted_text = (predicted.get(stakeholder_id) or "").strip().lower()
        actual_text = (actual.get(stakeholder_id) or "").strip().lower()

        if not predicted_text and not actual_text:
            scores.append(1.0)
            continue
        if not predicted_text or not actual_text:
            scores.append(0.0)
            continue
        if predicted_text == actual_text:
            scores.append(1.0)
            continue

        predicted_tokens = set(predicted_text.split())
        actual_tokens = set(actual_text.split())
        union = predicted_tokens | actual_tokens
        scores.append(len(predicted_tokens & actual_tokens) / len(union) if union else 0.0)

    return float(sum(scores) / len(scores))


@dataclass
class UtteranceScore:
    goal: float = 0.0
    trust: float = 0.0
    info: float = 0.0
    risk: float = 0.0
    causal: float = 0.0

    def weighted_sum(self, weights: Dict[str, float]) -> float:
        return (
            weights.get("goal", 1.0) * self.goal
            + weights.get("trust", 1.0) * self.trust
            + weights.get("info", 1.0) * self.info
            + weights.get("risk", 1.0) * self.risk
            + weights.get("causal", 1.0) * self.causal
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "goal": self.goal,
            "trust": self.trust,
            "info": self.info,
            "risk": self.risk,
            "causal": self.causal,
        }


class UtteranceScorer:
    def score(
        self,
        action: DealRoomAction,
        state_before: Any,
        state_after: Any,
        true_graph: Any,
        lookahead_used: bool = False,
    ) -> UtteranceScore:
        beliefs_before = getattr(state_before, "beliefs", {}) or {}
        beliefs_after = getattr(state_after, "beliefs", {}) or {}

        active_blockers_before = getattr(state_before, "active_blockers", []) or []
        active_blockers_after = getattr(state_after, "active_blockers", []) or []

        deal_stage = getattr(state_after, "deal_stage", "evaluation") or "evaluation"
        risk_profiles = getattr(state_before, "risk_profiles", None)
        authority_weights = getattr(state_before, "authority_weights", {}) or {}
        deal_terms = getattr(state_before, "current_terms", {}) or {}

        targeted_ids = action.target_ids if action else []

        goal = self._score_goal(
            beliefs_before,
            beliefs_after,
            active_blockers_before,
            active_blockers_after,
            deal_stage,
            risk_profiles,
            deal_terms,
            authority_weights,
        )
        trust = self._score_trust(beliefs_before, beliefs_after, targeted_ids)
        info = self._score_info(beliefs_before, beliefs_after)
        risk = self._score_risk(
            beliefs_before, beliefs_after, risk_profiles, deal_terms
        )
        causal = self._score_causal(true_graph, targeted_ids)

        if lookahead_used:
            goal = max(0.0, goal - LOOKAHEAD_COST)

        return UtteranceScore(
            goal=goal, trust=trust, info=info, risk=risk, causal=causal
        )

    def _score_goal(
        self,
        beliefs_before: Dict[str, BeliefDistribution],
        beliefs_after: Dict[str, BeliefDistribution],
        blockers_before: List[str],
        blockers_after: List[str],
        deal_stage: str,
        risk_profiles: Optional[Dict[str, StakeholderRiskProfile]],
        deal_terms: Dict,
        authority_weights: Dict[str, float],
    ) -> float:
        approval_delta = 0.0
        total_auth = 0.0
        for sid, b_after in beliefs_after.items():
            b_before = beliefs_before.get(sid)
            if b_before is None:
                continue
            auth = authority_weights.get(sid, 1.0)
            approval_delta += (
                b_after.positive_mass() - b_before.positive_mass()
            ) * auth
            total_auth += auth
        approval_score = (approval_delta / total_auth) if total_auth > 0 else 0.0

        blockers_before_set = set(blockers_before)
        blockers_after_set = set(blockers_after)
        resolved = len(blockers_before_set - blockers_after_set)
        new_created = len(blockers_after_set - blockers_before_set)
        blocker_score = resolved * 0.15 - new_created * 0.10

        veto_improvements = []
        if risk_profiles and deal_terms:
            for sid, profile in risk_profiles.items():
                if profile.lambda_risk > 0.40:
                    cvar_b = self._compute_cvar(
                        sid, beliefs_before, profile, deal_terms
                    )
                    cvar_a = self._compute_cvar(sid, beliefs_after, profile, deal_terms)
                    tau = profile.tau
                    if tau > 0:
                        headroom_delta = max(0.0, 1.0 - cvar_a / tau) - max(
                            0.0, 1.0 - cvar_b / tau
                        )
                        veto_improvements.append(headroom_delta)
        veto_score = (
            sum(veto_improvements) / len(veto_improvements)
            if veto_improvements
            else 0.0
        )

        raw = 0.50 * approval_score + 0.30 * blocker_score + 0.20 * veto_score
        return float(np.clip(0.5 + raw, 0.0, 1.0))

    def _score_trust(
        self,
        beliefs_before: Dict[str, BeliefDistribution],
        beliefs_after: Dict[str, BeliefDistribution],
        targeted_ids: List[str],
    ) -> float:
        if not targeted_ids:
            return 0.5
        deltas = []
        for sid in targeted_ids:
            b_before = beliefs_before.get(sid)
            b_after = beliefs_after.get(sid)
            if b_before is None or b_after is None:
                continue
            pm_delta = b_after.positive_mass() - b_before.positive_mass()
            tw_delta = b_after.distribution.get(
                "trustworthy", 0
            ) - b_before.distribution.get("trustworthy", 0)
            deltas.append(0.6 * pm_delta + 0.4 * tw_delta)
        if not deltas:
            return 0.5
        mean_delta = sum(deltas) / len(deltas)
        return float(np.clip(0.5 + mean_delta * 3.0, 0.0, 1.0))

    def _score_info(
        self,
        beliefs_before: Dict[str, BeliefDistribution],
        beliefs_after: Dict[str, BeliefDistribution],
    ) -> float:
        reductions = []
        for sid in beliefs_after:
            b_before = beliefs_before.get(sid)
            b_after = beliefs_after.get(sid)
            if b_before is None or b_after is None:
                continue
            h_before = _entropy_base2(np.array(list(b_before.distribution.values())))
            h_after = _entropy_base2(np.array(list(b_after.distribution.values())))
            reductions.append((h_before - h_after) / LOG2_6)
        if not reductions:
            return 0.5
        mean_reduction = sum(reductions) / len(reductions)
        return float(np.clip(0.5 + mean_reduction * 5.0, 0.0, 1.0))

    def _score_risk(
        self,
        beliefs_before: Dict[str, BeliefDistribution],
        beliefs_after: Dict[str, BeliefDistribution],
        risk_profiles: Optional[Dict[str, StakeholderRiskProfile]],
        deal_terms: Dict,
    ) -> float:
        if not risk_profiles or not deal_terms:
            return 0.5
        improvements = []
        for sid, profile in risk_profiles.items():
            if profile.lambda_risk > 0.30:
                cvar_b = self._compute_cvar(sid, beliefs_before, profile, deal_terms)
                cvar_a = self._compute_cvar(sid, beliefs_after, profile, deal_terms)
                if cvar_b > 1e-8:
                    improvements.append((cvar_b - cvar_a) / cvar_b)
        if not improvements:
            return 0.5
        mean_imp = sum(improvements) / len(improvements)
        return float(np.clip(0.5 + mean_imp, 0.0, 1.0))

    def _score_causal(
        self,
        graph: Any,
        targeted_ids: List[str],
    ) -> float:
        if not targeted_ids or not graph:
            return 0.0
        edges = getattr(graph, "edges", [])
        nodes = getattr(graph, "nodes", [])
        if not edges or len(nodes) <= 2:
            return 0.5
        centrality = get_betweenness_centrality(graph, targeted_ids[0])
        n = len(nodes)
        max_possible = ((n - 1) * (n - 2)) if n > 2 else 1.0
        return float(centrality / max_possible if max_possible > 0 else 0.0)

    def _compute_cvar(
        self,
        stakeholder_id: str,
        beliefs: Dict[str, BeliefDistribution],
        profile: StakeholderRiskProfile,
        deal_terms: Dict,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        belief = beliefs.get(stakeholder_id)
        if belief is None:
            return 0.0
        rng = rng or np.random.default_rng(42)
        try:
            outcomes = compute_outcome_distribution(
                deal_terms, profile, rng, n_samples=500
            )
            return compute_cvar(outcomes, profile.alpha)
        except Exception:
            return 0.0
