"""
Utterance scorer for DealRoom v3 - five-dimensional reward scoring with LLM judge.
"""

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from models import DealRoomAction, DealRoomObservation

from deal_room.environment.llm_client import score_utterance_dimensions


LOOKAHEAD_COST = 0.07

CACHE: Dict[str, float] = {}

LIKELIHOOD_TABLE = {
    "send_document(DPA)_proactive": {
        "competent": 0.85,
        "incompetent": 0.15,
        "trustworthy": 0.80,
        "deceptive": 0.20,
        "aligned": 0.80,
        "misaligned": 0.20,
    },
    "send_document(security_cert)_proactive": {
        "competent": 0.80,
        "incompetent": 0.20,
        "trustworthy": 0.75,
        "deceptive": 0.25,
        "aligned": 0.75,
        "misaligned": 0.25,
    },
    "send_document(roi_model)_to_finance": {
        "competent": 0.75,
        "incompetent": 0.25,
        "trustworthy": 0.60,
        "deceptive": 0.40,
        "aligned": 0.70,
        "misaligned": 0.30,
    },
    "send_document(implementation_timeline)": {
        "competent": 0.70,
        "incompetent": 0.30,
        "trustworthy": 0.60,
        "deceptive": 0.40,
        "aligned": 0.70,
        "misaligned": 0.30,
    },
    "direct_message_role_specific": {
        "competent": 0.70,
        "incompetent": 0.30,
        "trustworthy": 0.65,
        "deceptive": 0.35,
        "aligned": 0.70,
        "misaligned": 0.30,
    },
    "send_document(DPA)_requested": {
        "competent": 0.80,
        "incompetent": 0.20,
        "trustworthy": 0.75,
        "deceptive": 0.25,
        "aligned": 0.75,
        "misaligned": 0.25,
    },
    "send_document(vendor_packet)": {
        "competent": 0.70,
        "incompetent": 0.30,
        "trustworthy": 0.65,
        "deceptive": 0.35,
        "aligned": 0.70,
        "misaligned": 0.30,
    },
    "send_document(support_plan)": {
        "competent": 0.70,
        "incompetent": 0.30,
        "trustworthy": 0.65,
        "deceptive": 0.35,
        "aligned": 0.70,
        "misaligned": 0.30,
    },
}


@dataclass
class UtteranceScore:
    goal: float = 0.0
    trust: float = 0.0
    information: float = 0.0
    risk: float = 0.0
    causal: float = 0.0
    prediction_accuracy: Optional[float] = None
    lookahead_used: bool = False


@dataclass
class ScoringMetrics:
    prediction_accuracy: Optional[float] = None
    lookahead_used: bool = False
    cache_hit: bool = False


def compute_prediction_accuracy(
    predicted_responses: Dict[str, str], actual_responses: Dict[str, str]
) -> float:
    if not predicted_responses or not actual_responses:
        return 0.0

    accuracies = []
    for stakeholder_id in predicted_responses:
        if stakeholder_id in actual_responses:
            pred = predicted_responses[stakeholder_id]
            actual = actual_responses[stakeholder_id]
            if pred == actual:
                accuracies.append(1.0)
            else:
                common_len = min(len(pred), len(actual))
                matches = sum(
                    1 for p, a in zip(pred[:common_len], actual[:common_len]) if p == a
                )
                accuracies.append(matches / max(common_len, 1))

    return sum(accuracies) / len(accuracies) if accuracies else 0.0


def _get_cache_key(message: str, stakeholder_id: str, belief_state_hash: str) -> str:
    return hashlib.sha256(
        f"{message}:{stakeholder_id}:{belief_state_hash}".encode()
    ).hexdigest()[:16]


def _score_risk_heuristic(action: DealRoomAction, obs: DealRoomObservation) -> float:
    if obs.veto_precursors:
        return 0.2
    if action.action_type == "send_document":
        return 0.6
    return 0.5


def _score_causal_heuristic(
    action: DealRoomAction, graph: Any, engagement_levels: Dict[str, float]
) -> float:
    if not action.target_ids or not graph:
        return 0.3
    target = action.target_ids[0]
    if target in engagement_levels and engagement_levels[target] > 0.5:
        return 0.7
    return 0.4


def _get_llm_client():
    try:
        from openai import OpenAI

        api_key = (
            os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")
        )
        api_base = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
        if not api_key:
            return None
        return OpenAI(api_key=api_key, base_url=api_base)
    except Exception:
        return None


class UtteranceScorer:
    def __init__(self, rng: Optional[np.random.Generator] = None):
        self.rng = rng or np.random.default_rng()

    def score(
        self,
        action: DealRoomAction,
        obs: DealRoomObservation,
        stakeholder_id: str,
        engagement_levels: Optional[Dict[str, float]] = None,
        graph: Any = None,
        lookahead_used: bool = False,
        predicted_responses: Optional[Dict[str, str]] = None,
        actual_responses: Optional[Dict[str, str]] = None,
        belief_state_hash: Optional[str] = None,
    ) -> Tuple[UtteranceScore, ScoringMetrics]:
        cache_key = _get_cache_key(
            action.message, stakeholder_id, belief_state_hash or ""
        )
        if cache_key in CACHE:
            cached = CACHE[cache_key]
            metrics = ScoringMetrics(cache_hit=True, lookahead_used=lookahead_used)
            if lookahead_used:
                cached = max(0.0, cached - LOOKAHEAD_COST)
                metrics.prediction_accuracy = compute_prediction_accuracy(
                    predicted_responses or {}, actual_responses or {}
                )
            return UtteranceScore(
                goal=cached,
                trust=cached,
                information=cached,
                risk=cached,
                causal=cached,
                prediction_accuracy=metrics.prediction_accuracy,
                lookahead_used=lookahead_used,
            ), metrics

        role = obs.stakeholders.get(stakeholder_id, {}).get("role", "")
        state_summary = f"stage={obs.deal_stage}, blockers={obs.active_blockers}, momentum={obs.deal_momentum}"
        scoring_prompt = (
            f"You are evaluating a vendor's message in an enterprise B2B negotiation.\n"
            f"State: {state_summary}\n"
            f"Blockers: {obs.active_blockers}\n"
            f"Message: {message}\n"
            f"Role: {role}\n"
            f"Responses: {list(obs.stakeholder_messages.values())}\n"
            f"Return JSON with three scores 0.0-1.0: goal (progress toward deal close), "
            f"trust (relationship quality), info (information gain for the buyer)."
        )
        context = f"utterance_scorer round {getattr(obs, 'round_number', '?')} {stakeholder_id}"
        scores = score_utterance_dimensions(
            scoring_prompt=scoring_prompt, context=context
        )
        goal = scores["goal"]
        trust = scores["trust"]
        information = scores["info"]

        risk = _score_risk_heuristic(action, obs)
        causal = _score_causal_heuristic(action, graph, engagement_levels or {})

        if lookahead_used:
            goal = max(0.0, goal - LOOKAHEAD_COST)

        metrics = ScoringMetrics(lookahead_used=lookahead_used)
        if lookahead_used and predicted_responses and actual_responses:
            metrics.prediction_accuracy = compute_prediction_accuracy(
                predicted_responses, actual_responses
            )

        CACHE[cache_key] = goal

        return UtteranceScore(
            goal=goal,
            trust=trust,
            information=information,
            risk=risk,
            causal=causal,
            prediction_accuracy=metrics.prediction_accuracy,
            lookahead_used=lookahead_used,
        ), metrics
