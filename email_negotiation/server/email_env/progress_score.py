import math
from typing import Dict, List, Tuple


STAKEHOLDER_WEIGHTS = {
    "ExecSponsor": 0.35,
    "Legal": 0.25,
    "Finance": 0.20,
    "CTO": 0.20,
    "TechLead": 0.10,
    "Procurement": 0.10,
    "Operations": 0.05,
}

DEAL_STAGE_ORDER = ["initial", "qualification", "discovery", "proposal", "negotiation", "closing", "closed"]


def compute_progress_score(
    alignment_scores: Dict[str, float],
    docs_delivered: List[str],
    docs_required: List[str],
    deal_stage: str,
    concerns_resolved: int,
    concerns_total: int,
) -> float:
    S = _compute_stakeholder_alignment(alignment_scores)
    D = _compute_doc_completeness(docs_delivered, docs_required)
    G = _compute_deal_stage_progress(deal_stage)
    R = _compute_concern_resolution(concerns_resolved, concerns_total)

    score = (S ** 0.4) * (D ** 0.2) * (G ** 0.2) * (R ** 0.2)
    return max(0.0, min(1.0, score))


def _compute_stakeholder_alignment(alignment_scores: Dict[str, float]) -> float:
    if not alignment_scores:
        return 0.0
    total = 0.0
    for sid, weight in STAKEHOLDER_WEIGHTS.items():
        total += weight * alignment_scores.get(sid, 0.3)
    return total


def _compute_doc_completeness(docs_delivered: List[str], docs_required: List[str]) -> float:
    if not docs_required:
        return 1.0
    delivered_set = set(docs_delivered)
    required_set = set(docs_required)
    return len(delivered_set & required_set) / len(required_set)


def _compute_deal_stage_progress(deal_stage: str) -> float:
    try:
        idx = DEAL_STAGE_ORDER.index(deal_stage)
        return idx / (len(DEAL_STAGE_ORDER) - 1)
    except ValueError:
        return 0.0


def _compute_concern_resolution(resolved: int, total: int) -> float:
    if total == 0:
        return 1.0
    return resolved / total


def compute_shaping_reward(current_score: float, previous_score: float) -> float:
    delta = current_score - previous_score
    shaping = max(-0.5, min(0.5, 2.0 * delta))
    if previous_score > 0.05 and delta < 0.005:
        shaping -= 0.05
    return shaping


def apply_progress_gating(tone_base: float, progress_score: float) -> float:
    return tone_base * (progress_score ** 0.5)


def check_early_stopping(delta_history: List[float], window: int = 3, threshold: float = 0.01) -> bool:
    if len(delta_history) < window:
        return False
    recent = delta_history[-window:]
    avg = sum(recent) / len(recent)
    return abs(avg) < threshold
