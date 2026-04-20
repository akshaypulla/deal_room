"""
CVaR-based preference model for DealRoom v3 stakeholders.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class StakeholderRiskProfile:
    stakeholder_id: str
    role: str
    alpha: float
    tau: float
    lambda_risk: float
    utility_weights: Dict[str, float] = field(default_factory=dict)
    uncertainty_domains: List[str] = field(default_factory=list)


def compute_outcome_distribution(
    deal_terms: Dict,
    stakeholder_profile: StakeholderRiskProfile,
    rng: np.random.Generator,
    n_samples: int = 1000,
) -> np.ndarray:
    domain = (
        stakeholder_profile.uncertainty_domains[0]
        if stakeholder_profile.uncertainty_domains
        else "generic"
    )

    base_success_prob = 0.75
    if "compliance" in domain or "regulatory" in domain or "data_protection" in domain:
        base_success_prob = 0.80
        if deal_terms.get("has_dpa") and deal_terms.get("has_security_cert"):
            base_success_prob = 0.92
        elif deal_terms.get("has_dpa") or deal_terms.get("has_security_cert"):
            base_success_prob = 0.86
    elif "cost" in domain or "payment" in domain:
        base_success_prob = 0.70
    elif "implementation" in domain or "operational" in domain:
        base_success_prob = 0.75

    price = deal_terms.get("price", 100000)
    timeline = deal_terms.get("timeline_weeks", 12)
    liability_cap = deal_terms.get("liability_cap", 1000000)

    outcome_adjustment = 0.0
    if liability_cap and liability_cap < 500000:
        outcome_adjustment -= 0.05

    outcomes = []
    for _ in range(n_samples):
        if rng.random() < base_success_prob:
            outcome = 1.0 - (0.1 * rng.random()) + outcome_adjustment
        else:
            severity = rng.random()
            if severity < 0.3:
                outcome = 0.6 + 0.1 * rng.random() + outcome_adjustment
            elif severity < 0.7:
                outcome = 0.3 + 0.2 * rng.random() + outcome_adjustment
            else:
                outcome = 0.0 + 0.2 * rng.random() + outcome_adjustment
        outcomes.append(max(0.0, min(1.0, outcome)))

    return np.array(outcomes)


def compute_cvar(outcomes: np.ndarray, alpha: float) -> float:
    if len(outcomes) == 0:
        return 0.0

    sorted_outcomes = np.sort(outcomes)
    cutoff_index = int(len(sorted_outcomes) * (1 - alpha))
    if cutoff_index >= len(sorted_outcomes):
        cutoff_index = len(sorted_outcomes) - 1

    if cutoff_index < 0:
        return 1.0

    tail_losses = 1.0 - sorted_outcomes[: cutoff_index + 1]
    tail_probs = np.ones(len(tail_losses)) / len(sorted_outcomes)

    total_tail_prob = sum(tail_probs)
    if total_tail_prob < 1e-8:
        return 0.0

    cvar = sum(l * p for l, p in zip(tail_losses, tail_probs)) / total_tail_prob
    return float(cvar)


def compute_expected_utility(outcomes: np.ndarray) -> float:
    if len(outcomes) == 0:
        return 0.0
    return float(np.mean(outcomes))


def evaluate_deal(
    deal_terms: Dict,
    stakeholder_profile: StakeholderRiskProfile,
    rng: np.random.Generator,
    n_samples: int = 500,
) -> Tuple[float, float]:
    outcomes = compute_outcome_distribution(
        deal_terms, stakeholder_profile, rng, n_samples
    )

    expected_utility = compute_expected_utility(outcomes)
    cvar_loss = compute_cvar(outcomes, stakeholder_profile.alpha)

    return expected_utility, cvar_loss


def check_veto_trigger(
    cvar_loss: float, stakeholder_profile: StakeholderRiskProfile
) -> bool:
    return cvar_loss > stakeholder_profile.tau


def get_observable_signals(
    stakeholder_profile: StakeholderRiskProfile, deal_terms: Dict
) -> Dict[str, float]:
    signals = {}

    for domain in stakeholder_profile.uncertainty_domains:
        if "compliance" in domain or "data_protection" in domain:
            signals["compliance_concern"] = (
                0.8 if not deal_terms.get("has_dpa") else 0.2
            )
            break

    if any(
        "cost" in d or "payment" in d for d in stakeholder_profile.uncertainty_domains
    ):
        signals["cost_sensitivity"] = 0.7 if deal_terms.get("price") else 0.3

    signals["risk_tolerance"] = 1.0 - stakeholder_profile.lambda_risk
    signals["tail_sensitivity"] = stakeholder_profile.alpha

    return signals


def compute_deal_quality_score(
    deal_terms: Dict,
    stakeholder_profile: StakeholderRiskProfile,
    rng: np.random.Generator,
) -> float:
    expected_utility, cvar_loss = evaluate_deal(deal_terms, stakeholder_profile, rng)

    lambda_risk = stakeholder_profile.lambda_risk
    quality_score = (1 - lambda_risk) * expected_utility - lambda_risk * cvar_loss

    return max(0.0, min(1.0, quality_score))
