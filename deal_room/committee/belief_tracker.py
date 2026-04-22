"""
BeliefTracker for DealRoom v3 - Bayesian belief updates for stakeholder model.
"""

import math
from typing import Dict, List, Tuple

from .causal_graph import BeliefDistribution, VENDOR_TYPES


LOG2_6 = math.log(6) / math.log(2)

ACTION_LIKELIHOODS = {
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
    "default": {
        "competent": 0.50,
        "incompetent": 0.50,
        "trustworthy": 0.50,
        "deceptive": 0.50,
        "aligned": 0.50,
        "misaligned": 0.50,
    },
}


def _get_likelihood(
    action_type: str, documents: List[Dict], stakeholder_role: str
) -> Dict[str, float]:
    for key, likelihoods in ACTION_LIKELIHOODS.items():
        if key in action_type:
            return likelihoods

    if documents:
        doc_names = [d.get("name", "") for d in documents]
        for key, likelihoods in ACTION_LIKELIHOODS.items():
            for doc in doc_names:
                if doc.lower() in key.lower():
                    return likelihoods

    return ACTION_LIKELIHOODS["default"]


def bayesian_update(
    belief: BeliefDistribution,
    action_type: str,
    documents: List[Dict],
    stakeholder_role: str,
    is_targeted: bool,
) -> BeliefDistribution:
    likelihoods = _get_likelihood(action_type, documents, stakeholder_role)
    damping = 1.0 if is_targeted else 0.7

    new_dist = {}
    for vendor_type, prior_prob in belief.distribution.items():
        likelihood = likelihoods.get(vendor_type, 0.5)
        dampened_likelihood = 1.0 + damping * (likelihood - 1.0)
        new_dist[vendor_type] = prior_prob * dampened_likelihood

    total = sum(new_dist.values())
    new_dist = {k: max(0.01, v / total) for k, v in new_dist.items()}

    probs = [new_dist.get(t, 0.01) for t in VENDOR_TYPES]
    entropy = -sum(p * math.log(p, 2) for p in probs if p > 0)
    confidence = 1.0 - (entropy / LOG2_6 if LOG2_6 > 0 else 0)

    return BeliefDistribution(
        distribution=new_dist,
        stakeholder_role=belief.stakeholder_role,
        confidence=confidence,
        history=belief.history + [(action_type, damping)],
    )


def compute_engagement_level(belief: BeliefDistribution) -> float:
    return belief.positive_mass() - belief.negative_mass()
