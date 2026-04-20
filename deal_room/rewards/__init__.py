"""
Rewards module for DealRoom v3 - utterance scoring and Pareto efficiency.
"""

from .pareto_efficiency import check_pareto_optimality, compute_terminal_reward
from .utterance_scorer import (
    UtteranceScore,
    UtteranceScorer,
    compute_prediction_accuracy,
)

__all__ = [
    "UtteranceScore",
    "UtteranceScorer",
    "check_pareto_optimality",
    "compute_terminal_reward",
    "compute_prediction_accuracy",
]
