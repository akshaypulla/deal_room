"""
Stakeholders module for DealRoom v3 - CVaR preferences and archetypes.
"""

from .archetypes import (
    ARCHETYPE_PROFILES,
    get_all_archetypes,
    get_archetype,
)
from .cvar_preferences import (
    StakeholderRiskProfile,
    check_veto_trigger,
    compute_cvar,
    compute_deal_quality_score,
    compute_expected_utility,
    compute_outcome_distribution,
    evaluate_deal,
    get_observable_signals,
)

__all__ = [
    "StakeholderRiskProfile",
    "ARCHETYPE_PROFILES",
    "get_archetype",
    "get_all_archetypes",
    "compute_outcome_distribution",
    "compute_cvar",
    "compute_expected_utility",
    "evaluate_deal",
    "check_veto_trigger",
    "get_observable_signals",
    "compute_deal_quality_score",
]
