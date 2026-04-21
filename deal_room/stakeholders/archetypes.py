"""
Stakeholder risk profiles (archetypes) for DealRoom v3.
"""

from dataclasses import dataclass, field
from typing import Dict, List

from .cvar_preferences import StakeholderRiskProfile


ARCHETYPE_PROFILES: Dict[str, StakeholderRiskProfile] = {}


def _init_archetypes() -> Dict[str, StakeholderRiskProfile]:
    profiles = {
        "Legal": StakeholderRiskProfile(
            stakeholder_id="Legal",
            role="General Counsel / Legal",
            alpha=0.95,
            tau=0.10,
            lambda_risk=0.70,
            veto_power=True,
            utility_weights={
                "compliance_coverage": 0.40,
                "liability_limitation": 0.30,
                "data_protection": 0.20,
                "exit_clauses": 0.10,
            },
            uncertainty_domains=[
                "compliance_breach",
                "data_protection_failure",
                "contractual_ambiguity",
            ],
        ),
        "Finance": StakeholderRiskProfile(
            stakeholder_id="Finance",
            role="CFO / Finance",
            alpha=0.90,
            tau=0.15,
            lambda_risk=0.50,
            veto_power=True,
            utility_weights={
                "roi_clarity": 0.35,
                "payment_terms": 0.25,
                "cost_predictability": 0.25,
                "budget_approval_path": 0.15,
            },
            uncertainty_domains=[
                "payment_default",
                "cost_overrun",
                "budget_reallocation",
            ],
        ),
        "TechLead": StakeholderRiskProfile(
            stakeholder_id="TechLead",
            role="CTO / Technical Lead",
            alpha=0.80,
            tau=0.25,
            lambda_risk=0.30,
            utility_weights={
                "implementation_feasibility": 0.40,
                "integration_quality": 0.30,
                "support_model": 0.20,
                "vendor_technical_depth": 0.10,
            },
            uncertainty_domains=[
                "implementation_failure",
                "integration_complexity",
                "technical_debt",
            ],
        ),
        "Procurement": StakeholderRiskProfile(
            stakeholder_id="Procurement",
            role="Head of Procurement",
            alpha=0.85,
            tau=0.20,
            lambda_risk=0.45,
            utility_weights={
                "contract_compliance": 0.35,
                "price_competitiveness": 0.30,
                "vendor_qualification": 0.25,
                "process_adherence": 0.10,
            },
            uncertainty_domains=[
                "contract_enforceability",
                "vendor_viability",
                "procurement_policy_breach",
            ],
        ),
        "Operations": StakeholderRiskProfile(
            stakeholder_id="Operations",
            role="VP Operations / COO",
            alpha=0.80,
            tau=0.30,
            lambda_risk=0.35,
            utility_weights={
                "operational_continuity": 0.40,
                "implementation_timeline": 0.30,
                "change_management": 0.20,
                "support_responsiveness": 0.10,
            },
            uncertainty_domains=[
                "operational_disruption",
                "timeline_delay",
                "change_management_failure",
            ],
        ),
        "ExecSponsor": StakeholderRiskProfile(
            stakeholder_id="ExecSponsor",
            role="CEO / Executive Sponsor",
            alpha=0.70,
            tau=0.40,
            lambda_risk=0.25,
            veto_power=True,
            utility_weights={
                "strategic_alignment": 0.40,
                "organizational_consensus": 0.30,
                "reputational_risk": 0.20,
                "competitive_advantage": 0.10,
            },
            uncertainty_domains=[
                "reputational_damage",
                "strategic_misalignment",
                "political_risk",
            ],
        ),
    }
    return profiles


ARCHETYPE_PROFILES = _init_archetypes()


def get_archetype(stakeholder_id: str) -> StakeholderRiskProfile:
    return ARCHETYPE_PROFILES.get(stakeholder_id)


def get_all_archetypes() -> Dict[str, StakeholderRiskProfile]:
    return dict(ARCHETYPE_PROFILES)
