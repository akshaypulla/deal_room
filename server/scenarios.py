"""
Task configurations for all 3 DealRoom tasks.
Every key listed here is required — KeyError otherwise.
"""

STAKEHOLDER_IDS = ["CFO", "CTO", "Legal", "Procurement", "Ops"]

SCENARIOS = {
    "aligned": {
        "max_rounds": 8,
        "veto_threshold": 0.68,
        "block_threshold": 0.28,
        "shock_prob": 0.04,
        "round_3_hint": None,
        "days_to_deadline": 45,
        "initial_beliefs": {
            "CFO": {"competence": 0.55, "risk_tolerance": 0.52, "pricing_rigor": 0.50},
            "CTO": {"competence": 0.58, "risk_tolerance": 0.55, "pricing_rigor": 0.48},
            "Legal": {
                "competence": 0.50,
                "risk_tolerance": 0.45,
                "pricing_rigor": 0.52,
            },
            "Procurement": {
                "competence": 0.53,
                "risk_tolerance": 0.50,
                "pricing_rigor": 0.55,
            },
            "Ops": {"competence": 0.60, "risk_tolerance": 0.58, "pricing_rigor": 0.45},
        },
        "initial_satisfaction": {
            "CFO": 0.54,
            "CTO": 0.56,
            "Legal": 0.48,
            "Procurement": 0.52,
            "Ops": 0.60,
        },
        "coalition_tension": None,
        "description": (
            "Low-friction enterprise deal. All stakeholders broadly favorable. "
            "Minor concerns from Legal (liability) and CFO (ROI timeline). "
            "Tests: correct document sequencing, stakeholder engagement order."
        ),
    },
    "conflicted": {
        "max_rounds": 10,
        "veto_threshold": 0.52,
        "block_threshold": 0.32,
        "shock_prob": 0.07,
        "round_3_hint": None,
        "days_to_deadline": 30,
        "initial_beliefs": {
            "CFO": {"competence": 0.42, "risk_tolerance": 0.35, "pricing_rigor": 0.48},
            "CTO": {"competence": 0.44, "risk_tolerance": 0.38, "pricing_rigor": 0.42},
            "Legal": {
                "competence": 0.38,
                "risk_tolerance": 0.32,
                "pricing_rigor": 0.50,
            },
            "Procurement": {
                "competence": 0.40,
                "risk_tolerance": 0.35,
                "pricing_rigor": 0.52,
            },
            "Ops": {"competence": 0.55, "risk_tolerance": 0.52, "pricing_rigor": 0.40},
        },
        "initial_satisfaction": {
            "CFO": 0.50,
            "CTO": 0.52,
            "Legal": 0.46,
            "Procurement": 0.48,
            "Ops": 0.62,
        },
        "coalition_tension": {
            "cto_cfo": "conflict",
            "legal_procurement": "alliance",
        },
        "description": (
            "Active CTO-CFO tension from failed prior project. "
            "Legal-Procurement blocking alliance. Ops champion isolated. "
            "Tests: coalition sequencing, independent credibility building, veto avoidance."
        ),
    },
    "hostile_acquisition": {
        "max_rounds": 10,
        "veto_threshold": 0.40,
        "block_threshold": 0.32,
        "shock_prob": 0.11,
        "round_3_hint": (
            "AE note: Post-acquisition compliance team from acquiring EU parent has joined review. "
            "Expect heightened data sovereignty scrutiny. "
            "Align all messaging with GDPR baseline requirements immediately."
        ),
        "days_to_deadline": 20,
        "initial_beliefs": {
            "CFO": {"competence": 0.40, "risk_tolerance": 0.32, "pricing_rigor": 0.45},
            "CTO": {"competence": 0.42, "risk_tolerance": 0.35, "pricing_rigor": 0.40},
            "Legal": {
                "competence": 0.35,
                "risk_tolerance": 0.28,
                "pricing_rigor": 0.48,
            },
            "Procurement": {
                "competence": 0.38,
                "risk_tolerance": 0.32,
                "pricing_rigor": 0.50,
            },
            "Ops": {"competence": 0.50, "risk_tolerance": 0.46, "pricing_rigor": 0.38},
        },
        "initial_satisfaction": {
            "CFO": 0.52,
            "CTO": 0.54,
            "Legal": 0.48,
            "Procurement": 0.52,
            "Ops": 0.62,
        },
        "coalition_tension": {
            "cto_cfo": "conflict",
            "legal_procurement": "alliance",
        },
        "description": (
            "Post-acquisition authority shift. New EU compliance requirements. "
            "Compressed timeline. "
            "Tests: adaptive stakeholder mapping, GDPR framing, precision under uncertainty."
        ),
    },
}


def expand_targets(target: str) -> list:
    """Expand target string to list of individual stakeholder IDs."""
    VALID_SUBGROUPS = {
        "cto_cfo": ["CTO", "CFO"],
        "legal_procurement": ["Legal", "Procurement"],
    }
    ALL_STAKEHOLDER_IDS = ["CFO", "CTO", "Legal", "Procurement", "Ops"]
    t = target.lower().strip()
    if t == "all":
        return list(ALL_STAKEHOLDER_IDS)
    if t in VALID_SUBGROUPS:
        return VALID_SUBGROUPS[t]
    for sid in ALL_STAKEHOLDER_IDS:
        if sid.lower() == t:
            return [sid]
    return []
