"""
CCIGrader — Contract Closure Index v3

Measures sustainable, implementable consensus across 4 dimensions:
  Consensus (40%)         — weighted satisfaction avg with weakest-link penalty
  Implementation Risk     — multiplicative: CTO+Ops satisfaction post-signature
  Efficiency (15%)        — pacing penalty, not raw speed
  Execution Penalty       — malformed output penalty

Weighs are STAGE-DEPENDENT. Who can block changes through the deal lifecycle.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models import DealRoomState

STAGE_WEIGHTS = {
    "evaluation": {
        "CFO": 0.35,
        "CTO": 0.30,
        "Legal": 0.15,
        "Procurement": 0.15,
        "Ops": 0.05,
    },
    "negotiation": {
        "CFO": 0.30,
        "CTO": 0.25,
        "Legal": 0.20,
        "Procurement": 0.15,
        "Ops": 0.10,
    },
    "legal_review": {
        "CFO": 0.25,
        "CTO": 0.15,
        "Legal": 0.35,
        "Procurement": 0.20,
        "Ops": 0.05,
    },
    "final_approval": {
        "CFO": 0.40,
        "CTO": 0.15,
        "Legal": 0.25,
        "Procurement": 0.10,
        "Ops": 0.10,
    },
}


class CCIGrader:
    @staticmethod
    def compute(state: "DealRoomState") -> float:
        """
        Returns CCI in [0.0, 1.0].
        Returns 0.0 if deal not closed or deal failed (veto, timeout, mass blocking).
        """
        if not state.deal_closed or state.deal_failed:
            return 0.0

        stage = state.deal_stage
        if stage not in STAGE_WEIGHTS:
            stage = "final_approval"
        weights = STAGE_WEIGHTS[stage]

        total_w = sum(weights.values())
        assert abs(total_w - 1.0) < 0.01, f"Weights sum to {total_w}, must be 1.0"

        sat = {k: state.satisfaction.get(k, 0.5) for k in weights}
        min_sat = min(sat.values())

        weighted_avg = sum(sat[k] * weights[k] for k in weights)

        weakest_link = 0.6 + 0.4 * min(1.0, min_sat / 0.35)
        consensus = max(0.0, min(1.0, weighted_avg * weakest_link))

        sat_cto = sat.get("CTO", 0.5)
        sat_ops = sat.get("Ops", 0.5)
        impl_risk = max(0.5, 1.0 - (0.45 * (1.0 - sat_cto) + 0.35 * (1.0 - sat_ops)))

        efficiency = max(
            0.1, 1.0 - ((state.round_number / state.max_rounds) ** 1.3) * 0.4
        )

        exec_penalty = min(0.20, state.validation_failures * 0.04)

        raw = (consensus * impl_risk * efficiency) - exec_penalty
        return round(max(0.0, min(1.0, raw)), 4)
