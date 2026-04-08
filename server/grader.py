"""
Deterministic terminal grader for DealRoom V2.5.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models import DealRoomState


class CCIGrader:
    MIN_SCORE = 0.01
    MAX_SCORE = 0.99

    WEIGHTS = {
        "approval_completeness": 0.35,
        "constraint_satisfaction": 0.25,
        "term_feasibility": 0.15,
        "relationship_durability": 0.15,
        "efficiency": 0.10,
    }

    @classmethod
    def compute(cls, state: "DealRoomState") -> float:
        if not state.deal_closed or state.deal_failed:
            return cls.MIN_SCORE
        if not state.feasibility_state.get("is_feasible", False):
            return cls.MIN_SCORE
        if any(not constraint.get("resolved") for constraint in state.hidden_constraints.values()):
            return cls.MIN_SCORE

        mandatory_ids = [
            stakeholder_id
            for stakeholder_id, payload in state.stakeholder_private.items()
            if payload.get("mandatory")
        ]
        for stakeholder_id in mandatory_ids:
            private = state.stakeholder_private[stakeholder_id]
            if private["approval"] < 0.62 or private["private_resistance"] > 0.65:
                return cls.MIN_SCORE
        for stakeholder_id, private in state.stakeholder_private.items():
            if private.get("veto_power") and private["private_resistance"] > 0.65:
                return cls.MIN_SCORE

        approval_score = cls._approval_completeness(state, mandatory_ids)
        constraint_score = cls._constraint_satisfaction(state)
        feasibility_score = cls._term_feasibility(state)
        relationship_score = cls._relationship_durability(state)
        efficiency_score = cls._efficiency(state)

        score = (
            approval_score * cls.WEIGHTS["approval_completeness"]
            + constraint_score * cls.WEIGHTS["constraint_satisfaction"]
            + feasibility_score * cls.WEIGHTS["term_feasibility"]
            + relationship_score * cls.WEIGHTS["relationship_durability"]
            + efficiency_score * cls.WEIGHTS["efficiency"]
        )
        return round(max(cls.MIN_SCORE, min(cls.MAX_SCORE, score)), 4)

    @staticmethod
    def _approval_completeness(state: "DealRoomState", mandatory_ids) -> float:
        if not mandatory_ids:
            return 1.0
        approvals = [state.stakeholder_private[item]["approval"] for item in mandatory_ids]
        return min(1.0, sum(approvals) / len(approvals))

    @staticmethod
    def _constraint_satisfaction(state: "DealRoomState") -> float:
        constraints = list(state.hidden_constraints.values())
        if not constraints:
            return 1.0
        resolved = sum(1 for item in constraints if item.get("resolved"))
        return resolved / len(constraints)

    @staticmethod
    def _term_feasibility(state: "DealRoomState") -> float:
        penalty = min(0.20, 0.05 * len(state.feasibility_state.get("violations", [])))
        return max(0.0, 1.0 - penalty)

    @staticmethod
    def _relationship_durability(state: "DealRoomState") -> float:
        trusts = [payload["trust"] for payload in state.stakeholder_private.values()]
        mark_penalty = 0.0
        for payload in state.stakeholder_private.values():
            mark_penalty += 0.03 * len(payload.get("permanent_marks", []))
        average = sum(trusts) / len(trusts) if trusts else 0.0
        return max(0.0, min(1.0, average - mark_penalty))

    @staticmethod
    def _efficiency(state: "DealRoomState") -> float:
        if state.max_rounds <= 0:
            return 0.0
        return max(0.1, 1.0 - ((state.round_number / state.max_rounds) ** 1.25) * 0.45)
