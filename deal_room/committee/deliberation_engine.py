"""
Committee deliberation engine for DealRoom v3 - belief propagation with optional MiniMax summary.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .causal_graph import BeliefDistribution, propagate_beliefs
from deal_room.environment.llm_client import generate_deliberation_summary


DELIBERATION_STEPS = {
    "aligned": 3,
    "conflicted": 3,
    "hostile_acquisition": 4,
}


@dataclass
class DeliberationResult:
    updated_beliefs: Dict[str, BeliefDistribution]
    summary_dialogue: Optional[str]
    propagation_deltas: Dict[str, float]


def _minimax_call(prompt: str, max_tokens: int = 100, temperature: float = 0.8) -> str:
    return generate_deliberation_summary(prompt=prompt, context="deliberation")


class CommitteeDeliberationEngine:
    def __init__(
        self,
        graph,  # CausalGraph
        n_deliberation_steps: int = 3,
    ):
        self.graph = graph
        self.n_steps = n_deliberation_steps

    def run(
        self,
        vendor_action,  # DealRoomAction
        beliefs_before_action: Dict[str, BeliefDistribution],
        beliefs_after_vendor_action: Dict[str, BeliefDistribution],
        render_summary: bool = True,
    ) -> DeliberationResult:
        updated_beliefs = propagate_beliefs(
            graph=self.graph,
            beliefs_before_action=beliefs_before_action,
            beliefs_after_action=beliefs_after_vendor_action,
            n_steps=self.n_steps,
        )

        summary = None
        if render_summary and vendor_action.target_ids:
            summary = self._generate_summary(
                beliefs_before=beliefs_before_action,
                beliefs_after=updated_beliefs,
                targeted_stakeholder=vendor_action.target_ids[0],
            )

        return DeliberationResult(
            updated_beliefs=updated_beliefs,
            summary_dialogue=summary,
            propagation_deltas={
                sid: (
                    updated_beliefs[sid].positive_mass()
                    - beliefs_before_action[sid].positive_mass()
                )
                for sid in updated_beliefs
            },
        )

    def _generate_summary(
        self,
        beliefs_before: Dict[str, BeliefDistribution],
        beliefs_after: Dict[str, BeliefDistribution],
        targeted_stakeholder: str,
    ) -> str:
        deltas = {
            sid: abs(
                beliefs_after[sid].positive_mass() - beliefs_before[sid].positive_mass()
            )
            for sid in beliefs_after
            if sid != targeted_stakeholder
        }
        if not deltas:
            return ""

        top_two = sorted(deltas.keys(), key=lambda x: deltas[x], reverse=True)[:2]
        if len(top_two) < 2:
            return ""

        s1, s2 = top_two[0], top_two[1]
        d1 = beliefs_after[s1].positive_mass() - beliefs_before[s1].positive_mass()
        d2 = beliefs_after[s2].positive_mass() - beliefs_before[s2].positive_mass()

        sentiment1 = (
            "cautiously positive"
            if d1 > 0.05
            else ("concerned" if d1 < -0.05 else "neutral")
        )
        sentiment2 = (
            "cautiously positive"
            if d2 > 0.05
            else ("concerned" if d2 < -0.05 else "neutral")
        )

        role_map = {
            "Legal": "General Counsel / Legal",
            "Finance": "CFO / Finance",
            "TechLead": "CTO / Technical Lead",
            "Procurement": "Head of Procurement",
            "Operations": "VP Operations / COO",
            "ExecSponsor": "CEO / Executive Sponsor",
        }
        r1 = role_map.get(s1, s1)
        r2 = role_map.get(s2, s2)

        prompt = f"""Two committee members briefly discuss the vendor after their latest communication.
{s1} ({r1}) is currently {sentiment1} about the vendor.
{s2} ({r2}) is currently {sentiment2} about the vendor.
Write 2-3 turns of realistic internal discussion (no vendor present).
Under 80 words total. Do not invent facts not stated."""

        return _minimax_call(prompt, max_tokens=100, temperature=0.8)
