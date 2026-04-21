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


def _minimax_call(
    prompt: str, max_tokens: int = 220, temperature: float = 0.8, timeout: float = 5.0
) -> str:
    return generate_deliberation_summary(
        prompt=prompt, context="deliberation", timeout=timeout
    )


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
        target_belief_before = beliefs_before.get(targeted_stakeholder)
        target_belief_after = beliefs_after.get(targeted_stakeholder)
        if target_belief_before is None or target_belief_after is None:
            return ""

        pm_before = target_belief_before.positive_mass()
        pm_after = target_belief_after.positive_mass()
        pm_delta = pm_after - pm_before

        confidence_before = getattr(target_belief_before, "confidence", 0.5)
        confidence_after = getattr(target_belief_after, "confidence", 0.5)
        conf_delta = confidence_after - confidence_before

        other_deltas = {}
        for sid, b_after in beliefs_after.items():
            if sid == targeted_stakeholder:
                continue
            b_before = beliefs_before.get(sid)
            if b_before is not None:
                delta = b_after.positive_mass() - b_before.positive_mass()
                if abs(delta) > 0.01:
                    other_deltas[sid] = delta

        prompt = (
            f"Deliberation summary for deal room committee discussion:\n\n"
            f"Targeted stakeholder: {targeted_stakeholder}\n"
            f"Belief shift for targeted: positive mass {pm_before:.2f} -> {pm_after:.2f} (delta {pm_delta:+.2f})\n"
            f"Confidence shift: {confidence_before:.2f} -> {confidence_after:.2f} (delta {conf_delta:+.2f})\n"
        )
        if other_deltas:
            prompt += "Other stakeholder deltas:\n"
            for sid, delta in sorted(other_deltas.items()):
                prompt += f"  {sid}: {delta:+.2f}\n"
        prompt += (
            f"\nSummarize in 2-4 sentences how the committee's understanding evolved. "
            f"Focus on the targeted stakeholder's changed perception and downstream effects."
        )
        try:
            return _minimax_call(prompt, max_tokens=220, temperature=0.8, timeout=5.0)
        except Exception:
            return ""
