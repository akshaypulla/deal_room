"""
Stakeholder engine for DealRoom V2.5.

This module owns role-specific update rules, response generation, banding, and
the bounded relationship propagation model.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, List

import numpy as np

from server.scenarios import ROLE_LIBRARY

DOCUMENT_EFFECTS = {
    "roi_model": {"finance": 0.10, "executive_sponsor": 0.06, "procurement": 0.04},
    "implementation_timeline": {"technical": 0.09, "operations": 0.10, "executive_sponsor": 0.03},
    "security_cert": {"technical": 0.06, "legal_compliance": 0.10, "procurement": 0.04},
    "dpa": {"legal_compliance": 0.12, "procurement": 0.04},
    "vendor_packet": {"procurement": 0.10, "finance": 0.03},
    "reference_case": {"finance": 0.04, "operations": 0.05, "executive_sponsor": 0.05},
    "support_plan": {"operations": 0.08, "technical": 0.05},
}

OPENING_MESSAGES = {
    "finance": "I need clarity on business impact and spend discipline before this can move.",
    "technical": "I need to understand implementation feasibility and the delivery burden on my team.",
    "legal_compliance": "We will need defensible compliance language and clear contractual commitments.",
    "procurement": "Process fit matters here. We need the right documentation and a clean approval path.",
    "operations": "Our team needs a realistic rollout and confidence that this will not disrupt execution.",
    "executive_sponsor": "I care about political safety, momentum, and whether this can survive internal scrutiny.",
}

RESPONSE_TEMPLATES = {
    "supporter": {
        "finance": "The financial case is getting stronger. If we keep this disciplined, I can help move it forward.",
        "technical": "This is looking more workable. Keep the implementation detail tight and I can support it.",
        "legal_compliance": "This is becoming easier to defend. We still need precision, but the direction is solid.",
        "procurement": "This is moving in the right direction. If process stays clean, I can help accelerate it.",
        "operations": "The rollout story is improving. I can advocate for this if delivery stays credible.",
        "executive_sponsor": "The internal case is becoming easier to sponsor. Keep reducing risk and ambiguity.",
    },
    "workable": {
        "finance": "I can see the outline, but I still need more confidence on spend and downside protection.",
        "technical": "This is directionally fine, but I still need evidence that delivery is realistic.",
        "legal_compliance": "We are closer, but compliance precision still matters more than optimism.",
        "procurement": "This is workable if the missing process pieces are handled properly.",
        "operations": "I can work with this, but I still need confidence that the rollout fits our window.",
        "executive_sponsor": "This is not blocked, but it is not yet easy to sponsor internally.",
    },
    "neutral": {
        "finance": "I still do not have enough clarity to defend this internally.",
        "technical": "I still see unanswered implementation risk here.",
        "legal_compliance": "I still need more precise commitments before this is reviewable.",
        "procurement": "The process is not clean enough yet for me to advance this.",
        "operations": "I still need a more grounded implementation story.",
        "executive_sponsor": "There is still too much uncertainty for me to put my name behind this.",
    },
    "blocker": {
        "finance": "I am not prepared to advance this. The commercial and governance risk is too high.",
        "technical": "This is not credible enough technically for me to support.",
        "legal_compliance": "This is too risky and under-specified to take forward.",
        "procurement": "This is off-process and not ready to progress.",
        "operations": "I do not trust the delivery picture right now.",
        "executive_sponsor": "This is politically unsafe in its current form.",
    },
}

ROLE_TONE_WEIGHTS = {
    "finance": {"credible": 1.2, "specific": 1.2, "pushy": -1.2, "evasive": -1.0, "adaptive": 0.6},
    "technical": {"credible": 1.0, "specific": 1.3, "pushy": -0.8, "evasive": -0.9, "adaptive": 1.1},
    "legal_compliance": {"credible": 1.3, "specific": 1.0, "pushy": -1.3, "evasive": -1.4, "adaptive": 0.5},
    "procurement": {"credible": 0.8, "specific": 0.7, "pushy": -0.8, "evasive": -0.7, "adaptive": 0.6},
    "operations": {"credible": 0.8, "specific": 1.0, "pushy": -0.7, "evasive": -0.8, "adaptive": 1.2},
    "executive_sponsor": {"credible": 1.0, "specific": 0.8, "pushy": -1.0, "evasive": -1.0, "adaptive": 0.9},
}


def approval_band(approval: float, resistance: float) -> str:
    if resistance > 0.65 or approval < 0.48:
        return "blocker"
    if approval >= 0.72:
        return "supporter"
    if approval >= 0.62:
        return "workable"
    return "neutral"


class StakeholderEngine:
    def __init__(self):
        self.state = None
        self.rng = None

    def reset(self, state, rng: np.random.Generator):
        self.state = state
        self.rng = rng

    def generate_opening(self) -> Dict[str, str]:
        return {
            stakeholder_id: OPENING_MESSAGES[payload["role"]]
            for stakeholder_id, payload in self.state.stakeholders.items()
        }

    def apply_action(
        self,
        action_dict: Dict[str, object],
        analysis: Dict[str, object],
    ) -> Dict[str, object]:
        target_ids: List[str] = list(action_dict.get("target_ids", []))
        tone_scores = analysis.get("tone_scores", {})
        artifact_matches = analysis.get("artifact_matches", [])
        request_matches = analysis.get("request_matches", [])
        deltas: Dict[str, Dict[str, float]] = {}
        satisfied_requests: List[Dict[str, str]] = []
        touched_bands: Dict[str, str] = {}

        for stakeholder_id in target_ids:
            private = self.state.stakeholder_private[stakeholder_id]
            role = private["role"]
            prior_band = approval_band(private["approval"], private["private_resistance"])

            trust_delta = self._tone_impact(role, tone_scores)
            approval_delta = trust_delta * 0.7
            fit_delta = max(0.0, tone_scores.get("specific", 0.0) - tone_scores.get("evasive", 0.0)) * 0.04
            resistance_delta = -0.25 * max(0.0, trust_delta)

            if action_dict.get("action_type") == "backchannel":
                trust_delta += 0.03
            if action_dict.get("action_type") == "exec_escalation":
                approval_delta -= 0.02
                trust_delta -= 0.03

            for artifact in artifact_matches:
                artifact_delta = DOCUMENT_EFFECTS.get(artifact, {}).get(role, 0.0)
                approval_delta += artifact_delta
                fit_delta += artifact_delta * 0.6

            for matched in request_matches:
                if matched["stakeholder_id"] != stakeholder_id:
                    continue
                satisfied_requests.append(matched)
                approval_delta += 0.03
                trust_delta += 0.02
                remaining = self.state.requested_artifacts.get(stakeholder_id, [])
                if matched["artifact"] in remaining:
                    remaining.remove(matched["artifact"])

            private["trust"] = self._clamp(private["trust"] + trust_delta)
            private["approval"] = self._clamp(
                min(private["approval"] + approval_delta, self.state.approval_caps.get(stakeholder_id, 1.0))
            )
            private["perceived_fit"] = self._clamp(private["perceived_fit"] + fit_delta)
            private["private_resistance"] = self._clamp(private["private_resistance"] + resistance_delta)
            private["band_history"].append(prior_band)

            deltas[stakeholder_id] = {
                "trust": round(trust_delta, 4),
                "approval": round(approval_delta, 4),
                "perceived_fit": round(fit_delta, 4),
                "private_resistance": round(resistance_delta, 4),
            }
            touched_bands[stakeholder_id] = approval_band(
                private["approval"], private["private_resistance"]
            )

        propagation = self._propagate_relationships(deltas)
        touched_bands.update(
            {
                stakeholder_id: approval_band(
                    private["approval"], private["private_resistance"]
                )
                for stakeholder_id, private in self.state.stakeholder_private.items()
            }
        )
        return {
            "deltas": deltas,
            "propagation": propagation,
            "satisfied_requests": satisfied_requests,
            "bands": touched_bands,
        }

    def _propagate_relationships(self, deltas: Dict[str, Dict[str, float]]) -> List[Dict[str, float]]:
        applied = []
        for edge in self.state.relationship_edges:
            source = edge["source"]
            target = edge["target"]
            if source not in deltas or target not in self.state.stakeholder_private:
                continue
            source_delta = deltas[source]
            target_private = self.state.stakeholder_private[target]
            approval_delta = 0.0
            resistance_delta = 0.0
            if edge["type"] == "alliance":
                if abs(source_delta["approval"]) > 0.08:
                    approval_delta += source_delta["approval"] * 0.4
                if abs(source_delta["private_resistance"]) > 0.08:
                    resistance_delta += source_delta["private_resistance"] * 0.4
            elif edge["type"] == "conflict":
                if source_delta["approval"] < -0.02 or source_delta["private_resistance"] > 0.02:
                    resistance_delta += 0.04
            elif edge["type"] == "sponsor":
                source_private = self.state.stakeholder_private[source]
                if source_private["trust"] >= 0.72 and source_private["approval"] >= 0.72:
                    approval_delta += 0.03

            if approval_delta or resistance_delta:
                target_private["approval"] = self._clamp(
                    min(
                        target_private["approval"] + approval_delta,
                        self.state.approval_caps.get(target, 1.0),
                    )
                )
                target_private["private_resistance"] = self._clamp(
                    target_private["private_resistance"] + resistance_delta
                )
                applied.append(
                    {
                        "source": source,
                        "target": target,
                        "type": edge["type"],
                        "approval_delta": round(approval_delta, 4),
                        "resistance_delta": round(resistance_delta, 4),
                    }
                )
        return applied

    def generate_responses(self, target_ids: List[str]) -> Dict[str, str]:
        responses = {}
        visible_targets = target_ids or list(self.state.stakeholders.keys())
        for stakeholder_id in visible_targets:
            if stakeholder_id not in self.state.stakeholders:
                continue
            private = self.state.stakeholder_private[stakeholder_id]
            band = approval_band(private["approval"], private["private_resistance"])
            role = private["role"]
            response = RESPONSE_TEMPLATES[band][role]
            if self.state.requested_artifacts.get(stakeholder_id):
                missing = self.state.requested_artifacts[stakeholder_id][0].replace("_", " ")
                response = f"{response} I still need the {missing} to move this forward."
            responses[stakeholder_id] = response
        return responses

    def _tone_impact(self, role: str, tone_scores: Dict[str, float]) -> float:
        weights = ROLE_TONE_WEIGHTS[role]
        collaborative = tone_scores.get("collaborative", 0.0) * 0.05
        adaptive = tone_scores.get("adaptive", 0.0) * 0.04 * weights.get("adaptive", 1.0)
        credible = tone_scores.get("credible", 0.0) * 0.05 * weights.get("credible", 1.0)
        specific = tone_scores.get("specific", 0.0) * 0.05 * weights.get("specific", 1.0)
        pushy = tone_scores.get("pushy", 0.0) * 0.05 * abs(weights.get("pushy", -1.0))
        evasive = tone_scores.get("evasive", 0.0) * 0.05 * abs(weights.get("evasive", -1.0))
        return round(collaborative + adaptive + credible + specific - pushy - evasive, 4)

    @staticmethod
    def _clamp(value: float) -> float:
        return round(float(np.clip(value, 0.0, 1.0)), 4)
