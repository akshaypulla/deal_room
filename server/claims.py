"""
Commitment ledger for DealRoom V2.5.

The ledger keeps a small rolling history of semantic commitments and detects
contradictions across numeric and qualitative slots.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, List

from server.scenarios import expand_targets

NUMERIC_TOLERANCES = {
    "price": 0.08,
    "timeline_weeks": 0.15,
}

POLARITY_SLOTS = {
    "security_posture",
    "liability",
    "support_level",
    "implementation_commitment",
}


class CommitmentLedger:
    def __init__(self, max_claims: int = 12):
        self.max_claims = max_claims
        self.claims: List[Dict[str, object]] = []

    def reset(self):
        self.claims = []

    def ingest(
        self,
        stakeholder_ids: List[str],
        claim_candidates: List[Dict[str, object]],
        threshold_jitter: Dict[str, float],
    ) -> Dict[str, List[Dict[str, object]]]:
        contradictions: List[Dict[str, object]] = []
        recorded: List[Dict[str, object]] = []
        for stakeholder_id in stakeholder_ids:
            for claim in claim_candidates:
                entry = deepcopy(claim)
                entry["stakeholder_id"] = stakeholder_id
                entry["slot_threshold"] = round(
                    0.78 + float(threshold_jitter.get(str(claim["slot"]), 0.0)), 4
                )
                prior = self._latest_for(stakeholder_id, str(claim["slot"]))
                if prior and self._is_contradiction(prior, entry):
                    contradictions.append(
                        {
                            "stakeholder_id": stakeholder_id,
                            "slot": entry["slot"],
                            "previous": prior,
                            "current": entry,
                        }
                    )
                self.claims.append(entry)
                recorded.append(entry)
        if len(self.claims) > self.max_claims:
            self.claims = self.claims[-self.max_claims :]
        return {"contradictions": contradictions, "recorded": recorded}

    def _latest_for(self, stakeholder_id: str, slot: str) -> Dict[str, object] | None:
        for item in reversed(self.claims):
            if item["stakeholder_id"] == stakeholder_id and item["slot"] == slot:
                return item
        return None

    def _is_contradiction(
        self,
        prior: Dict[str, object],
        current: Dict[str, object],
    ) -> bool:
        slot = str(current["slot"])
        if slot in NUMERIC_TOLERANCES:
            previous_value = float(prior["value"])
            current_value = float(current["value"])
            if previous_value == 0:
                return False
            return abs(previous_value - current_value) / previous_value > NUMERIC_TOLERANCES[slot]
        if slot in POLARITY_SLOTS:
            return prior.get("value") != current.get("value") or prior.get("polarity") != current.get("polarity")
        return False
