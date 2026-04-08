"""
ClaimsTracker — Regex-only numerical contradiction detection.

DESIGN DECISION (Issue 1):
Target expansion is done by the CALLER (environment), not here.
This function receives a single stakeholder ID string only.
Calling expand_targets() here was redundant and violated single responsibility.
"""

import re
from typing import Dict, List

CLAIM_PATTERNS = {
    "implementation_weeks": (
        r"(?i)(?:go.?live|deploy|implementation|deliver|complete)"
        r"(?:\s+(?:in|within|takes?|will\s+take))?\s+(\d+)\s*(?:weeks?|wk)"
    ),
    "team_size": (
        r"(?i)(?:team|resources?)\s+of\s+(\d+)"
        r"|(\d+)\s*(?:dedicated\s+)?(?:engineers?|members?)"
    ),
    "price_commit": r"(?i)\$\s*(\d{1,3}(?:,\d{3})*|\d{4,})",
}

DEVIATION_TOLERANCE = 0.15

VALID_SUBGROUPS: Dict[str, List[str]] = {
    "cto_cfo": ["CTO", "CFO"],
    "legal_procurement": ["Legal", "Procurement"],
}

ALL_STAKEHOLDER_IDS = ["CFO", "CTO", "Legal", "Procurement", "Ops"]


def expand_targets(target: str) -> List[str]:
    """
    Expand target string to list of individual stakeholder IDs.

    Uses explicit subgroup registry — never splits on underscore.
    Splitting on underscore would break stakeholder names containing underscores
    and would produce lowercase IDs that don't match the dict keys.

    Returns empty list for unknown targets — caller decides how to handle.
    """
    t = target.lower().strip()
    if t == "all":
        return list(ALL_STAKEHOLDER_IDS)
    if t in VALID_SUBGROUPS:
        return VALID_SUBGROUPS[t]
    for sid in ALL_STAKEHOLDER_IDS:
        if sid.lower() == t:
            return [sid]
    return []


class ClaimsTracker:
    """
    Tracks numerical commitments per stakeholder.
    Returns True when a new value deviates >15% from the prior commitment.

    Receives individual stakeholder IDs only.
    Target expansion is done by the environment before calling this.
    """

    def __init__(self):
        self.claims: Dict[str, List[float]] = {}

    def reset(self):
        self.claims = {}

    def extract_and_track(self, target: str, message: str) -> bool:
        """
        target: single stakeholder ID (e.g. "CFO"), already expanded by caller.
        Returns True if a contradiction is detected.
        """
        if not message or not target:
            return False

        triggered = False
        for claim_type, pattern in CLAIM_PATTERNS.items():
            match = re.search(pattern, message)
            if not match:
                continue
            raw_val = next((g for g in match.groups() if g is not None), None)
            if raw_val is None:
                continue
            val = float(raw_val.replace(",", ""))
            key = f"{target}:{claim_type}"
            if key in self.claims and self.claims[key]:
                last = self.claims[key][-1]
                if last > 0 and abs(val - last) / last > DEVIATION_TOLERANCE:
                    triggered = True
            self.claims.setdefault(key, []).append(val)

        return triggered
