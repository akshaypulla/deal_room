"""
OutputValidator — 3-layer output parsing. Strict mode only.
No LLM extraction layer — training must be fully deterministic.

Layer 1: JSON extraction from markdown code blocks or inline JSON
Layer 2: Heuristic keyword extraction from free text
Layer 3: Safe fallback — never raises, always returns a valid dict
"""

import re
import json
from typing import Tuple, Dict, Optional

VALID_ACTION_TYPES = {
    "direct_message",
    "group_proposal",
    "backchannel",
    "send_document",
    "concession",
    "walkaway_signal",
    "reframe_value_prop",
    "exec_escalation",
}

VALID_TARGETS = {
    "cfo",
    "cto",
    "legal",
    "procurement",
    "ops",
    "all",
    "cto_cfo",
    "legal_procurement",
}


class OutputValidator:
    def __init__(self, mode: str = "strict"):
        self.mode = mode

    def validate(self, raw: str) -> Tuple[Dict, float]:
        """
        Returns (normalized_dict, confidence).
        confidence 1.0 = clean JSON parse
        confidence 0.6 = heuristic extraction
        confidence 0.0 = fallback used
        """
        if not raw:
            return self._fallback(raw), 0.0

        for pattern in [
            r"```json\s*(.*?)\s*```",
            r"```\s*(.*?)\s*```",
            r'(\{[^{}]*"action_type"[^{}]*\})',
        ]:
            match = re.search(pattern, raw, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(1).strip())
                    return self._normalize(data), 1.0
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

        action_type = self._extract_action_type(raw)
        target = self._extract_target(raw)
        if action_type:
            return self._normalize(
                {
                    "action_type": action_type,
                    "target": target or "all",
                    "message": raw[:300].strip(),
                }
            ), 0.6

        return self._fallback(raw), 0.0

    def _normalize(self, data: Dict) -> Dict:
        at = data.get("action_type", "direct_message")
        if at not in VALID_ACTION_TYPES:
            at = "direct_message"
        tgt = str(data.get("target", "all"))
        if tgt.lower() not in VALID_TARGETS:
            tgt = "all"
        return {
            "action_type": at,
            "target": tgt,
            "message": str(data.get("message", ""))[:500],
            "channel": data.get("channel", "formal"),
            "mode": data.get("mode", "async_email"),
            "documents": data.get("documents", []),
            "proposed_terms": data.get("proposed_terms", None),
        }

    def _fallback(self, raw: str) -> Dict:
        return {
            "action_type": "direct_message",
            "target": "all",
            "message": (raw[:200] if raw else "Continuing our discussion."),
            "channel": "formal",
            "mode": "async_email",
            "documents": [],
            "proposed_terms": None,
        }

    def _extract_action_type(self, raw: str) -> Optional[str]:
        raw_lower = raw.lower()
        for at in VALID_ACTION_TYPES:
            if at.replace("_", " ") in raw_lower or at in raw_lower:
                return at
        return None

    def _extract_target(self, raw: str) -> Optional[str]:
        raw_lower = raw.lower()
        for t in [
            "cto_cfo",
            "legal_procurement",
            "cfo",
            "cto",
            "legal",
            "procurement",
            "ops",
            "all",
        ]:
            if t in raw_lower:
                return t
        return "all"
