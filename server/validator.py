"""
Output validator for DealRoom V2.5.

JSON-first parsing is preserved, but target validation is dynamic and tied to the
active episode roster.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Tuple

from server.scenarios import expand_targets

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

VALID_PROPOSED_TERM_KEYS = {
    "price",
    "timeline_weeks",
    "security_commitments",
    "support_level",
    "liability_cap",
}


class OutputValidator:
    def __init__(self, mode: str = "strict"):
        self.mode = mode

    def validate(
        self,
        raw: str,
        available_targets: Optional[List[str]] = None,
    ) -> Tuple[Dict[str, object], float]:
        available_targets = available_targets or []
        if not raw:
            return self._fallback("empty_message"), 0.0

        for pattern in [
            r"```json\s*(.*?)\s*```",
            r"```\s*(.*?)\s*```",
            r"(\{.*\})",
        ]:
            match = re.search(pattern, raw, re.DOTALL)
            if not match:
                continue
            try:
                payload = json.loads(match.group(1).strip())
                return self._normalize(payload, available_targets), 1.0
            except json.JSONDecodeError:
                continue

        heuristic = {
            "action_type": self._extract_action_type(raw),
            "target": self._extract_target(raw, available_targets),
            "message": raw[:500].strip(),
        }
        if heuristic["action_type"]:
            return self._normalize(heuristic, available_targets), 0.6

        return self._fallback("unparseable"), 0.0

    def _normalize(
        self,
        payload: Dict[str, object],
        available_targets: List[str],
    ) -> Dict[str, object]:
        action_type = str(payload.get("action_type", "direct_message"))
        if action_type not in VALID_ACTION_TYPES:
            action_type = "direct_message"

        target = str(payload.get("target", "all"))
        target_ids = payload.get("target_ids", [])
        normalized_ids = self._resolve_target_ids(target, target_ids, available_targets)
        malformed = False
        error = None
        if available_targets and target != "all" and not normalized_ids:
            malformed = True
            error = f"unknown_target:{target}"

        proposed_terms = payload.get("proposed_terms") or {}
        normalized_terms = {
            key: value
            for key, value in proposed_terms.items()
            if key in VALID_PROPOSED_TERM_KEYS
        }

        documents = payload.get("documents") or []
        if not isinstance(documents, list):
            documents = []

        return {
            "action_type": action_type,
            "target": target,
            "target_ids": normalized_ids,
            "message": str(payload.get("message", ""))[:1200],
            "channel": str(payload.get("channel", "formal")),
            "mode": str(payload.get("mode", "async_email")),
            "documents": documents,
            "proposed_terms": normalized_terms or None,
            "malformed_action": malformed,
            "error": error,
        }

    def _resolve_target_ids(
        self,
        target: str,
        target_ids: object,
        available_targets: List[str],
    ) -> List[str]:
        normalized: List[str] = []
        if isinstance(target_ids, list):
            for value in target_ids:
                normalized.extend(expand_targets(str(value), available_targets))
        if target and target != "all":
            normalized.extend(expand_targets(target, available_targets))
        if target == "all":
            normalized.extend(available_targets)
        normalized = [item for item in normalized if item in available_targets]
        return list(dict.fromkeys(normalized))

    def _fallback(self, error: str) -> Dict[str, object]:
        return {
            "action_type": "direct_message",
            "target": "all",
            "target_ids": [],
            "message": "",
            "channel": "formal",
            "mode": "async_email",
            "documents": [],
            "proposed_terms": None,
            "malformed_action": True,
            "error": error,
        }

    def _extract_action_type(self, raw: str) -> Optional[str]:
        lowered = raw.lower()
        aliases = {
            "send_document": ["send document", "share document", "attach"],
            "group_proposal": ["group proposal", "move forward", "proceed together"],
            "backchannel": ["backchannel", "quiet check in", "off the record"],
            "concession": ["concession", "we can adjust", "we can offer"],
            "walkaway_signal": ["walk away", "pause the deal", "step back"],
            "reframe_value_prop": ["reframe", "different value", "reposition"],
            "exec_escalation": ["executive escalation", "exec escalation", "bring in leadership"],
        }
        for action_type in VALID_ACTION_TYPES:
            if action_type in lowered or action_type.replace("_", " ") in lowered:
                return action_type
        for action_type, examples in aliases.items():
            if any(example in lowered for example in examples):
                return action_type
        return None

    def _extract_target(self, raw: str, available_targets: List[str]) -> str:
        lowered = raw.lower()
        if " all " in f" {lowered} ":
            return "all"
        for target in available_targets:
            if target in lowered:
                return target
        for alias in ["cfo", "cto", "legal", "procurement", "ops", "cto_cfo", "legal_procurement"]:
            if alias in lowered:
                return alias
        return "all"
