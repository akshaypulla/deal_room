from typing import Any, Dict, List, Optional, Tuple


INTENTS = {"address_concern", "offer_document", "make_concession", "escalate_to_exec", "group_proposal", "walkaway"}
TARGETS = {"Legal", "Finance", "CTO", "TechLead", "Procurement", "Operations", "ExecSponsor"}
TONES = {"formal", "reassuring", "urgent"}
DOCS = {"DPA", "roi_model", "security_cert", "implementation_timeline", "vendor_packet"}
CONCESSION_TERMS = {"price", "timeline", "liability_cap", "payment_terms", "support_terms"}
MAX_CC = 2


class AntiGamingValidator:
    def __init__(self):
        self._action_history: List[Dict] = []
        self._intent_counts: Dict[str, int] = {}
        self._target_counts: Dict[str, int] = {}

    def validate_action(self, action: Dict) -> Tuple[bool, Optional[str]]:
        reason = self._check_action_validity(action)
        if reason:
            return False, reason

        reason = self._check_policy_constraints(action)
        if reason:
            return False, reason

        self._record_action(action)
        return True, None

    def _check_action_validity(self, action: Dict) -> Optional[str]:
        intent = action.get("intent", "")
        if intent not in INTENTS:
            return f"Invalid intent: {intent}"

        target = action.get("target", "")
        if target not in TARGETS:
            return f"Invalid target: {target}"

        tone = action.get("tone", "")
        if tone not in TONES:
            return f"Invalid tone: {tone}"

        doc = action.get("include_document")
        if doc and doc not in DOCS:
            return f"Invalid document: {doc}"

        concession_term = action.get("concession_term")
        if concession_term and concession_term not in CONCESSION_TERMS:
            return f"Invalid concession_term: {concession_term}"

        cc = action.get("cc", [])
        if len(cc) > MAX_CC:
            return f"Too many CC recipients: {len(cc)} (max {MAX_CC})"

        return None

    def _check_policy_constraints(self, action: Dict) -> Optional[str]:
        if self._action_history:
            last = self._action_history[-1]

            intent = action.get("intent", "")
            target = action.get("target", "")
            last_intent = last.get("intent", "")
            last_target = last.get("target", "")

            if action == last:
                return "Identical consecutive action"

            if intent == last_intent and target == last_target:
                return f"Same intent+target as last step ({intent} -> {target})"

            recent_3 = self._action_history[-3:] if len(self._action_history) >= 3 else list(self._action_history)
            target_recent_count = sum(1 for a in recent_3 if a.get("target") == target)
            if target_recent_count >= 2 and len(recent_3) >= 2:
                return f"Target {target} already targeted 2x in last 3 steps"

            if intent == "walkaway" and last_intent == "walkaway":
                return "Consecutive walkaway"

        intent = action.get("intent", "")
        self._intent_counts[intent] = self._intent_counts.get(intent, 0) + 1
        if self._intent_counts[intent] > 3:
            intent_total = sum(self._intent_counts.values())
            if intent_total > 5 and self._intent_counts[intent] / intent_total > 0.6:
                return f"Overused intent: {intent}"

        return None

    def _record_action(self, action: Dict) -> None:
        self._action_history.append(action.copy())
        if len(self._action_history) > 20:
            self._action_history.pop(0)

    def apply_diminishing_returns(self, base_reward: float, signal_type: str) -> float:
        decay_factors = {
            "sentiment_positive": 0.9,
            "doc_acknowledged": 0.85,
            "engagement_questions": 0.8,
            "concern_resolved": 0.75,
        }
        factor = decay_factors.get(signal_type, 0.95)
        return base_reward * factor

    def clip_reward(self, reward: float, reward_type: str = "dense") -> float:
        if reward_type == "dense":
            return max(-0.3, min(0.3, reward))
        elif reward_type == "milestone":
            return max(-0.5, min(0.5, reward))
        return reward

    def check_cta_no_response_penalty(
        self,
        action: Dict,
        response_data: Dict,
        step_count: int,
    ) -> Tuple[float, bool]:
        has_cta = any(word in action.get("intent", "") for word in ["propose", "confirm", "decide"])
        no_response = len(response_data.get("reply", "")) < 20

        if has_cta and no_response:
            penalty = -0.1 * (1 + max(0, step_count - 2))
            return penalty, True
        return 0.0, False

    def reset(self) -> None:
        self._action_history.clear()
        self._intent_counts.clear()
        self._target_counts.clear()
