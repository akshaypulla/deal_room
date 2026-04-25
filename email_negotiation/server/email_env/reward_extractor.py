from typing import Any, Dict, List, Tuple


POSITIVE_MARKERS = {"thanks", "yes", "approve", "proceed", "agreed", "acceptable", "clear", "resolved"}
NEGATIVE_MARKERS = {"cannot accept", "reject", "unwilling", "concern", "problem", "issue", "violation", "non-compliant"}
KEY_TERMS = {"liability", "price", "budget", "ROI", "compliance", "data protection", "DPA", "security", "integration", "timeline"}
DOC_REFERENCES = {"DPA", "ROI model", "security cert", "implementation timeline", "vendor packet"}


class RewardExtractor:
    def __init__(self):
        self.last_reward = 0.0

    def extract(self, response_data: Dict, action_data: Dict) -> Tuple[float, Dict[str, Any]]:
        reply = response_data.get("reply", "").lower()
        concerns_raised = response_data.get("concerns_raised", [])
        sentiment = response_data.get("sentiment", "neutral")
        terms_mentioned = response_data.get("terms_mentioned", [])
        escalation = response_data.get("escalation_detected", False)

        reward_breakdown = {}
        total = 0.0

        sentiment_reward, sentiment_key = self._extract_sentiment_trust(sentiment, action_data)
        reward_breakdown[sentiment_key] = sentiment_reward
        total += sentiment_reward

        engagement_reward, engagement_key = self._extract_engagement(reply, concerns_raised)
        reward_breakdown[engagement_key] = engagement_reward
        total += engagement_reward

        doc_reward, doc_key = self._extract_document(reply, action_data)
        reward_breakdown[doc_key] = doc_reward
        total += doc_reward

        objection_reward, objection_key = self._extract_objection(reply)
        reward_breakdown[objection_key] = objection_reward
        total += objection_reward

        cc_reward, cc_key = self._extract_cc_penalty(reply, action_data)
        reward_breakdown[cc_key] = cc_reward
        total += cc_reward

        concern_reward, concern_key = self._extract_concern_resolution(
            reply, concerns_raised, sentiment
        )
        reward_breakdown[concern_key] = concern_reward
        total += concern_reward

        self.last_reward = total
        return total, reward_breakdown

    def _extract_sentiment_trust(self, sentiment: str, action_data: Dict = None) -> Tuple[float, str]:
        if sentiment == "positive":
            return 0.1, "sentiment_positive"
        elif sentiment == "skeptical":
            return -0.15, "sentiment_skeptical"
        return 0.0, "sentiment_neutral"

    def _extract_engagement(self, reply: str, concerns_raised: List[str]) -> Tuple[float, str]:
        question_count = reply.count("?")
        if question_count > 0:
            return 0.1, "engagement_questions"
        if not concerns_raised and question_count == 0:
            return 0.0, "engagement_none"
        return 0.0, "engagement_neutral"

    def _extract_document(self, reply: str, action_data: Dict) -> Tuple[float, str]:
        doc = action_data.get("include_document")
        if not doc:
            return 0.0, "doc_none"
        ref_terms = {
            "DPA": ["dpa", "data protection", "privacy", "liability"],
            "ROI_MODEL": ["roi", "return", "financial", "budget", "justified"],
            "SECURITY_CERT": ["security", "cert", "compliance", "soc2", "audit"],
            "IMPLEMENTATION_TIMELINE": ["timeline", "schedule", "milestone", "plan"],
            "VENDOR_PACKET": ["vendor", "packet", "overview", "capabilities"],
        }
        doc_key = doc.upper().replace("_", "_")
        relevant_terms = ref_terms.get(doc_key, [])
        mentioned = any(t in reply for t in relevant_terms)
        acknowledged = ("attached" in reply or "received" in reply or "review" in reply)
        if mentioned and acknowledged:
            return 0.2, "doc_acknowledged"
        elif mentioned:
            return 0.1, "doc_mentioned"
        elif acknowledged:
            return 0.05, "doc_sent"
        return 0.0, "doc_sent"

    def _extract_objection(self, reply: str) -> Tuple[float, str]:
        obj_count = sum(1 for marker in NEGATIVE_MARKERS if marker in reply)
        if obj_count >= 2:
            return -0.5, "objection_strong"
        elif obj_count == 1:
            return -0.2, "objection_mild"
        return 0.0, "objection_none"

    def _extract_cc_penalty(self, reply: str, action_data: Dict) -> Tuple[float, str]:
        cc = action_data.get("cc", [])
        if not cc:
            return 0.0, "cc_none"
        if "thank you" not in reply and "received" not in reply:
            return -0.05, "cc_unacknowledged"
        return 0.0, "cc_acknowledged"

    def _extract_concern_resolution(
        self, reply: str, concerns_raised: List[str], sentiment: str
    ) -> Tuple[float, str]:
        if not concerns_raised:
            if sentiment == "positive":
                return 0.2, "concern_resolved"
            return 0.0, "concern_none"
        if sentiment == "skeptical":
            return -0.15, "concern_raised_skeptical"
        return -0.1, "concern_raised"


def compute_terminal_reward(outcome: str) -> float:
    if outcome == "deal_closed":
        return 3.0
    elif outcome == "veto":
        return -3.0
    elif outcome == "max_rounds":
        return -1.5
    return 0.0
