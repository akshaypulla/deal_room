from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import random


ARCHETYPE_PROFILES = {
    "Legal": {
        "archetype": "compliance/risk",
        "weight": 0.25,
        "primary_topics": ["liability", "data_protection", "contract_terms", "compliance"],
        "decision_power": "veto",
        "alignment_base": 0.4,
    },
    "Finance": {
        "archetype": "cost/ROI",
        "weight": 0.20,
        "primary_topics": ["pricing", "payment_terms", "ROI", "budget"],
        "decision_power": "influencer",
        "alignment_base": 0.5,
    },
    "CTO": {
        "archetype": "technical_fit",
        "weight": 0.20,
        "primary_topics": ["integration", "security", "performance", "architecture"],
        "decision_power": "influencer",
        "alignment_base": 0.6,
    },
    "TechLead": {
        "archetype": "technical_fit",
        "weight": 0.10,
        "primary_topics": ["integration", "security", "performance", "API"],
        "decision_power": "influencer",
        "alignment_base": 0.55,
    },
    "Procurement": {
        "archetype": "vendor_management",
        "weight": 0.10,
        "primary_topics": ["pricing", "contract_terms", "SLAs", "vendor_risk"],
        "decision_power": "allocator",
        "alignment_base": 0.45,
    },
    "Operations": {
        "archetype": "operational_impact",
        "weight": 0.05,
        "primary_topics": ["implementation", "training", "support", "migration"],
        "decision_power": "influencer",
        "alignment_base": 0.5,
    },
    "ExecSponsor": {
        "archetype": "business_outcomes",
        "weight": 0.35,
        "primary_topics": ["ROI", "strategic_fit", "timeline", "business_case"],
        "decision_power": "final_approver",
        "alignment_base": 0.3,
    },
}


@dataclass
class StatefulArchetypeAgent:
    stakeholder_id: str
    archetype: str
    memory: List[str] = field(default_factory=list)
    current_concerns: List[str] = field(default_factory=list)
    engagement_level: float = 0.5
    alignment_score: float = 0.5
    docs_received: List[str] = field(default_factory=list)
    docs_acknowledged: List[str] = field(default_factory=list)
    sentiment_history: List[str] = field(default_factory=list)
    escalation_detected: bool = False
    _llm: Optional[Any] = None

    @classmethod
    def from_archetype(cls, stakeholder_id: str, profile: Dict) -> "StatefulArchetypeAgent":
        return cls(
            stakeholder_id=stakeholder_id,
            archetype=profile["archetype"],
            alignment_score=profile.get("alignment_base", 0.5),
            engagement_level=0.5,
        )

    def generate_email_response(
        self,
        sender_action: Dict,
        inbox_messages: List[str],
    ) -> Dict[str, Any]:
        profile = ARCHETYPE_PROFILES.get(self.stakeholder_id, {})
        topics = profile.get("primary_topics", [])

        intent = sender_action.get("intent", "")
        target = sender_action.get("target", "")
        tone = sender_action.get("tone", "formal")
        document = sender_action.get("include_document")

        concerns_raised = []
        sentiment = "neutral"
        terms_mentioned = []
        escalation = False

        if document:
            self.docs_received.append(document)

        if intent == "address_concern":
            if random.random() > 0.5:
                concerns_raised = [random.choice(topics)] if topics else []
                sentiment = "skeptical"
            else:
                if self.current_concerns:
                    self.current_concerns.pop(0)
                sentiment = "positive"
                self.alignment_score = min(1.0, self.alignment_score + 0.1)

        elif intent == "offer_document":
            sentiment = "positive"
            self.alignment_score = min(1.0, self.alignment_score + 0.05)
            if document:
                self.docs_acknowledged.append(document)

        elif intent == "make_concession":
            sentiment = "positive"
            self.alignment_score = min(1.0, self.alignment_score + 0.15)
            terms_mentioned.append(sender_action.get("concession_term", "price"))

        elif intent == "walkaway":
            sentiment = "skeptical"
            escalation = True
            self.alignment_score = max(0.0, self.alignment_score - 0.2)

        elif intent == "escalate_to_exec":
            escalation = True

        self.engagement_level = min(1.0, self.engagement_level + 0.05)
        self.sentiment_history.append(sentiment)

        response_body = self._build_response_body(sentiment, concerns_raised, intent, tone)

        return {
            "reply": response_body,
            "concerns_raised": concerns_raised,
            "sentiment": sentiment,
            "terms_mentioned": terms_mentioned,
            "escalation_detected": escalation,
            "alignment_delta": self._compute_alignment_delta(sentiment, concerns_raised, escalation),
        }

    def _build_response_body(self, sentiment: str, concerns: List[str], intent: str, tone: str) -> str:
        greeting = "Dear Seller," if tone == "formal" else "Hi,"
        if sentiment == "positive":
            return f"{greeting}\n\nThank you for the update. We appreciate the progress and look forward to reviewing the details.\n\nBest regards,\n{self.stakeholder_id}"
        elif sentiment == "skeptical":
            concern_str = ", ".join(concerns) if concerns else "the current terms"
            return f"{greeting}\n\nWe still have concerns regarding {concern_str}. Please provide additional clarification before we can proceed.\n\nBest regards,\n{self.stakeholder_id}"
        else:
            return f"{greeting}\n\nWe are reviewing the materials and will respond shortly.\n\nBest regards,\n{self.stakeholder_id}"

    def _compute_alignment_delta(self, sentiment: str, concerns: List[str], escalation: bool) -> float:
        delta = 0.0
        if sentiment == "positive":
            delta = 0.05
        elif sentiment == "skeptical":
            delta = -0.05 * len(concerns) if concerns else -0.05
        if escalation:
            delta -= 0.15
        return delta

    def update_memory(self, entry: str) -> None:
        self.memory.append(entry)

    def get_state(self) -> Dict[str, Any]:
        return {
            "stakeholder_id": self.stakeholder_id,
            "archetype": self.archetype,
            "alignment_score": self.alignment_score,
            "engagement_level": self.engagement_level,
            "current_concerns": self.current_concerns,
            "docs_received": self.docs_received,
            "docs_acknowledged": self.docs_acknowledged,
            "sentiment_history": self.sentiment_history[-5:],
            "escalation_detected": self.escalation_detected,
        }
