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
        concession_term = sender_action.get("concession_term", "")
        inbox_text = " ".join(inbox_messages).lower() if inbox_messages else ""

        concerns_raised = []
        sentiment = "neutral"
        terms_mentioned = []
        escalation = False

        if document:
            self.docs_received.append(document)

        positive_patterns = [
            "thank", "appreciate", "received", "reviewed", "looks good",
            "clear", "acceptable", "agreed", "approved", "proceed",
        ]
        skeptical_patterns = [
            "concern", "issue", "problem", "unclear", "cannot", "reject",
            "insufficient", "violation", "non-compliant", "risk",
        ]

        doc_refs = {
            "DPA": ["dpa", "data protection", "privacy"],
            "roi_model": ["roi", "return", "financial", "budget"],
            "security_cert": ["security", "cert", "compliance", "soc2"],
            "implementation_timeline": ["timeline", "schedule", "milestone"],
            "vendor_packet": ["vendor", "packet", "overview"],
        }

        if intent == "address_concern":
            action_message = sender_action.get("message", "").lower()
            mentions_topic = any(t in action_message for t in topics)
            prior_skeptical = any(s in inbox_text for s in skeptical_patterns)

            if mentions_topic and prior_skeptical:
                sentiment = "positive"
                if self.current_concerns:
                    self.current_concerns.pop(0)
                self.alignment_score = min(1.0, self.alignment_score + 0.1)
                concerns_raised = []
            elif mentions_topic:
                sentiment = "positive"
                if self.current_concerns:
                    self.current_concerns.pop(0)
                self.alignment_score = min(1.0, self.alignment_score + 0.08)
                concerns_raised = []
            else:
                concerns_raised = [random.choice(topics)] if topics else []
                sentiment = "skeptical"
                self.alignment_score = max(0.0, self.alignment_score - 0.05)

        elif intent == "offer_document":
            doc_mentioned_in_reply = False
            if document:
                ref_terms = doc_refs.get(document.upper(), [])
                doc_mentioned_in_reply = any(t in inbox_text for t in ref_terms)
                self.docs_acknowledged.append(document)

            if doc_mentioned_in_reply or not self.current_concerns:
                sentiment = "positive"
                self.alignment_score = min(1.0, self.alignment_score + 0.08)
            else:
                concerns_raised = [random.choice(topics)] if topics else []
                sentiment = "skeptical"
                self.alignment_score = max(0.0, self.alignment_score - 0.03)

        elif intent == "make_concession":
            if concession_term:
                terms_mentioned.append(concession_term)
            prior_request = any(t in inbox_text for t in [concession_term] if concession_term)
            sentiment = "positive"
            self.alignment_score = min(1.0, self.alignment_score + 0.12)
            if self.current_concerns:
                self.current_concerns.pop(0)

        elif intent == "walkaway":
            sentiment = "skeptical"
            escalation = True
            self.alignment_score = max(0.0, self.alignment_score - 0.2)

        elif intent == "escalate_to_exec":
            escalation = True
            sentiment = "neutral"
            self.alignment_score = max(0.0, self.alignment_score - 0.05)

        elif intent == "group_proposal":
            cc = sender_action.get("cc", [])
            if len(cc) >= 2:
                sentiment = "positive"
                self.alignment_score = min(1.0, self.alignment_score + 0.1)
            elif len(cc) == 1:
                sentiment = "skeptical"
                self.alignment_score = min(1.0, self.alignment_score + 0.03)
            else:
                sentiment = "neutral"

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

    def _build_response_body(
        self, sentiment: str, concerns: List[str], intent: str, tone: str
    ) -> str:
        greeting = "Dear Seller," if tone == "formal" else "Hi,"
        document = self.docs_received[-1] if self.docs_received else None
        doc_mentions = {
            "DPA": "the Data Protection Agreement",
            "ROI_MODEL": "the ROI model",
            "SECURITY_CERT": "the security certification",
            "IMPLEMENTATION_TIMELINE": "the implementation timeline",
            "VENDOR_PACKET": "the vendor capability overview",
            "DPA": "the DPA",
            "ROI_MODEL": "the ROI analysis",
            "SECURITY_CERT": "the security certificate",
            "IMPLEMENTATION_TIMELINE": "the implementation plan",
            "VENDOR_PACKET": "the vendor packet",
        }

        if sentiment == "positive":
            if intent == "offer_document" and document:
                doc_ref = doc_mentions.get(document.upper(), document)
                return (
                    f"{greeting}\n\n"
                    f"Thank you for providing {doc_ref}. We have reviewed the materials "
                    f"and find them acceptable. We look forward to proceeding.\n\n"
                    f"Best regards,\n{self.stakeholder_id}"
                )
            if intent == "make_concession":
                return (
                    f"{greeting}\n\n"
                    f"Thank you for the concession. We appreciate your flexibility and "
                    f"are prepared to move forward on this basis.\n\n"
                    f"Best regards,\n{self.stakeholder_id}"
                )
            if intent == "group_proposal" and len(self.docs_acknowledged) >= 2:
                return (
                    f"{greeting}\n\n"
                    f"Thank you for the comprehensive proposal. We have reviewed the materials "
                    f"across all dimensions and are satisfied with the terms.\n\n"
                    f"Best regards,\n{self.stakeholder_id}"
                )
            if self.docs_acknowledged:
                return (
                    f"{greeting}\n\n"
                    f"Thank you for the update. We appreciate the progress and look "
                    f"forward to reviewing the details.\n\n"
                    f"Best regards,\n{self.stakeholder_id}"
                )
            return (
                f"{greeting}\n\n"
                f"Thank you for the update. We appreciate the progress and look "
                f"forward to reviewing the details.\n\n"
                f"Best regards,\n{self.stakeholder_id}"
            )
        elif sentiment == "skeptical":
            concern_str = ", ".join(concerns) if concerns else "the current terms"
            if intent == "offer_document" and document:
                doc_ref = doc_mentions.get(document.upper(), document)
                return (
                    f"{greeting}\n\n"
                    f"We have reviewed {doc_ref}, but still have concerns regarding "
                    f"{concern_str}. Please provide additional clarification.\n\n"
                    f"Best regards,\n{self.stakeholder_id}"
                )
            if intent == "make_concession":
                return (
                    f"{greeting}\n\n"
                    f"We appreciate the concession, but {concern_str} still needs to be "
                    f"addressed before we can proceed.\n\n"
                    f"Best regards,\n{self.stakeholder_id}"
                )
            return (
                f"{greeting}\n\n"
                f"We still have concerns regarding {concern_str}. "
                f"Please provide additional clarification before we can proceed.\n\n"
                f"Best regards,\n{self.stakeholder_id}"
            )
        else:
            if intent == "offer_document" and document:
                doc_ref = doc_mentions.get(document.upper(), document)
                return (
                    f"{greeting}\n\n"
                    f"We have received {doc_ref} and will review it shortly. "
                    f"We will follow up with any questions.\n\n"
                    f"Best regards,\n{self.stakeholder_id}"
                )
            return (
                f"{greeting}\n\n"
                f"We are reviewing the materials and will respond shortly.\n\n"
                f"Best regards,\n{self.stakeholder_id}"
            )

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
