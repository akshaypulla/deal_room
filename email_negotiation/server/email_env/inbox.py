from dataclasses import dataclass, field
from typing import Dict, List, Optional
from collections import defaultdict
from .email_message import EmailMessage


@dataclass
class StakeholderInbox:
    stakeholder_id: str
    messages: List[EmailMessage] = field(default_factory=list)
    engagement_status: str = "silent"
    last_message_time: Optional[str] = None
    current_concerns: List[str] = field(default_factory=list)
    alignment_score: float = 0.5
    docs_received: List[str] = field(default_factory=list)
    docs_acknowledged: List[str] = field(default_factory=list)
    memory: List[str] = field(default_factory=list)

    def add_message(self, msg: EmailMessage) -> None:
        self.messages.append(msg)
        self.last_message_time = msg.timestamp.isoformat()

    def get_thread(self) -> str:
        if not self.messages:
            return f"[{self.stakeholder_id.upper()}] ○ Silent\nStatus: Awaiting response\n"
        last = self.messages[-1]
        lines = [
            f"[{self.stakeholder_id.upper()}] ● Active (last: {self.last_message_time})",
            f"Status: Concern unresolved ({', '.join(self.current_concerns) if self.current_concerns else 'none'})",
            f"Last message: {last.body[:80]!r}",
        ]
        return "\n".join(lines)

    def apply_weak_update(self, msg: EmailMessage) -> None:
        self.memory.append(f"CC'd: {msg.subject[:50]}")


class EmailInbox:
    def __init__(self):
        self.stakeholder_inboxes: Dict[str, StakeholderInbox] = {}
        self.all_messages: List[EmailMessage] = []

    def add_stakeholder(self, stakeholder_id: str, initial_concerns: Optional[List[str]] = None) -> None:
        self.stakeholder_inboxes[stakeholder_id] = StakeholderInbox(
            stakeholder_id=stakeholder_id,
            current_concerns=initial_concerns or [],
        )

    def deliver_email(self, msg: EmailMessage) -> None:
        self.all_messages.append(msg)
        if msg.to_addr in self.stakeholder_inboxes:
            self.stakeholder_inboxes[msg.to_addr].add_message(msg)
            if msg.document_type:
                self.stakeholder_inboxes[msg.to_addr].docs_received.append(msg.document_type)
        for cc_addr in msg.cc:
            if cc_addr in self.stakeholder_inboxes:
                self.stakeholder_inboxes[cc_addr].apply_weak_update(msg)

    def get_full_summary(self, progress_score: float, deal_stage: str, unresolved: List[str]) -> str:
        lines = ["=== BUYER ORGANIZATION ==="]
        for sid, inbox in self.stakeholder_inboxes.items():
            lines.append(inbox.get_thread())
        lines.append("\n=== DEAL STATE ===")
        lines.append(f"Stage: {deal_stage}")
        lines.append(f"Progress: {progress_score:.2f}")
        lines.append(f"Unresolved: {unresolved}")
        lines.append("\n=== YOUR ACTION ===")
        lines.append("Choose: target, intent, tone, attachments, cc")
        return "\n".join(lines)

    def get_active_stakeholders(self) -> List[str]:
        return [sid for sid, ib in self.stakeholder_inboxes.items() if ib.engagement_status == "active"]

    def get_silent_stakeholders(self) -> List[str]:
        return [sid for sid, ib in self.stakeholder_inboxes.items() if ib.engagement_status == "silent"]
