from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime


@dataclass
class EmailMessage:
    msg_id: str = ""
    from_addr: str = "seller@company.com"
    to_addr: str = ""
    cc: List[str] = field(default_factory=list)
    subject: str = ""
    body: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    is_read: bool = False
    thread_id: str = ""
    attachments: List[str] = field(default_factory=list)
    document_type: Optional[str] = None

    def to_summary(self) -> str:
        lines = [
            f"From: {self.from_addr}",
            f"To: {self.to_addr}",
        ]
        if self.cc:
            lines.append(f"CC: {', '.join(self.cc)}")
        lines.append(f"Subject: {self.subject}")
        lines.append(f"\n{self.body[:300]}")
        if self.attachments:
            lines.append(f"Attachments: {', '.join(self.attachments)}")
        return "\n".join(lines)


@dataclass
class EmailAttachment:
    name: str
    doc_type: str
    path: Optional[str] = None
    delivered: bool = False
    acknowledged: bool = False


VALID_DOC_TYPES = {"DPA", "roi_model", "security_cert", "implementation_timeline", "vendor_packet"}

VALID_INTENTS = {"address_concern", "offer_document", "make_concession", "escalate_to_exec", "group_proposal", "walkaway"}

VALID_TARGETS = {"Legal", "Finance", "CTO", "TechLead", "Procurement", "Operations", "ExecSponsor"}

VALID_TONES = {"formal", "reassuring", "urgent"}
