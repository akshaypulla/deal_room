"""
DealRoom v3 — Prompt Templates and Action Parser for LLM interaction.

Provides:
1. build_situation_prompt(obs) — converts DealRoomObservation to text prompt
2. parse_action_text(text) — converts LLM text output to DealRoomAction
"""

from __future__ import annotations

import json
import random
import re
from typing import List, Optional, Tuple

from models import DealRoomAction, DealRoomObservation

DOCUMENT_TYPES = [
    "dpa",
    "security_cert",
    "roi_model",
    "implementation_timeline",
    "compliance_report",
]
STAKEHOLDER_NAMES = [
    "Legal",
    "Finance",
    "TechLead",
    "Procurement",
    "Operations",
    "ExecSponsor",
]


def build_situation_prompt(obs: DealRoomObservation) -> str:
    """
    Convert a DealRoomObservation into a text prompt for the LLM agent.
    This is the prompt the LLM sees at each step.
    """
    parts = []

    parts.append(
        "You are an AI vendor negotiating a B2B software deal with a buying committee."
    )
    parts.append("")

    parts.append("=== CURRENT SITUATION ===")
    parts.append(f"- Round: {obs.round_number}/{obs.max_rounds}")
    parts.append(f"- Deal Stage: {obs.deal_stage}")
    parts.append(f"- Deal Momentum: {obs.deal_momentum}")

    active_blockers = obs.active_blockers or []
    parts.append(
        f"- Active Blockers: {', '.join(active_blockers) if active_blockers else 'None'}"
    )

    days = getattr(obs, "days_to_deadline", 30)
    parts.append(f"- Days to Deadline: {days}")

    veto_precursors = obs.veto_precursors or {}
    if veto_precursors:
        veto_lines = [f"  {sid}: {reason}" for sid, reason in veto_precursors.items()]
        parts.append(f"- Veto Warnings:")
        parts.extend(veto_lines)

    parts.append("")

    parts.append("=== COMMITTEE MEMBERS ===")
    engagement = obs.engagement_level or {}
    stakeholder_msgs = obs.stakeholder_messages or {}
    approval_progress = getattr(obs, "approval_path_progress", {})

    for sid in STAKEHOLDER_NAMES:
        role_info = obs.stakeholders.get(sid, {})
        role = (
            role_info.get("role", sid)
            if isinstance(role_info, dict)
            else str(role_info)
        )
        eng = engagement.get(sid, 0.5)
        eng_pct = int(eng * 100)
        band = "neutral"
        if approval_progress and sid in approval_progress:
            band = approval_progress[sid].get("band", "neutral")
        last_msg = stakeholder_msgs.get(sid, "")
        msg_snippet = f' Last said: "{last_msg[:80]}..."' if last_msg else ""

        parts.append(
            f"  - {sid} ({role}): engagement={eng_pct}%, approval={band}.{msg_snippet}"
        )

    parts.append("")

    weak_signals = obs.weak_signals or {}
    if weak_signals:
        parts.append("=== WEAK SIGNALS (hints about hidden concerns) ===")
        for sid, signals in weak_signals.items():
            if signals:
                for sig in signals[:2]:
                    parts.append(f"  - {sid}: {sig}")
        parts.append("")

    requested = getattr(obs, "requested_artifacts", {})
    if requested:
        parts.append("=== REQUESTED DOCUMENTS ===")
        for sid, docs in requested.items():
            if docs:
                parts.append(f"  - {sid} wants: {', '.join(docs)}")
        parts.append("")

    cross_echoes = getattr(obs, "cross_stakeholder_echoes", [])
    if cross_echoes:
        parts.append("=== CROSS-COMMITTEE OBSERVATIONS ===")
        for echo in cross_echoes[:3]:
            if isinstance(echo, dict):
                parts.append(
                    f"  - {echo.get('from', '?')} noticed {echo.get('about', 'something')} from {echo.get('to', '?')}"
                )
        parts.append("")

    parts.append("=== AVAILABLE ACTIONS ===")
    parts.append("Choose ONE action. END with EOS token (stop generating).")
    parts.append("")
    parts.append("  send_document <target> <doc_type> [message]")
    parts.append("    Example: send_document Finance roi_model Here is our ROI model.")
    parts.append("    Available doc types: " + ", ".join(DOCUMENT_TYPES))
    parts.append("    NOTE: Send the document to address the stakeholder's concerns.")
    parts.append("")
    parts.append("  direct_message <target> <message> ###")
    parts.append(
        "    Example: direct_message Legal I want to address your concerns.###"
    )
    parts.append("")
    parts.append("  concession <target> <term>=<value> ###")
    parts.append("    Example: concession Finance liability_cap=1500000###")
    parts.append("    Example: concession Operations timeline_weeks=12###")
    parts.append("")
    parts.append("  group_proposal <message> ###")
    parts.append("    Example: group_proposal I propose we finalize the terms.###")
    parts.append("")
    parts.append("  exec_escalation <message> ###")
    parts.append("    Example: exec_escalation Requesting executive meeting.###")
    parts.append("")
    parts.append(
        "IMPORTANT: For send_document, write ONE line and STOP. Do not add explanations or extra text. Other actions must end with ###."
    )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Action Parser
# ---------------------------------------------------------------------------

ACTION_PATTERNS = [
    (
        re.compile(
            r"^\s*send_document\s+(\w+)\s+(\w+)(?:\s+([^#]+))?(?:\s*###\s*)?$",
            re.IGNORECASE | re.DOTALL,
        ),
        "send_document",
    ),
    (
        re.compile(
            r"^\s*direct_message\s+(\w+)\s+(.+?)\s*###\s*$",
            re.IGNORECASE | re.DOTALL,
        ),
        "direct_message",
    ),
    (
        re.compile(
            r"^\s*concession\s+(\w+)\s+(\w+)=([\d.]+)(?:\s+(.+?))?\s*###\s*$",
            re.IGNORECASE | re.DOTALL,
        ),
        "concession",
    ),
    (
        re.compile(
            r"^\s*group_proposal\s+(.+?)\s*###\s*$",
            re.IGNORECASE | re.DOTALL,
        ),
        "group_proposal",
    ),
    (
        re.compile(
            r"^\s*exec_escalation\s+(.+?)\s*###\s*$",
            re.IGNORECASE | re.DOTALL,
        ),
        "exec_escalation",
    ),
    (
        re.compile(
            r"^\s*exec_escalation\s+(.+?)$",
            re.IGNORECASE | re.DOTALL,
        ),
        "exec_escalation",
    ),
]


def _normalize_target(target: str) -> str:
    """Normalize target name to one of the standard stakeholder names."""
    target_lower = target.lower()
    mapping = {
        "legal": "Legal",
        "finance": "Finance",
        "techlead": "TechLead",
        "procurement": "Procurement",
        "operations": "Operations",
        "execsponsor": "ExecSponsor",
        "exec": "ExecSponsor",
        "legal_team": "Legal",
        "finance_team": "Finance",
        "all": "all",
        "committee": "all",
        "everyone": "all",
    }
    return mapping.get(target_lower, target.capitalize())


def _validate_doc_type(doc_type: str) -> str:
    """Validate and normalize document type."""
    doc_lower = doc_type.lower()
    mapping = {
        "dpa": "dpa",
        "gdpr": "dpa",
        "security_cert": "security_cert",
        "security_certificate": "security_cert",
        "cert": "security_cert",
        "roi_model": "roi_model",
        "roi": "roi_model",
        "implementation_timeline": "implementation_timeline",
        "timeline": "implementation_timeline",
        "implementation_plan": "implementation_timeline",
        "plan": "implementation_timeline",
        "compliance_report": "compliance_report",
        "compliance": "compliance_report",
    }
    return mapping.get(doc_lower, "dpa")


def parse_action_text(text: str) -> DealRoomAction:
    """
    Parse LLM output text into a DealRoomAction.
    Falls back to a safe direct_message if no pattern matches.
    """
    text = text.strip()

    for pattern, action_type in ACTION_PATTERNS:
        m = pattern.match(text)
        if m:
            groups = m.groups()
            if action_type == "send_document":
                target = _normalize_target(groups[0])
                doc_type = _validate_doc_type(groups[1])
                message = (groups[2] or "").strip()[:500]
                return DealRoomAction(
                    action_type="send_document",
                    target=target,
                    target_ids=[target] if target != "all" else STAKEHOLDER_NAMES,
                    message=message or f"Sending {doc_type} document to {target}.",
                    documents=[{"type": doc_type, "name": doc_type.upper()}],
                )
            elif action_type == "direct_message":
                target = _normalize_target(groups[0])
                message = groups[1].strip()[:500]
                return DealRoomAction(
                    action_type="direct_message",
                    target=target,
                    target_ids=[target] if target != "all" else STAKEHOLDER_NAMES,
                    message=message,
                )
            elif action_type == "concession":
                target = _normalize_target(groups[0])
                term_key = groups[1].strip().lower()
                term_value = float(groups[2])
                message = (
                    groups[3] or f"Concession offered on {term_key}={term_value}"
                ).strip()[:500]
                return DealRoomAction(
                    action_type="concession",
                    target=target,
                    target_ids=[target],
                    message=message,
                    proposed_terms={term_key: term_value},
                )
            elif action_type == "group_proposal":
                message = groups[0].strip()[:500]
                return DealRoomAction(
                    action_type="group_proposal",
                    target="all",
                    target_ids=STAKEHOLDER_NAMES,
                    message=message,
                )
            elif action_type == "exec_escalation":
                message = groups[0].strip()[:500]
                return DealRoomAction(
                    action_type="exec_escalation",
                    target="ExecSponsor",
                    target_ids=["ExecSponsor"],
                    message=message,
                )

    return _fallback_action(text)


def _fallback_action(text: str) -> DealRoomAction:
    """Parse failure fallback — extract what we can or use safe default."""
    text = text.strip()
    text = text.split("###")[0].strip()

    text_lower = text.lower()

    for stakeholder in STAKEHOLDER_NAMES:
        if stakeholder.lower() in text_lower:
            return DealRoomAction(
                action_type="direct_message",
                target=stakeholder,
                target_ids=[stakeholder],
                message=text[:200],
            )

    return DealRoomAction(
        action_type="direct_message",
        target="all",
        target_ids=STAKEHOLDER_NAMES,
        message=text[:200] if text else "Acknowledged.",
    )


# ---------------------------------------------------------------------------
# Stakeholder response prompt builder
# ---------------------------------------------------------------------------


def build_stakeholder_prompt(
    stakeholder_name: str, context: str, role: str = ""
) -> str:
    """Build a prompt for generating stakeholder responses via GPT-4o-mini."""
    role = role or stakeholder_name
    return f"""You are扮演 {stakeholder_name}，一位在企业软件采购决策中的关键人物（角色：{role}）。

Given the following negotiation context:
{context}

Respond as {stakeholder_name} would in a business negotiation. Keep your response brief (2-3 sentences, under 100 words). Be consistent with the role and situation described above.

Respond in English with a single paragraph."""


# ---------------------------------------------------------------------------
# Template-based fallback responses (deterministic, no API needed)
# ---------------------------------------------------------------------------

TEMPLATE_RESPONSES = {
    "Legal": {
        "supportive": "Thank you for the document. The liability safeguards look reasonable. I can support moving forward with the DPA.",
        "neutral": "I've reviewed the materials. There are some compliance considerations we should discuss before I can fully endorse this proposal.",
        "skeptical": "I have significant concerns about the risk allocation in this proposal. We need stronger liability protections before I can support this.",
        "hostile": "This proposal raises serious compliance concerns. I cannot support any arrangement that doesn't fully address our legal requirements.",
    },
    "Finance": {
        "supportive": "The ROI model looks solid. The downside scenarios are well-covered. I can support the commercial terms as proposed.",
        "neutral": "The financial model is interesting. I'd like to see more detail on the implementation costs before I can commit.",
        "skeptical": "The cost structure concerns me. We need tighter commercial safeguards before I can support this from a finance perspective.",
        "hostile": "The financial terms are unacceptable. The risk-adjusted return doesn't meet our investment threshold.",
    },
    "TechLead": {
        "supportive": "The implementation plan addresses our technical concerns well. The delivery milestones are realistic. I support this direction.",
        "neutral": "I can see the merit in the approach. Let me review the technical specifications more carefully before committing.",
        "skeptical": "There are integration concerns I need to raise. The technical approach needs more detail before I can endorse it.",
        "hostile": "This technical approach doesn't meet our standards. I cannot support a solution that doesn't address our core technical requirements.",
    },
    "Procurement": {
        "supportive": "The procurement terms look clean and auditable. I can support proceeding with this contract structure.",
        "neutral": "I see some areas that need clarification in the procurement terms. Let me review the contract structure more carefully.",
        "skeptical": "The procurement terms need revision. We need stronger audit rights and termination clauses.",
        "hostile": "This contract structure is not acceptable. We need significant changes before I can support this procurement.",
    },
    "Operations": {
        "supportive": "The operational impact looks manageable. I can support the implementation approach.",
        "neutral": "I have some concerns about the operational overhead. Let me assess the resource requirements more carefully.",
        "skeptical": "The operational impact is significant. We need a clearer plan for managing the transition.",
        "hostile": "This implementation plan creates unacceptable operational risk. I cannot support this without major revisions.",
    },
    "ExecSponsor": {
        "supportive": "The strategic rationale is clear and compelling. I fully endorse this proposal.",
        "neutral": "I understand the business case. Let's ensure we have the right risk mitigation in place before final approval.",
        "skeptical": "I see potential here but there are execution risks that concern me. Let's address the key blockers before I commit.",
        "hostile": "This proposal doesn't meet our strategic threshold. We need to see significant improvements before executive support.",
    },
}


def get_template_response(stakeholder_name: str, stance: str = "neutral") -> str:
    """Return a deterministic template response for a stakeholder."""
    templates = TEMPLATE_RESPONSES.get(stakeholder_name, TEMPLATE_RESPONSES["Finance"])
    return templates.get(stance, templates["neutral"])
