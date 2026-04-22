"""
DealRoom v3 — Stakeholder LLM Generation with GPT-4o-mini + Fallback.

Provides generate_stakeholder_response() which:
1. Tries GPT-4o-mini via OpenAI API with 1.5s timeout
2. Falls back to template-based response on any failure (timeout, auth, rate limit, empty)

Zero API dependency for demo reliability.
"""

from __future__ import annotations

import os
import random
import time
from typing import Optional

from .prompts import build_stakeholder_prompt, get_template_response

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")


def _use_llm() -> bool:
    return bool(OPENAI_API_KEY)


def _call_gpt4o_mini(prompt: str, timeout: float = 30.0) -> Optional[str]:
    """Call GPT-4o-mini via llm_call_text for stats tracking. Returns None on failure."""
    from deal_room.environment.llm_client import llm_call_text

    result = llm_call_text(
        prompt=prompt,
        call_type="stakeholder_response",
        temperature=0.7,
        context="stakeholder_response",
        allow_skip=True,
        timeout=timeout,
    )
    return result


def generate_stakeholder_response(
    stakeholder_name: str,
    context: str,
    role: str = "",
    stance: str = "neutral",
) -> str:
    """
    Generate a stakeholder response for the negotiation.

    Tries GPT-4o-mini first. On any failure (timeout, rate limit, auth error,
    empty response), falls back to template-based response.

    Args:
        stakeholder_name: Name of the stakeholder (e.g., "Finance", "Legal")
        context: Negotiation context to include in the prompt
        role: Optional role description
        stance: Stance for template fallback ("supportive", "neutral", "skeptical", "hostile")

    Returns:
        A response string from the stakeholder
    """
    if _use_llm():
        prompt = build_stakeholder_prompt(stakeholder_name, context, role)
        result = _call_gpt4o_mini(prompt, timeout=1.5)
        if result:
            return result

    return get_template_response(stakeholder_name, stance)
