"""DealRoom Custom Lab - visual guided round-table conference interface."""

from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
from openenv.core.env_server.types import EnvironmentMetadata

from models import DealRoomAction, DealRoomObservation, DealRoomState
from server.grader import CCIGrader
from server.session_pool import DealRoomSessionPool
from server.walkthrough_data import GUIDE_DATA

_LLM_MODEL = None
_LLM_TOKENIZER = None
_LLM_MODEL_PATH = os.environ.get("DEALROOM_MODEL_PATH", "dealroom-qwen-3b-negotiation")


def _get_llm_model_and_tokenizer():
    global _LLM_MODEL, _LLM_TOKENIZER
    if _LLM_MODEL is not None:
        return _LLM_MODEL, _LLM_TOKENIZER

    try:
        import torch
        from unsloth import FastLanguageModel
    except ImportError:
        return None, None

    path = _LLM_MODEL_PATH
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=path,
            max_seq_length=512,
            dtype=torch.float16,
            load_in_4bit=True,
        )
    except Exception:
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name="Qwen/Qwen2.5-3B-Instruct",
                max_seq_length=512,
                dtype=torch.float16,
                load_in_4bit=True,
            )
        except Exception:
            return None, None

    FastLanguageModel.for_inference(model)
    _LLM_MODEL = model
    _LLM_TOKENIZER = tokenizer
    return model, tokenizer


def _build_llm_action_from_observation(
    observation: Dict[str, Any], model, tokenizer
) -> DealRoomAction:
    from deal_room.environment.prompts import build_situation_prompt, parse_action_text

    obs_obj = DealRoomObservation(**observation)
    prompt = build_situation_prompt(obs_obj)
    if not prompt:
        return DealRoomAction(
            action_type="direct_message",
            target="all",
            target_ids=[],
            message="Let me think about the best approach here.",
        )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    ).strip()
    parsed = parse_action_text(response)
    return parsed


TASK_ORDER = ["aligned", "conflicted", "hostile_acquisition"]
LEVEL_LABELS = {
    "simple": "Simple",
    "medium": "Medium",
    "hard": "Hard",
}
ROLE_ICONS = {
    "finance": "💰",
    "technical": "🛠️",
    "legal_compliance": "⚖️",
    "procurement": "📦",
    "operations": "⚙️",
    "executive_sponsor": "🎯",
}

CUSTOM_CSS = """
.dealroom-lab {
    background: #0d1117;
    border-radius: 12px;
    padding: 16px;
    font-family: "IBM Plex Sans", system-ui, sans-serif;
    min-height: 100vh;
    max-height: 100vh;
    overflow-y: auto;
}
.dealroom-lab * {
    color: #e5e7eb;
}
.dealroom-lab h1, .dealroom-lab h2, .dealroom-lab h3 {
    color: #f3f4f6;
}
.lab-title {
    text-align: center;
    padding: 8px 0 12px;
    margin-bottom: 0;
}
.lab-title h1 {
    margin: 0;
    font-size: 1.3rem;
    color: #f3f4f6;
}
.lab-title p {
    margin: 4px 0 0;
    color: #9ca3af;
    font-size: 0.8rem;
}
.step-indicator {
    display: flex;
    gap: 8px;
    padding: 10px 14px;
    background: #10161d;
    border-radius: 10px;
    margin-bottom: 12px;
}
.step-btn {
    flex: 1;
    text-align: center;
    padding: 8px 10px !important;
    border-radius: 8px !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    border: 2px solid transparent !important;
    opacity: 0.5;
    background: #1f2937 !important;
    color: #6b7280 !important;
}
.step-btn.active {
    opacity: 1;
    background: rgba(255, 106, 0, 0.1) !important;
    border-color: #FF6A00 !important;
    color: #FF6A00 !important;
}
.step-btn.unlocked {
    opacity: 0.8;
    background: #1f2937 !important;
    border-color: #4b5563 !important;
    color: #d1d5db !important;
}
.step-hint {
    display: block;
    font-size: 0.65rem;
    font-weight: 400;
    margin-top: 2px;
    opacity: 0.7;
}
.start-btn {
    display: block;
    width: 100%;
    padding: 12px 20px !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #FF6A00 0%, #e55a00 100%) !important;
    border: none !important;
    border-radius: 10px !important;
    color: #fff !important;
    box-shadow: 0 4px 20px rgba(255, 106, 0, 0.3);
}
.start-btn:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 24px rgba(255, 106, 0, 0.4);
}
.main-layout {
    display: grid;
    grid-template-columns: 1fr 320px;
    gap: 16px;
    min-height: 500px;
    height: auto;
}
.left-panel {
    display: flex;
    flex-direction: column;
    gap: 12px;
    min-height: 0;
    overflow-y: auto;
}
.right-panel {
    display: flex;
    flex-direction: column;
    gap: 12px;
    min-height: 0;
    overflow-y: auto;
}
.round-container {
    background: radial-gradient(ellipse at center, #1a1f2e 0%, #0d1117 70%);
    border-radius: 50%;
    width: 100%;
    max-width: 340px;
    height: 280px;
    margin: 0 auto;
    border: 2px solid #2a3142;
    box-shadow: 0 0 40px rgba(255, 106, 0, 0.1), inset 0 0 50px rgba(0,0,0,0.4);
    position: relative;
    flex-shrink: 0;
}
.round-center {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 80px;
    height: 80px;
    background: rgba(13, 17, 23, 0.9);
    border: 2px solid #FF6A00;
    border-radius: 50%;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
}
.round-center strong {
    font-size: 0.85rem;
    color: #FF6A00;
}
.round-center span {
    font-size: 0.65rem;
    color: #9ca3af;
}
.seat {
    position: absolute;
    width: 60px;
    height: 60px;
    border-radius: 50%;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    transition: all 0.3s ease;
    cursor: pointer;
    border: 3px solid transparent;
    box-shadow: 0 4px 16px rgba(0,0,0,0.4);
}
.seat:hover {
    transform: scale(1.12);
    z-index: 10;
    box-shadow: 0 0 20px rgba(255, 106, 0, 0.3);
}
.seat.selected {
    border-color: #FF6A00;
    box-shadow: 0 0 24px rgba(255, 106, 0, 0.5);
}
.seat.blocking {
    background: linear-gradient(135deg, #7f1d1d 0%, #450a0a 100%);
    border-color: #ef4444;
}
.seat.uncertain {
    background: linear-gradient(135deg, #78350f 0%, #451a03 100%);
    border-color: #f59e0b;
}
.seat.aligned {
    background: linear-gradient(135deg, #065f46 0%, #022c22 100%);
    border-color: #22c55e;
}
.seat.dimmed {
    opacity: 0.4;
}
.seat-icon {
    font-size: 1.2rem;
}
.seat-name {
    font-size: 0.6rem;
    font-weight: 700;
    color: #fff;
    text-transform: uppercase;
    margin-top: 2px;
}
.popup-card {
    background: #0f141b;
    border: 1px solid #304154;
    border-radius: 10px;
    padding: 12px;
    position: relative;
    width: 100%;
    z-index: 20;
    animation: popup-in 0.3s ease-out;
    box-shadow: 0 4px 16px rgba(0,0,0,0.3);
    margin-top: 10px;
}
@keyframes popup-in {
    from { opacity: 0; transform: translate(-50%, -40%); }
    to { opacity: 1; transform: translate(-50%, -50%); }
}
.popup-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
    padding-bottom: 6px;
    border-bottom: 1px solid #1f2937;
}
.popup-icon {
    font-size: 1.2rem;
}
.popup-name {
    font-weight: 700;
    color: #f3f4f6;
    font-size: 0.85rem;
}
.popup-role {
    font-size: 0.65rem;
    color: #9ca3af;
}
.popup-quote {
    background: #0a1016;
    border-left: 3px solid #FF6A00;
    padding: 8px 10px;
    border-radius: 0 6px 6px 0;
    margin: 8px 0;
    color: #e5e7eb;
    font-style: italic;
    font-size: 0.8rem;
}
.popup-status {
    display: flex;
    align-items: center;
    gap: 6px;
    margin: 6px 0;
    font-size: 0.8rem;
}
.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
}
.status-dot.red { background: #ef4444; }
.status-dot.amber { background: #f59e0b; }
.status-dot.green { background: #22c55e; }
.popup-request {
    background: rgba(255, 106, 0, 0.08);
    border: 1px solid rgba(255, 106, 0, 0.2);
    border-radius: 6px;
    padding: 6px 8px;
    margin-top: 6px;
    font-size: 0.75rem;
}
.popup-request strong {
    color: #FF6A00;
}
.popup-request p {
    color: #d1d5db;
    margin: 2px 0 0;
    font-size: 0.75rem;
}
.chips-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 6px;
    margin: 8px 0;
}
.chip-btn {
    padding: 8px 10px !important;
    border-radius: 6px !important;
    border: 1px solid #374151 !important;
    background: #1f2937 !important;
    color: #e5e7eb !important;
    font-size: 0.75rem !important;
    cursor: pointer;
    transition: all 0.2s ease;
    text-align: center;
}
.chip-btn:hover {
    border-color: #FF6A00 !important;
    background: rgba(255, 106, 0, 0.1) !important;
}
.chip-btn.selected {
    border-color: #FF6A00 !important;
    background: rgba(255, 106, 0, 0.2) !important;
    color: #FF6A00 !important;
}
.chip-btn.suggested {
    border-color: #FF6A00 !important;
    background: rgba(255, 106, 0, 0.15) !important;
}
.action-bar {
    background: #10161d;
    border: 1px solid #1f2937;
    border-radius: 10px;
    padding: 12px;
    flex: 1;
    min-height: 0;
    display: flex;
    flex-direction: column;
}
.action-bar h3 {
    margin: 0 0 8px;
    font-size: 0.85rem;
}
.message-input {
    width: 100%;
    background: #0d1117 !important;
    border: 1px solid #2b3746 !important;
    border-radius: 6px !important;
    color: #e5e7eb !important;
    padding: 8px 10px !important;
    font-size: 0.8rem !important;
    margin-bottom: 8px;
}
.send-btn {
    width: 100%;
    padding: 10px !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    background: linear-gradient(135deg, #FF6A00 0%, #e55a00 100%) !important;
    border: none !important;
    border-radius: 6px !important;
    color: #fff !important;
}
.send-btn:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(255, 106, 0, 0.3);
}
.sim-controls {
    display: flex;
    gap: 6px;
    margin-top: 8px;
}
.run-btn {
    flex: 1;
    padding: 10px !important;
    font-weight: 600 !important;
    background: linear-gradient(135deg, #FF6A00 0%, #e55a00 100%) !important;
    border: none !important;
    border-radius: 6px !important;
    color: #fff !important;
}
.step-btn-sm {
    padding: 10px 14px !important;
    background: #1f2937 !important;
    border: 1px solid #374151 !important;
    border-radius: 6px !important;
    color: #e5e7eb !important;
}
.advanced-toggle {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 10px;
    background: #1f2937;
    border: 1px solid #374151;
    border-radius: 6px;
    color: #9ca3af;
    font-size: 0.75rem;
    cursor: pointer;
    margin-top: 8px;
}
.advanced-toggle:hover {
    color: #e5e7eb;
}
.advanced-controls {
    display: none;
    margin-top: 8px;
    padding: 8px;
    background: #0d1117;
    border-radius: 6px;
}
.advanced-controls.open {
    display: block;
}
.speed-select {
    width: 100%;
    padding: 6px !important;
    background: #1f2937 !important;
    border: 1px solid #374151 !important;
    border-radius: 6px !important;
    color: #e5e7eb !important;
    margin-bottom: 6px;
}
.auto-btns {
    display: flex;
    gap: 6px;
}
.auto-btn {
    flex: 1;
    padding: 8px !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
}
.pause-btn {
    background: #f59e0b !important;
    border: none !important;
    color: #0d1117 !important;
}
.stop-btn {
    background: #374151 !important;
    border: none !important;
    color: #e5e7eb !important;
}
.score-panel {
    background: #10161d;
    border: 1px solid #1f2937;
    border-radius: 10px;
    padding: 12px;
}
.score-display {
    text-align: center;
    padding: 12px;
    background: linear-gradient(135deg, #1a1f2e 0%, #0f1419 100%);
    border: 2px solid #FF6A00;
    border-radius: 12px;
    margin-bottom: 10px;
}
.score-value {
    font-size: 2.2rem;
    font-weight: 700;
    color: #FF6A00;
    text-shadow: 0 0 20px rgba(255, 106, 0, 0.5);
    line-height: 1;
}
.score-label {
    font-size: 0.75rem;
    color: #9ca3af;
    margin-top: 4px;
}
.score-delta {
    font-size: 0.9rem;
    margin-top: 4px;
}
.signals-section {
    margin-bottom: 10px;
}
.signals-section h4 {
    font-size: 0.7rem;
    color: #9ca3af;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin: 0 0 6px;
}
.signals-area {
    font-size: 0.8rem;
    color: #d1d5db;
    padding: 6px 0;
    border-bottom: 1px solid #1f2937;
}
.signal-tag {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 3px 6px;
    border-radius: 999px;
    font-size: 0.7rem;
    background: rgba(255, 106, 0, 0.1);
    color: #FF6A00;
    margin: 2px 2px 2px 0;
}
.why-panel {
    border-top: 1px solid #1f2937;
    padding-top: 8px;
}
.why-header {
    color: #9ca3af;
    font-size: 0.8rem;
    font-weight: 600;
    margin-bottom: 6px;
}
.why-content {
    margin-top: 6px;
    padding: 8px;
    background: #0d1117;
    border-radius: 6px;
    font-size: 0.75rem;
    color: #d1d5db;
    line-height: 1.5;
}
.why-content.open {
    display: block;
}
.why-score-breakdown {
    margin-top: 8px;
    padding-top: 8px;
    border-top: 1px solid #1f2937;
}
.why-score-item {
    display: flex;
    justify-content: space-between;
    margin-bottom: 4px;
}
.why-score-label {
    color: #9ca3af;
}
.why-score-value {
    color: #FF6A00;
    font-weight: 600;
}
.why-score-weight {
    color: #6b7280;
    font-size: 0.7rem;
}
.hint-box {
    background: rgba(255, 106, 0, 0.1);
    border: 1px solid rgba(255, 106, 0, 0.3);
    border-radius: 6px;
    padding: 8px 12px;
    margin: 8px 0;
    font-size: 0.8rem;
    color: #FF6A00;
    text-align: center;
}
.hint-box::before {
    content: "👉 ";
}
.auto-status {
    text-align: center;
    padding: 6px;
    background: rgba(255, 106, 0, 0.1);
    border-radius: 6px;
    font-size: 0.75rem;
    color: #FF6A00;
    margin-bottom: 6px;
}
.blocker-tag {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 3px 6px;
    border-radius: 999px;
    font-size: 0.7rem;
    background: rgba(239, 68, 68, 0.15);
    color: #ef4444;
    margin: 2px 2px 2px 0;
}
.how-it-works {
    background: linear-gradient(135deg, rgba(255, 106, 0, 0.08) 0%, rgba(255, 106, 0, 0.03) 100%);
    border: 1px solid rgba(255, 106, 0, 0.2);
    border-radius: 10px;
    padding: 14px;
    margin-bottom: 12px;
}
.how-it-works h3 {
    margin: 0 0 10px;
    font-size: 0.9rem;
    color: #FF6A00;
}
.how-it-works ol {
    margin: 0;
    padding-left: 18px;
    font-size: 0.8rem;
    color: #d1d5db;
    line-height: 1.6;
}
.how-it-works li {
    margin-bottom: 4px;
}
.scoring-info {
    background: #0d1117;
    border-radius: 6px;
    padding: 8px;
    margin-top: 10px;
}
.scoring-info h4 {
    margin: 0 0 6px;
    font-size: 0.75rem;
    color: #FF6A00;
}
.scoring-item {
    display: flex;
    justify-content: space-between;
    font-size: 0.75rem;
    margin-bottom: 3px;
}
.scoring-item-name {
    color: #9ca3af;
}
.scoring-item-weight {
    color: #6b7280;
}
.hidden {
    display: none !important;
}
.causal-graph-area {
    margin-top: 10px;
    padding: 8px;
    background: #0d1117;
    border-radius: 8px;
    border: 1px solid #1f2937;
}
.causal-graph-area svg {
    max-width: 100%;
    height: auto;
}
.llm-agent-toggle {
    margin-top: 8px;
}
.seed-preset-dropdown {
    margin-top: 6px;
}
"""


class DealRoomWebManager:
    def __init__(self, pool: DealRoomSessionPool, metadata: EnvironmentMetadata):
        self.pool = pool
        self.metadata = metadata
        self._playground_session_id: Optional[str] = None

    def reset_session(
        self,
        task_id: str,
        seed: int,
        session_id: Optional[str] = None,
    ) -> Tuple[str, DealRoomObservation, DealRoomState]:
        return self.pool.reset(task_id=task_id, seed=seed, session_id=session_id)

    def step_session(
        self,
        session_id: str,
        action: DealRoomAction,
    ) -> Tuple[DealRoomObservation, float, bool, Dict[str, Any], DealRoomState]:
        return self.pool.step(session_id, action)

    def get_state_for_session(self, session_id: str) -> Dict[str, Any]:
        return self.pool.state(session_id).model_dump()

    def get_beliefs_for_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self.pool.get_beliefs(session_id)


def load_metadata() -> EnvironmentMetadata:
    readme_path = Path("README.md")
    readme = readme_path.read_text(encoding="utf-8") if readme_path.exists() else None
    return EnvironmentMetadata(
        name="deal-room",
        description="A realistic multi-stakeholder enterprise negotiation environment.",
        version="1.0.0",
        author="akshaypulla",
        readme_content=readme,
    )


SEAT_POSITIONS = [
    {"top": "8%", "left": "50%", "transform": "translateX(-50%)"},
    {"top": "32%", "left": "6%"},
    {"top": "32%", "right": "6%"},
    {"bottom": "8%", "left": "50%", "transform": "translateX(-50%)"},
]


def build_custom_tab(
    web_manager: DealRoomWebManager,
    action_fields: List[Dict[str, Any]],
    metadata: EnvironmentMetadata,
    is_chat_env: bool,
    title: str,
    quick_start_md: Optional[str],
) -> gr.Blocks:
    del action_fields, metadata, is_chat_env, title, quick_start_md

    def default_view_state() -> Dict[str, Any]:
        return {
            "task": "aligned",
            "seed": 42,
            "level": "simple",
            "source": "custom",
            "session_id": None,
            "selected_stakeholder": None,
            "current_observation": None,
            "current_state": None,
            "trace": [],
            "status_message": "Click Start Simple Round to begin.",
            "popup_queue": [],
            "popup_index": 0,
            "unlocked_levels": ["simple"],
            "round_started": False,
            "show_hint": False,
            "hint_text": "",
            "selected_chip": None,
            "message_text": "",
            "auto_playing": False,
            "auto_paused": False,
            "auto_speed": "medium",
            "last_score": 0.0,
            "score_delta": None,
            "show_advanced": False,
            "round_complete": False,
            "use_llm_agent": False,
            "seed_preset": None,
        }

    def _escape(value: Any) -> str:
        return html.escape(str(value))

    def _normalize_view_state(view_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        base = default_view_state()
        if not isinstance(view_state, dict):
            return base
        merged = dict(base)
        merged.update(view_state)
        if not isinstance(merged.get("trace"), list):
            merged["trace"] = []
        if not isinstance(merged.get("popup_queue"), list):
            merged["popup_queue"] = []
        if not isinstance(merged.get("unlocked_levels"), list):
            merged["unlocked_levels"] = ["simple"]
        if "use_llm_agent" not in merged:
            merged["use_llm_agent"] = False
        if "seed_preset" not in merged:
            merged["seed_preset"] = None
        return merged

    def _normalize_saved_runs(saved_runs: Any) -> List[Dict[str, Any]]:
        if not isinstance(saved_runs, list):
            return []
        return [item for item in saved_runs if isinstance(item, dict)]

    def _coerce_observation(data: Dict[str, Any]) -> DealRoomObservation:
        return DealRoomObservation.model_validate(data)

    def _first_stakeholder_id(observation: Dict[str, Any]) -> Optional[str]:
        stakeholders = observation.get("stakeholders", {})
        return next(iter(stakeholders), None)

    def _keep_valid_selected(view_state: Dict[str, Any]) -> None:
        observation = view_state.get("current_observation") or {}
        stakeholders = observation.get("stakeholders", {})
        selected = view_state.get("selected_stakeholder")
        if selected not in stakeholders:
            view_state["selected_stakeholder"] = _first_stakeholder_id(observation)

    def _approval_band(observation: Dict[str, Any], stakeholder_id: str) -> str:
        progress = observation.get("approval_path_progress", {}).get(stakeholder_id, {})
        band = str(progress.get("band", "neutral"))
        if band in ("supporter", "workable"):
            return "aligned"
        elif band == "blocker":
            return "blocking"
        return "uncertain"

    def _generate_chips(view_state: Dict[str, Any]) -> List[str]:
        view_state = _normalize_view_state(view_state)
        observation = view_state.get("current_observation") or {}
        selected = view_state.get("selected_stakeholder")
        if not selected or not observation:
            return [
                "Address concern",
                "Ask for details",
                "Provide evidence",
                "Delay decision",
            ]
        payload = observation.get("stakeholders", {}).get(selected, {})
        role = payload.get("role", "")
        band = _approval_band(observation, selected)
        requested = observation.get("requested_artifacts", {}).get(selected, [])
        chips = []
        if role == "finance" or "budget" in str(requested) or "roi" in str(requested):
            chips = [
                "Provide ROI justification",
                "Address budget concern",
                "Ask for cost breakdown",
                "Propose payment terms",
            ]
        elif role == "legal_compliance":
            chips = [
                "Clarify contract terms",
                "Address compliance gap",
                "Provide legal documentation",
                "Suggest alternative clauses",
            ]
        elif role == "procurement":
            chips = [
                "Justify vendor selection",
                "Compare alternatives",
                "Address delivery concerns",
                "Request timeline flexibility",
            ]
        elif role == "executive_sponsor":
            chips = [
                "Summarize business case",
                "Highlight key benefits",
                "Address risk concerns",
                "Propose next steps",
            ]
        elif band == "blocking":
            chips = [
                "Address their concern",
                "Request clarification",
                "Provide supporting evidence",
                "Escalate to manager",
            ]
        elif band == "uncertain":
            chips = [
                "Clarify your position",
                "Provide more details",
                "Address specific worries",
                "Build confidence",
            ]
        else:
            chips = [
                "Acknowledge their point",
                "Build on alignment",
                "Move discussion forward",
                "Confirm understanding",
            ]
        return chips[:4]

    def _policy_action(observation: Dict[str, Any]) -> DealRoomAction:
        obs = _coerce_observation(observation)
        for stakeholder_id, artifacts in obs.requested_artifacts.items():
            if artifacts:
                artifact = artifacts[0]
                return DealRoomAction(
                    action_type="send_document",
                    target=stakeholder_id,
                    target_ids=[stakeholder_id],
                    message=f"Sharing the requested {artifact.replace('_', ' ')}.",
                    documents=[{"type": artifact, "specificity": "high"}],
                )
        if obs.active_blockers or not obs.known_constraints:
            target_id = next(
                iter(obs.active_blockers), next(iter(obs.stakeholders), "all")
            )
            return DealRoomAction(
                action_type="direct_message",
                target=target_id,
                target_ids=[target_id] if target_id != "all" else [],
                message="Help me understand the real concern we need to address.",
            )
        return DealRoomAction(
            action_type="direct_message",
            target="all",
            target_ids=list(obs.stakeholders.keys()),
            message="Let me know if there's anything else you need before we proceed.",
        )

    def _run_reset(
        task: str,
        seed: int,
        level: str,
        view_state: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        current = _normalize_view_state(view_state)
        session_id, obs, state = web_manager.reset_session(
            task_id=task,
            seed=int(seed),
            session_id=current.get("session_id"),
        )
        observation = obs.model_dump()
        stakeholders = list(observation.get("stakeholders", {}).keys())
        updated = dict(current)
        updated.update(
            {
                "task": task,
                "seed": int(seed),
                "level": level,
                "source": "custom",
                "session_id": session_id,
                "current_observation": observation,
                "current_state": state.model_dump(),
                "selected_stakeholder": stakeholders[0] if stakeholders else None,
                "popup_queue": [{"stakeholder_id": s} for s in stakeholders],
                "popup_index": 0,
                "round_started": True,
                "show_hint": True,
                "hint_text": "Click a stakeholder to see their response",
                "trace": [],
                "last_score": 0.0,
                "score_delta": None,
                "round_complete": False,
                "selected_chip": None,
                "message_text": "",
            }
        )
        return updated

    def _record_step(
        view_state: Dict[str, Any],
        action: DealRoomAction,
        obs: DealRoomObservation,
        reward: float,
        done: bool,
        info: Dict[str, Any],
        state: Dict[str, Any],
    ) -> Dict[str, Any]:
        updated = _normalize_view_state(view_state)
        trace = list(updated.get("trace", []))
        old_score = updated.get("last_score", 0.0)
        step_num = len([item for item in trace if item.get("kind") == "step"]) + 1
        trace.append(
            {
                "kind": "step",
                "step": step_num,
                "action": action.model_dump(),
                "reward": reward,
                "done": done,
                "stage": obs.deal_stage,
                "blockers": list(obs.active_blockers),
            }
        )
        updated["trace"] = trace
        updated["current_observation"] = obs.model_dump()
        updated["current_state"] = state
        updated["score_delta"] = reward - old_score if old_score else None
        updated["last_score"] = reward
        updated["round_complete"] = done
        if done:
            updated["status_message"] = f"Round complete! Final score: {reward:.2f}"
            updated["show_hint"] = False
        else:
            updated["status_message"] = f"Step {step_num} | Reward: {reward:.2f}"
        return updated

    def _save_run_if_complete(
        view_state: Dict[str, Any], saved_runs: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        view_state = _normalize_view_state(view_state)
        saved_runs = _normalize_saved_runs(saved_runs)
        observation = view_state.get("current_observation") or {}
        if not observation.get("done"):
            return saved_runs, view_state
        score = CCIGrader.compute(
            DealRoomState.model_validate(view_state.get("current_state") or {})
        )
        run_id = f"{view_state['level']}-{view_state['task']}-{view_state['seed']}-{view_state['source']}-{len(saved_runs) + 1}"
        saved_runs = [item for item in saved_runs if item.get("id") != run_id]
        saved_runs.append(
            {
                "id": run_id,
                "task": view_state["task"],
                "level": view_state["level"],
                "seed": view_state["seed"],
                "source": view_state["source"],
                "score": score,
            }
        )
        saved_runs = saved_runs[-8:]

        level_order = ["simple", "medium", "hard"]
        current_index = (
            level_order.index(view_state["level"])
            if view_state["level"] in level_order
            else 0
        )
        if current_index < len(level_order) - 1:
            next_level = level_order[current_index + 1]
            if next_level not in view_state.get("unlocked_levels", []):
                view_state["unlocked_levels"] = view_state.get(
                    "unlocked_levels", []
                ) + [next_level]

        return saved_runs, view_state

    def _build_round_table(view_state: Dict[str, Any]) -> str:
        view_state = _normalize_view_state(view_state)
        observation = view_state.get("current_observation") or {}
        selected = view_state.get("selected_stakeholder")
        seats_html = []
        stakeholder_list = list(observation.get("stakeholders", {}).items())
        for index, (stakeholder_id, payload) in enumerate(stakeholder_list):
            if index >= len(SEAT_POSITIONS):
                break
            band = _approval_band(observation, stakeholder_id)
            pos = SEAT_POSITIONS[index]
            pos_style = "; ".join(f"{k}: {v}" for k, v in pos.items())
            is_selected = stakeholder_id == selected
            seat_class = f"seat {band}"
            if is_selected:
                seat_class += " selected"
            elif selected and stakeholder_id != selected:
                seat_class += " dimmed"
            seats_html.append(
                f"<div class='{seat_class}' style='{pos_style}' data-stakeholder='{_escape(stakeholder_id)}'>"
                f"<div class='seat-icon'>{ROLE_ICONS.get(payload.get('role', ''), '👤')}</div>"
                f"<div class='seat-name'>{_escape(payload.get('display_name', stakeholder_id)[:8])}</div>"
                f"</div>"
            )
        level_text = LEVEL_LABELS.get(view_state["level"], "Simple")
        return (
            "<div class='round-container'>"
            + "".join(seats_html)
            + (
                "<div class='round-center'>"
                "<strong>DEAL</strong>"
                f"<span>{level_text}</span>"
                "</div>"
            )
            + "</div>"
        )

    def _build_popup(view_state: Dict[str, Any]) -> str:
        view_state = _normalize_view_state(view_state)
        observation = view_state.get("current_observation") or {}
        popup_queue = view_state.get("popup_queue", [])
        current_idx = view_state.get("popup_index", 0)
        if not popup_queue or current_idx >= len(popup_queue):
            return "<div class='popup-card hidden'></div>"
        stakeholder_id = popup_queue[current_idx].get("stakeholder_id")
        if not stakeholder_id or stakeholder_id not in observation.get(
            "stakeholders", {}
        ):
            return "<div class='popup-card hidden'></div>"
        payload = observation["stakeholders"][stakeholder_id]
        message = observation.get("stakeholder_messages", {}).get(
            stakeholder_id, "No message yet."
        )
        requested = observation.get("requested_artifacts", {}).get(stakeholder_id, [])
        band = _approval_band(observation, stakeholder_id)
        status_class = (
            "red"
            if band == "blocking"
            else ("amber" if band == "uncertain" else "green")
        )
        status_text = (
            "Blocking"
            if band == "blocking"
            else ("Uncertain" if band == "uncertain" else "Aligned")
        )
        request_text = (
            ", ".join(r.replace("_", " ") for r in requested)
            if requested
            else "Nothing specific"
        )
        return (
            "<div class='popup-card'>"
            "<div class='popup-header'>"
            f"<div class='popup-icon'>{ROLE_ICONS.get(payload.get('role', ''), '👤')}</div>"
            f"<div><div class='popup-name'>{_escape(payload.get('display_name', stakeholder_id))}</div>"
            f"<div class='popup-role'>{_escape(payload.get('role', ''))}</div></div>"
            "</div>"
            f"<div class='popup-quote'>\"{_escape(message)}\"</div>"
            "<div class='popup-status'>"
            f"<div class='status-dot {status_class}'></div>"
            f"<span>Status: {status_text}</span>"
            "</div>"
            "<div class='popup-request'>"
            "<strong>Needs:</strong>"
            f"<p>{_escape(request_text)}</p>"
            "</div>"
            "</div>"
        )

    def _build_chips(view_state: Dict[str, Any]) -> Tuple[str, str, str, str]:
        chips = _generate_chips(view_state)
        selected = view_state.get("selected_chip")
        chip_values = []
        for i in range(4):
            if i < len(chips):
                chip_values.append(chips[i])
            else:
                chip_values.append("")
        return tuple(chip_values)  # type: ignore

    def _build_score_panel(view_state: Dict[str, Any]) -> str:
        view_state = _normalize_view_state(view_state)
        observation = view_state.get("current_observation") or {}
        current_score = view_state.get("last_score", 0.0)
        score_delta = view_state.get("score_delta")
        done = observation.get("done", False)
        if done:
            final_score = current_score
            return (
                "<div class='score-display'>"
                f"<div class='score-value'>{final_score:.2f}</div>"
                "<div class='score-label'>Final Score</div>"
                "</div>"
            )
        delta_html = ""
        if score_delta is not None:
            sign = "+" if score_delta >= 0 else ""
            color = "#FF6A00" if score_delta >= 0 else "#ef4444"
            delta_html = f"<div class='score-delta' style='color: {color};'>{sign}{score_delta:.2f}</div>"
        blockers = observation.get("active_blockers", [])
        blocker_html = ""
        if blockers:
            tags = "".join(
                f"<span class='blocker-tag'>⚠️ {_escape(b)}</span>" for b in blockers
            )
            blocker_html = f"<div style='margin-top:8px;'>{tags}</div>"
        return (
            "<div class='score-display'>"
            f"<div class='score-value'>{current_score:.2f}</div>"
            "<div class='score-label'>Current Score</div>"
            f"{delta_html}"
            f"{blocker_html}"
            "</div>"
        )

    def _build_signals(view_state: Dict[str, Any]) -> str:
        view_state = _normalize_view_state(view_state)
        observation = view_state.get("current_observation") or {}
        if not observation:
            return "<div class='signals-area'>No signals yet</div>"
        signals = []
        for stakeholder_id, artifacts in observation.get(
            "requested_artifacts", {}
        ).items():
            if artifacts:
                for art in artifacts:
                    signals.append(
                        f"<span class='signal-tag'>📋 {_escape(art.replace('_', ' '))}</span>"
                    )
        if not signals:
            return "<div class='signals-area'>No pending requests</div>"
        return f"<div class='signals-area'>{''.join(signals)}</div>"

    def _compute_score_breakdown(view_state: Dict[str, Any]) -> Dict[str, float]:
        state = view_state.get("current_state") or {}
        mandatory_ids = [
            sid
            for sid, p in state.get("stakeholder_private", {}).items()
            if p.get("mandatory")
        ]
        approval_score = 0.0
        if mandatory_ids:
            approvals = [
                state["stakeholder_private"][sid]["approval"] for sid in mandatory_ids
            ]
            approval_score = min(1.0, sum(approvals) / len(approvals))
        constraints = list(state.get("hidden_constraints", {}).values())
        constraint_score = 0.0
        if constraints:
            resolved = sum(1 for c in constraints if c.get("resolved"))
            constraint_score = resolved / len(constraints)
        violations = state.get("feasibility_state", {}).get("violations", [])
        penalty = min(0.20, 0.05 * len(violations))
        feasibility_score = max(0.0, 1.0 - penalty)
        trusts = [p["trust"] for p in state.get("stakeholder_private", {}).values()]
        mark_penalty = sum(
            0.03 * len(p.get("permanent_marks", []))
            for p in state.get("stakeholder_private", {}).values()
        )
        average_trust = sum(trusts) / len(trusts) if trusts else 0.0
        relationship_score = max(0.0, min(1.0, average_trust - mark_penalty))
        max_rounds = state.get("max_rounds", 20)
        round_num = state.get("round_number", 0)
        efficiency_score = (
            max(0.1, 1.0 - ((round_num / max_rounds) ** 1.25) * 0.45)
            if max_rounds > 0
            else 0.0
        )
        return {
            "approval_completeness": approval_score,
            "constraint_satisfaction": constraint_score,
            "term_feasibility": feasibility_score,
            "relationship_durability": relationship_score,
            "efficiency": efficiency_score,
        }

    def _build_why(view_state: Dict[str, Any]) -> str:
        view_state = _normalize_view_state(view_state)
        observation = view_state.get("current_observation") or {}
        state = view_state.get("current_state") or {}
        blockers = observation.get("active_blockers", [])
        if not observation:
            content = (
                "<strong>Your goal:</strong> Get all stakeholders to approve the deal.<br><br>"
                "<strong>How to score higher:</strong><br>"
                "• Satisfy stakeholder requests (artifacts)<br>"
                "• Resolve hidden constraints<br>"
                "• Keep negotiations efficient<br>"
                "• Maintain stakeholder trust"
            )
        elif blockers:
            reasons = [f"• {_escape(b)} is blocking progress" for b in blockers]
            content = "<br>".join(reasons)
        else:
            content = "No active blockers. The negotiation is progressing well."
        breakdown = _compute_score_breakdown(view_state)
        weights = CCIGrader.WEIGHTS
        breakdown_html = ""
        for key, weight in weights.items():
            label = key.replace("_", " ").title()
            value = breakdown.get(key, 0.0)
            pct = int(value * 100)
            breakdown_html += f"""
            <div class="why-score-item">
                <span class="why-score-label">{label}</span>
                <span>
                    <span class="why-score-value">{pct}%</span>
                    <span class="why-score-weight">({int(weight * 100)}%)</span>
                </span>
            </div>"""
        return (
            "<div class='why-panel'>"
            "<div class='why-header'>Why this score?</div>"
            f"<div class='why-content open'>{content}"
            f"<div class='why-score-breakdown'><strong>Score Breakdown:</strong>{breakdown_html}</div>"
            "</div>"
            "</div>"
        )

    def _build_causal_graph(view_state: Dict[str, Any]) -> str:
        view_state = _normalize_view_state(view_state)
        session_id = view_state.get("session_id")
        if not session_id:
            return (
                "<div class='causal-graph-area'>Start a round to see belief graph</div>"
            )

        beliefs = web_manager.get_beliefs_for_session(session_id)
        if not beliefs:
            return "<div class='causal-graph-area'>No belief data yet</div>"

        try:
            import io
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import networkx as nx
        except Exception:
            return "<div class='causal-graph-area'>Graph unavailable (networkx/matplotlib not installed)</div>"

        G = nx.DiGraph()
        stake_roles = {
            "Finance": "finance",
            "Legal": "legal_compliance",
            "TechLead": "technical",
            "Procurement": "procurement",
            "Operations": "operations",
            "ExecSponsor": "executive_sponsor",
        }
        role_icons = {
            "finance": "💰",
            "legal_compliance": "⚖️",
            "technical": "🛠️",
            "procurement": "📦",
            "operations": "⚙️",
            "executive_sponsor": "🎯",
        }

        for sid, belief_dist in beliefs.items():
            if not hasattr(belief_dist, "distribution"):
                continue
            pos = belief_dist.positive_mass()
            neg = belief_dist.negative_mass()
            role = getattr(belief_dist, "stakeholder_role", sid)
            icon = role_icons.get(role, "👤")
            color = "#22c55e" if pos > neg else "#ef4444" if neg > pos else "#eab308"
            G.add_node(sid, pos=pos, neg=neg, icon=icon, color=color)

        state = view_state.get("current_state") or {}
        edges = state.get("relationship_edges", [])
        for edge in edges:
            src = edge.get("from") or edge.get("source")
            dst = edge.get("to") or edge.get("target")
            if src and dst and src in G.nodes and dst in G.nodes:
                G.add_edge(src, dst, weight=edge.get("strength", 0.5))

        if G.number_of_nodes() == 0:
            return "<div class='causal-graph-area'>Building belief graph...</div>"

        fig, ax = plt.subplots(1, 1, figsize=(4, 3), dpi=80)
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

        node_colors = [G.nodes[n].get("color", "#6b7280") for n in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800, ax=ax)
        nx.draw_networkx_labels(
            G,
            pos,
            labels={n: G.nodes[n].get("icon", "") for n in G.nodes()},
            font_size=12,
            ax=ax,
        )
        edge_weights = [G[u][v].get("weight", 0.5) for u, v in G.edges()]
        nx.draw_networkx_edges(
            G,
            pos,
            edge_color=edge_weights,
            width=[w * 2 for w in edge_weights],
            alpha=0.6,
            ax=ax,
        )

        ax.set_title("Belief Graph", fontsize=10)
        ax.axis("off")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="svg", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        svg_data = buf.read().decode("utf-8")
        buf.close()

        return f"<div class='causal-graph-area'>{svg_data}</div>"

    def _build_how_it_works(view_state: Dict[str, Any]) -> str:
        view_state = _normalize_view_state(view_state)
        round_started = view_state.get("round_started", False)
        hidden_class = "hidden" if round_started else ""
        if round_started:
            return f"<div class='how-it-works {hidden_class}'></div>"
        return """
        <div class="how-it-works">
            <h3>Judge Mode: 3 Things To Watch</h3>
            <ol>
                <li><strong>Graph inference:</strong> target one stakeholder and watch cross-stakeholder echoes reveal hidden influence.</li>
                <li><strong>Reward improvement:</strong> compare careful evidence-sharing against aggressive escalation.</li>
                <li><strong>CVaR veto:</strong> hostile, risky pressure can trigger a stakeholder-specific veto.</li>
            </ol>
            <p><strong>Recommended path:</strong> Start Simple Round, send evidence to Finance or Legal, then run the round.</p>
            <div class="scoring-info">
                <h4>Advanced details are hidden by default</h4>
                <div class="scoring-item"><span class="scoring-item-name">Detailed controls</span><span class="scoring-item-weight">Advanced panel</span></div>
                <div class="scoring-item"><span class="scoring-item-name">Lookahead/curriculum diagnostics</span><span class="scoring-item-weight">Developer report</span></div>
            </div>
        </div>
        """

    def _build_advanced_instructions() -> str:
        return """
        <div class="how-it-works">
            <h3>Full Controls</h3>
            <ol>
                <li><strong>Click "Start Simple Round"</strong> to begin the negotiation</li>
                <li><strong>Click a stakeholder seat</strong> to see their concerns and what they need</li>
                <li><strong>Send responses</strong> to address their concerns or select a suggestion chip</li>
                <li><strong>Run Round</strong> to see how stakeholders respond to your actions</li>
                <li><strong>Goal:</strong> Get all stakeholders to "Aligned" (green) to close the deal</li>
            </ol>
            <div class="scoring-info">
                <h4>📊 Score Components</h4>
                <div class="scoring-item"><span class="scoring-item-name">Approval Completeness</span><span class="scoring-item-weight">35%</span></div>
                <div class="scoring-item"><span class="scoring-item-name">Constraint Satisfaction</span><span class="scoring-item-weight">25%</span></div>
                <div class="scoring-item"><span class="scoring-item-name">Term Feasibility</span><span class="scoring-item-weight">15%</span></div>
                <div class="scoring-item"><span class="scoring-item-name">Relationship Durability</span><span class="scoring-item-weight">15%</span></div>
                <div class="scoring-item"><span class="scoring-item-name">Efficiency</span><span class="scoring-item-weight">10%</span></div>
            </div>
        </div>
        """

    def _build_hint(view_state: Dict[str, Any]) -> str:
        view_state = _normalize_view_state(view_state)
        round_started = view_state.get("round_started", False)
        if not round_started:
            return "<div class='hint-box'>👆 Click 'Start Simple Round' to begin!</div>"

        observation = view_state.get("current_observation") or {}
        stakeholders = observation.get("stakeholders", {})
        selected = view_state.get("selected_stakeholder")

        if not stakeholders:
            return "<div class='hint-box'>⏳ Waiting for game to load...</div>"

        aligned_stakeholders = []
        uncertain_stakeholders = []
        for sid, data in stakeholders.items():
            status = _approval_band(observation, sid)
            if status == "green":
                aligned_stakeholders.append(data.get("display_name", sid))
            else:
                uncertain_stakeholders.append(data.get("display_name", sid))

        if len(aligned_stakeholders) == len(stakeholders):
            return "<div class='hint-box'>🎉 All stakeholders aligned! Click 'Run Round' to finalize and see your score!</div>"

        if not selected or selected not in stakeholders:
            next_uncertain = (
                uncertain_stakeholders[0]
                if uncertain_stakeholders
                else list(stakeholders.values())[0].get("display_name", "a stakeholder")
            )
            return f"<div class='hint-box'>👆 Select '{next_uncertain}' from the dropdown to address their concerns</div>"

        selected_data = stakeholders.get(selected, {})
        selected_name = selected_data.get("display_name", selected)
        selected_status = _approval_band(observation, selected)

        if selected_status == "green":
            remaining = len(uncertain_stakeholders)
            if remaining > 0:
                next_uncertain = uncertain_stakeholders[0]
                return f"<div class='hint-box'>✅ {selected_name} is aligned! Now select '{next_uncertain}' from dropdown to continue</div>"
            else:
                return "<div class='hint-box'>✅ All stakeholders aligned! Click 'Run Round' to finalize</div>"
        else:
            return f"<div class='hint-box'>📝 {selected_name} needs attention. Use chips or type a message, then click 'Send Response'</div>"

    def _render_all_outputs(
        view_state: Dict[str, Any], saved_runs: List[Dict[str, Any]]
    ) -> Tuple[Any, ...]:
        view_state = _normalize_view_state(view_state)
        saved_runs = _normalize_saved_runs(saved_runs)
        _keep_valid_selected(view_state)
        chip_0, chip_1, chip_2, chip_3 = _build_chips(view_state)

        observation = view_state.get("current_observation") or {}
        stakeholders = observation.get("stakeholders", {})
        stakeholder_choices = [
            f"{ROLE_ICONS.get(p.get('role', ''), '👤')} {p.get('display_name', sid)} ({sid})"
            for sid, p in stakeholders.items()
        ]
        selected = view_state.get("selected_stakeholder")
        selected_choice = None
        if selected and selected in stakeholders:
            p = stakeholders[selected]
            selected_choice = f"{ROLE_ICONS.get(p.get('role', ''), '👤')} {p.get('display_name', selected)} ({selected})"

        round_started = view_state.get("round_started", False)

        return (
            _build_how_it_works(view_state),
            _build_round_table(view_state),
            _build_popup(view_state),
            gr.update(
                value=selected_choice,
                choices=stakeholder_choices,
                visible=round_started and bool(stakeholder_choices),
            ),
            _build_hint(view_state),
            gr.update(value=chip_0, visible=bool(chip_0)),
            gr.update(value=chip_1, visible=bool(chip_1)),
            gr.update(value=chip_2, visible=bool(chip_2)),
            gr.update(value=chip_3, visible=bool(chip_3)),
            gr.update(value=view_state.get("message_text", "")),
            _build_score_panel(view_state),
            _build_signals(view_state),
            _build_why(view_state),
            _build_causal_graph(view_state),
        )

    def handle_llm_toggle_change(
        use_llm: bool, view_state: Dict[str, Any], saved_runs: List[Dict[str, Any]]
    ) -> Tuple[Any, ...]:
        view_state = _normalize_view_state(view_state)
        saved_runs = _normalize_saved_runs(saved_runs)
        updated = dict(view_state)
        updated["use_llm_agent"] = use_llm
        return (updated, saved_runs) + _render_all_outputs(updated, saved_runs)

    def handle_seed_preset_change(
        preset_value: str, view_state: Dict[str, Any], saved_runs: List[Dict[str, Any]]
    ) -> Tuple[Any, ...]:
        view_state = _normalize_view_state(view_state)
        saved_runs = _normalize_saved_runs(saved_runs)
        updated = dict(view_state)
        if preset_value:
            parts = preset_value.split("/")
            if len(parts) == 2:
                updated["task"] = parts[0]
                try:
                    updated["seed"] = int(parts[1])
                except ValueError:
                    pass
        return (updated, saved_runs) + _render_all_outputs(updated, saved_runs)

    def handle_send_action_with_llm(
        message: str,
        use_llm: bool,
        view_state: Dict[str, Any],
        saved_runs: List[Dict[str, Any]],
    ) -> Tuple[Any, ...]:
        view_state = _normalize_view_state(view_state)
        saved_runs = _normalize_saved_runs(saved_runs)
        if not view_state.get("current_observation") or not view_state.get(
            "session_id"
        ):
            updated = _run_reset(
                "aligned", int(view_state.get("seed", 42)), "simple", view_state
            )
            return (updated, saved_runs) + _render_all_outputs(updated, saved_runs)

        if use_llm:
            model, tokenizer = _get_llm_model_and_tokenizer()
            if model is not None and tokenizer is not None:
                try:
                    action = _build_llm_action_from_observation(
                        view_state["current_observation"], model, tokenizer
                    )
                except Exception:
                    selected = view_state.get("selected_stakeholder", "all")
                    action = DealRoomAction(
                        action_type="direct_message",
                        target=selected,
                        target_ids=[selected] if selected != "all" else [],
                        message=message or "Understood.",
                    )
            else:
                selected = view_state.get("selected_stakeholder", "all")
                action = DealRoomAction(
                    action_type="direct_message",
                    target=selected,
                    target_ids=[selected] if selected != "all" else [],
                    message=message or "Understood.",
                )
        else:
            selected = view_state.get("selected_stakeholder", "all")
            action = DealRoomAction(
                action_type="direct_message",
                target=selected,
                target_ids=[selected] if selected != "all" else [],
                message=message or "Understood.",
            )
        obs, reward, done, info, state = web_manager.step_session(
            view_state["session_id"], action
        )
        updated = _record_step(
            view_state, action, obs, reward, done, info, state.model_dump()
        )
        stakeholders = list(
            updated["current_observation"].get("stakeholders", {}).keys()
        )
        current_selected = view_state.get("selected_stakeholder")
        if current_selected not in stakeholders:
            current_selected = stakeholders[0] if stakeholders else None
        updated["selected_stakeholder"] = current_selected
        updated["popup_index"] = 0
        updated["popup_queue"] = (
            [{"stakeholder_id": current_selected}] if current_selected else []
        )
        if updated.get("round_complete"):
            updated["show_hint"] = False
        else:
            updated["show_hint"] = True
            updated["hint_text"] = "Click a stakeholder to continue"
        updated["selected_chip"] = None
        updated["message_text"] = ""
        saved_runs, updated = _save_run_if_complete(updated, saved_runs)
        return (updated, saved_runs) + _render_all_outputs(updated, saved_runs)

    def handle_start_round(
        level: str, view_state: Dict[str, Any], saved_runs: List[Dict[str, Any]]
    ) -> Tuple[Any, ...]:
        view_state = _normalize_view_state(view_state)
        saved_runs = _normalize_saved_runs(saved_runs)
        task_map = {
            "simple": "aligned",
            "medium": GUIDE_DATA["task"],
            "hard": "hostile_acquisition",
        }
        task = task_map.get(level, "aligned")
        seed = view_state.get("seed", 42)
        updated = _run_reset(task, int(seed), level, view_state)
        return (updated, saved_runs) + _render_all_outputs(updated, saved_runs)

    def handle_step(
        view_state: Dict[str, Any], saved_runs: List[Dict[str, Any]]
    ) -> Tuple[Any, ...]:
        view_state = _normalize_view_state(view_state)
        saved_runs = _normalize_saved_runs(saved_runs)
        if not view_state.get("current_observation") or not view_state.get(
            "session_id"
        ):
            updated = _run_reset(
                "aligned", int(view_state.get("seed", 42)), "simple", view_state
            )
            return (updated, saved_runs) + _render_all_outputs(updated, saved_runs)
        action = _policy_action(view_state["current_observation"])
        obs, reward, done, info, state = web_manager.step_session(
            view_state["session_id"], action
        )
        updated = _record_step(
            view_state, action, obs, reward, done, info, state.model_dump()
        )
        updated["popup_index"] = 0
        stakeholders = list(
            updated["current_observation"].get("stakeholders", {}).keys()
        )
        updated["popup_queue"] = [{"stakeholder_id": s} for s in stakeholders]
        if updated.get("round_complete"):
            updated["show_hint"] = False
        else:
            updated["show_hint"] = True
            updated["hint_text"] = "Click a stakeholder to see their response"
        updated["selected_chip"] = None
        updated["message_text"] = ""
        saved_runs, updated = _save_run_if_complete(updated, saved_runs)
        return (updated, saved_runs) + _render_all_outputs(updated, saved_runs)

    def handle_seat_click(
        index_or_id: Any, view_state: Dict[str, Any], saved_runs: List[Dict[str, Any]]
    ) -> Tuple[Any, ...]:
        view_state = _normalize_view_state(view_state)
        saved_runs = _normalize_saved_runs(saved_runs)
        stakeholders = list(
            (view_state.get("current_observation") or {}).get("stakeholders", {}).keys()
        )

        stakeholder_id = None
        if isinstance(index_or_id, int):
            if index_or_id < len(stakeholders):
                stakeholder_id = stakeholders[index_or_id]
        else:
            for sid in stakeholders:
                if sid in str(index_or_id):
                    stakeholder_id = sid
                    break

        if stakeholder_id:
            view_state["selected_stakeholder"] = stakeholder_id
            view_state["popup_index"] = 0
            view_state["popup_queue"] = [{"stakeholder_id": stakeholder_id}]
            view_state["show_hint"] = False
            view_state["selected_chip"] = None
            view_state["message_text"] = ""
        return (view_state, saved_runs) + _render_all_outputs(view_state, saved_runs)

    def handle_chip_select(
        chip_index: int, view_state: Dict[str, Any], saved_runs: List[Dict[str, Any]]
    ) -> Tuple[Any, ...]:
        view_state = _normalize_view_state(view_state)
        saved_runs = _normalize_saved_runs(saved_runs)
        chips = _generate_chips(view_state)
        if 0 <= chip_index < len(chips):
            chip = chips[chip_index]
            view_state["selected_chip"] = chip
            view_state["message_text"] = chip
        return (view_state, saved_runs) + _render_all_outputs(view_state, saved_runs)

    def handle_toggle_advanced(
        view_state: Dict[str, Any], saved_runs: List[Dict[str, Any]]
    ) -> Tuple[Any, ...]:
        view_state = _normalize_view_state(view_state)
        saved_runs = _normalize_saved_runs(saved_runs)
        view_state["show_advanced"] = not view_state.get("show_advanced", False)
        return (view_state, saved_runs) + _render_all_outputs(view_state, saved_runs)

    def handle_toggle_auto_play(
        view_state: Dict[str, Any], saved_runs: List[Dict[str, Any]]
    ) -> Tuple[Any, ...]:
        view_state = _normalize_view_state(view_state)
        saved_runs = _normalize_saved_runs(saved_runs)
        view_state["auto_playing"] = not view_state.get("auto_playing", False)
        view_state["auto_paused"] = False
        return (view_state, saved_runs) + _render_all_outputs(view_state, saved_runs)

    def handle_toggle_pause(
        view_state: Dict[str, Any], saved_runs: List[Dict[str, Any]]
    ) -> Tuple[Any, ...]:
        view_state = _normalize_view_state(view_state)
        saved_runs = _normalize_saved_runs(saved_runs)
        view_state["auto_paused"] = not view_state.get("auto_paused", False)
        return (view_state, saved_runs) + _render_all_outputs(view_state, saved_runs)

    def handle_stop_auto(
        view_state: Dict[str, Any], saved_runs: List[Dict[str, Any]]
    ) -> Tuple[Any, ...]:
        view_state = _normalize_view_state(view_state)
        saved_runs = _normalize_saved_runs(saved_runs)
        view_state["auto_playing"] = False
        view_state["auto_paused"] = False
        return (view_state, saved_runs) + _render_all_outputs(view_state, saved_runs)

    def handle_speed_change(
        speed: str, view_state: Dict[str, Any], saved_runs: List[Dict[str, Any]]
    ) -> Tuple[Any, ...]:
        view_state = _normalize_view_state(view_state)
        saved_runs = _normalize_saved_runs(saved_runs)
        view_state["auto_speed"] = speed
        return (view_state, saved_runs) + _render_all_outputs(view_state, saved_runs)

    def handle_level_select(
        level: str, view_state: Dict[str, Any], saved_runs: List[Dict[str, Any]]
    ) -> Tuple[Any, ...]:
        view_state = _normalize_view_state(view_state)
        saved_runs = _normalize_saved_runs(saved_runs)
        if level not in view_state.get("unlocked_levels", []):
            return (view_state, saved_runs) + _render_all_outputs(
                view_state, saved_runs
            )
        return handle_start_round(level, view_state, saved_runs)

    demo = gr.Blocks(elem_classes=["dealroom-lab"])
    with demo:
        gr.HTML(f"<style>{CUSTOM_CSS}</style>")

        view_state = gr.State(default_view_state())
        saved_runs = gr.State([])

        with gr.Column(elem_classes=["lab-title"]):
            gr.HTML(
                "<h1>🎯 DealRoom Lab</h1>"
                "<p>Negotiate with stakeholders • Close the deal</p>"
            )

        with gr.Row(elem_classes=["step-indicator"]):
            step_simple = gr.Button(
                "🔓 Simple\nStart here",
                elem_classes=["step-btn", "active"],
                variant="secondary",
            )
            step_medium = gr.Button(
                "⚪ Medium\nAvailable after Simple",
                elem_classes=["step-btn"],
                variant="secondary",
            )
            step_hard = gr.Button(
                "⚪ Hard\nAvailable after Medium",
                elem_classes=["step-btn"],
                variant="secondary",
            )

        how_it_works_html = gr.HTML()

        start_btn = gr.Button(
            "▶ Start Simple Round", elem_classes=["start-btn"], variant="primary"
        )

        with gr.Row(elem_classes=["main-layout"]):
            with gr.Column(elem_classes=["left-panel"]):
                table_html = gr.HTML()
                stakeholder_select = gr.Dropdown(
                    label="👇 Select a stakeholder to see their message:",
                    choices=[],
                    value=None,
                    interactive=True,
                    visible=False,
                )
                popup_html = gr.HTML()
                hint_html = gr.HTML(value="")

                with gr.Column(elem_classes=["action-bar"]):
                    gr.HTML("<h3>💬 Your Response</h3>")
                    with gr.Row(elem_classes=["chips-grid"]):
                        chip_btn_0 = gr.Button(
                            "",
                            variant="secondary",
                            elem_classes=["chip-btn"],
                            visible=False,
                        )
                        chip_btn_1 = gr.Button(
                            "",
                            variant="secondary",
                            elem_classes=["chip-btn"],
                            visible=False,
                        )
                        chip_btn_2 = gr.Button(
                            "",
                            variant="secondary",
                            elem_classes=["chip-btn"],
                            visible=False,
                        )
                        chip_btn_3 = gr.Button(
                            "",
                            variant="secondary",
                            elem_classes=["chip-btn"],
                            visible=False,
                        )
                    message_input = gr.Textbox(
                        placeholder="Type your message or select a suggestion above...",
                        lines=2,
                        elem_classes=["message-input"],
                    )
                    send_btn = gr.Button(
                        "Send Response", elem_classes=["send-btn"], variant="primary"
                    )

                    gr.HTML(
                        "<div style='margin-top:10px;'>▶ Run Round to see stakeholder responses</div>"
                    )
                    with gr.Row(elem_classes=["sim-controls"]):
                        run_btn = gr.Button(
                            "▶ Run Round", elem_classes=["run-btn"], variant="primary"
                        )
                        step_btn = gr.Button(
                            "⏭ Step", elem_classes=["step-btn-sm"], variant="secondary"
                        )

                    gr.HTML(
                        "<div class='advanced-toggle' onclick='this.nextElementSibling.classList.toggle(\"open\")'>⚙️ Advanced Controls ▼</div>"
                    )
                    with gr.Column(
                        elem_classes=["advanced-controls"]
                    ) as advanced_panel:
                        speed_select = gr.Dropdown(
                            ["slow", "medium", "fast"],
                            value="medium",
                            label="Auto-play Speed",
                            elem_classes=["speed-select"],
                        )
                        with gr.Row(elem_classes=["auto-btns"]):
                            auto_start_btn = gr.Button(
                                "▶ Auto-play", elem_classes=["auto-btn"]
                            )
                            pause_btn = gr.Button(
                                "⏸ Pause", elem_classes=["auto-btn", "pause-btn"]
                            )
                            stop_btn = gr.Button(
                                "⏹ Stop", elem_classes=["auto-btn", "stop-btn"]
                            )
                        llm_agent_toggle = gr.Checkbox(
                            label="🤖 Use LLM Agent (auto-negotiate)",
                            value=False,
                            elem_classes=["llm-agent-toggle"],
                        )
                        seed_preset_dropdown = gr.Dropdown(
                            label="Seed Preset (Quick Start)",
                            choices=[
                                "hostile_acquisition/7",
                                "hostile_acquisition/13",
                                "hostile_acquisition/42",
                                "hostile_acquisition/99",
                                "aligned/7",
                                "aligned/42",
                                "aligned/99",
                                "conflicted/13",
                                "conflicted/42",
                                "conflicted/99",
                            ],
                            value=None,
                            interactive=True,
                            elem_classes=["seed-preset-dropdown"],
                        )

            with gr.Column(elem_classes=["right-panel"]):
                score_html = gr.HTML()
                signals_html = gr.HTML()
                why_html = gr.HTML()
                causal_graph_html = gr.HTML()

        outputs = [
            view_state,
            saved_runs,
            how_it_works_html,
            table_html,
            popup_html,
            stakeholder_select,
            hint_html,
            chip_btn_0,
            chip_btn_1,
            chip_btn_2,
            chip_btn_3,
            message_input,
            score_html,
            signals_html,
            why_html,
            causal_graph_html,
        ]

        def render_initial(vs, sr):
            vs = _normalize_view_state(vs)
            sr = _normalize_saved_runs(sr)
            return (vs, sr) + _render_all_outputs(vs, sr)

        demo.load(fn=render_initial, inputs=[view_state, saved_runs], outputs=outputs)

        start_btn.click(
            fn=handle_start_round,
            inputs=[gr.State("simple"), view_state, saved_runs],
            outputs=outputs,
        )

        stakeholder_select.change(
            fn=handle_seat_click,
            inputs=[stakeholder_select, view_state, saved_runs],
            outputs=outputs,
        )

        step_simple.click(
            fn=handle_level_select,
            inputs=[gr.State("simple"), view_state, saved_runs],
            outputs=outputs,
        )
        step_medium.click(
            fn=handle_level_select,
            inputs=[gr.State("medium"), view_state, saved_runs],
            outputs=outputs,
        )
        step_hard.click(
            fn=handle_level_select,
            inputs=[gr.State("hard"), view_state, saved_runs],
            outputs=outputs,
        )

        run_btn.click(
            fn=handle_step,
            inputs=[view_state, saved_runs],
            outputs=outputs,
        )

        step_btn.click(
            fn=handle_step,
            inputs=[view_state, saved_runs],
            outputs=outputs,
        )

        send_btn.click(
            fn=handle_send_action_with_llm,
            inputs=[message_input, llm_agent_toggle, view_state, saved_runs],
            outputs=outputs,
        )

        llm_agent_toggle.change(
            fn=handle_llm_toggle_change,
            inputs=[llm_agent_toggle, view_state, saved_runs],
            outputs=outputs,
        )

        seed_preset_dropdown.change(
            fn=handle_seed_preset_change,
            inputs=[seed_preset_dropdown, view_state, saved_runs],
            outputs=outputs,
        )

        chip_btn_0.click(
            fn=handle_chip_select,
            inputs=[gr.State(0), view_state, saved_runs],
            outputs=outputs,
        )
        chip_btn_1.click(
            fn=handle_chip_select,
            inputs=[gr.State(1), view_state, saved_runs],
            outputs=outputs,
        )
        chip_btn_2.click(
            fn=handle_chip_select,
            inputs=[gr.State(2), view_state, saved_runs],
            outputs=outputs,
        )
        chip_btn_3.click(
            fn=handle_chip_select,
            inputs=[gr.State(3), view_state, saved_runs],
            outputs=outputs,
        )

        auto_start_btn.click(
            fn=handle_toggle_auto_play,
            inputs=[view_state, saved_runs],
            outputs=outputs,
        )

        pause_btn.click(
            fn=handle_toggle_pause,
            inputs=[view_state, saved_runs],
            outputs=outputs,
        )

        stop_btn.click(
            fn=handle_stop_auto,
            inputs=[view_state, saved_runs],
            outputs=outputs,
        )

        speed_select.change(
            fn=handle_speed_change,
            inputs=[speed_select, view_state, saved_runs],
            outputs=outputs,
        )

    return demo
