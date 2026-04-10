"""DealRoom custom Gradio tab - visual round-table conference interface."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
from openenv.core.env_server.types import EnvironmentMetadata

from models import DealRoomAction, DealRoomObservation, DealRoomState
from server.grader import CCIGrader
from server.session_pool import DealRoomSessionPool
from server.walkthrough_data import GUIDE_DATA

TASK_ORDER = ["aligned", "conflicted", "hostile_acquisition"]
TASK_DISPLAY = {
    "aligned": "Simple Round",
    "conflicted": "Medium Round",
    "hostile_acquisition": "Hard Round",
}
LEVEL_TO_TASK = {
    "simple": "aligned",
    "medium": GUIDE_DATA["task"],
    "hard": "hostile_acquisition",
}
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
STAGE_ORDER = ["evaluation", "negotiation", "legal_review", "final_approval", "closed"]

CUSTOM_CSS = """
.dealroom-custom {
    background: #0d1117;
    border-radius: 16px;
    padding: 20px;
    font-family: "IBM Plex Sans", system-ui, sans-serif;
    min-height: 700px;
}
.dealroom-custom * {
    color: #e5e7eb;
}
.dealroom-custom h1, .dealroom-custom h2, .dealroom-custom h3 {
    color: #f3f4f6;
}
.split-container {
    display: grid;
    grid-template-columns: 1fr 280px;
    gap: 20px;
    min-height: 550px;
}
.left-panel {
    display: flex;
    flex-direction: column;
    gap: 16px;
}
.right-panel {
    background: #10161d;
    border: 1px solid #1f2937;
    border-radius: 12px;
    padding: 16px;
    height: fit-content;
}
.progress-strip {
    display: flex;
    gap: 8px;
    padding: 12px 16px;
    background: #10161d;
    border-radius: 10px;
    margin-bottom: 8px;
}
.progress-step {
    flex: 1;
    text-align: center;
    padding: 8px 12px;
    border-radius: 8px;
    font-size: 0.85rem;
    background: #1f2937;
    color: #9ca3af;
    border: 1px solid #374151;
}
.progress-step.active {
    background: rgba(34, 197, 94, 0.15);
    border-color: #22c55e;
    color: #22c55e;
}
.progress-step.locked {
    opacity: 0.5;
}
.progress-step .unlock-hint {
    display: block;
    font-size: 0.7rem;
    margin-top: 4px;
    opacity: 0.7;
}
.round-area {
    background: radial-gradient(ellipse at center, #1a1f2e 0%, #0d1117 70%);
    border-radius: 50%;
    width: 100%;
    max-width: 420px;
    height: 340px;
    margin: 0 auto;
    border: 2px solid #2a3142;
    box-shadow: 0 0 60px rgba(34, 197, 94, 0.08), inset 0 0 60px rgba(0,0,0,0.5);
    position: relative;
}
.round-center {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 100px;
    height: 100px;
    background: rgba(13, 17, 23, 0.9);
    border: 2px solid #304154;
    border-radius: 50%;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
}
.round-center strong {
    font-size: 0.85rem;
    color: #22c55e;
}
.round-center span {
    font-size: 0.7rem;
    color: #9ca3af;
}
.seat {
    position: absolute;
    width: 72px;
    height: 72px;
    border-radius: 50%;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    transition: all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
    cursor: pointer;
    border: 3px solid transparent;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
}
.seat:hover {
    transform: scale(1.1);
    z-index: 10;
}
.seat.selected {
    border-color: #60a5fa;
    box-shadow: 0 0 20px rgba(96, 165, 250, 0.4);
}
.seat.supporter {
    background: linear-gradient(135deg, #065f46 0%, #064e3b 100%);
    border-color: #22c55e;
}
.seat.blocker {
    background: linear-gradient(135deg, #7f1d1d 0%, #450a0a 100%);
    border-color: #ef4444;
    animation: pulse-blocker 2s ease-in-out infinite;
}
.seat.uncertain {
    background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
    border-color: #f59e0b;
}
.seat.dimmed {
    opacity: 0.4;
}
.seat-icon {
    font-size: 1.3rem;
    margin-bottom: 2px;
}
.seat-name {
    font-size: 0.65rem;
    font-weight: 700;
    color: #fff;
    text-transform: uppercase;
}
.seat-role {
    font-size: 0.6rem;
    color: #9ca3af;
}
@keyframes pulse-blocker {
    0%, 100% { box-shadow: 0 4px 20px rgba(239, 68, 68, 0.3); }
    50% { box-shadow: 0 4px 30px rgba(239, 68, 68, 0.6); }
}
.anchored-popup {
    position: absolute;
    width: 260px;
    background: #0f141b;
    border: 1px solid #304154;
    border-radius: 12px;
    padding: 14px;
    z-index: 20;
    animation: popup-appear 0.3s ease-out;
    box-shadow: 0 8px 32px rgba(0,0,0,0.5);
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
}
@keyframes popup-appear {
    from { opacity: 0; transform: translate(-50%, -40%); }
    to { opacity: 1; transform: translate(-50%, -50%); }
}
.popup-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 10px;
    padding-bottom: 8px;
    border-bottom: 1px solid #1f2937;
}
.popup-icon {
    font-size: 1.3rem;
}
.popup-name {
    font-weight: 700;
    color: #f3f4f6;
    font-size: 0.9rem;
}
.popup-role {
    font-size: 0.7rem;
    color: #9ca3af;
}
.popup-quote {
    background: #0a1016;
    border-left: 3px solid #3b82f6;
    padding: 10px 12px;
    border-radius: 0 6px 6px 0;
    margin: 8px 0;
    color: #e5e7eb;
    font-style: italic;
    font-size: 0.85rem;
}
.popup-status {
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 6px 0;
}
.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
}
.status-dot.green { background: #22c55e; }
.status-dot.amber { background: #f59e0b; }
.status-dot.red { background: #ef4444; }
.popup-request {
    background: rgba(34, 197, 94, 0.08);
    border: 1px solid rgba(34, 197, 94, 0.2);
    border-radius: 6px;
    padding: 8px 10px;
    margin-top: 8px;
}
.popup-request strong {
    color: #22c55e;
    font-size: 0.75rem;
}
.popup-request p {
    color: #9ca3af;
    font-size: 0.8rem;
    margin: 2px 0 0;
}
.action-bar {
    background: #10161d;
    border: 1px solid #1f2937;
    border-radius: 12px;
    padding: 14px;
}
.action-bar h3 {
    margin: 0 0 10px;
    font-size: 0.95rem;
}
.chat-input-row {
    display: flex;
    gap: 8px;
    margin-bottom: 8px;
}
.quick-message {
    flex: 1;
    background: #0d1117 !important;
    border: 1px solid #2b3746 !important;
    border-radius: 8px !important;
    color: #e5e7eb !important;
    padding: 10px 12px !important;
}
.send-btn {
    background: #1f8b4c !important;
    border: none !important;
    border-radius: 8px !important;
    color: #fff !important;
    font-weight: 600 !important;
}
.run-btn {
    background: linear-gradient(135deg, #1f8b4c 0%, #166534 100%) !important;
    border: none !important;
    border-radius: 8px !important;
    color: #fff !important;
    font-weight: 600 !important;
    padding: 12px 20px !important;
}
.auto-btn {
    background: #1f2937 !important;
    border: 1px solid #374151 !important;
    border-radius: 8px !important;
    color: #e5e7eb !important;
}
.score-panel {
    background: linear-gradient(135deg, #1a1f2e 0%, #0f1419 100%);
    border: 2px solid #22c55e;
    border-radius: 14px;
    padding: 16px;
    text-align: center;
    margin-bottom: 14px;
}
.score-value {
    font-size: 2.5rem;
    font-weight: 700;
    color: #22c55e;
    text-shadow: 0 0 20px rgba(34, 197, 94, 0.5);
    line-height: 1;
}
.score-label {
    font-size: 0.8rem;
    color: #9ca3af;
    margin-top: 4px;
}
.score-delta {
    font-size: 1rem;
    color: #22c55e;
    margin-top: 6px;
}
.signals-list {
    margin-top: 12px;
}
.signal-item {
    display: flex;
    align-items: flex-start;
    gap: 6px;
    padding: 6px 0;
    border-bottom: 1px solid #1f2937;
    font-size: 0.8rem;
}
.signal-icon {
    font-size: 0.85rem;
}
.signal-text {
    color: #d1d5db;
}
.why-collapsible {
    margin-top: 12px;
    border-top: 1px solid #1f2937;
    padding-top: 10px;
}
.why-toggle {
    background: none;
    border: none;
    color: #9ca3af;
    font-size: 0.8rem;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 2px 0;
}
.why-toggle:hover {
    color: #e5e7eb;
}
.why-content {
    display: none;
    margin-top: 8px;
    padding: 8px;
    background: #0d1117;
    border-radius: 6px;
    font-size: 0.8rem;
    color: #d1d5db;
}
.why-content.open {
    display: block;
}
.blocker-tag {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 3px 8px;
    border-radius: 999px;
    font-size: 0.75rem;
    background: rgba(239, 68, 68, 0.15);
    color: #ef4444;
    margin: 3px 3px 3px 0;
}
.request-tag {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 3px 8px;
    border-radius: 999px;
    font-size: 0.75rem;
    background: rgba(59, 130, 246, 0.15);
    color: #60a5fa;
    margin: 3px 3px 3px 0;
}
.insights-section {
    margin-bottom: 12px;
}
.insights-section h4 {
    font-size: 0.8rem;
    color: #9ca3af;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin: 0 0 8px;
}
.step-accordion {
    border: 1px solid #1f2937;
    border-radius: 8px;
    margin-bottom: 6px;
    overflow: hidden;
}
.step-accordion-header {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 12px 14px;
    background: #10161d;
    cursor: pointer;
    transition: background 0.2s;
}
.step-accordion-header:hover {
    background: #1a2332;
}
.step-accordion-header.active {
    background: rgba(34, 197, 94, 0.08);
    border-left: 3px solid #22c55e;
}
.step-accordion-header.locked {
    opacity: 0.6;
    cursor: not-allowed;
}
.step-number {
    width: 24px;
    height: 24px;
    border-radius: 50%;
    background: #1f2937;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 0.8rem;
}
.step-number.active {
    background: #22c55e;
    color: #0d1117;
}
.step-number.locked {
    background: #374151;
    color: #6b7280;
}
.step-info {
    flex: 1;
}
.step-title {
    font-weight: 600;
    color: #f3f4f6;
    font-size: 0.9rem;
}
.step-desc {
    font-size: 0.75rem;
    color: #9ca3af;
    margin-top: 2px;
}
.step-lock-icon {
    font-size: 0.9rem;
}
.divider-line {
    height: 1px;
    background: #1f2937;
    margin: 12px 0;
}
.gr-input input, .gr-input textarea, .gr-input select {
    background: #0d1117 !important;
    border: 1px solid #2b3746 !important;
    color: #e5e7eb !important;
    border-radius: 6px !important;
}
.gr-input label {
    color: #9ca3af !important;
    font-size: 0.8rem !important;
}
""".strip()


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


def build_custom_tab(
    web_manager: DealRoomWebManager,
    action_fields: List[Dict[str, Any]],
    metadata: EnvironmentMetadata,
    is_chat_env: bool,
    title: str,
    quick_start_md: Optional[str],
) -> gr.Blocks:
    del action_fields, metadata, is_chat_env, title, quick_start_md

    SEAT_POSITIONS = [
        {"top": "10%", "left": "50%", "transform": "translateX(-50%)"},
        {"top": "35%", "left": "8%"},
        {"top": "35%", "right": "8%"},
        {"bottom": "10%", "left": "50%", "transform": "translateX(-50%)"},
    ]

    def default_view_state() -> Dict[str, Any]:
        return {
            "task": "aligned",
            "seed": 42,
            "level": "simple",
            "source": "custom",
            "guide_step": 0,
            "session_id": None,
            "selected_stakeholder": None,
            "current_observation": None,
            "current_state": None,
            "trace": [],
            "status_message": "Click Open Simple to start the negotiation.",
            "popup_queue": [],
            "popup_index": 0,
            "auto_advance": False,
            "auto_delay": 5,
            "unlocked_levels": ["simple"],
            "show_suggestions": False,
            "last_score": 0.0,
            "score_delta": None,
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
        return str(progress.get("band", "neutral"))

    def _run_reset(
        task: str,
        seed: int,
        level: str,
        source: str,
        view_state: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        current = _normalize_view_state(view_state)
        session_id, obs, state = web_manager.reset_session(
            task_id=task,
            seed=int(seed),
            session_id=current.get("session_id"),
        )
        observation = obs.model_dump()
        updated = dict(current)
        updated.update(
            {
                "task": task,
                "seed": int(seed),
                "level": level,
                "source": source,
                "guide_step": 0,
                "session_id": session_id,
                "current_observation": observation,
                "current_state": state.model_dump(),
                "selected_stakeholder": _first_stakeholder_id(observation),
                "popup_queue": [],
                "popup_index": 0,
                "trace": [
                    {
                        "kind": "reset",
                        "task": task,
                        "seed": int(seed),
                        "level": level,
                        "stage": obs.deal_stage,
                        "blockers": list(obs.active_blockers),
                    }
                ],
                "status_message": f"Round ready. Click Run to simulate.",
                "last_score": 0.0,
                "score_delta": None,
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
        trace.append(
            {
                "kind": "step",
                "step": len([item for item in trace if item.get("kind") == "step"]) + 1,
                "action": action.model_dump(),
                "reward": reward,
                "done": done,
                "stage": obs.deal_stage,
                "blockers": list(obs.active_blockers),
                "dense_reward_breakdown": info.get("dense_reward_breakdown", {}),
                "relationship_effects": info.get("relationship_effects", []),
            }
        )
        updated["trace"] = trace
        updated["current_observation"] = obs.model_dump()
        updated["current_state"] = state
        updated["score_delta"] = reward - old_score if old_score else None
        updated["last_score"] = reward
        _keep_valid_selected(updated)
        updated["popup_queue"] = []
        updated["popup_index"] = 0
        updated["status_message"] = (
            f"Step {trace[-1]['step']} | Reward: {reward:.2f} | "
            f"{'Done!' if done else 'Continue...'}"
        )
        return updated

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
            action_type="group_proposal",
            target="all",
            target_ids=list(obs.stakeholders.keys()),
            message="I believe we have enough aligned evidence to move to final approval.",
            proposed_terms={
                "price": 180000,
                "timeline_weeks": 14,
                "support_level": "named_support_lead",
                "liability_cap": "mutual_cap",
            },
        )

    def _run_action(
        payload: Dict[str, Any],
        source: str,
        view_state: Dict[str, Any],
        saved_runs: List[Dict[str, Any]],
    ) -> Tuple[Any, ...]:
        view_state = _normalize_view_state(view_state)
        saved_runs = _normalize_saved_runs(saved_runs)
        if not view_state.get("current_observation") or not view_state.get(
            "session_id"
        ):
            view_state = _run_reset(
                view_state["task"],
                int(view_state["seed"]),
                view_state["level"],
                source,
                view_state,
            )
        action = DealRoomAction.model_validate(payload)
        obs, reward, done, info, state = web_manager.step_session(
            view_state["session_id"], action
        )
        updated = _record_step(
            view_state, action, obs, reward, done, info, state.model_dump()
        )
        updated["source"] = source
        saved_runs = _save_run_if_complete(updated, saved_runs)
        return (updated, saved_runs) + _render_all_outputs(updated, saved_runs)

    def _save_run_if_complete(
        view_state: Dict[str, Any], saved_runs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        view_state = _normalize_view_state(view_state)
        saved_runs = _normalize_saved_runs(saved_runs)
        observation = view_state.get("current_observation") or {}
        if not observation.get("done"):
            return saved_runs
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
        return saved_runs[-8:]

    def _target_choices(view_state: Dict[str, Any]) -> List[str]:
        observation = _normalize_view_state(view_state).get("current_observation") or {}
        return ["all"] + list(observation.get("stakeholders", {}).keys())

    def _build_round_table(view_state: Dict[str, Any]) -> str:
        view_state = _normalize_view_state(view_state)
        observation = view_state.get("current_observation") or {}
        selected = view_state.get("selected_stakeholder")
        popup_queue = view_state.get("popup_queue", [])
        current_idx = view_state.get("popup_index", 0)
        seats_html = []
        stakeholder_list = list(observation.get("stakeholders", {}).items())
        for index, (stakeholder_id, payload) in enumerate(stakeholder_list):
            if index >= len(SEAT_POSITIONS):
                break
            band = _approval_band(observation, stakeholder_id)
            pos = SEAT_POSITIONS[index]
            pos_style = "; ".join(f"{k}: {v}" for k, v in pos.items())
            is_selected = stakeholder_id == selected
            is_speaking = (
                len(popup_queue) > 0
                and current_idx < len(popup_queue)
                and popup_queue[current_idx].get("stakeholder_id") == stakeholder_id
            )
            seat_class = "seat"
            if band == "supporter" or band == "workable":
                seat_class += " supporter"
            elif band == "blocker":
                seat_class += " blocker"
            elif band == "uncertain":
                seat_class += " uncertain"
            if is_selected or is_speaking:
                seat_class += " selected"
            elif selected and stakeholder_id != selected:
                seat_class += " dimmed"
            seats_html.append(
                f"<div class='{seat_class}' style='{pos_style}' data-stakeholder='{_escape(stakeholder_id)}'>"
                f"<div class='seat-icon'>{ROLE_ICONS.get(payload.get('role', ''), '👤')}</div>"
                f"<div class='seat-name'>{_escape(payload.get('display_name', stakeholder_id)[:8])}</div>"
                f"<div class='seat-role'>{_escape(payload.get('role', '')[:10])}</div>"
                f"</div>"
            )
        level_text = {"simple": "Basic", "medium": "Medium", "hard": "Hard"}.get(
            view_state["level"], "Basic"
        )
        return (
            "<div class='round-area'>"
            + "".join(seats_html)
            + (
                "<div class='round-center'>"
                f"<strong>DEAL</strong>"
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
            if not observation:
                return "<div class='anchored-popup' style='display:none;'></div>"
            selected = view_state.get("selected_stakeholder")
            if not selected or selected not in observation.get("stakeholders", {}):
                return "<div class='anchored-popup' style='display:none;'></div>"
            stakeholder_id = selected
            payload = observation["stakeholders"][stakeholder_id]
        else:
            stakeholder_id = popup_queue[current_idx].get("stakeholder_id")
            payload = observation.get("stakeholders", {}).get(stakeholder_id, {})
        message = observation.get("stakeholder_messages", {}).get(
            stakeholder_id, "No message yet."
        )
        requested = observation.get("requested_artifacts", {}).get(stakeholder_id, [])
        progress = observation.get("approval_path_progress", {}).get(stakeholder_id, {})
        band = progress.get("band", "neutral")
        status_class = (
            "green"
            if band in ("supporter", "workable")
            else ("red" if band == "blocker" else "amber")
        )
        status_text = {
            "supporter": "Aligned",
            "workable": "Aligned",
            "blocker": "Blocking",
            "uncertain": "Uncertain",
            "neutral": "Neutral",
        }.get(band, "Neutral")
        request_text = (
            ", ".join(r.replace("_", " ") for r in requested)
            if requested
            else "Nothing requested"
        )
        popup_content = (
            "<div class='popup-header'>"
            f"<div class='popup-icon'>{ROLE_ICONS.get(payload.get('role', ''), '👤')}</div>"
            f"<div><div class='popup-name'>{_escape(payload.get('display_name', stakeholder_id))}</div>"
            f"<div class='popup-role'>{_escape(payload.get('role', ''))}</div></div>"
            "</div>"
            f"<div class='popup-quote'>\"{_escape(message)}\"</div>"
            "<div class='popup-status'>"
            f"<div class='status-dot {status_class}'></div>"
            f"<span style='color: #e5e7eb;'>{status_text}</span>"
            "</div>"
            f"<div class='popup-request'>"
            f"<strong>Needs:</strong>"
            f"<p>{_escape(request_text)}</p>"
            f"</div>"
        )
        return f"<div class='anchored-popup'>{popup_content}</div>"

    def _build_score_panel(view_state: Dict[str, Any]) -> str:
        view_state = _normalize_view_state(view_state)
        observation = view_state.get("current_observation") or {}
        state = view_state.get("current_state") or {}
        current_score = view_state.get("last_score", 0.0)
        score_delta = view_state.get("score_delta")
        done = observation.get("done", False)
        if done:
            final_score = CCIGrader.compute(DealRoomState.model_validate(state))
            return (
                "<div class='score-panel'>"
                f"<div class='score-value'>{final_score:.2f}</div>"
                "<div class='score-label'>Final Score</div>"
                "</div>"
            )
        delta_html = ""
        if score_delta is not None:
            sign = "+" if score_delta >= 0 else ""
            delta_color = "#22c55e" if score_delta >= 0 else "#ef4444"
            delta_html = f"<div class='score-delta' style='color: {delta_color};'>{sign}{score_delta:.2f}</div>"
        blockers = observation.get("active_blockers", [])
        blocker_html = ""
        if blockers:
            blocker_tags = "".join(
                f"<span class='blocker-tag'>⚠️ {_escape(b)}</span>" for b in blockers
            )
            blocker_html = f"<div style='margin-top:10px;'>{blocker_tags}</div>"
        return (
            "<div class='score-panel'>"
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
            return "<div class='signals-list'><div class='signal-item'><span class='signal-icon'>📊</span><span class='signal-text'>No signals yet</span></div></div>"
        signals = []
        for stakeholder_id, artifacts in observation.get(
            "requested_artifacts", {}
        ).items():
            if artifacts:
                for art in artifacts:
                    signals.append(
                        f"<span class='request-tag'>📋 {_escape(art.replace('_', ' '))}</span>"
                    )
        if not signals:
            return "<div class='signals-list'><div class='signal-item'><span class='signal-icon'>📊</span><span class='signal-text'>No pending requests</span></div></div>"
        return f"<div class='signals-list'>{''.join(signals)}</div>"

    def _build_why(view_state: Dict[str, Any]) -> str:
        view_state = _normalize_view_state(view_state)
        observation = view_state.get("current_observation") or {}
        blockers = observation.get("active_blockers", [])
        if not blockers:
            return (
                "<div class='why-collapsible'>"
                "<button class='why-toggle' onclick='this.nextElementSibling.classList.toggle(\"open\")'>"
                "▼ Why this score?"
                "</button>"
                "<div class='why-content'>"
                "<p>No active blockers. The negotiation is progressing well.</p>"
                "</div>"
                "</div>"
            )
        reasons = [f"• {_escape(b)} is blocking progress" for b in blockers]
        content = "<p>" + "<br>".join(reasons) + "</p>"
        return (
            "<div class='why-collapsible'>"
            "<button class='why-toggle' onclick='this.nextElementSibling.classList.toggle(\"open\")'>"
            "▼ Why this score?"
            "</button>"
            f"<div class='why-content'>{content}</div>"
            "</div>"
        )

    def _render_all_outputs(
        view_state: Dict[str, Any], saved_runs: List[Dict[str, Any]]
    ) -> Tuple[Any, ...]:
        view_state = _normalize_view_state(view_state)
        saved_runs = _normalize_saved_runs(saved_runs)
        _keep_valid_selected(view_state)
        targets = _target_choices(view_state)
        return (
            _build_round_table(view_state),
            _build_popup(view_state),
            _build_score_panel(view_state),
            _build_signals(view_state),
            _build_why(view_state),
            gr.update(choices=targets, value="all"),
        )

    def _render_insights_panel(
        view_state: Dict[str, Any], saved_runs: List[Dict[str, Any]]
    ) -> Tuple[Any, ...]:
        view_state = _normalize_view_state(view_state)
        return (
            _build_score_panel(view_state),
            _build_signals(view_state),
            _build_why(view_state),
        )

    def handle_reset(
        task: str,
        seed: int,
        level: str,
        view_state: Dict[str, Any],
        saved_runs: List[Dict[str, Any]],
    ) -> Tuple[Any, ...]:
        view_state = _normalize_view_state(view_state)
        saved_runs = _normalize_saved_runs(saved_runs)
        updated = _run_reset(task, int(seed), level, level, view_state)
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
                view_state["task"],
                int(view_state["seed"]),
                view_state["level"],
                view_state["level"],
                view_state,
            )
            return (updated, saved_runs) + _render_all_outputs(updated, saved_runs)
        action = _policy_action(view_state["current_observation"])
        return _run_action(action.model_dump(), "auto", view_state, saved_runs)

    def handle_send_message(
        message: str, view_state: Dict[str, Any], saved_runs: List[Dict[str, Any]]
    ) -> Tuple[Any, ...]:
        view_state = _normalize_view_state(view_state)
        saved_runs = _normalize_saved_runs(saved_runs)
        if not view_state.get("current_observation") or not view_state.get(
            "session_id"
        ):
            updated = _run_reset(
                view_state["task"],
                int(view_state["seed"]),
                view_state["level"],
                view_state["level"],
                view_state,
            )
            return (updated, saved_runs) + _render_all_outputs(updated, saved_runs)
        action = DealRoomAction(
            action_type="direct_message",
            target="all",
            target_ids=[],
            message=message,
        )
        return _run_action(action.model_dump(), "manual", view_state, saved_runs)

    def handle_seat_click(
        stakeholder_id: str,
        view_state: Dict[str, Any],
        saved_runs: List[Dict[str, Any]],
    ) -> Tuple[Any, ...]:
        view_state = _normalize_view_state(view_state)
        saved_runs = _normalize_saved_runs(saved_runs)
        view_state["selected_stakeholder"] = stakeholder_id
        view_state["popup_index"] = 0
        view_state["popup_queue"] = [{"stakeholder_id": stakeholder_id}]
        return (view_state, saved_runs) + _render_all_outputs(view_state, saved_runs)

    def handle_next_popup(
        view_state: Dict[str, Any], saved_runs: List[Dict[str, Any]]
    ) -> Tuple[Any, ...]:
        view_state = _normalize_view_state(view_state)
        saved_runs = _normalize_saved_runs(saved_runs)
        popup_queue = view_state.get("popup_queue", [])
        current_idx = view_state.get("popup_index", 0)
        if current_idx < len(popup_queue) - 1:
            view_state["popup_index"] = current_idx + 1
        return (view_state, saved_runs) + _render_all_outputs(view_state, saved_runs)

    def handle_auto_advance_toggle(
        view_state: Dict[str, Any], saved_runs: List[Dict[str, Any]]
    ) -> Tuple[Any, ...]:
        view_state = _normalize_view_state(view_state)
        saved_runs = _normalize_saved_runs(saved_runs)
        view_state["auto_advance"] = not view_state.get("auto_advance", False)
        return (view_state, saved_runs) + _render_all_outputs(view_state, saved_runs)

    def handle_focus_seat(
        index: int, view_state: Dict[str, Any], saved_runs: List[Dict[str, Any]]
    ) -> Tuple[Any, ...]:
        view_state = _normalize_view_state(view_state)
        saved_runs = _normalize_saved_runs(saved_runs)
        stakeholders = list(
            (view_state.get("current_observation") or {}).get("stakeholders", {}).keys()
        )
        if index < len(stakeholders):
            stakeholder_id = stakeholders[index]
            view_state["selected_stakeholder"] = stakeholder_id
            view_state["popup_index"] = 0
            view_state["popup_queue"] = [{"stakeholder_id": stakeholder_id}]
        return (view_state, saved_runs) + _render_all_outputs(view_state, saved_runs)

    def handle_open_level(
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
        updated = _run_reset(task, int(seed), level, level, view_state)
        if level not in updated.get("unlocked_levels", []):
            updated["unlocked_levels"] = updated.get("unlocked_levels", []) + [level]
        return (updated, saved_runs) + _render_all_outputs(updated, saved_runs)

    def handle_unlock_level(
        level: str, view_state: Dict[str, Any], saved_runs: List[Dict[str, Any]]
    ) -> Tuple[Any, ...]:
        view_state = _normalize_view_state(view_state)
        saved_runs = _normalize_saved_runs(saved_runs)
        if level not in view_state.get("unlocked_levels", []):
            view_state["unlocked_levels"] = view_state.get("unlocked_levels", []) + [
                level
            ]
        return (view_state, saved_runs) + _render_all_outputs(view_state, saved_runs)

    demo = gr.Blocks(elem_classes=["dealroom-custom"])
    with demo:
        gr.HTML(f"<style>{CUSTOM_CSS}</style>")

        view_state = gr.State(default_view_state())
        saved_runs = gr.State([])

        with gr.Column():
            gr.HTML(
                "<div style='text-align:center;padding:16px 0 8px;'>"
                "<h1 style='margin:0;color:#f3f4f6;font-size:1.5rem;'>🎯 DealRoom Lab</h1>"
                "<p style='margin:6px 0 0;color:#9ca3af;font-size:0.9rem;'>Negotiate with stakeholders • Close the deal</p>"
                "</div>"
            )

            progress_html = gr.HTML()

            with gr.Row():
                with gr.Column(scale=7):
                    table_html = gr.HTML()
                    popup_html = gr.HTML()

                    gr.HTML("<div class='action-bar'>")
                    gr.HTML("<h3>🎮 Actions</h3>")

                    with gr.Row(elem_classes=["chat-input-row"]):
                        message_input = gr.Textbox(
                            placeholder="Type your message...",
                            lines=2,
                            elem_classes=["quick-message"],
                        )
                        send_btn = gr.Button(
                            "📤 Send", elem_classes=["send-btn"], variant="primary"
                        )

                    gr.HTML("<div class='divider-line'></div>")

                    with gr.Row():
                        run_btn = gr.Button(
                            "▶ Run Round", elem_classes=["run-btn"], variant="primary"
                        )
                        step_btn = gr.Button("⏭ Step", elem_classes=["auto-btn"])
                        auto_toggle_btn = gr.Button("⏵ Auto", elem_classes=["auto-btn"])

                    gr.HTML("</div>")

                with gr.Column(scale=3):
                    score_html = gr.HTML()
                    signals_html = gr.HTML()
                    why_html = gr.HTML()

        outputs = [
            view_state,
            saved_runs,
            table_html,
            popup_html,
            score_html,
            signals_html,
            why_html,
        ]

        def render_initial(vs, sr):
            vs = _normalize_view_state(vs)
            sr = _normalize_saved_runs(sr)
            return (vs, sr) + _render_all_outputs(vs, sr)

        def render_progress(vs, sr):
            vs = _normalize_view_state(vs)
            sr = _normalize_saved_runs(sr)
            unlocked = vs.get("unlocked_levels", ["simple"])
            current = vs.get("level", "simple")
            steps_html = ""
            for level in ["simple", "medium", "hard"]:
                is_active = level == current
                is_unlocked = level in unlocked
                cls = "active" if is_active else ("locked" if not is_unlocked else "")
                icon = "🔓" if is_unlocked else "🔒"
                label = LEVEL_LABELS.get(level, level)
                hint = {
                    "simple": "Start here",
                    "medium": "Complete Simple",
                    "hard": "Complete Medium",
                }.get(level, "")
                hint_html = (
                    f'<span class="unlock-hint">{hint}</span>'
                    if hint and not is_unlocked
                    else ""
                )
                steps_html += (
                    f"<div class='progress-step {cls}' data-level='{level}'>"
                    f"{icon} {label}"
                    f"{hint_html}"
                    f"</div>"
                )
            return f"<div class='progress-strip'>{steps_html}</div>"

        demo.load(fn=render_initial, inputs=[view_state, saved_runs], outputs=outputs)

        def on_load_update(vs, sr):
            vs = _normalize_view_state(vs)
            sr = _normalize_saved_runs(sr)
            return (render_progress(vs, sr),) + tuple(_render_insights_panel(vs, sr))

        demo.load(
            fn=on_load_update,
            inputs=[view_state, saved_runs],
            outputs=[progress_html, score_html, signals_html, why_html],
        )

        simple_open_btn = gr.Button("Open Simple", visible=True)
        medium_open_btn = gr.Button("Open Medium", visible=True)
        hard_open_btn = gr.Button("Open Hard", visible=True)

        def update_buttons(vs, sr):
            vs = _normalize_view_state(vs)
            sr = _normalize_saved_runs(sr)
            unlocked = vs.get("unlocked_levels", ["simple"])
            return (
                gr.update(visible="simple" in unlocked),
                gr.update(visible="medium" in unlocked),
                gr.update(visible="hard" in unlocked),
            )

        demo.load(
            fn=update_buttons,
            inputs=[view_state, saved_runs],
            outputs=[simple_open_btn, medium_open_btn, hard_open_btn],
        )

        simple_open_btn.click(
            fn=handle_open_level,
            inputs=[gr.State("simple"), view_state, saved_runs],
            outputs=outputs,
        )
        medium_open_btn.click(
            fn=handle_open_level,
            inputs=[gr.State("medium"), view_state, saved_runs],
            outputs=outputs,
        )
        hard_open_btn.click(
            fn=handle_open_level,
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

        auto_toggle_btn.click(
            fn=handle_auto_advance_toggle,
            inputs=[view_state, saved_runs],
            outputs=outputs,
        )

        send_btn.click(
            fn=handle_send_message,
            inputs=[message_input, view_state, saved_runs],
            outputs=outputs,
        )

        for i in range(4):
            btn = gr.Button(f"Seat {i + 1}", visible=True)
            btn.click(
                fn=handle_focus_seat,
                inputs=[gr.State(i), view_state, saved_runs],
                outputs=outputs,
            )

    return demo
