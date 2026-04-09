"""DealRoom custom Gradio tab with a progressive round-table learning flow."""

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
SEAT_POSITIONS = [
    "top: 7%; left: 50%; transform: translateX(-50%);",
    "top: 36%; left: 8%;",
    "top: 36%; right: 8%;",
    "bottom: 7%; left: 50%; transform: translateX(-50%);",
]
LEVEL_EXPLANATIONS = {
    "simple": {
        "title": "Simple round-table discussion",
        "body": (
            "This round keeps the committee small and the hidden structure light. "
            "You can see the basic negotiation loop: hear concerns, share one relevant artifact, "
            "and watch approval move."
        ),
        "limits": (
            "This is still too easy for a realistic enterprise deal. There are fewer stakeholders, "
            "less internal politics, and only one hidden feasibility issue."
        ),
    },
    "medium": {
        "title": "Medium round-table with hidden friction",
        "body": (
            "Now the committee has competing incentives. New artifacts matter, blockers can persist "
            "across rounds, and the right move is often to clarify before proposing."
        ),
        "limits": (
            "This still abstracts the full enterprise mess. Relationship propagation is bounded and "
            "stakeholders remain deterministic rather than fully strategic planners."
        ),
    },
    "hard": {
        "title": "Hard round-table close to the real workflow",
        "body": (
            "This is the full lab. Multiple hard constraints, authority shifts, lower tolerance for bad sequencing, "
            "and a terminal grader that only rewards feasible closure."
        ),
        "limits": (
            "This is the realistic end of the environment we ship today. It is still a deterministic hybrid simulator, "
            "not a free-running multi-LLM society."
        ),
    },
}

CUSTOM_CSS = """
#dealroom-custom-root {
  background: #0d1117;
  border: 1px solid #243041;
  border-radius: 14px;
  padding: 18px;
}
#dealroom-custom-root .dealroom-shell,
#dealroom-custom-root .dealroom-shell * {
  font-family: "IBM Plex Sans", "Inter", system-ui, sans-serif;
}
#dealroom-custom-root .hero {
  background: linear-gradient(135deg, #0f141b 0%, #141b24 100%);
  border: 1px solid #243041;
  border-radius: 12px;
  padding: 18px 20px;
  margin-bottom: 16px;
}
#dealroom-custom-root .hero h1 {
  margin: 0 0 8px;
  color: #f3f4f6;
  font-size: 1.4rem;
}
#dealroom-custom-root .hero p {
  margin: 0;
  color: #9ba7b6;
  line-height: 1.45;
}
#dealroom-custom-root .proof-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 12px;
}
#dealroom-custom-root .chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 10px;
  border-radius: 999px;
  border: 1px solid #304154;
  background: #111822;
  color: #d8e1ea;
  font-size: 0.8rem;
}
#dealroom-custom-root .chip--green {
  background: rgba(22, 163, 74, 0.12);
  border-color: rgba(34, 197, 94, 0.35);
  color: #86efac;
}
#dealroom-custom-root .chip--amber {
  background: rgba(245, 158, 11, 0.12);
  border-color: rgba(245, 158, 11, 0.35);
  color: #fcd34d;
}
#dealroom-custom-root .chip--red {
  background: rgba(239, 68, 68, 0.12);
  border-color: rgba(239, 68, 68, 0.35);
  color: #fca5a5;
}
#dealroom-custom-root .step-strip {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
  margin-top: 14px;
}
#dealroom-custom-root .step-card {
  background: #10161d;
  border: 1px solid #283240;
  border-radius: 10px;
  padding: 12px;
}
#dealroom-custom-root .step-card strong {
  display: block;
  color: #f3f4f6;
  margin-bottom: 4px;
}
#dealroom-custom-root .step-card p {
  margin: 0;
  color: #9ba7b6;
  font-size: 0.85rem;
}
#dealroom-custom-root .status-panel,
#dealroom-custom-root .panel {
  background: #10161d;
  border: 1px solid #283240;
  border-radius: 10px;
  padding: 14px;
}
#dealroom-custom-root .status-panel h3,
#dealroom-custom-root .panel h3,
#dealroom-custom-root .panel h4 {
  margin-top: 0;
  color: #f3f4f6;
}
#dealroom-custom-root .panel + .panel {
  margin-top: 12px;
}
#dealroom-custom-root .soft {
  color: #9ba7b6;
}
#dealroom-custom-root .round-area {
  background: radial-gradient(circle at center, #192233 0%, #0d1117 68%);
  border: 1px solid #243041;
  border-radius: 14px;
  padding: 20px;
  min-height: 420px;
}
#dealroom-custom-root .round-table {
  position: relative;
  width: min(100%, 420px);
  height: 320px;
  margin: 0 auto 14px;
  border-radius: 50%;
  background: radial-gradient(circle at center, #111827 0%, #0b1118 70%);
  border: 2px solid #273244;
  box-shadow: inset 0 0 50px rgba(0,0,0,0.45);
}
#dealroom-custom-root .round-center {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  text-align: center;
  width: 180px;
  padding: 14px;
  border-radius: 12px;
  background: rgba(13, 17, 23, 0.86);
  border: 1px solid #304154;
}
#dealroom-custom-root .round-center strong {
  display: block;
  color: #f3f4f6;
}
#dealroom-custom-root .round-center span {
  color: #9ba7b6;
  font-size: 0.86rem;
}
#dealroom-custom-root .seat {
  position: absolute;
  width: 86px;
  min-height: 82px;
  border-radius: 16px;
  padding: 10px 8px;
  text-align: center;
  background: #121a25;
  border: 1px solid #324254;
  box-shadow: 0 6px 18px rgba(0,0,0,0.35);
}
#dealroom-custom-root .seat.selected {
  border-color: #60a5fa;
  box-shadow: 0 0 0 2px rgba(96, 165, 250, 0.2);
}
#dealroom-custom-root .seat.supporter {
  border-color: rgba(34, 197, 94, 0.5);
}
#dealroom-custom-root .seat.blocker {
  border-color: rgba(239, 68, 68, 0.55);
}
#dealroom-custom-root .seat-icon {
  font-size: 1.35rem;
  line-height: 1;
}
#dealroom-custom-root .seat-name {
  color: #f3f4f6;
  font-size: 0.76rem;
  font-weight: 600;
  margin-top: 6px;
}
#dealroom-custom-root .seat-band {
  color: #9ba7b6;
  font-size: 0.72rem;
  margin-top: 3px;
}
#dealroom-custom-root .speaker-popup {
  background: #0f141b;
  border: 1px solid #304154;
  border-radius: 12px;
  padding: 16px;
}
#dealroom-custom-root .speaker-popup h3 {
  margin: 0 0 8px;
  color: #f3f4f6;
}
#dealroom-custom-root .quote {
  background: #0a1016;
  border-left: 3px solid #3b82f6;
  padding: 12px 14px;
  border-radius: 8px;
  color: #e5e7eb;
  margin: 10px 0 12px;
}
#dealroom-custom-root .speaker-grid,
#dealroom-custom-root .metrics-grid,
#dealroom-custom-root .timeline-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
  gap: 10px;
}
#dealroom-custom-root .metric,
#dealroom-custom-root .timeline-item {
  background: #0b1118;
  border: 1px solid #2b3746;
  border-radius: 8px;
  padding: 10px 12px;
}
#dealroom-custom-root .metric strong,
#dealroom-custom-root .timeline-item strong {
  display: block;
  color: #f3f4f6;
  margin-bottom: 4px;
  font-size: 0.9rem;
}
#dealroom-custom-root .metric p,
#dealroom-custom-root .timeline-item p {
  margin: 0;
  color: #9ba7b6;
  font-size: 0.84rem;
}
#dealroom-custom-root .confidence-track {
  width: 100%;
  height: 8px;
  background: #1b2430;
  border-radius: 999px;
  overflow: hidden;
  margin-top: 8px;
}
#dealroom-custom-root .confidence-fill {
  height: 100%;
  background: linear-gradient(90deg, #f59e0b, #22c55e);
}
#dealroom-custom-root .action-note {
  background: rgba(59, 130, 246, 0.08);
  border: 1px solid rgba(59, 130, 246, 0.2);
  color: #bfdbfe;
  border-radius: 8px;
  padding: 12px 14px;
}
#dealroom-custom-root .warning-box {
  background: rgba(245, 158, 11, 0.10);
  border: 1px solid rgba(245, 158, 11, 0.22);
  color: #fde68a;
  border-radius: 8px;
  padding: 12px 14px;
}
#dealroom-custom-root .gr-button {
  border-radius: 9px !important;
  box-shadow: none !important;
}
#dealroom-custom-root .gr-button-primary {
  background: #1f8b4c !important;
  border-color: #1f8b4c !important;
}
#dealroom-custom-root .seat-button-row {
  margin-top: 6px;
}
#dealroom-custom-root input,
#dealroom-custom-root textarea,
#dealroom-custom-root select {
  background: #0b1118 !important;
  color: #e5e7eb !important;
  border-color: #2b3746 !important;
}
"""


class DealRoomWebManager:
    """Shared manager that lets both Playground and Custom drive the same pool."""

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

    async def reset_environment(
        self, reset_kwargs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        reset_kwargs = reset_kwargs or {}
        self._playground_session_id, obs, state = self.pool.reset(
            task_id=reset_kwargs.get("task_id", "aligned"),
            seed=reset_kwargs.get("seed"),
            session_id=reset_kwargs.get("episode_id") or self._playground_session_id,
        )
        obs.metadata["session_id"] = self._playground_session_id
        obs_dict = obs.model_dump(exclude={"reward", "metadata"})
        return {
            "observation": obs_dict,
            "reward": obs.reward,
            "done": obs.done,
            "session_id": self._playground_session_id,
            "state": state.model_dump(),
        }

    async def step_environment(self, action_data: Dict[str, Any]) -> Dict[str, Any]:
        if not self._playground_session_id:
            raise RuntimeError("Reset the environment before stepping.")
        action = DealRoomAction.model_validate(action_data)
        obs, reward, done, _info, state = self.pool.step(
            self._playground_session_id, action
        )
        obs.reward = reward
        obs.done = done
        obs.metadata["session_id"] = self._playground_session_id
        obs_dict = obs.model_dump(exclude={"reward", "metadata"})
        return {
            "observation": obs_dict,
            "reward": reward,
            "done": done,
            "session_id": self._playground_session_id,
            "state": state.model_dump(),
        }

    def get_state(self) -> Dict[str, Any]:
        if not self._playground_session_id:
            return DealRoomState().model_dump()
        return self.pool.state(self._playground_session_id).model_dump()


def load_metadata() -> EnvironmentMetadata:
    readme_path = Path("README.md")
    readme = readme_path.read_text(encoding="utf-8") if readme_path.exists() else None
    return EnvironmentMetadata(
        name="deal-room",
        description=(
            "A realistic multi-stakeholder enterprise negotiation environment with "
            "hidden constraints, bounded political dynamics, and deterministic grading."
        ),
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
            "status_message": "Start with the simple round-table, then move to medium and hard.",
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
        if not isinstance(merged.get("current_observation"), dict):
            merged["current_observation"] = None
        if not isinstance(merged.get("current_state"), dict):
            merged["current_state"] = None
        if merged.get("level") not in LEVEL_LABELS:
            merged["level"] = base["level"]
        if not isinstance(merged.get("selected_stakeholder"), (str, type(None))):
            merged["selected_stakeholder"] = None
        if not isinstance(merged.get("guide_step"), int):
            merged["guide_step"] = 0
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
                "status_message": (
                    f"{LEVEL_LABELS[level]} round ready on {task} with seed {seed}. "
                    f"Focus a stakeholder seat to inspect what they are saying."
                ),
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
                "last_action_error": info.get("last_action_error"),
            }
        )
        updated["trace"] = trace
        updated["current_observation"] = obs.model_dump()
        updated["current_state"] = state
        _keep_valid_selected(updated)
        updated["status_message"] = (
            f"Step {trace[-1]['step']} processed. Reward {reward:.2f}. "
            f"{'The episode is complete.' if done else 'Continue the discussion.'}"
        )
        return updated

    def _approval_band(observation: Dict[str, Any], stakeholder_id: str) -> str:
        progress = observation.get("approval_path_progress", {}).get(stakeholder_id, {})
        return str(progress.get("band", "neutral"))

    def _band_chip(band: str) -> str:
        css = "chip"
        if band in {"supporter", "workable"}:
            css += " chip--green"
        elif band == "blocker":
            css += " chip--red"
        else:
            css += " chip--amber"
        return f"<span class='{css}'>{_escape(band)}</span>"

    def _render_status_panel(view_state: Dict[str, Any]) -> str:
        view_state = _normalize_view_state(view_state)
        observation = view_state.get("current_observation") or {}
        level = view_state["level"]
        explanation = LEVEL_EXPLANATIONS[level]
        proof_chips = "".join(
            f"<span class='chip'>{_escape(text)}</span>"
            for text in [
                "dynamic stakeholders",
                "hidden constraints",
                "relationship propagation",
                "partial observability",
                "deterministic grading",
            ]
        )
        steps = []
        for current in ("simple", "medium", "hard"):
            klass = "step-card"
            if current == level:
                klass += " chip--green"
            steps.append(
                "<div class='step-card'>"
                f"<strong>{_escape(LEVEL_LABELS[current])}</strong>"
                f"<p>{_escape(LEVEL_EXPLANATIONS[current]['title'])}</p>"
                "</div>"
            )
        stage = observation.get("deal_stage", "not_started")
        blockers = ", ".join(observation.get("active_blockers", [])) or "none"
        known = ", ".join(item["id"] for item in observation.get("known_constraints", [])) or "none"
        return (
            "<div class='status-panel'>"
            f"<h3>{_escape(explanation['title'])}</h3>"
            f"<p class='soft'>{_escape(view_state.get('status_message', 'Ready.'))}</p>"
            f"<div class='proof-row'>{proof_chips}</div>"
            f"<div class='step-strip'>{''.join(steps)}</div>"
            "<div class='metrics-grid' style='margin-top:12px;'>"
            f"<div class='metric'><strong>Current level</strong><p>{_escape(LEVEL_LABELS[level])}</p></div>"
            f"<div class='metric'><strong>Stage</strong><p>{_escape(stage)}</p></div>"
            f"<div class='metric'><strong>Visible blockers</strong><p>{_escape(blockers)}</p></div>"
            f"<div class='metric'><strong>Known constraints</strong><p>{_escape(known)}</p></div>"
            "</div>"
            "</div>"
        )

    def _render_round_table(view_state: Dict[str, Any]) -> str:
        view_state = _normalize_view_state(view_state)
        observation = view_state.get("current_observation") or {}
        if not observation:
            return (
                "<div class='round-area'>"
                "<div class='round-table'>"
                "<div class='round-center'><strong>Round table not started</strong><span>Run one of the three levels to begin.</span></div>"
                "</div></div>"
            )
        selected = view_state.get("selected_stakeholder")
        seats = []
        for index, (stakeholder_id, payload) in enumerate(observation.get("stakeholders", {}).items()):
            if index >= len(SEAT_POSITIONS):
                break
            band = _approval_band(observation, stakeholder_id)
            selected_css = " selected" if stakeholder_id == selected else ""
            seats.append(
                f"<div class='seat {band}{selected_css}' style='{SEAT_POSITIONS[index]}'>"
                f"<div class='seat-icon'>{ROLE_ICONS.get(payload.get('role', ''), '👤')}</div>"
                f"<div class='seat-name'>{_escape(payload.get('display_name', stakeholder_id))}</div>"
                f"<div class='seat-band'>{_escape(band)}</div>"
                "</div>"
            )
        level = view_state["level"]
        center_text = {
            "simple": "Basic grounds",
            "medium": "Hidden friction",
            "hard": "Realistic close pressure",
        }[level]
        return (
            "<div class='round-area'>"
            "<div class='round-table'>"
            + "".join(seats)
            + (
                "<div class='round-center'>"
                f"<strong>{_escape(TASK_DISPLAY.get(view_state['task'], view_state['task']))}</strong>"
                f"<span>{_escape(center_text)}</span>"
                "</div>"
            )
            + "</div></div>"
        )

    def _render_popup(view_state: Dict[str, Any]) -> str:
        view_state = _normalize_view_state(view_state)
        observation = view_state.get("current_observation") or {}
        state = view_state.get("current_state") or {}
        selected = view_state.get("selected_stakeholder")
        if not observation or not selected or selected not in observation.get("stakeholders", {}):
            return (
                "<div class='speaker-popup'>"
                "<h3>Stakeholder popup</h3>"
                "<p class='soft'>Pick a seat below the round table to inspect that stakeholder's message, current asks, and approval state.</p>"
                "</div>"
            )
        payload = observation["stakeholders"][selected]
        progress = observation.get("approval_path_progress", {}).get(selected, {})
        message = observation.get("stakeholder_messages", {}).get(selected, "No direct message yet.")
        requested = observation.get("requested_artifacts", {}).get(selected, [])
        weak_signals = observation.get("weak_signals", {}).get(selected, [])
        private = state.get("stakeholder_private", {}).get(selected, {})
        marks = private.get("permanent_marks", [])
        mandatory_badge = "<span class='chip chip--red'>mandatory</span>" if progress.get("mandatory") else ""
        veto_badge = "<span class='chip chip--red'>veto</span>" if payload.get("veto_power") else ""
        return (
            "<div class='speaker-popup'>"
            f"<h3>{ROLE_ICONS.get(payload.get('role', ''), '👤')} {_escape(payload.get('display_name', selected))}</h3>"
            f"<div class='proof-row'>{_band_chip(progress.get('band', 'neutral'))}"
            f"<span class='chip'>authority {float(payload.get('authority', 0.0)):.2f}</span>"
            f"{mandatory_badge}"
            f"{veto_badge}"
            "</div>"
            f"<div class='quote'>{_escape(message)}</div>"
            "<div class='speaker-grid'>"
            f"<div class='metric'><strong>Why they matter</strong><p>{_escape(payload.get('role', selected))}</p></div>"
            f"<div class='metric'><strong>Requested evidence</strong><p>{_escape(', '.join(requested) or 'none')}</p></div>"
            f"<div class='metric'><strong>Weak signals</strong><p>{_escape('; '.join(weak_signals) or 'none')}</p></div>"
            f"<div class='metric'><strong>Trust damage</strong><p>{_escape(', '.join(marks) or 'none')}</p></div>"
            "</div>"
            "</div>"
        )

    def _render_timeline(view_state: Dict[str, Any]) -> str:
        trace = _normalize_view_state(view_state).get("trace", [])
        if not trace:
            return "<div class='panel'><p class='soft'>No conversation yet.</p></div>"
        cards = []
        for entry in trace[-8:]:
            if entry["kind"] == "reset":
                cards.append(
                    "<div class='timeline-item'>"
                    f"<strong>Reset · {_escape(entry['task'])}</strong>"
                    f"<p>seed {entry['seed']} · stage {_escape(entry['stage'])} · blockers {_escape(', '.join(entry.get('blockers', [])) or 'none')}</p>"
                    "</div>"
                )
            else:
                action = entry.get("action", {})
                cards.append(
                    "<div class='timeline-item'>"
                    f"<strong>Step {entry['step']} · {_escape(action.get('action_type', 'action'))}</strong>"
                    f"<p>reward {float(entry.get('reward', 0.0)):.2f} · stage {_escape(entry.get('stage', 'n/a'))} · blockers {_escape(', '.join(entry.get('blockers', [])) or 'none')}</p>"
                    f"<p class='soft'>{_escape(action.get('message', '(no message)'))}</p>"
                    "</div>"
                )
        return "<div class='timeline-grid'>" + "".join(cards) + "</div>"

    def _constraint_confidence(observation: Dict[str, Any], state: Dict[str, Any]) -> str:
        cards = []
        known = {item["id"] for item in observation.get("known_constraints", [])}
        for constraint_id, payload in state.get("hidden_constraints", {}).items():
            if constraint_id in known or payload.get("status") == "known":
                confidence = 0.95
            elif payload.get("status") == "hinted":
                confidence = 0.65
            else:
                confidence = 0.28
            label = "high" if confidence >= 0.85 else ("medium" if confidence >= 0.55 else "low")
            cards.append(
                "<div class='metric'>"
                f"<strong>{_escape(payload.get('label', constraint_id))}</strong>"
                f"<p>{_escape(label)} confidence · status {_escape(payload.get('status', 'hidden'))}</p>"
                "<div class='confidence-track'>"
                f"<div class='confidence-fill' style='width:{confidence * 100:.0f}%'></div>"
                "</div>"
                "</div>"
            )
        return "".join(cards) or "<div class='metric'><strong>No constraints yet</strong><p>Reset a scenario to inspect constraint confidence.</p></div>"

    def _render_signals(view_state: Dict[str, Any]) -> str:
        view_state = _normalize_view_state(view_state)
        observation = view_state.get("current_observation") or {}
        state = view_state.get("current_state") or {}
        if not observation:
            return "<div class='panel'><p class='soft'>No signals yet.</p></div>"
        weak_cards = []
        for stakeholder_id, signals in observation.get("weak_signals", {}).items():
            weak_cards.append(
                "<div class='metric'>"
                f"<strong>{_escape(stakeholder_id)}</strong>"
                f"<p>{_escape('; '.join(signals) or 'none')}</p>"
                "</div>"
            )
        requested_cards = []
        for stakeholder_id, artifacts in observation.get("requested_artifacts", {}).items():
            requested_cards.append(
                "<div class='metric'>"
                f"<strong>{_escape(stakeholder_id)}</strong>"
                f"<p>{_escape(', '.join(artifacts) or 'none')}</p>"
                "</div>"
            )
        weak_html = "".join(weak_cards) or "<div class='metric'><strong>Weak signals</strong><p>none</p></div>"
        requested_html = "".join(requested_cards) or "<div class='metric'><strong>Requested artifacts</strong><p>none</p></div>"
        return (
            "<div class='panel'>"
            "<h3>Signals and evidence</h3>"
            "<div class='speaker-grid'>"
            f"{weak_html}"
            f"{requested_html}"
            f"{_constraint_confidence(observation, state)}"
            "</div>"
            "</div>"
        )

    def _render_judge_lens(view_state: Dict[str, Any]) -> str:
        view_state = _normalize_view_state(view_state)
        observation = view_state.get("current_observation") or {}
        state = view_state.get("current_state") or {}
        if not observation:
            return "<div class='panel'><p class='soft'>No judge lens data yet.</p></div>"
        latest = None
        for entry in reversed(view_state.get("trace", [])):
            if entry.get("kind") == "step":
                latest = entry
                break
        breakdown = latest.get("dense_reward_breakdown", {}) if latest else {}
        reward_rows = "".join(
            f"<div class='metric'><strong>{_escape(key)}</strong><p>+{float(value):.2f}</p></div>"
            for key, value in breakdown.items()
            if float(value) > 0
        ) or "<div class='metric'><strong>No milestone</strong><p>No dense-reward milestone on the latest turn.</p></div>"
        feasibility = state.get("feasibility_state", {})
        return (
            "<div class='panel'>"
            "<h3>Judge lens</h3>"
            "<div class='metrics-grid'>"
            f"<div class='metric'><strong>Visible blockers</strong><p>{_escape(', '.join(observation.get('active_blockers', [])) or 'none')}</p></div>"
            f"<div class='metric'><strong>Known constraints</strong><p>{_escape(', '.join(item['id'] for item in observation.get('known_constraints', [])) or 'none')}</p></div>"
            f"<div class='metric'><strong>Hidden constraints</strong><p>{_escape(', '.join(state.get('hidden_constraints', {}).keys()) or 'none')}</p></div>"
            f"<div class='metric'><strong>Feasibility</strong><p>{'ready' if feasibility.get('is_feasible') else 'not ready'} · {_escape(', '.join(feasibility.get('violations', [])) or 'no visible violations')}</p></div>"
            "</div>"
            "<h4>Latest reward breakdown</h4>"
            f"<div class='metrics-grid'>{reward_rows}</div>"
            "</div>"
        )

    def _render_debrief(view_state: Dict[str, Any], saved_runs: List[Dict[str, Any]]) -> str:
        view_state = _normalize_view_state(view_state)
        saved_runs = _normalize_saved_runs(saved_runs)
        observation = view_state.get("current_observation") or {}
        state = view_state.get("current_state") or {}
        if not observation.get("done"):
            return (
                "<div class='panel'>"
                "<h3>Debrief</h3>"
                "<p class='soft'>Finish a run to compare the terminal score and see how the round-table journey ended.</p>"
                "</div>"
            )
        score = CCIGrader.compute(DealRoomState.model_validate(state))
        recent_runs = saved_runs[-2:]
        comparison = ""
        if len(recent_runs) >= 2:
            comparison = (
                f"<p class='soft'>Recent diff: {recent_runs[-2]['score']:.2f} → {recent_runs[-1]['score']:.2f}</p>"
            )
        return (
            "<div class='panel'>"
            f"<h3>Final score {score:.2f}</h3>"
            f"{comparison}"
            f"<div class='metrics-grid'>"
            f"<div class='metric'><strong>Closure</strong><p>{'closed' if state.get('deal_closed') else 'failed'}</p></div>"
            f"<div class='metric'><strong>Reason</strong><p>{_escape(state.get('failure_reason') or 'successful close')}</p></div>"
            f"<div class='metric'><strong>Rounds used</strong><p>{state.get('round_number', 0)} / {state.get('max_rounds', 0)}</p></div>"
            f"<div class='metric'><strong>Remaining blockers</strong><p>{_escape(', '.join(state.get('active_blockers', [])) or 'none')}</p></div>"
            "</div>"
            "</div>"
        )

    def _render_level_notes(view_state: Dict[str, Any]) -> str:
        level = _normalize_view_state(view_state)["level"]
        details = LEVEL_EXPLANATIONS[level]
        return (
            f"### {details['title']}\n\n"
            f"{details['body']}\n\n"
            f"> Remaining gap: {details['limits']}"
        )

    def _save_run_if_complete(view_state: Dict[str, Any], saved_runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        view_state = _normalize_view_state(view_state)
        saved_runs = _normalize_saved_runs(saved_runs)
        observation = view_state.get("current_observation") or {}
        if not observation.get("done"):
            return saved_runs
        score = CCIGrader.compute(DealRoomState.model_validate(view_state.get("current_state") or {}))
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

    def _seat_updates(view_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        observation = _normalize_view_state(view_state).get("current_observation") or {}
        stakeholders = list(observation.get("stakeholders", {}).items())
        selected = _normalize_view_state(view_state).get("selected_stakeholder")
        updates: List[Dict[str, Any]] = []
        for index in range(4):
            if index < len(stakeholders):
                stakeholder_id, payload = stakeholders[index]
                updates.append(
                    gr.update(
                        visible=True,
                        value=f"{ROLE_ICONS.get(payload.get('role', ''), '👤')} {payload.get('display_name', stakeholder_id)}",
                        variant="primary" if stakeholder_id == selected else "secondary",
                    )
                )
            else:
                updates.append(gr.update(visible=False, value=f"Seat {index + 1}", variant="secondary"))
        return updates

    def _render_bundle(view_state: Dict[str, Any], saved_runs: List[Dict[str, Any]]) -> Tuple[Any, ...]:
        return (
            _render_status_panel(view_state),
            _render_round_table(view_state),
            _render_popup(view_state),
            _render_timeline(view_state),
            _render_signals(view_state),
            _render_judge_lens(view_state),
            _render_level_notes(view_state),
            _render_debrief(view_state, saved_runs),
            gr.update(choices=_target_choices(view_state), value="all"),
            *_seat_updates(view_state),
        )

    def _policy_action(observation: Dict[str, Any]) -> DealRoomAction:
        obs = _coerce_observation(observation)
        for stakeholder_id, artifacts in obs.requested_artifacts.items():
            if artifacts:
                artifact = artifacts[0]
                return DealRoomAction(
                    action_type="send_document",
                    target=stakeholder_id,
                    target_ids=[stakeholder_id],
                    message=f"Sharing the requested {artifact.replace('_', ' ')} so your team can review this cleanly.",
                    documents=[{"type": artifact, "specificity": "high"}],
                )
        if obs.active_blockers or not obs.known_constraints:
            target_id = next(iter(obs.active_blockers), next(iter(obs.stakeholders), "all"))
            return DealRoomAction(
                action_type="direct_message",
                target=target_id,
                target_ids=[target_id] if target_id != "all" else [],
                message="Help me understand the real approval or feasibility issue we still need to address.",
            )
        return DealRoomAction(
            action_type="group_proposal",
            target="all",
            target_ids=list(obs.stakeholders.keys()),
            message="I believe we have enough aligned evidence to move to final approval on concrete terms.",
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
        if not view_state.get("current_observation") or not view_state.get("session_id"):
            view_state = _run_reset(
                view_state["task"],
                int(view_state["seed"]),
                view_state["level"],
                source,
                view_state,
            )
        action = DealRoomAction.model_validate(payload)
        obs, reward, done, info, state = web_manager.step_session(view_state["session_id"], action)
        updated = _record_step(view_state, action, obs, reward, done, info, state.model_dump())
        updated["source"] = source
        saved_runs = _save_run_if_complete(updated, saved_runs)
        return (updated, saved_runs) + _render_bundle(updated, saved_runs)

    def _render_all_state(view_state: Dict[str, Any], saved_runs: List[Dict[str, Any]]) -> Tuple[Any, ...]:
        view_state = _normalize_view_state(view_state)
        saved_runs = _normalize_saved_runs(saved_runs)
        _keep_valid_selected(view_state)
        return (view_state, saved_runs) + _render_bundle(view_state, saved_runs)

    def start_simple(seed: int, view_state: Dict[str, Any], saved_runs: List[Dict[str, Any]]) -> Tuple[Any, ...]:
        updated = _run_reset("aligned", int(seed), "simple", "simple", view_state)
        return _render_all_state(updated, saved_runs)

    def advance_simple(view_state: Dict[str, Any], saved_runs: List[Dict[str, Any]]) -> Tuple[Any, ...]:
        view_state = _normalize_view_state(view_state)
        if view_state["level"] != "simple" or not view_state.get("current_observation"):
            view_state = _run_reset("aligned", int(view_state["seed"]), "simple", "simple", view_state)
        action = _policy_action(view_state["current_observation"])
        return _run_action(action.model_dump(), "simple", view_state, saved_runs)

    def start_medium(seed: int, view_state: Dict[str, Any], saved_runs: List[Dict[str, Any]]) -> Tuple[Any, ...]:
        updated = _run_reset(GUIDE_DATA["task"], int(seed), "medium", "guide", view_state)
        updated["guide_step"] = 0
        updated["status_message"] = "Medium round started. Use Next Guided Move to walk through the committee."
        return _render_all_state(updated, saved_runs)

    def next_medium(view_state: Dict[str, Any], saved_runs: List[Dict[str, Any]]) -> Tuple[Any, ...]:
        view_state = _normalize_view_state(view_state)
        if view_state["level"] != "medium" or not view_state.get("current_observation"):
            view_state = _run_reset(GUIDE_DATA["task"], GUIDE_DATA["seed"], "medium", "guide", view_state)
        step_index = min(view_state.get("guide_step", 0), len(GUIDE_DATA["steps"]) - 1)
        step = GUIDE_DATA["steps"][step_index]
        if step.get("action") and not view_state["current_observation"].get("done"):
            payload = DealRoomAction.model_validate(step["action"]).model_dump()
            result = _run_action(payload, "guide", view_state, saved_runs)
            updated = result[0]
            updated["guide_step"] = min(step_index + 1, len(GUIDE_DATA["steps"]) - 1)
            updated["status_message"] = f"Medium guided step {updated['guide_step']} completed: {step['title']}."
            saved = result[1]
            return _render_all_state(updated, saved)
        view_state["guide_step"] = min(step_index + 1, len(GUIDE_DATA["steps"]) - 1)
        view_state["status_message"] = f"Medium explanation step: {step['title']}."
        return _render_all_state(view_state, saved_runs)

    def take_over_medium(view_state: Dict[str, Any], saved_runs: List[Dict[str, Any]]) -> Tuple[Any, ...]:
        view_state = _normalize_view_state(view_state)
        view_state["level"] = "hard"
        view_state["source"] = "manual"
        view_state["status_message"] = "You are now driving the same committee manually in hard-lab mode."
        return _render_all_state(view_state, saved_runs)

    def open_hard(task: str, seed: int, view_state: Dict[str, Any], saved_runs: List[Dict[str, Any]]) -> Tuple[Any, ...]:
        updated = _run_reset(task, int(seed), "hard", "manual", view_state)
        updated["status_message"] = "Hard round ready. Use actions, baseline, or bad-move demos."
        return _render_all_state(updated, saved_runs)

    def refresh_hard(view_state: Dict[str, Any], saved_runs: List[Dict[str, Any]]) -> Tuple[Any, ...]:
        view_state = _normalize_view_state(view_state)
        session_id = view_state.get("session_id")
        if session_id and web_manager.pool.has_session(session_id):
            view_state["current_state"] = web_manager.get_state_for_session(session_id)
            view_state["status_message"] = "Refreshed the state from the active round."
        return _render_all_state(view_state, saved_runs)

    def focus_seat(index: int, view_state: Dict[str, Any], saved_runs: List[Dict[str, Any]]) -> Tuple[Any, ...]:
        view_state = _normalize_view_state(view_state)
        stakeholders = list((view_state.get("current_observation") or {}).get("stakeholders", {}).keys())
        if index < len(stakeholders):
            view_state["selected_stakeholder"] = stakeholders[index]
            view_state["status_message"] = f"Focused {stakeholders[index]}."
        return _render_all_state(view_state, saved_runs)

    def suggest_action(view_state: Dict[str, Any]) -> str:
        view_state = _normalize_view_state(view_state)
        if not view_state.get("current_observation"):
            return "Open a round first to generate a suggested move."
        if view_state["current_observation"].get("done"):
            return "The round is already complete."
        action = _policy_action(view_state["current_observation"])
        return (
            f"**Suggested next action:** `{action.action_type}` to "
            f"`{', '.join(action.target_ids) or action.target}`\n\n{action.message}"
        )

    def step_baseline(view_state: Dict[str, Any], saved_runs: List[Dict[str, Any]]) -> Tuple[Any, ...]:
        view_state = _normalize_view_state(view_state)
        if not view_state.get("current_observation"):
            task = view_state.get("task") or LEVEL_TO_TASK.get(view_state["level"], "hostile_acquisition")
            view_state = _run_reset(task, int(view_state["seed"]), view_state["level"], "baseline", view_state)
        action = _policy_action(view_state["current_observation"])
        return _run_action(action.model_dump(), "baseline", view_state, saved_runs)

    def run_baseline(task: str, seed: int, view_state: Dict[str, Any], saved_runs: List[Dict[str, Any]]) -> Tuple[Any, ...]:
        working = _run_reset(task, int(seed), "hard", "baseline", view_state)
        step_guard = 0
        while (
            working.get("current_observation")
            and not working["current_observation"].get("done")
            and step_guard < int(working["current_observation"].get("max_rounds", 10)) + 2
        ):
            action = _policy_action(working["current_observation"])
            obs, reward, done, info, state = web_manager.step_session(working["session_id"], action)
            working = _record_step(working, action, obs, reward, done, info, state.model_dump())
            working["source"] = "baseline"
            step_guard += 1
            if done:
                break
        saved_runs = _save_run_if_complete(working, saved_runs)
        return _render_all_state(working, saved_runs)

    def submit_quick_action(
        task: str,
        seed: int,
        action_type: str,
        target: str,
        message: str,
        document_type: str,
        price: float,
        timeline_weeks: float,
        support_level: str,
        liability_cap: str,
        view_state: Dict[str, Any],
        saved_runs: List[Dict[str, Any]],
    ) -> Tuple[Any, ...]:
        view_state = _normalize_view_state(view_state)
        if (
            not view_state.get("current_observation")
            or view_state.get("task") != task
            or int(view_state.get("seed", 0)) != int(seed)
        ):
            view_state = _run_reset(task, int(seed), "hard", "manual", view_state)
        payload: Dict[str, Any] = {
            "action_type": action_type,
            "target": target,
            "target_ids": [] if target == "all" else [target],
            "message": message,
        }
        if document_type and document_type != "none":
            payload["documents"] = [{"type": document_type, "specificity": "high"}]
        if action_type == "group_proposal":
            payload["proposed_terms"] = {
                "price": int(price) if price else 180000,
                "timeline_weeks": int(timeline_weeks) if timeline_weeks else 14,
                "support_level": support_level or "named_support_lead",
                "liability_cap": liability_cap or "mutual_cap",
            }
        return _run_action(payload, "manual", view_state, saved_runs)

    def submit_advanced_json(raw_action: str, view_state: Dict[str, Any], saved_runs: List[Dict[str, Any]]) -> Tuple[Any, ...]:
        view_state = _normalize_view_state(view_state)
        try:
            payload = json.loads(raw_action)
        except json.JSONDecodeError as exc:
            view_state["status_message"] = f"Invalid JSON action: {exc}"
            return _render_all_state(view_state, saved_runs)
        return _run_action(payload, "advanced_json", view_state, saved_runs)

    def load_bad_move(kind: str) -> str:
        if kind == "close":
            payload = {
                "action_type": "group_proposal",
                "target": "all",
                "target_ids": [],
                "message": "We should sign now and handle the remaining details later.",
            }
        elif kind == "ignore":
            payload = {
                "action_type": "direct_message",
                "target": "finance",
                "target_ids": ["finance"],
                "message": "Let us finalize commercials now and come back to legal after approval.",
            }
        else:
            payload = {
                "action_type": "send_document",
                "target": "finance",
                "target_ids": ["finance"],
                "message": "Sending something generic even though it was not requested.",
                "documents": [{"type": "support_plan", "specificity": "low"}],
            }
        return json.dumps(payload, indent=2)

    with gr.Blocks(elem_id="dealroom-custom-root") as demo:
        gr.HTML(f"<style>{CUSTOM_CSS}</style>")
        view_state = gr.State(default_view_state())
        saved_runs = gr.State([])

        gr.HTML(
            """
            <div class="dealroom-shell">
              <div class="hero">
                <h1>DealRoom Custom Lab</h1>
                <p>
                  This page teaches the environment as a round-table conference. Start with a basic discussion,
                  move into hidden-friction committee dynamics, and finish in the full enterprise lab.
                  The Playground stays untouched; this tab is the guided learning layer.
                </p>
                <div class="proof-row">
                  <span class="chip">round-table view</span>
                  <span class="chip">stakeholder popups</span>
                  <span class="chip">simple → medium → hard</span>
                  <span class="chip">judge lens</span>
                </div>
              </div>
            </div>
            """
        )

        status_html = gr.HTML()

        with gr.Row():
            with gr.Column(scale=4):
                with gr.Accordion("Step 1 · Simple Round", open=True):
                    gr.Markdown(
                        "Run a small aligned committee first. This is the basic conversation loop: listen, clarify, and answer with one concrete artifact."
                    )
                    simple_seed = gr.Number(value=42, precision=0, label="Simple seed", info="Keeps the simple committee reproducible.")
                    with gr.Row():
                        simple_run_btn = gr.Button("Run Simple Round", variant="primary")
                        simple_step_btn = gr.Button("Advance Simple Turn")
                    gr.Markdown(
                        "> Limitation: this is still a light version of enterprise negotiation. Fewer stakeholders and less internal politics make it easier than the real thing."
                    )

                with gr.Accordion("Step 2 · Medium Round", open=False):
                    gr.Markdown(
                        "Use the curated conflicted walkthrough to see hidden constraints, blocker sequencing, and why committee management matters."
                    )
                    medium_seed = gr.Number(value=GUIDE_DATA["seed"], precision=0, label="Medium seed", info="Seeded walkthrough for the conflicted scenario.")
                    with gr.Row():
                        medium_run_btn = gr.Button("Run Medium Round", variant="primary")
                        medium_next_btn = gr.Button("Next Guided Move")
                    medium_takeover_btn = gr.Button("Take Over Manually")
                    gr.Markdown(
                        f"**Walkthrough focus:** {GUIDE_DATA['summary']}\n\n"
                        "> Limitation: this adds real friction, but stakeholders are still bounded state-update agents rather than fully autonomous strategic planners."
                    )

                with gr.Accordion("Step 3 · Hard Round", open=False):
                    gr.Markdown(
                        "The hard lab is the closest to the full environment. Use manual actions, baseline actions, or deliberately bad moves to understand what the grader is rewarding."
                    )
                    hard_task = gr.Dropdown(
                        TASK_ORDER,
                        value="hostile_acquisition",
                        label="Hard task",
                        info="Choose the task you want to drive manually in the hard lab.",
                    )
                    hard_seed = gr.Number(value=42, precision=0, label="Hard seed", info="Use a fixed seed when you want clean comparisons.")
                    with gr.Row():
                        hard_open_btn = gr.Button("Open Hard Lab", variant="primary")
                        hard_refresh_btn = gr.Button("Refresh Current Round")

                    suggestion_md = gr.Markdown("Open a hard round to get a suggested next move.")
                    with gr.Row():
                        suggest_btn = gr.Button("Suggest Next Action")
                        baseline_step_btn = gr.Button("Baseline Step")
                        baseline_run_btn = gr.Button("Run Baseline Episode")

                    gr.Markdown("### Hard-lab action composer")
                    action_type = gr.Dropdown(
                        [
                            "direct_message",
                            "send_document",
                            "backchannel",
                            "group_proposal",
                            "concession",
                            "walkaway_signal",
                            "reframe_value_prop",
                            "exec_escalation",
                        ],
                        value="direct_message",
                        label="Action type",
                        info="The move you want the agent to make this turn.",
                    )
                    target_dropdown = gr.Dropdown(["all"], value="all", label="Target", info="Choose which stakeholder seat this move addresses.")
                    message_box = gr.Textbox(
                        value="I want to make sure we are addressing the real approval concern before we push this forward.",
                        lines=3,
                        label="Message",
                        info="This is the text the agent sends to the committee.",
                    )
                    document_type = gr.Dropdown(
                        ["none", "roi_model", "implementation_timeline", "security_cert", "dpa", "vendor_packet", "reference_case", "support_plan"],
                        value="none",
                        label="Document",
                        info="Attach a specific artifact when the move is document-driven.",
                    )
                    with gr.Row():
                        price_input = gr.Number(value=180000, precision=0, label="Price")
                        timeline_input = gr.Number(value=14, precision=0, label="Timeline weeks")
                    with gr.Row():
                        support_level = gr.Dropdown(
                            ["named_support_lead", "priority", "standard"],
                            value="named_support_lead",
                            label="Support level",
                        )
                        liability_cap = gr.Dropdown(
                            ["mutual_cap", "standard_cap", "custom_cap"],
                            value="mutual_cap",
                            label="Liability cap",
                        )
                    hard_send_btn = gr.Button("Send Hard-Lab Action", variant="primary")

                    gr.Markdown("### Counterfactual bad moves")
                    with gr.Row():
                        bad_close_btn = gr.Button("Close Too Early")
                        bad_ignore_btn = gr.Button("Ignore Legal")
                        bad_irrelevant_btn = gr.Button("Send Wrong Artifact")
                    advanced_json = gr.Textbox(
                        label="Advanced JSON action",
                        lines=10,
                        info="Use this when you want to submit a raw structured action payload.",
                    )
                    advanced_submit_btn = gr.Button("Submit Advanced JSON")

            with gr.Column(scale=6):
                gr.HTML("<div class='panel'><h3>Round table conference</h3><p class='soft'>Click a stakeholder seat to open their popup and inspect what they are saying in the current round.</p></div>")
                with gr.Row(elem_classes=["seat-button-row"]):
                    gr.Column(scale=2)
                    seat_btn_0 = gr.Button("Seat 1", visible=False)
                    gr.Column(scale=2)
                with gr.Row():
                    seat_btn_1 = gr.Button("Seat 2", visible=False)
                    table_html = gr.HTML()
                    seat_btn_2 = gr.Button("Seat 3", visible=False)
                with gr.Row(elem_classes=["seat-button-row"]):
                    gr.Column(scale=2)
                    seat_btn_3 = gr.Button("Seat 4", visible=False)
                    gr.Column(scale=2)

                popup_html = gr.HTML()
                level_notes = gr.Markdown()
                timeline_html = gr.HTML()
                signals_html = gr.HTML()
                judge_html = gr.HTML()
                debrief_html = gr.HTML()

        render_outputs = [
            status_html,
            table_html,
            popup_html,
            timeline_html,
            signals_html,
            judge_html,
            level_notes,
            debrief_html,
            target_dropdown,
            seat_btn_0,
            seat_btn_1,
            seat_btn_2,
            seat_btn_3,
        ]

        full_outputs = [view_state, saved_runs] + render_outputs

        demo.load(
            fn=lambda vs, sr: _render_all_state(vs, sr),
            inputs=[view_state, saved_runs],
            outputs=full_outputs,
        )

        simple_run_btn.click(
            fn=start_simple,
            inputs=[simple_seed, view_state, saved_runs],
            outputs=full_outputs,
        )
        simple_step_btn.click(
            fn=advance_simple,
            inputs=[view_state, saved_runs],
            outputs=full_outputs,
        )

        medium_run_btn.click(
            fn=start_medium,
            inputs=[medium_seed, view_state, saved_runs],
            outputs=full_outputs,
        )
        medium_next_btn.click(
            fn=next_medium,
            inputs=[view_state, saved_runs],
            outputs=full_outputs,
        )
        medium_takeover_btn.click(
            fn=take_over_medium,
            inputs=[view_state, saved_runs],
            outputs=full_outputs,
        )

        hard_open_btn.click(
            fn=open_hard,
            inputs=[hard_task, hard_seed, view_state, saved_runs],
            outputs=full_outputs,
        )
        hard_refresh_btn.click(
            fn=refresh_hard,
            inputs=[view_state, saved_runs],
            outputs=full_outputs,
        )
        suggest_btn.click(fn=suggest_action, inputs=[view_state], outputs=[suggestion_md])
        baseline_step_btn.click(
            fn=step_baseline,
            inputs=[view_state, saved_runs],
            outputs=full_outputs,
        )
        baseline_run_btn.click(
            fn=run_baseline,
            inputs=[hard_task, hard_seed, view_state, saved_runs],
            outputs=full_outputs,
        )
        hard_send_btn.click(
            fn=submit_quick_action,
            inputs=[
                hard_task,
                hard_seed,
                action_type,
                target_dropdown,
                message_box,
                document_type,
                price_input,
                timeline_input,
                support_level,
                liability_cap,
                view_state,
                saved_runs,
            ],
            outputs=full_outputs,
        )
        advanced_submit_btn.click(
            fn=submit_advanced_json,
            inputs=[advanced_json, view_state, saved_runs],
            outputs=full_outputs,
        )

        bad_close_btn.click(fn=lambda: load_bad_move("close"), outputs=[advanced_json])
        bad_ignore_btn.click(fn=lambda: load_bad_move("ignore"), outputs=[advanced_json])
        bad_irrelevant_btn.click(fn=lambda: load_bad_move("artifact"), outputs=[advanced_json])

        seat_btn_0.click(fn=lambda vs, sr: focus_seat(0, vs, sr), inputs=[view_state, saved_runs], outputs=full_outputs)
        seat_btn_1.click(fn=lambda vs, sr: focus_seat(1, vs, sr), inputs=[view_state, saved_runs], outputs=full_outputs)
        seat_btn_2.click(fn=lambda vs, sr: focus_seat(2, vs, sr), inputs=[view_state, saved_runs], outputs=full_outputs)
        seat_btn_3.click(fn=lambda vs, sr: focus_seat(3, vs, sr), inputs=[view_state, saved_runs], outputs=full_outputs)

    return demo
