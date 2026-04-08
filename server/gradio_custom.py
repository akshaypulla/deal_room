"""OpenEnv custom tab for the DealRoom judge experience."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
from openenv.core.env_server.types import EnvironmentMetadata

from models import DealRoomAction, DealRoomObservation
from server.deal_room_environment import DealRoomEnvironment
from server.grader import CCIGrader
from server.walkthrough_data import GUIDE_DATA

TASK_ORDER = ["aligned", "conflicted", "hostile_acquisition"]
TASK_DISPLAY = {
    "aligned": "Aligned",
    "conflicted": "Conflicted",
    "hostile_acquisition": "Hostile Acquisition",
}
TASK_DESCRIPTIONS = {
    "aligned": "Low-friction committee where sequencing and feasibility discipline still matter.",
    "conflicted": "Conflicting incentives, approval drag, and coalition-sensitive sequencing.",
    "hostile_acquisition": "Compressed timeline, authority shock, and harder recovery from mistakes.",
}
PROOF_CHIPS = [
    "dynamic stakeholders",
    "hidden constraints",
    "relationship propagation",
    "irreversible mistakes",
    "deterministic grading",
]
RL_CHIPS = [
    "partial observability",
    "delayed reward",
    "language-sensitive dynamics",
    "long-horizon planning",
]
STAGE_ORDER = ["evaluation", "negotiation", "legal_review", "final_approval", "closed"]
CUSTOM_CSS = """
.dealroom-custom { color: #102033; }
.dealroom-custom .hero {
  background: linear-gradient(135deg, rgba(11,34,58,.98), rgba(18,70,78,.95));
  color: #f6f2e9;
  border-radius: 22px;
  padding: 24px;
  margin-bottom: 16px;
  box-shadow: 0 18px 45px rgba(16,32,51,.18);
}
.dealroom-custom .hero h1 { margin: 0 0 10px; font-size: 2rem; line-height: 1.1; }
.dealroom-custom .hero p { margin: 0; color: rgba(246,242,233,.88); }
.dealroom-custom .chip-row, .dealroom-custom .badge-row {
  display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px;
}
.dealroom-custom .chip {
  display: inline-flex;
  padding: 6px 10px;
  border-radius: 999px;
  background: #eef3f7;
  color: #183149;
  border: 1px solid rgba(16,32,51,.08);
  font-size: 0.86rem;
}
.dealroom-custom .chip--teal { background: #d9f1ef; color: #0d5551; }
.dealroom-custom .chip--amber { background: #fff2d8; color: #8d5b00; }
.dealroom-custom .chip--risk { background: #ffe0d9; color: #9b3412; }
.dealroom-custom .panel {
  background: #fffdf9;
  border: 1px solid rgba(16,32,51,.08);
  border-radius: 18px;
  padding: 18px;
  box-shadow: 0 10px 30px rgba(16,32,51,.06);
}
.dealroom-custom .panel h3, .dealroom-custom .panel h4 {
  margin-top: 0;
  color: #102033;
}
.dealroom-custom .soft { color: #5c6c7d; }
.dealroom-custom .grid-2 {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 12px;
}
.dealroom-custom .grid-3 {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 12px;
}
.dealroom-custom .stakeholder-card,
.dealroom-custom .metric-card,
.dealroom-custom .timeline-card,
.dealroom-custom .score-card,
.dealroom-custom .diff-card {
  border-radius: 14px;
  padding: 14px;
  background: #f9f6ef;
  border: 1px solid rgba(16,32,51,.08);
}
.dealroom-custom .timeline-card { margin-bottom: 10px; }
.dealroom-custom .warning {
  margin-top: 12px;
  padding: 12px 14px;
  border-radius: 14px;
  background: #fff0eb;
  color: #8e2f19;
  border: 1px solid rgba(156,52,18,.14);
}
.dealroom-custom .stage-rail {
  display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0;
}
.dealroom-custom .stage-pill {
  padding: 6px 10px;
  border-radius: 999px;
  background: #eef3f7;
  color: #435465;
  font-size: 0.84rem;
}
.dealroom-custom .stage-pill.active {
  background: #113451;
  color: #fffdf9;
}
.dealroom-custom .confidence {
  margin-top: 10px;
}
.dealroom-custom .confidence__track {
  width: 100%;
  height: 10px;
  border-radius: 999px;
  background: #ece6da;
  overflow: hidden;
}
.dealroom-custom .confidence__fill {
  height: 100%;
  background: linear-gradient(90deg, #df8c29, #11806f);
}
.dealroom-custom .code-block {
  padding: 10px 12px;
  border-radius: 12px;
  background: #11283d;
  color: #f6f2e9;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  white-space: pre-wrap;
  word-break: break-word;
}
.dealroom-custom ul { margin: 0; padding-left: 18px; }
"""


class DealRoomWebManager:
    """Minimal manager that lets the stock OpenEnv playground drive our env."""

    def __init__(self, env: DealRoomEnvironment, metadata: EnvironmentMetadata):
        self.env = env
        self.metadata = metadata

    async def reset_environment(
        self, reset_kwargs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        reset_kwargs = reset_kwargs or {}
        obs = self.env.reset(
            seed=reset_kwargs.get("seed"),
            task_id=reset_kwargs.get("task_id", "aligned"),
            episode_id=reset_kwargs.get("episode_id"),
        )
        obs_dict = obs.model_dump(exclude={"reward", "metadata"})
        return {"observation": obs_dict, "reward": obs.reward, "done": obs.done}

    async def step_environment(self, action_data: Dict[str, Any]) -> Dict[str, Any]:
        action = DealRoomAction.model_validate(action_data)
        obs, reward, done, _info = self.env.step(action)
        obs.reward = reward
        obs.done = done
        obs_dict = obs.model_dump(exclude={"reward", "metadata"})
        return {"observation": obs_dict, "reward": reward, "done": done}

    def get_state(self) -> Dict[str, Any]:
        return self.env.state.model_dump()


def load_metadata() -> EnvironmentMetadata:
    readme_path = Path("README.md")
    readme = readme_path.read_text(encoding="utf-8") if readme_path.exists() else None
    return EnvironmentMetadata(
        name="deal-room",
        description=(
            "A realistic multi-stakeholder enterprise negotiation environment with "
            "hidden constraints, approval paths, and irreversible trust damage."
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
    del action_fields, is_chat_env, title, quick_start_md

    env = web_manager.env

    def default_view_state() -> Dict[str, Any]:
        return {
            "task": GUIDE_DATA["task"],
            "seed": GUIDE_DATA["seed"],
            "source": "custom",
            "guide_step": 0,
            "trace": [],
            "current_observation": None,
            "current_state": None,
            "status_message": "Choose a guided walkthrough, baseline demo, or live sandbox run.",
        }

    def default_saved_runs() -> List[Dict[str, Any]]:
        return []

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
        if not isinstance(merged.get("task"), str):
            merged["task"] = base["task"]
        if merged.get("seed") is None:
            merged["seed"] = base["seed"]
        if not isinstance(merged.get("source"), str):
            merged["source"] = base["source"]
        if not isinstance(merged.get("status_message"), str):
            merged["status_message"] = base["status_message"]
        if not isinstance(merged.get("guide_step"), int):
            merged["guide_step"] = base["guide_step"]
        if not isinstance(merged.get("current_observation"), dict):
            merged["current_observation"] = None
        if not isinstance(merged.get("current_state"), dict):
            merged["current_state"] = None
        return merged

    def _normalize_saved_runs(saved_runs: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        if not isinstance(saved_runs, list):
            return []
        return [item for item in saved_runs if isinstance(item, dict)]

    def _load_observation_dict(obs: DealRoomObservation) -> Dict[str, Any]:
        return obs.model_dump()

    def _coerce_observation(payload: Dict[str, Any]) -> DealRoomObservation:
        return DealRoomObservation.model_validate(payload)

    def _run_reset(
        task: str,
        seed: int,
        source: str,
        view_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        obs = env.reset(seed=int(seed), task_id=task)
        state = env.state.model_dump()
        new_state = _normalize_view_state(view_state)
        new_state.update(
            {
                "task": task,
                "seed": int(seed),
                "source": source,
                "guide_step": 0,
                "trace": [
                    {
                        "kind": "reset",
                        "task": task,
                        "seed": int(seed),
                        "stage": obs.deal_stage,
                        "blockers": list(obs.active_blockers),
                        "known_constraints": [item["id"] for item in obs.known_constraints],
                    }
                ],
                "current_observation": _load_observation_dict(obs),
                "current_state": state,
                "status_message": f"{TASK_DISPLAY.get(task, task)} seeded at {seed}.",
            }
        )
        return new_state

    def _record_step(
        view_state: Dict[str, Any],
        action: DealRoomAction,
        obs: DealRoomObservation,
        reward: float,
        done: bool,
        info: Dict[str, Any],
    ) -> Dict[str, Any]:
        trace = list(_normalize_view_state(view_state).get("trace", []))
        trace.append(
            {
                "kind": "step",
                "step": len([item for item in trace if item["kind"] == "step"]) + 1,
                "action": action.model_dump(),
                "reward": reward,
                "done": done,
                "stage": obs.deal_stage,
                "blockers": list(obs.active_blockers),
                "known_constraints": [item["id"] for item in obs.known_constraints],
                "dense_reward_breakdown": info.get("dense_reward_breakdown", {}),
                "relationship_effects": info.get("relationship_effects", []),
                "feasibility": info.get("feasibility", {}),
                "last_action_error": info.get("last_action_error"),
            }
        )
        updated = _normalize_view_state(view_state)
        updated["trace"] = trace
        updated["current_observation"] = obs.model_dump()
        updated["current_state"] = env.state.model_dump()
        updated["status_message"] = (
            f"Step {trace[-1]['step']} processed. Reward {reward:.2f}. "
            f"{'Episode complete.' if done else 'Ready for the next move.'}"
        )
        return updated

    def _approval_band_html(observation: Dict[str, Any], stakeholder_id: str) -> str:
        progress = observation.get("approval_path_progress", {}).get(stakeholder_id, {})
        band = progress.get("band", "unknown")
        cls = "chip chip--teal" if band in {"supporter", "workable"} else "chip chip--amber"
        if band == "blocker":
            cls = "chip chip--risk"
        return f"<span class='{cls}'>{_escape(band)}</span>"

    def _render_stage_rail(stage: str) -> str:
        items = []
        for item in STAGE_ORDER:
            klass = "stage-pill active" if item == stage else "stage-pill"
            items.append(f"<span class='{klass}'>{_escape(item)}</span>")
        return "<div class='stage-rail'>" + "".join(items) + "</div>"

    def _render_stakeholders(observation: Dict[str, Any]) -> str:
        cards = []
        for stakeholder_id, payload in observation.get("stakeholders", {}).items():
            badges = [
                f"<span class='chip'>{_escape(payload.get('role', stakeholder_id))}</span>",
                f"<span class='chip chip--amber'>authority {payload.get('authority', 0.0):.2f}</span>",
                _approval_band_html(observation, stakeholder_id),
            ]
            progress = observation.get("approval_path_progress", {}).get(stakeholder_id, {})
            if progress.get("mandatory"):
                badges.append("<span class='chip chip--risk'>mandatory</span>")
            if payload.get("veto_power"):
                badges.append("<span class='chip chip--risk'>veto</span>")
            cards.append(
                "<div class='stakeholder-card'>"
                f"<h4>{_escape(payload.get('display_name', stakeholder_id))}</h4>"
                f"<div class='badge-row'>{''.join(badges)}</div>"
                f"<p class='soft'>{_escape(observation.get('stakeholder_messages', {}).get(stakeholder_id, 'No visible message yet.'))}</p>"
                "</div>"
            )
        return "<div class='grid-2'>" + "".join(cards) + "</div>"

    def _render_timeline(trace: List[Dict[str, Any]]) -> str:
        if not trace:
            return "<div class='timeline-card'>No actions yet.</div>"
        cards = []
        for entry in trace:
            if entry["kind"] == "reset":
                cards.append(
                    "<div class='timeline-card'>"
                    f"<h4>Reset · {TASK_DISPLAY.get(entry['task'], entry['task'])} · seed {entry['seed']}</h4>"
                    f"<p class='soft'>Stage {_escape(entry['stage'])} · Blockers {_escape(', '.join(entry['blockers']) or 'none')}</p>"
                    "</div>"
                )
                continue
            action = entry["action"]
            targets = action.get("target_ids") or [action.get("target", "all")]
            doc_types = ", ".join(doc.get("type", "?") for doc in action.get("documents", []))
            details = action.get("message", "")
            if doc_types:
                details = f"{details}\n\nDocuments: {doc_types}".strip()
            if action.get("proposed_terms"):
                details = details + "\n\nTerms: " + json.dumps(action["proposed_terms"], sort_keys=True)
            cards.append(
                "<div class='timeline-card'>"
                f"<h4>Step {entry['step']} · {_escape(action.get('action_type', 'action'))}</h4>"
                f"<p class='soft'>Reward {entry['reward']:.2f} · Stage {_escape(entry['stage'])} · Targets {_escape(', '.join(targets))}</p>"
                f"<div class='code-block'>{_escape(details or '(no message)')}</div>"
                "</div>"
            )
        return "".join(cards)

    def _render_signals(observation: Dict[str, Any]) -> str:
        weak_rows = []
        for stakeholder_id, signals in observation.get("weak_signals", {}).items():
            weak_rows.append(
                "<div class='metric-card'>"
                f"<strong>{_escape(stakeholder_id)}</strong>"
                f"<p class='soft'>{_escape('; '.join(signals) or 'No weak signals')}</p>"
                "</div>"
            )
        requested_rows = []
        for stakeholder_id, artifacts in observation.get("requested_artifacts", {}).items():
            requested_rows.append(
                "<div class='metric-card'>"
                f"<strong>{_escape(stakeholder_id)}</strong>"
                f"<p class='soft'>{_escape(', '.join(artifacts) or 'No pending artifacts')}</p>"
                "</div>"
            )
        weak_html = "".join(weak_rows) or "<div class='metric-card'>No weak signals yet.</div>"
        requested_html = "".join(requested_rows) or "<div class='metric-card'>No requested artifacts right now.</div>"
        return (
            "<div class='grid-2'>"
            f"<div><h4>Weak signals</h4>{weak_html}</div>"
            f"<div><h4>Requested artifacts</h4>{requested_html}</div>"
            "</div>"
        )

    def _confidence_label(confidence: float) -> str:
        if confidence >= 0.85:
            return "high"
        if confidence >= 0.55:
            return "medium"
        return "low"

    def _render_constraint_confidence(observation: Dict[str, Any], state: Dict[str, Any]) -> str:
        hidden_constraints = state.get("hidden_constraints", {})
        known = {item["id"] for item in observation.get("known_constraints", [])}
        cards = []
        for constraint_id, constraint in hidden_constraints.items():
            status = constraint.get("status", "hidden")
            hinted = any(
                constraint_id in signal
                for signals in observation.get("weak_signals", {}).values()
                for signal in signals
            )
            if constraint_id in known or status == "known":
                confidence = 0.95
            elif hinted:
                confidence = 0.68
            elif constraint.get("required_artifact") and any(
                constraint.get("required_artifact") in artifacts
                for artifacts in observation.get("requested_artifacts", {}).values()
            ):
                confidence = 0.55
            else:
                confidence = 0.28
            label = _confidence_label(confidence)
            cards.append(
                "<div class='metric-card confidence'>"
                f"<strong>{_escape(constraint.get('label', constraint_id))}</strong>"
                f"<p class='soft'>status {_escape(status)} · confidence {_escape(label)}</p>"
                "<div class='confidence__track'>"
                f"<div class='confidence__fill' style='width:{confidence * 100:.0f}%'></div>"
                "</div>"
                "</div>"
            )
        return "".join(cards) or "<div class='metric-card'>No constraint confidence signals yet.</div>"

    def _score_breakdown(state: Dict[str, Any]) -> Dict[str, float]:
        score_state = env.state
        return {
            "approval_completeness": round(
                CCIGrader._approval_completeness(
                    score_state,
                    [
                        stakeholder_id
                        for stakeholder_id, payload in score_state.stakeholder_private.items()
                        if payload.get("mandatory")
                    ],
                ),
                4,
            ),
            "constraint_satisfaction": round(CCIGrader._constraint_satisfaction(score_state), 4),
            "term_feasibility": round(CCIGrader._term_feasibility(score_state), 4),
            "relationship_durability": round(CCIGrader._relationship_durability(score_state), 4),
            "efficiency": round(CCIGrader._efficiency(score_state), 4),
            "total": round(CCIGrader.compute(score_state), 4),
        }

    def _render_score_breakdown(state: Dict[str, Any]) -> str:
        breakdown = _score_breakdown(state)
        cards = []
        for key in [
            "approval_completeness",
            "constraint_satisfaction",
            "term_feasibility",
            "relationship_durability",
            "efficiency",
        ]:
            value = breakdown[key]
            cards.append(
                "<div class='score-card'>"
                f"<strong>{_escape(key.replace('_', ' '))}</strong>"
                f"<p class='soft'>{value:.2f}</p>"
                "<div class='confidence__track'>"
                f"<div class='confidence__fill' style='width:{value * 100:.0f}%'></div>"
                "</div>"
                "</div>"
            )
        return (
            "<div class='panel'>"
            f"<h3>Final score {breakdown['total']:.2f}</h3>"
            f"<div class='grid-2'>{''.join(cards)}</div>"
            "</div>"
        )

    def _collect_mistakes(view_state: Dict[str, Any]) -> Dict[str, List[str]]:
        view_state = _normalize_view_state(view_state)
        mistakes = {
            "Execution mistakes": [],
            "Negotiation mistakes": [],
            "Trust damage": [],
        }
        trace = view_state.get("trace", [])
        state = view_state.get("current_state") or {}
        for entry in trace:
            if entry["kind"] != "step":
                continue
            error = entry.get("last_action_error")
            if error:
                mistakes["Execution mistakes"].append(
                    f"Step {entry['step']}: malformed action handled softly ({error})."
                )
            action = entry["action"]
            if action.get("action_type") in {"group_proposal", "exec_escalation"} and entry.get("blockers"):
                mistakes["Negotiation mistakes"].append(
                    f"Step {entry['step']}: proposal/escalation attempted while blockers remained."
                )
        for stakeholder_id, payload in state.get("stakeholder_private", {}).items():
            for mark in payload.get("permanent_marks", []):
                mistakes["Trust damage"].append(
                    f"{stakeholder_id}: {mark.replace('_', ' ')}"
                )
        return {key: values for key, values in mistakes.items() if values}

    def _render_mistakes(view_state: Dict[str, Any]) -> str:
        mistakes = _collect_mistakes(view_state)
        if not mistakes:
            return "<div class='metric-card'>No tracked mistakes in this run.</div>"
        cards = []
        for category, values in mistakes.items():
            chips = "".join(f"<span class='chip chip--risk'>{_escape(value)}</span>" for value in values)
            cards.append(
                f"<div class='metric-card'><strong>{_escape(category)}</strong><div class='badge-row'>{chips}</div></div>"
            )
        return "".join(cards)

    def _render_judge_lens(view_state: Dict[str, Any]) -> str:
        view_state = _normalize_view_state(view_state)
        observation = view_state.get("current_observation") or {}
        state = view_state.get("current_state") or {}
        trace = view_state.get("trace", [])
        reward_breakdown = {}
        relationship_effects = []
        if trace and trace[-1]["kind"] == "step":
            reward_breakdown = trace[-1].get("dense_reward_breakdown", {})
            relationship_effects = trace[-1].get("relationship_effects", [])
        reward_html = "".join(
            (
                "<div class='metric-card'>"
                f"<strong>{_escape(key)}</strong><p class='soft'>+{float(value):.2f}</p>"
                "</div>"
            )
            for key, value in reward_breakdown.items()
            if float(value) > 0
        ) or "<div class='metric-card'>No dense reward milestones on the latest step.</div>"
        relationship_html = "".join(
            (
                "<div class='metric-card'>"
                f"<strong>{_escape(effect.get('type', 'effect'))}</strong>"
                f"<p class='soft'>{_escape(effect.get('source', '?'))} → {_escape(effect.get('target', '?'))} · "
                f"approval Δ {float(effect.get('approval_delta', 0.0)):.3f} · "
                f"resistance Δ {float(effect.get('resistance_delta', 0.0)):.3f}</p>"
                "</div>"
            )
            for effect in relationship_effects
        ) or "<div class='metric-card'>No relationship propagation on the latest step.</div>"
        feasibility = state.get("feasibility_state", {})
        return (
            "<div class='panel'>"
            "<h3>Judge lens</h3>"
            "<div class='grid-2'>"
            "<div class='metric-card'>"
            "<strong>Observable state</strong>"
            f"<p class='soft'>Stage {_escape(observation.get('deal_stage', 'n/a'))}<br>"
            f"Blockers {_escape(', '.join(observation.get('active_blockers', [])) or 'none')}<br>"
            f"Known constraints {_escape(', '.join(item['id'] for item in observation.get('known_constraints', [])) or 'none')}</p>"
            "</div>"
            "<div class='metric-card'>"
            "<strong>Hidden state view</strong>"
            f"<p class='soft'>Hidden constraints {_escape(', '.join(state.get('hidden_constraints', {}).keys()) or 'none')}<br>"
            f"Relationship edges {_escape(json.dumps(state.get('relationship_edges', [])))}<br>"
            f"Feasibility violations {_escape(', '.join(feasibility.get('violations', [])) or 'none')}</p>"
            "</div>"
            "</div>"
            "<h4>Constraint confidence</h4>"
            f"{_render_constraint_confidence(observation, state)}"
            "<h4>Reward breakdown</h4>"
            f"{reward_html}"
            "<h4>Relationship effects</h4>"
            f"{relationship_html}"
            "<h4>Mistake tracker</h4>"
            f"{_render_mistakes(view_state)}"
            "</div>"
        )

    def _render_overview(view_state: Dict[str, Any]) -> str:
        view_state = _normalize_view_state(view_state)
        proof_html = "".join(
            f"<span class='chip'>{_escape(item)}</span>" for item in PROOF_CHIPS
        )
        rl_html = "".join(
            f"<span class='chip chip--teal'>{_escape(item)}</span>" for item in RL_CHIPS
        )
        return (
            "<div class='hero'>"
            "<h1>A non-toy RL environment for enterprise negotiation.</h1>"
            "<p>"
            "DealRoom simulates the work humans actually do in software procurement: "
            "building committee consensus, discovering hidden blockers, sequencing evidence, "
            "and avoiding irreversible trust damage."
            "</p>"
            f"<div class='chip-row'>{proof_html}</div>"
            f"<div class='chip-row'>{rl_html}</div>"
            "</div>"
            "<div class='grid-2'>"
            "<div class='panel'>"
            "<h3>Why judges can trust this</h3>"
            "<ul>"
            "<li>Dynamic 2–4 stakeholder roster per episode.</li>"
            "<li>1–2 hidden hard constraints that must be inferred and resolved.</li>"
            "<li>Deterministic seeding and terminal grading.</li>"
            "<li>Dense reward helps learning, but cannot substitute for feasible closure.</li>"
            "</ul>"
            "</div>"
            "<div class='panel'>"
            "<h3>Current mode</h3>"
            f"<p class='soft'>Task {_escape(TASK_DISPLAY.get(view_state['task'], view_state['task']))} · seed {view_state['seed']} · source {_escape(view_state['source'])}</p>"
            f"<p class='soft'>{_escape(view_state.get('status_message', 'Ready.'))}</p>"
            "</div>"
            "</div>"
        )

    def _render_scenario_map(view_state: Dict[str, Any]) -> str:
        view_state = _normalize_view_state(view_state)
        observation = view_state.get("current_observation") or {}
        if not observation:
            return "<div class='panel'><p class='soft'>Reset a scenario to inspect the committee map.</p></div>"
        blockers = observation.get("active_blockers", [])
        return (
            "<div class='panel'>"
            f"<h3>{_escape(TASK_DISPLAY.get(view_state['task'], view_state['task']))} · seed {view_state['seed']}</h3>"
            f"<p class='soft'>{_escape(TASK_DESCRIPTIONS.get(view_state['task'], ''))}</p>"
            f"{_render_stage_rail(observation.get('deal_stage', 'evaluation'))}"
            "<div class='badge-row'>"
            f"<span class='chip chip--amber'>momentum {_escape(observation.get('deal_momentum', 'unknown'))}</span>"
            f"<span class='chip'>deadline {observation.get('days_to_deadline', 'n/a')} days</span>"
            f"<span class='chip chip--risk'>blockers {_escape(', '.join(blockers) or 'none')}</span>"
            "</div>"
            f"{_render_stakeholders(observation)}"
            "</div>"
        )

    def _render_guide(view_state: Dict[str, Any]) -> Tuple[str, str, str]:
        view_state = _normalize_view_state(view_state)
        step_index = min(view_state.get("guide_step", 0), len(GUIDE_DATA["steps"]) - 1)
        step = GUIDE_DATA["steps"][step_index]
        observation = view_state.get("current_observation") or {}
        stakeholder_html = (
            _render_stakeholders(observation)
            if observation
            else "<p class='soft'>Reset the walkthrough to begin.</p>"
        )
        guide_html = (
            "<div class='panel'>"
            f"<h3>Guided walkthrough · step {step_index + 1} / {len(GUIDE_DATA['steps'])}</h3>"
            f"<p class='soft'><strong>{_escape(step['title'])}</strong> · {_escape(step['concept'])}</p>"
            f"<p>{_escape(step['explanation'])}</p>"
            f"<div class='warning'>{_escape(step['counterfactual'])}</div>"
            "<h4>Judge highlights</h4>"
            f"<ul>{''.join(f'<li>{_escape(item)}</li>' for item in step['judge_highlights'])}</ul>"
            "</div>"
        )
        scene_html = (
            "<div class='panel'>"
            "<h3>Current walkthrough scene</h3>"
            f"<p class='soft'>Stage {_escape(observation.get('deal_stage', 'n/a'))} · "
            f"Blockers {_escape(', '.join(observation.get('active_blockers', [])) or 'none')} · "
            f"Known constraints {_escape(', '.join(item['id'] for item in observation.get('known_constraints', [])) or 'none')}</p>"
            f"{stakeholder_html}"
            "</div>"
        )
        next_action = step.get("action")
        if next_action:
            next_md = (
                f"### Next walkthrough action\n"
                f"`{next_action['action_type']}` targeting `{', '.join(next_action.get('target_ids') or [next_action.get('target', 'all')])}`\n\n"
                f"{next_action.get('message', '')}"
            )
        else:
            next_md = "### Walkthrough start\nReset the seeded conflicted scenario to begin the guided episode."
        return guide_html, scene_html, next_md

    def _build_counterfactual_warnings(observation: Dict[str, Any], state: Dict[str, Any]) -> str:
        warnings: List[str] = []
        blockers = observation.get("active_blockers", [])
        if blockers:
            warnings.append(
                f"Early closure is unsafe because blockers remain: {', '.join(blockers)}."
            )
        unresolved = [
            constraint_id
            for constraint_id, payload in state.get("hidden_constraints", {}).items()
            if not payload.get("resolved")
        ]
        if unresolved:
            warnings.append(
                f"Known or hidden feasibility constraints are still unresolved: {', '.join(unresolved)}."
            )
        requested = observation.get("requested_artifacts", {})
        for stakeholder_id, artifacts in requested.items():
            if artifacts:
                warnings.append(
                    f"{stakeholder_id} is still waiting for evidence ({artifacts[0]} first), so irrelevant artifacts are likely wasted turns."
                )
                break
        if not warnings:
            warnings.append("No obvious counterfactual warning right now. The next move should focus on efficient closure.")
        return "\n".join(f"- {item}" for item in warnings)

    def _pick_probe_target(obs: DealRoomObservation) -> str:
        weakest = None
        weakest_score = 10.0
        rank_map = {"blocker": 0, "neutral": 1, "workable": 2, "supporter": 3}
        for stakeholder_id, payload in obs.approval_path_progress.items():
            score = rank_map.get(payload.get("band", "neutral"), 1) - (0.2 if payload.get("mandatory") else 0.0)
            if score < weakest_score:
                weakest = stakeholder_id
                weakest_score = score
        return weakest or next(iter(obs.stakeholders))

    def _deterministic_policy_action(obs: DealRoomObservation) -> DealRoomAction:
        requested = obs.requested_artifacts
        for stakeholder_id, artifacts in requested.items():
            if artifacts:
                artifact = artifacts[0]
                fallback = {
                    "roi_model": "Here is the ROI model with explicit payback assumptions and downside cases.",
                    "reference_case": "Here is a reference case from a comparable deployment with measurable outcomes.",
                    "dpa": "Here is the DPA with GDPR-aligned privacy commitments and review-ready clauses.",
                    "security_cert": "Here are the requested security materials, audit artifacts, and control summaries.",
                    "vendor_packet": "Here is the supplier onboarding packet including process, insurance, and vendor details.",
                    "implementation_timeline": "Here is the implementation timeline with milestones, owners, and delivery guardrails.",
                    "support_plan": "Here is the support plan with named coverage, escalation paths, and ongoing ownership.",
                }.get(artifact, "Here is the requested material.")
                return DealRoomAction(
                    action_type="send_document",
                    target=stakeholder_id,
                    target_ids=[stakeholder_id],
                    message=fallback,
                    documents=[{"type": artifact, "specificity": "high"}],
                )

        if obs.active_blockers or any(obs.weak_signals.values()) or not obs.known_constraints:
            target_id = _pick_probe_target(obs)
            role = obs.stakeholders[target_id]["role"]
            prompt_by_role = {
                "finance": "Help me understand the budget ceiling or board payback requirement we need to respect so I can tailor the terms responsibly.",
                "technical": "What delivery window or implementation constraint is truly non-negotiable for your team?",
                "legal_compliance": "Which privacy or compliance obligation is the real approval blocker right now?",
                "procurement": "What supplier-process requirement do we still need to satisfy to move this forward cleanly?",
                "operations": "What rollout window or support commitment is still risky from your side?",
                "executive_sponsor": "What internal approval risk do we need to de-risk before this is safe to sponsor?",
            }
            return DealRoomAction(
                action_type="direct_message",
                target=target_id,
                target_ids=[target_id],
                message=prompt_by_role.get(
                    role,
                    "Help me understand the real approval constraint we still need to respect.",
                ),
            )

        return DealRoomAction(
            action_type="group_proposal",
            target="all",
            target_ids=list(obs.stakeholders.keys()),
            message="I believe we have enough alignment to move to final approval on concrete, reviewable terms.",
            proposed_terms={
                "price": 180000,
                "timeline_weeks": 14,
                "security_commitments": ["gdpr", "audit rights"],
                "support_level": "named_support_lead",
                "liability_cap": "mutual_cap",
            },
        )

    def _save_run_if_complete(
        view_state: Dict[str, Any], saved_runs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        view_state = _normalize_view_state(view_state)
        saved_runs = _normalize_saved_runs(saved_runs)
        observation = view_state.get("current_observation") or {}
        if not observation.get("done"):
            return saved_runs
        score = CCIGrader.compute(env.state)
        run = {
            "id": f"{view_state['task']}-{view_state['seed']}-{view_state['source']}-{len(saved_runs) + 1}",
            "task": view_state["task"],
            "seed": view_state["seed"],
            "source": view_state["source"],
            "score": score,
            "trace": list(view_state.get("trace", [])),
            "state": json.loads(json.dumps(view_state.get("current_state") or {})),
        }
        deduped = [item for item in saved_runs if item["id"] != run["id"]]
        deduped.append(run)
        return deduped[-8:]

    def _render_debrief(view_state: Dict[str, Any]) -> str:
        view_state = _normalize_view_state(view_state)
        observation = view_state.get("current_observation") or {}
        if not observation.get("done"):
            return (
                "<div class='panel'><h3>Debrief</h3>"
                "<p class='soft'>Complete an episode to view score breakdowns, mistakes, and constraint progression.</p>"
                "</div>"
            )
        journey_rows = []
        for entry in view_state.get("trace", []):
            if entry["kind"] == "reset":
                continue
            journey_rows.append(
                "<div class='timeline-card'>"
                f"<h4>Step {entry['step']}</h4>"
                f"<p class='soft'>Known constraints {_escape(', '.join(entry.get('known_constraints', [])) or 'none')} · "
                f"Blockers {_escape(', '.join(entry.get('blockers', [])) or 'none')}</p>"
                "</div>"
            )
        return (
            _render_score_breakdown(view_state.get("current_state") or {})
            + "<div class='panel'><h3>Constraint and blocker journey</h3>"
            + ("".join(journey_rows) or "<div class='timeline-card'>No journey captured.</div>")
            + "<h3>Tracked mistakes</h3>"
            + _render_mistakes(view_state)
            + "</div>"
        )

    def _render_run_label(run: Dict[str, Any]) -> str:
        return f"{TASK_DISPLAY.get(run['task'], run['task'])} | seed {run['seed']} | {run['source']} | score {run['score']:.2f}"

    def _render_diff(saved_runs: List[Dict[str, Any]], left_id: Optional[str], right_id: Optional[str]) -> str:
        if len(saved_runs) < 2:
            return "<div class='panel'><p class='soft'>Complete at least two runs to compare trajectories.</p></div>"
        left = next((item for item in saved_runs if item["id"] == left_id), saved_runs[-2])
        right = next((item for item in saved_runs if item["id"] == right_id), saved_runs[-1])
        divergence = "No divergence found."
        for left_entry, right_entry in zip(left["trace"], right["trace"]):
            if left_entry.get("action") != right_entry.get("action"):
                divergence = (
                    f"First divergence at step {left_entry.get('step', 0)}: "
                    f"{left_entry.get('action', {}).get('action_type', 'n/a')} vs "
                    f"{right_entry.get('action', {}).get('action_type', 'n/a')}"
                )
                break
        left_blockers = left["trace"][-1].get("blockers", []) if left["trace"] else []
        right_blockers = right["trace"][-1].get("blockers", []) if right["trace"] else []
        left_reward = sum(item.get("reward", 0.0) for item in left["trace"] if item["kind"] == "step")
        right_reward = sum(item.get("reward", 0.0) for item in right["trace"] if item["kind"] == "step")
        return (
            "<div class='panel'>"
            f"<h3>{_escape(_render_run_label(left))}</h3>"
            f"<p class='soft'>versus {_escape(_render_run_label(right))}</p>"
            f"<div class='diff-card'><strong>First divergence</strong><p class='soft'>{_escape(divergence)}</p></div>"
            f"<div class='diff-card'><strong>Final score delta</strong><p class='soft'>{left['score']:.2f} vs {right['score']:.2f} (Δ {(left['score'] - right['score']):+.2f})</p></div>"
            f"<div class='diff-card'><strong>Blocker delta</strong><p class='soft'>{_escape(', '.join(left_blockers) or 'none')} vs {_escape(', '.join(right_blockers) or 'none')}</p></div>"
            f"<div class='diff-card'><strong>Reward delta</strong><p class='soft'>{left_reward:.2f} vs {right_reward:.2f}</p></div>"
            "</div>"
        )

    def _target_choices(view_state: Dict[str, Any]) -> List[str]:
        view_state = _normalize_view_state(view_state)
        observation = view_state.get("current_observation") or {}
        roster = list(observation.get("stakeholders", {}).keys())
        return ["all"] + roster if roster else ["all"]

    def _render_all(
        view_state: Dict[str, Any], saved_runs: List[Dict[str, Any]]
    ) -> Tuple[Any, ...]:
        view_state = _normalize_view_state(view_state)
        saved_runs = _normalize_saved_runs(saved_runs)
        overview_html = _render_overview(view_state)
        guide_html, scene_html, next_action_md = _render_guide(view_state)
        scenario_map_html = _render_scenario_map(view_state)
        timeline_html = _render_timeline(view_state.get("trace", []))
        observation = view_state.get("current_observation") or {}
        state = view_state.get("current_state") or {}
        signals_html = _render_signals(observation) if observation else "<div class='panel'><p class='soft'>No observation yet.</p></div>"
        judge_lens_html = _render_judge_lens(view_state) if observation else "<div class='panel'><p class='soft'>No judge lens data yet.</p></div>"
        counterfactual_md = _build_counterfactual_warnings(observation, state) if observation else "Reset an episode to generate counterfactual warnings."
        suggestion = _deterministic_policy_action(_coerce_observation(observation)) if observation and not observation.get("done") else None
        if suggestion is None:
            suggestion_md = "No suggested next action right now."
        else:
            suggestion_md = (
                f"**Suggested next action:** `{suggestion.action_type}` targeting "
                f"`{', '.join(suggestion.target_ids) or suggestion.target}`\n\n"
                f"{suggestion.message}"
            )
        debrief_html = _render_debrief(view_state)
        run_choices = [(_render_run_label(run), run["id"]) for run in saved_runs]
        left_value = run_choices[-2][1] if len(run_choices) >= 2 else None
        right_value = run_choices[-1][1] if len(run_choices) >= 1 else None
        diff_html = _render_diff(saved_runs, left_value, right_value)
        target_update = gr.update(choices=_target_choices(view_state), value="all")
        return (
            view_state.get("status_message", "Ready."),
            overview_html,
            guide_html,
            scene_html,
            next_action_md,
            scenario_map_html,
            timeline_html,
            signals_html,
            judge_lens_html,
            counterfactual_md,
            suggestion_md,
            debrief_html,
            gr.update(choices=run_choices, value=left_value),
            gr.update(choices=run_choices, value=right_value),
            diff_html,
            target_update,
        )

    def start_walkthrough(view_state: Dict[str, Any], saved_runs: List[Dict[str, Any]]):
        saved_runs = _normalize_saved_runs(saved_runs)
        new_state = _run_reset(GUIDE_DATA["task"], GUIDE_DATA["seed"], "guide", view_state)
        return (new_state, saved_runs) + _render_all(new_state, saved_runs)

    def next_walkthrough_step(view_state: Dict[str, Any], saved_runs: List[Dict[str, Any]]):
        view_state = _normalize_view_state(view_state)
        saved_runs = _normalize_saved_runs(saved_runs)
        if not view_state.get("current_observation"):
            view_state = _run_reset(GUIDE_DATA["task"], GUIDE_DATA["seed"], "guide", view_state)
        step_index = min(view_state.get("guide_step", 0), len(GUIDE_DATA["steps"]) - 1)
        step = GUIDE_DATA["steps"][step_index]
        action_payload = step.get("action")
        if action_payload is not None and not view_state.get("current_observation", {}).get("done"):
            action = DealRoomAction.model_validate(action_payload)
            obs, reward, done, info = env.step(action)
            view_state = _record_step(view_state, action, obs, reward, done, info)
        view_state["guide_step"] = min(step_index + 1, len(GUIDE_DATA["steps"]) - 1)
        saved_runs = _save_run_if_complete(view_state, saved_runs)
        return (view_state, saved_runs) + _render_all(view_state, saved_runs)

    def take_over(view_state: Dict[str, Any], saved_runs: List[Dict[str, Any]]):
        saved_runs = _normalize_saved_runs(saved_runs)
        updated = _normalize_view_state(view_state)
        updated["source"] = "manual"
        updated["status_message"] = "Custom tab control is now manual. Use the live sandbox composer."
        return (updated, saved_runs) + _render_all(updated, saved_runs)

    def open_sandbox(task: str, seed: int, view_state: Dict[str, Any], saved_runs: List[Dict[str, Any]]):
        saved_runs = _normalize_saved_runs(saved_runs)
        new_state = _run_reset(task, seed, "manual", view_state)
        return (new_state, saved_runs) + _render_all(new_state, saved_runs)

    def refresh_from_env(view_state: Dict[str, Any], saved_runs: List[Dict[str, Any]]):
        saved_runs = _normalize_saved_runs(saved_runs)
        updated = _normalize_view_state(view_state)
        if env.state.episode_id:
            updated["current_state"] = env.state.model_dump()
            if updated.get("current_observation") is None:
                updated["status_message"] = "Environment exists, but no custom-tab observation has been captured yet."
            else:
                updated["status_message"] = "Synced judge lens from the current environment state."
        return (updated, saved_runs) + _render_all(updated, saved_runs)

    def apply_action_payload(
        payload: Dict[str, Any],
        source: str,
        view_state: Dict[str, Any],
        saved_runs: List[Dict[str, Any]],
    ):
        view_state = _normalize_view_state(view_state)
        saved_runs = _normalize_saved_runs(saved_runs)
        if not view_state.get("current_observation"):
            view_state = _run_reset(view_state["task"], view_state["seed"], source, view_state)
        action = DealRoomAction.model_validate(payload)
        obs, reward, done, info = env.step(action)
        updated = _record_step(view_state, action, obs, reward, done, info)
        updated["source"] = source
        saved_runs = _save_run_if_complete(updated, saved_runs)
        return (updated, saved_runs) + _render_all(updated, saved_runs)

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
    ):
        if not view_state.get("current_observation") or view_state["task"] != task or int(view_state["seed"]) != int(seed):
            view_state = _run_reset(task, seed, "manual", view_state)
        target_ids = [] if target == "all" else [target]
        payload: Dict[str, Any] = {
            "action_type": action_type,
            "target": target,
            "target_ids": target_ids,
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
        return apply_action_payload(payload, "manual", view_state, saved_runs)

    def submit_advanced_json(
        raw_action: str,
        view_state: Dict[str, Any],
        saved_runs: List[Dict[str, Any]],
    ):
        view_state = _normalize_view_state(view_state)
        saved_runs = _normalize_saved_runs(saved_runs)
        try:
            payload = json.loads(raw_action)
        except json.JSONDecodeError as exc:
            updated = _normalize_view_state(view_state)
            updated["status_message"] = f"Invalid JSON payload: {exc}"
            return (updated, saved_runs) + _render_all(updated, saved_runs)
        return apply_action_payload(payload, "advanced_json", view_state, saved_runs)

    def suggested_next_action(view_state: Dict[str, Any]):
        view_state = _normalize_view_state(view_state)
        observation = view_state.get("current_observation")
        if not observation or observation.get("done"):
            return "No suggestion available until a live episode is running."
        action = _deterministic_policy_action(_coerce_observation(observation))
        return (
            f"**Suggested next action:** `{action.action_type}` targeting "
            f"`{', '.join(action.target_ids) or action.target}`\n\n"
            f"{action.message}"
        )

    def step_agent_once(view_state: Dict[str, Any], saved_runs: List[Dict[str, Any]]):
        view_state = _normalize_view_state(view_state)
        saved_runs = _normalize_saved_runs(saved_runs)
        observation = view_state.get("current_observation")
        if not observation:
            view_state = _run_reset(view_state["task"], view_state["seed"], "baseline", view_state)
            observation = view_state["current_observation"]
        obs_model = _coerce_observation(observation)
        if obs_model.done:
            view_state["status_message"] = "Episode already complete."
            return (view_state, saved_runs) + _render_all(view_state, saved_runs)
        action = _deterministic_policy_action(obs_model)
        return apply_action_payload(action.model_dump(), "baseline", view_state, saved_runs)

    def run_agent_episode(task: str, seed: int, view_state: Dict[str, Any], saved_runs: List[Dict[str, Any]]):
        saved_runs = _normalize_saved_runs(saved_runs)
        working_state = _run_reset(task, seed, "baseline", view_state)
        steps = 0
        while not working_state["current_observation"]["done"] and steps < working_state["current_observation"]["max_rounds"] + 2:
            obs_model = _coerce_observation(working_state["current_observation"])
            action = _deterministic_policy_action(obs_model)
            obs, reward, done, info = env.step(action)
            working_state = _record_step(working_state, action, obs, reward, done, info)
            working_state["source"] = "baseline"
            steps += 1
            if done:
                break
        saved_runs = _save_run_if_complete(working_state, saved_runs)
        return (working_state, saved_runs) + _render_all(working_state, saved_runs)

    def load_bad_move(kind: str) -> str:
        if kind == "close":
            payload = {
                "action_type": "group_proposal",
                "target": "all",
                "target_ids": [],
                "message": "We should sign now and compress the remaining review.",
            }
        elif kind == "ignore_legal":
            payload = {
                "action_type": "direct_message",
                "target": "finance",
                "target_ids": ["finance"],
                "message": "Let us finalize commercials now and handle legal later.",
            }
        else:
            payload = {
                "action_type": "send_document",
                "target": "finance",
                "target_ids": ["finance"],
                "message": "Sharing something generic even though it was not requested.",
                "documents": [{"type": "support_plan", "specificity": "low"}],
            }
        return json.dumps(payload, indent=2)

    def update_diff(saved_runs: List[Dict[str, Any]], left_id: Optional[str], right_id: Optional[str]):
        return _render_diff(_normalize_saved_runs(saved_runs), left_id, right_id)

    with gr.Blocks() as demo:
        gr.HTML(f"<style>{CUSTOM_CSS}</style>")
        gr.Markdown(f"## Custom Judge View for `{metadata.name}`", elem_classes=["dealroom-custom"])

        view_state = gr.State(default_view_state())
        saved_runs = gr.BrowserState(default_saved_runs(), storage_key="dealroom_saved_runs")

        with gr.Row():
            start_guide_btn = gr.Button("Watch Guided Walkthrough", variant="primary")
            baseline_btn = gr.Button("Watch Baseline Agent")
            open_sandbox_btn = gr.Button("Open Live Sandbox")

        status_box = gr.Markdown("Ready.", elem_classes=["dealroom-custom"])

        with gr.Tabs():
            with gr.Tab("Overview"):
                overview_html = gr.HTML(elem_classes=["dealroom-custom"])

            with gr.Tab("Guided Walkthrough"):
                with gr.Row():
                    with gr.Column(scale=5):
                        guide_html = gr.HTML(elem_classes=["dealroom-custom"])
                        guide_scene_html = gr.HTML(elem_classes=["dealroom-custom"])
                    with gr.Column(scale=3):
                        guide_action_md = gr.Markdown()
                        with gr.Row():
                            guide_reset_btn = gr.Button("Reset Walkthrough")
                            guide_next_btn = gr.Button("Next Step", variant="primary")
                            guide_take_over_btn = gr.Button("Take Over")

            with gr.Tab("Live Sandbox"):
                with gr.Row():
                    task_dropdown = gr.Dropdown(TASK_ORDER, value="conflicted", label="Task")
                    seed_input = gr.Number(value=64, precision=0, label="Seed")
                    sandbox_reset_btn = gr.Button("Reset Scenario", variant="primary")
                    sandbox_refresh_btn = gr.Button("Refresh From Current Episode")
                with gr.Row():
                    with gr.Column(scale=3):
                        scenario_map_html = gr.HTML(elem_classes=["dealroom-custom"])
                    with gr.Column(scale=4):
                        timeline_html = gr.HTML(elem_classes=["dealroom-custom"])
                    with gr.Column(scale=3):
                        judge_lens_html = gr.HTML(elem_classes=["dealroom-custom"])

                with gr.Row():
                    signals_html = gr.HTML(elem_classes=["dealroom-custom"])

                with gr.Row():
                    with gr.Column(scale=4):
                        counterfactual_md = gr.Markdown(label="Counterfactual warnings")
                    with gr.Column(scale=4):
                        suggestion_md = gr.Markdown(label="Suggested next action")
                    with gr.Column(scale=2):
                        suggest_btn = gr.Button("Suggested Next Action")
                        baseline_step_btn = gr.Button("Step Agent Once")
                        baseline_run_btn = gr.Button("Watch Baseline Agent")

                with gr.Group():
                    gr.Markdown("### Quick Action")
                    with gr.Row():
                        action_type = gr.Dropdown(
                            [
                                "direct_message",
                                "backchannel",
                                "send_document",
                                "group_proposal",
                                "exec_escalation",
                            ],
                            value="direct_message",
                            label="Action type",
                        )
                        target_dropdown = gr.Dropdown(["all"], value="all", label="Target")
                        document_type = gr.Dropdown(
                            [
                                "none",
                                "roi_model",
                                "reference_case",
                                "dpa",
                                "security_cert",
                                "vendor_packet",
                                "implementation_timeline",
                                "support_plan",
                            ],
                            value="none",
                            label="Document",
                        )
                    message_box = gr.Textbox(
                        label="Message",
                        lines=4,
                        value="Help me understand the real approval constraint we still need to respect.",
                    )
                    with gr.Row():
                        price_input = gr.Number(value=180000, label="Price")
                        timeline_input = gr.Number(value=14, label="Timeline weeks")
                        support_level = gr.Textbox(value="named_support_lead", label="Support level")
                        liability_cap = gr.Textbox(value="mutual_cap", label="Liability cap")
                    with gr.Row():
                        quick_submit_btn = gr.Button("Send Action", variant="primary")
                        bad_close_btn = gr.Button("Close Too Early")
                        bad_ignore_legal_btn = gr.Button("Ignore Legal")
                        bad_wrong_artifact_btn = gr.Button("Send Wrong Artifact")

                with gr.Accordion("Advanced JSON", open=False):
                    advanced_json = gr.Code(language="json", value="{}", label="Raw action payload")
                    advanced_submit_btn = gr.Button("Submit JSON")

            with gr.Tab("Debrief & Replay"):
                debrief_html = gr.HTML(elem_classes=["dealroom-custom"])
                with gr.Row():
                    diff_left = gr.Dropdown([], label="Left run")
                    diff_right = gr.Dropdown([], label="Right run")
                    diff_refresh_btn = gr.Button("Compare Runs")
                diff_html = gr.HTML(elem_classes=["dealroom-custom"])

        shared_outputs = [
            view_state,
            saved_runs,
            status_box,
            overview_html,
            guide_html,
            guide_scene_html,
            guide_action_md,
            scenario_map_html,
            timeline_html,
            signals_html,
            judge_lens_html,
            counterfactual_md,
            suggestion_md,
            debrief_html,
            diff_left,
            diff_right,
            diff_html,
            target_dropdown,
        ]

        demo.load(
            fn=lambda vs, sr: (vs, sr) + _render_all(vs, sr),
            inputs=[view_state, saved_runs],
            outputs=shared_outputs,
        )
        start_guide_btn.click(
            fn=start_walkthrough,
            inputs=[view_state, saved_runs],
            outputs=shared_outputs,
        )
        guide_reset_btn.click(
            fn=start_walkthrough,
            inputs=[view_state, saved_runs],
            outputs=shared_outputs,
        )
        guide_next_btn.click(
            fn=next_walkthrough_step,
            inputs=[view_state, saved_runs],
            outputs=shared_outputs,
        )
        guide_take_over_btn.click(
            fn=take_over,
            inputs=[view_state, saved_runs],
            outputs=shared_outputs,
        )
        open_sandbox_btn.click(
            fn=open_sandbox,
            inputs=[task_dropdown, seed_input, view_state, saved_runs],
            outputs=shared_outputs,
        )
        sandbox_reset_btn.click(
            fn=open_sandbox,
            inputs=[task_dropdown, seed_input, view_state, saved_runs],
            outputs=shared_outputs,
        )
        sandbox_refresh_btn.click(
            fn=refresh_from_env,
            inputs=[view_state, saved_runs],
            outputs=shared_outputs,
        )
        baseline_btn.click(
            fn=run_agent_episode,
            inputs=[task_dropdown, seed_input, view_state, saved_runs],
            outputs=shared_outputs,
        )
        baseline_run_btn.click(
            fn=run_agent_episode,
            inputs=[task_dropdown, seed_input, view_state, saved_runs],
            outputs=shared_outputs,
        )
        baseline_step_btn.click(
            fn=step_agent_once,
            inputs=[view_state, saved_runs],
            outputs=shared_outputs,
        )
        quick_submit_btn.click(
            fn=submit_quick_action,
            inputs=[
                task_dropdown,
                seed_input,
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
            outputs=shared_outputs,
        )
        advanced_submit_btn.click(
            fn=submit_advanced_json,
            inputs=[advanced_json, view_state, saved_runs],
            outputs=shared_outputs,
        )
        suggest_btn.click(
            fn=suggested_next_action,
            inputs=[view_state],
            outputs=[suggestion_md],
        )
        bad_close_btn.click(fn=lambda: load_bad_move("close"), outputs=[advanced_json])
        bad_ignore_legal_btn.click(fn=lambda: load_bad_move("ignore_legal"), outputs=[advanced_json])
        bad_wrong_artifact_btn.click(fn=lambda: load_bad_move("wrong_artifact"), outputs=[advanced_json])
        diff_refresh_btn.click(
            fn=update_diff,
            inputs=[saved_runs, diff_left, diff_right],
            outputs=[diff_html],
        )

    return demo
