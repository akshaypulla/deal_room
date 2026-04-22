"""
DealRoomV3 environment for DealRoom v3 - OpenEnv wrapper with causal graph inference.
"""

import hashlib
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..committee.belief_tracker import bayesian_update
from ..committee.causal_graph import (
    BeliefDistribution,
    compute_engagement_level,
    sample_graph,
)
from ..committee.deliberation_engine import CommitteeDeliberationEngine
from ..environment.constants import (
    DEFAULT_MAX_ROUNDS,
    REWARD_WEIGHTS,
    SUMMARY_TIMEOUT_SECONDS,
    VETO_WARNING_THRESHOLD_RATIO,
)
from ..environment.lookahead import LookaheadSimulator
from models import DealRoomAction, DealRoomObservation, DealRoomState
from ..rewards.pareto_efficiency import compute_terminal_reward
from ..rewards.utterance_scorer import (
    LOOKAHEAD_COST,
    UtteranceScorer,
    compute_prediction_accuracy,
)
from ..stakeholders.archetypes import ARCHETYPE_PROFILES, get_archetype
from ..stakeholders.cvar_preferences import compute_cvar, compute_outcome_distribution


OBS_CONFIG = None  # Set in __init__


def _init_obs_config():
    from dataclasses import dataclass

    @dataclass
    class ObservationConfig:
        engagement_noise_sigma: float = 0.03
        echo_recall_probability: float = 0.70
        weak_signal_hard_threshold: float = 0.12
        weak_signal_soft_lower: float = 0.08
        weak_signal_soft_probability: float = 0.70
        reference_injection_threshold: float = 0.10
        minimax_base_reference_target: float = 0.60
        engagement_history_window: int = 5

    return ObservationConfig()


STANDARD_STAKEHOLDERS = [
    "Legal",
    "Finance",
    "TechLead",
    "Procurement",
    "Operations",
    "ExecSponsor",
]
STANDARD_HIERARCHY = {
    "Legal": 3,
    "Finance": 3,
    "TechLead": 2,
    "Procurement": 2,
    "Operations": 2,
    "ExecSponsor": 5,
}

INITIAL_BELIEFS = {
    "aligned": {
        "default": {
            "competent": 0.25,
            "incompetent": 0.10,
            "trustworthy": 0.25,
            "deceptive": 0.10,
            "aligned": 0.20,
            "misaligned": 0.10,
        }
    },
    "conflicted": {
        "cost_cluster": {
            "competent": 0.20,
            "incompetent": 0.15,
            "trustworthy": 0.20,
            "deceptive": 0.15,
            "aligned": 0.15,
            "misaligned": 0.15,
        },
        "risk_cluster": {
            "competent": 0.15,
            "incompetent": 0.20,
            "trustworthy": 0.15,
            "deceptive": 0.20,
            "aligned": 0.10,
            "misaligned": 0.20,
        },
        "impl_cluster": {
            "competent": 0.25,
            "incompetent": 0.10,
            "trustworthy": 0.25,
            "deceptive": 0.10,
            "aligned": 0.20,
            "misaligned": 0.10,
        },
    },
    "hostile_acquisition": {
        "default": {
            "competent": 0.12,
            "incompetent": 0.22,
            "trustworthy": 0.12,
            "deceptive": 0.22,
            "aligned": 0.10,
            "misaligned": 0.22,
        }
    },
}


def _get_initial_beliefs(task_id: str, stakeholder_id: str) -> Dict[str, float]:
    if task_id == "conflicted":
        if stakeholder_id in ["Finance", "Procurement"]:
            return dict(INITIAL_BELIEFS["conflicted"]["cost_cluster"])
        elif stakeholder_id in ["Legal", "Compliance"]:
            return dict(INITIAL_BELIEFS["conflicted"]["risk_cluster"])
        else:
            return dict(INITIAL_BELIEFS["conflicted"]["impl_cluster"])
    return dict(
        INITIAL_BELIEFS.get(task_id, {}).get(
            "default",
            {
                "competent": 0.17,
                "incompetent": 0.17,
                "trustworthy": 0.17,
                "deceptive": 0.17,
                "aligned": 0.16,
                "misaligned": 0.16,
            },
        )
    )


@dataclass
class StateSnapshot:
    beliefs: Dict[str, BeliefDistribution]
    active_blockers: List[str]
    risk_profiles: Dict[str, Any]
    authority_weights: Dict[str, float]
    current_terms: Optional[Any]
    round_number: int = 0
    deal_stage: str = "evaluation"
    deal_momentum: str = "progressing"

    def stable_hash(self) -> str:
        import hashlib, json

        state_repr = {
            "round": self.round_number,
            "deal_stage": self.deal_stage,
            "deal_momentum": self.deal_momentum,
            "active_blockers": sorted(self.active_blockers or []),
            "belief_fp": {
                sid: round(b.positive_mass(), 2)
                for sid, b in sorted((self.beliefs or {}).items())
            },
        }
        return hashlib.sha256(
            json.dumps(state_repr, sort_keys=True).encode()
        ).hexdigest()[:12]


@dataclass
class ScenarioConfig:
    task_id: str
    max_rounds: int = 10
    seed: Optional[int] = None


class DealRoomV3:
    def __init__(self, use_llm_stakeholders: bool = False):
        global OBS_CONFIG
        OBS_CONFIG = _init_obs_config()
        self._rng: Optional[np.random.Generator] = None
        self._scenario: Optional[ScenarioConfig] = None
        self._state: Optional[DealRoomState] = None
        self._graph = None
        self._beliefs: Dict[str, BeliefDistribution] = {}
        self._noisy_engagement: Dict[str, float] = {}
        self._engagement_history: Dict[str, List[float]] = {}
        self._utterance_scorer = UtteranceScorer()
        self._lookahead_simulator: Optional[LookaheadSimulator] = None
        self._step_count: int = 0
        self._round_number: int = 0
        self._episode_id: str = ""
        self._veto_precursor_streaks: Dict[str, int] = {}
        self._use_llm_stakeholders = use_llm_stakeholders

    @property
    def action_space(self) -> List[DealRoomAction]:
        """Safe reusable action templates for simple baselines and smoke tests."""
        return [
            DealRoomAction(
                action_type="direct_message",
                target="Finance",
                target_ids=["Finance"],
                message="I want to understand the business case concerns.",
            ),
            DealRoomAction(
                action_type="send_document",
                target="Legal",
                target_ids=["Legal"],
                message="Sharing DPA and compliance safeguards.",
                documents=[{"name": "DPA", "content": "GDPR-aligned DPA"}],
            ),
            DealRoomAction(
                action_type="send_document",
                target="Finance",
                target_ids=["Finance"],
                message="Sharing ROI model and downside assumptions.",
                documents=[{"name": "roi", "content": "ROI model"}],
            ),
            DealRoomAction(
                action_type="send_document",
                target="TechLead",
                target_ids=["TechLead"],
                message="Sharing implementation plan with delivery safeguards.",
                documents=[
                    {"name": "implementation_timeline", "content": "16-week plan"}
                ],
            ),
        ]

    def reset(
        self, seed: Optional[int] = None, task_id: str = "aligned", **kwargs
    ) -> DealRoomObservation:
        self._rng = np.random.default_rng(seed)
        self._scenario = ScenarioConfig(task_id=task_id, seed=seed)
        self._episode_id = str(uuid.uuid4())[:8]
        self._step_count = 0
        self._round_number = 0
        self._lookahead_simulator = LookaheadSimulator(self._rng)
        self._veto_precursor_streaks = {sid: 0 for sid in STANDARD_STAKEHOLDERS}

        self._graph = sample_graph(
            stakeholder_set=STANDARD_STAKEHOLDERS,
            authority_hierarchy=STANDARD_HIERARCHY,
            scenario_type=task_id,
            rng=self._rng,
        )

        self._beliefs = {
            sid: BeliefDistribution(
                distribution=_get_initial_beliefs(task_id, sid),
                stakeholder_role=sid,
                confidence=1.0,
                history=[],
            )
            for sid in STANDARD_STAKEHOLDERS
        }

        self._noisy_engagement = {
            sid: float(
                np.clip(
                    0.5 + self._rng.normal(0, OBS_CONFIG.engagement_noise_sigma),
                    0.0,
                    1.0,
                )
            )
            for sid in STANDARD_STAKEHOLDERS
        }

        self._engagement_history = {
            sid: [self._noisy_engagement[sid]] * OBS_CONFIG.engagement_history_window
            for sid in STANDARD_STAKEHOLDERS
        }

        self._state = DealRoomState(
            episode_id=self._episode_id,
            step_count=0,
            task_id=task_id,
            round_number=0,
            max_rounds=DEFAULT_MAX_ROUNDS,
            stakeholders={
                sid: {"role": get_archetype(sid).role if get_archetype(sid) else sid}
                for sid in STANDARD_STAKEHOLDERS
            },
            stakeholder_private={
                sid: {
                    "trust": 0.5,
                    "approval": 0.3,
                    "perceived_fit": 0.5,
                    "private_resistance": 0.2,
                }
                for sid in STANDARD_STAKEHOLDERS
            },
            hidden_constraints={},
            relationship_edges=[],
            commitment_ledger=[],
            deferred_effects=[],
            offer_state=self._initial_offer_state(task_id),
            feasibility_state={
                "is_feasible": False,
                "violations": ["unresolved_constraints"],
            },
            active_blockers=[],
            deal_stage="evaluation",
            deal_momentum="stalling",
            rounds_since_last_contact={sid: 0 for sid in STANDARD_STAKEHOLDERS},
            approval_caps={},
            weak_signal_history={sid: [] for sid in STANDARD_STAKEHOLDERS},
            requested_artifacts={sid: [] for sid in STANDARD_STAKEHOLDERS},
        )

        risk_snapshot = self._evaluate_committee_risk(self._state.offer_state)
        return self._build_observation(
            vendor_action=None,
            is_reset=True,
            stakeholder_messages={},
            done=False,
            reward=None,
            info={},
            risk_snapshot=risk_snapshot,
        )

    def step(
        self, action: DealRoomAction
    ) -> Tuple[DealRoomObservation, float, bool, Dict[str, Any]]:
        if self._state is None or self._scenario is None:
            raise RuntimeError("Environment must be reset() before step().")

        action = self._normalize_action(action)
        lookahead_was_requested = action.lookahead is not None
        lookahead_result = (
            self._run_lookahead(action) if lookahead_was_requested else None
        )

        state_before = StateSnapshot(
            beliefs=dict(self._beliefs),
            active_blockers=list(self._state.active_blockers or []),
            risk_profiles={sid: get_archetype(sid) for sid in STANDARD_STAKEHOLDERS},
            authority_weights=self._graph.authority_weights if self._graph else {},
            current_terms=self._state.offer_state if self._state else {},
            round_number=self._round_number,
            deal_stage=self._state.deal_stage if self._state else "evaluation",
            deal_momentum=getattr(self._state, "deal_momentum", "progressing"),
        )

        self._step_count += 1
        self._round_number += 1
        self._apply_action_to_offer_state(action)
        self._update_deal_stage()

        previous_beliefs = {sid: b.copy() for sid, b in self._beliefs.items()}

        for sid in STANDARD_STAKEHOLDERS:
            is_targeted = sid in action.target_ids
            self._beliefs[sid] = bayesian_update(
                belief=self._beliefs[sid],
                action_type=action.action_type,
                documents=action.documents,
                stakeholder_role=sid,
                is_targeted=is_targeted,
            )

        try:
            deliberation_engine = CommitteeDeliberationEngine(
                graph=self._graph, n_deliberation_steps=3
            )
            deliberation_result = deliberation_engine.run(
                vendor_action=action,
                beliefs_before_action=previous_beliefs,
                beliefs_after_vendor_action=self._beliefs,
                render_summary=self._summary_rendering_enabled(),
            )
        except BaseException:
            deliberation_result = None

        if deliberation_result is not None:
            self._beliefs = deliberation_result.updated_beliefs
            deliberation_summary = deliberation_result.summary_dialogue
            propagation_deltas = deliberation_result.propagation_deltas
        else:
            self._beliefs = self._beliefs  # keep current beliefs
            deliberation_summary = None
            propagation_deltas = {}

        noisy_deltas = self._update_noisy_engagement(self._beliefs, previous_beliefs)

        stakeholder_messages = self._generate_stakeholder_responses(
            action, previous_beliefs
        )

        reward, reward_components = self._compute_reward(
            action,
            state_before,
            lookahead_was_requested,
        )

        risk_snapshot = self._evaluate_committee_risk(self._state.offer_state)
        precursors = self._compute_veto_precursors(risk_snapshot)
        self._update_veto_precursor_streaks(precursors)
        veto_triggered, veto_stakeholder = self._check_for_veto(risk_snapshot)
        max_rounds_reached = self._round_number >= self._state.max_rounds
        done = veto_triggered or max_rounds_reached

        terminal_reward = 0.0
        terminal_outcome = ""
        if done:
            terminal_reward, terminal_outcome = compute_terminal_reward(
                deal_closed=False,
                veto_triggered=veto_triggered,
                veto_stakeholder=veto_stakeholder or "",
                max_rounds_reached=max_rounds_reached,
                stage_regressions=self._state.stage_regressions,
                all_utilities=risk_snapshot["all_utilities"],
                cvar_losses=risk_snapshot["cvar_losses"],
                thresholds=risk_snapshot["thresholds"],
            )
            reward += terminal_reward

        self._state.active_blockers = sorted(precursors.keys())
        self._state.round_number = self._round_number
        self._state.step_count = self._step_count
        self._state.terminal_outcome = terminal_outcome
        self._state.veto_stakeholder = veto_stakeholder
        self._state.deal_failed = bool(
            done and terminal_outcome and "deal_closed" not in terminal_outcome
        )
        self._state.failure_reason = terminal_outcome
        self._state.deal_momentum = self._infer_deal_momentum(precursors)

        terminal_category = self._terminal_category(terminal_outcome, done)
        info = {
            "deliberation_summary": deliberation_summary,
            "propagation_deltas": propagation_deltas,
            "noisy_engagement_deltas": noisy_deltas,
            "reward_components": reward_components,
            "terminal_reward": terminal_reward,
            "terminal_outcome": terminal_outcome,
            "terminal_category": terminal_category,
            "veto_stakeholder": veto_stakeholder,
            "lookahead_used": lookahead_was_requested,
        }

        if lookahead_result is not None:
            prediction_accuracy = self._compute_lookahead_prediction_accuracy(
                lookahead_result=lookahead_result,
                previous_beliefs=previous_beliefs,
                stakeholder_messages=stakeholder_messages,
            )
            info.update(
                {
                    "lookahead_predicted_deltas": dict(
                        lookahead_result.predicted_belief_deltas
                    ),
                    "lookahead_predicted_responses": dict(
                        lookahead_result.predicted_responses
                    ),
                    "lookahead_cvar_impact": dict(lookahead_result.cvar_impact),
                    "prediction_accuracy": prediction_accuracy,
                    "lookahead_prediction_accuracy": prediction_accuracy,
                }
            )

        obs = self._build_observation(
            vendor_action=action,
            is_reset=False,
            stakeholder_messages=stakeholder_messages,
            done=done,
            reward=reward,
            info=info,
            risk_snapshot=risk_snapshot,
        )

        return obs, reward, done, info

    def _generate_stakeholder_responses(
        self, action: DealRoomAction, previous_beliefs: Dict[str, BeliefDistribution]
    ) -> Dict[str, str]:
        responses = {}
        for sid in STANDARD_STAKEHOLDERS:
            is_targeted = sid in action.target_ids
            belief = self._beliefs[sid]
            role = get_archetype(sid).role if get_archetype(sid) else sid

            if is_targeted:
                responses[sid] = self._generate_targeted_response(
                    sid, role, belief, action
                )
            else:
                responses[sid] = self._generate_non_targeted_response(
                    sid, role, belief, action, previous_beliefs
                )
        return responses

    def _generate_targeted_response(
        self, sid: str, role: str, belief: BeliefDistribution, action: DealRoomAction
    ) -> str:
        pos_mass = belief.positive_mass()
        stance = (
            "supportive"
            if pos_mass > 0.6
            else ("neutral" if pos_mass > 0.4 else "skeptical")
        )

        if self._use_llm_stakeholders:
            try:
                from .stakeholder_llm import generate_stakeholder_response

                action_desc = action.message or f"action: {action.action_type}"
                context = f"Vendor sent a {action.action_type} to {sid}: {action_desc}"
                return generate_stakeholder_response(
                    sid, context, role=role, stance=stance
                )
            except Exception:
                pass

        if pos_mass > 0.6:
            return f"Thank you for the {'document' if action.documents else 'message'}. I can see the merit in this approach and will review accordingly."
        elif pos_mass > 0.4:
            return f"I appreciate the information. Let me consider the implications for our evaluation before committing."
        else:
            return f"I have concerns about this direction. We need more detail before I can support this proposal."

    def _generate_non_targeted_response(
        self,
        sid: str,
        role: str,
        belief: BeliefDistribution,
        action: DealRoomAction,
        previous_beliefs: Dict[str, BeliefDistribution],
    ) -> str:
        delta = belief.positive_mass() - previous_beliefs[sid].positive_mass()
        if abs(delta) < 0.03:
            return ""
        if delta > 0:
            return f"I've noticed some positive developments in the evaluation. Monitoring the situation."
        else:
            return f"There are some concerns emerging. Will need to assess the full implications."

    def _update_noisy_engagement(
        self,
        true_beliefs_current: Dict[str, BeliefDistribution],
        true_beliefs_previous: Dict[str, BeliefDistribution],
    ) -> Dict[str, float]:
        noisy_deltas = {}
        for sid in STANDARD_STAKEHOLDERS:
            true_eng_current = compute_engagement_level(true_beliefs_current[sid])
            true_eng_previous = compute_engagement_level(true_beliefs_previous[sid])
            true_delta = true_eng_current - true_eng_previous

            noise = self._rng.normal(0, OBS_CONFIG.engagement_noise_sigma)
            noisy_delta = float(np.clip(true_delta + noise, -1.0, 1.0))

            self._noisy_engagement[sid] = float(
                np.clip(self._noisy_engagement[sid] + noisy_delta, 0.0, 1.0)
            )

            self._engagement_history[sid].pop(0)
            self._engagement_history[sid].append(self._noisy_engagement[sid])

            noisy_deltas[sid] = noisy_delta
        return noisy_deltas

    def _compute_reward(
        self,
        action: DealRoomAction,
        state_before: StateSnapshot,
        lookahead_used: bool,
    ) -> Tuple[float, Dict[str, float]]:
        state_after = StateSnapshot(
            beliefs=self._beliefs,
            active_blockers=list(self._state.active_blockers or []),
            risk_profiles={sid: get_archetype(sid) for sid in STANDARD_STAKEHOLDERS},
            authority_weights=self._graph.authority_weights if self._graph else {},
            current_terms=self._state.offer_state if self._state else {},
            round_number=self._round_number,
            deal_stage=self._state.deal_stage if self._state else "evaluation",
            deal_momentum=getattr(self._state, "deal_momentum", "progressing"),
        )

        score = self._utterance_scorer.score(
            action=action,
            state_before=state_before,
            state_after=state_after,
            true_graph=self._graph,
            lookahead_used=lookahead_used,
        )

        return float(score.weighted_sum(REWARD_WEIGHTS)), score.to_dict()

    def _build_observation(
        self,
        vendor_action: Optional[DealRoomAction],
        is_reset: bool,
        stakeholder_messages: Dict[str, str],
        done: bool,
        reward: Optional[float],
        info: Dict[str, Any],
        risk_snapshot: Dict[str, Any],
    ) -> DealRoomObservation:
        weak_signals = self._generate_weak_signals()
        veto_precursors = self._compute_veto_precursors(risk_snapshot)

        engagement_level_delta = {
            sid: self._noisy_engagement[sid]
            - (
                self._engagement_history[sid][-2]
                if len(self._engagement_history[sid]) > 1
                else self._noisy_engagement[sid]
            )
            for sid in STANDARD_STAKEHOLDERS
        }

        cross_echoes = (
            self._generate_cross_stakeholder_echoes(vendor_action)
            if vendor_action
            else []
        )
        primary_target = (
            vendor_action.target_ids[0]
            if vendor_action and vendor_action.target_ids
            else STANDARD_STAKEHOLDERS[0]
        )

        return DealRoomObservation(
            reward=reward,
            metadata={"graph_seed": self._graph.seed if self._graph else None},
            round_number=self._round_number,
            max_rounds=self._state.max_rounds if self._state else 10,
            stakeholders={
                sid: {"role": get_archetype(sid).role if get_archetype(sid) else sid}
                for sid in STANDARD_STAKEHOLDERS
            },
            stakeholder_messages=dict(stakeholder_messages),
            engagement_level=dict(self._noisy_engagement),
            weak_signals=weak_signals,
            known_constraints=[],
            requested_artifacts=dict(self._state.requested_artifacts)
            if self._state
            else {},
            approval_path_progress={
                sid: {"band": "neutral"} for sid in STANDARD_STAKEHOLDERS
            },
            deal_momentum=self._state.deal_momentum if self._state else "stalling",
            deal_stage=self._state.deal_stage if self._state else "evaluation",
            competitor_events=[],
            veto_precursors=veto_precursors,
            scenario_hint=None,
            active_blockers=self._state.active_blockers if self._state else [],
            days_to_deadline=(self._state.offer_state or {}).get("days_to_deadline", 30)
            if self._state
            else 30,
            done=done,
            info=info,
            engagement_level_delta=engagement_level_delta.get(primary_target, 0.0),
            engagement_history=[
                {sid: self._engagement_history[sid][-1]}
                for sid in STANDARD_STAKEHOLDERS
            ],
            cross_stakeholder_echoes=cross_echoes,
        )

    def _generate_weak_signals(self) -> Dict[str, List[str]]:
        weak_signals = {}
        for sid in STANDARD_STAKEHOLDERS:
            signals = []
            eng_level = self._noisy_engagement.get(sid, 0.5)
            delta = (
                self._engagement_history[sid][-1] - self._engagement_history[sid][0]
                if len(self._engagement_history[sid]) > 0
                else 0
            )

            if eng_level > 0.7:
                signals.append("high_engagement")
            elif eng_level < 0.3:
                signals.append("low_engagement")

            if delta > 0.1:
                signals.append("improving_engagement")
            elif delta < -0.1:
                signals.append("declining_engagement")

            belief = self._beliefs.get(sid)
            if belief and belief.confidence < 0.4:
                signals.append("high_uncertainty")

            weak_signals[sid] = signals if signals else ["neutral"]
        return weak_signals

    def _compute_veto_precursors(self, risk_snapshot: Dict[str, Any]) -> Dict[str, str]:
        precursors = {}
        for sid in STANDARD_STAKEHOLDERS:
            profile = get_archetype(sid)
            if not profile or not profile.veto_power:
                continue
            cvar_loss = risk_snapshot["cvar_losses"].get(sid, 0.0)
            if cvar_loss > profile.tau * VETO_WARNING_THRESHOLD_RATIO:
                if cvar_loss > profile.tau:
                    precursors[sid] = (
                        "Tail-risk concern is at block level; stronger safeguards are needed immediately."
                    )
                else:
                    precursors[sid] = (
                        "Tail-risk concern is rising; stronger safeguards may be required soon."
                    )
        return precursors

    def _generate_cross_stakeholder_echoes(
        self, action: Optional[DealRoomAction]
    ) -> List[Dict[str, str]]:
        echoes = []
        if not action or not action.target_ids:
            return echoes

        targeted = action.target_ids[0]
        for sid in STANDARD_STAKEHOLDERS:
            if sid == targeted:
                continue
            if self._rng.random() < OBS_CONFIG.echo_recall_probability:
                echoes.append(
                    {"from": targeted, "to": sid, "content": "cross_reference"}
                )
        return echoes

    def _initial_offer_state(self, task_id: str) -> Dict[str, Any]:
        base_terms = {
            "aligned": {
                "price": 95000,
                "timeline_weeks": 14,
                "security_commitments": ["gdpr", "audit_rights"],
                "support_level": "named_support_lead",
                "liability_cap": 1500000,
                "has_dpa": True,
                "has_security_cert": True,
            },
            "conflicted": {
                "price": 120000,
                "timeline_weeks": 12,
                "security_commitments": ["gdpr"],
                "support_level": "shared_success_team",
                "liability_cap": 800000,
                "has_dpa": False,
                "has_security_cert": True,
            },
            "hostile_acquisition": {
                "price": 160000,
                "timeline_weeks": 8,
                "security_commitments": [],
                "support_level": "best_effort",
                "liability_cap": 300000,
                "has_dpa": False,
                "has_security_cert": False,
            },
        }.get(task_id, {})
        return {
            **base_terms,
            "days_to_deadline": 30,
            "event_round": -1,
            "event_triggered": False,
        }

    def _normalize_action(self, action: DealRoomAction) -> DealRoomAction:
        canonical = {sid.lower(): sid for sid in STANDARD_STAKEHOLDERS}

        resolved_target_ids = [
            canonical.get(target.strip().lower(), target.strip())
            for target in action.target_ids
            if target and target.strip()
        ]

        if not resolved_target_ids and action.target:
            target = action.target.strip()
            if target.lower() == "all":
                resolved_target_ids = list(STANDARD_STAKEHOLDERS)
            else:
                resolved_target_ids = [
                    canonical.get(item.strip().lower(), item.strip())
                    for item in target.split(",")
                    if item.strip()
                ]

        normalized_target = action.target
        if normalized_target and normalized_target.lower() != "all":
            normalized_target = (
                ",".join(resolved_target_ids)
                if resolved_target_ids
                else normalized_target
            )

        return action.model_copy(
            update={
                "target_ids": resolved_target_ids,
                "target": normalized_target,
            }
        )

    def _apply_action_to_offer_state(self, action: DealRoomAction) -> None:
        offer_state = dict(self._state.offer_state or {})
        offer_state["days_to_deadline"] = max(
            0, offer_state.get("days_to_deadline", 30) - 1
        )

        for key, value in (action.proposed_terms or {}).items():
            offer_state[key] = value

        document_names = {
            str(document.get("type") or document.get("name") or "").lower()
            for document in action.documents
        }
        if any("dpa" in name for name in document_names):
            offer_state["has_dpa"] = True
        if any("security" in name or "cert" in name for name in document_names):
            offer_state["has_security_cert"] = True
        if any(
            "implementation" in name or "timeline" in name for name in document_names
        ):
            offer_state["timeline_weeks"] = min(
                offer_state.get("timeline_weeks", 12), 12
            )
        if any("roi" in name for name in document_names):
            offer_state["price"] = max(
                75000, int(offer_state.get("price", 100000) * 0.95)
            )

        if action.action_type == "concession":
            offer_state["price"] = max(
                70000, int(offer_state.get("price", 100000) * 0.90)
            )
            offer_state["liability_cap"] = max(
                offer_state.get("liability_cap", 1000000), 1500000
            )
        elif action.action_type == "exec_escalation":
            offer_state["price"] = int(offer_state.get("price", 100000) * 1.10)
            offer_state["liability_cap"] = min(
                offer_state.get("liability_cap", 1000000), 250000
            )
        elif action.action_type == "walkaway_signal":
            offer_state["price"] = int(offer_state.get("price", 100000) * 1.05)
            offer_state["liability_cap"] = min(
                offer_state.get("liability_cap", 1000000), 350000
            )
        elif action.action_type == "group_proposal" and not action.proposed_terms:
            offer_state["liability_cap"] = min(
                offer_state.get("liability_cap", 1000000), 500000
            )

        self._state.offer_state = offer_state

    def _update_deal_stage(self) -> None:
        if self._round_number < 3:
            self._state.deal_stage = "evaluation"
        elif self._round_number < 6:
            self._state.deal_stage = "negotiation"
        else:
            self._state.deal_stage = "final_review"

    def _infer_deal_momentum(self, precursors: Dict[str, str]) -> str:
        if self._state.terminal_outcome and "veto" in self._state.terminal_outcome:
            return "critical"
        if precursors:
            return "fragile"
        average_positive_mass = float(
            np.mean([belief.positive_mass() for belief in self._beliefs.values()])
        )
        return "progressing" if average_positive_mass >= 0.55 else "stalling"

    def _terminal_category(self, terminal_outcome: str, done: bool) -> str:
        if not done:
            return ""
        if terminal_outcome.startswith("veto"):
            return "veto"
        if terminal_outcome.startswith("deal_closed"):
            return "deal_closed"
        if terminal_outcome.startswith("max_rounds"):
            return "timeout"
        if terminal_outcome.startswith("stage_regression"):
            return "stage_regression"
        return terminal_outcome or "unknown"

    def _summary_rendering_enabled(self) -> bool:
        return os.getenv("DEALROOM_ENABLE_LLM_SUMMARY", "false").lower() in {
            "1",
            "true",
            "yes",
        }

    def _run_lookahead(self, action: DealRoomAction):
        if action.lookahead is None or self._lookahead_simulator is None:
            return None
        action_draft = action.lookahead.action_draft or action
        return self._lookahead_simulator.simulate(
            action_draft=action_draft,
            current_beliefs=self._beliefs,
            n_hypotheses=action.lookahead.n_hypotheses,
            depth=action.lookahead.depth,
        )

    def _context_rng(self, label: str, stakeholder_id: str) -> np.random.Generator:
        digest = hashlib.sha256(
            f"{self._episode_id}|{self._scenario.task_id}|{self._round_number}|{label}|{stakeholder_id}".encode()
        ).hexdigest()
        return np.random.default_rng(int(digest[:16], 16))

    def _evaluate_committee_risk(self, deal_terms: Dict[str, Any]) -> Dict[str, Any]:
        all_utilities: Dict[str, float] = {}
        cvar_losses: Dict[str, float] = {}
        thresholds: Dict[str, float] = {}
        scenario_multiplier = {
            "aligned": 0.12,
            "conflicted": 0.22,
            "hostile_acquisition": 0.42,
        }.get(self._scenario.task_id if self._scenario else "aligned", 0.22)

        for sid in STANDARD_STAKEHOLDERS:
            profile = get_archetype(sid)
            if profile is None:
                continue
            outcomes = compute_outcome_distribution(
                deal_terms,
                profile,
                self._context_rng("cvar", sid),
                n_samples=500,
            )
            belief = self._beliefs.get(sid)
            positive_mass = belief.positive_mass() if belief is not None else 0.5
            confidence_factor = 1.0 - (0.25 * np.clip(positive_mass, 0.0, 1.0))
            all_utilities[sid] = float(np.mean(outcomes)) if len(outcomes) else 0.0
            cvar_losses[sid] = float(
                compute_cvar(outcomes, profile.alpha)
                * scenario_multiplier
                * confidence_factor
            )
            thresholds[sid] = profile.tau

        return {
            "all_utilities": all_utilities,
            "cvar_losses": cvar_losses,
            "thresholds": thresholds,
        }

    def _update_veto_precursor_streaks(self, precursors: Dict[str, str]) -> None:
        for sid in STANDARD_STAKEHOLDERS:
            if sid in precursors:
                self._veto_precursor_streaks[sid] = (
                    self._veto_precursor_streaks.get(sid, 0) + 1
                )
            else:
                self._veto_precursor_streaks[sid] = 0

    def _check_for_veto(
        self, risk_snapshot: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        candidates: List[Tuple[float, str]] = []
        for sid in STANDARD_STAKEHOLDERS:
            profile = get_archetype(sid)
            if not profile or not profile.veto_power:
                continue
            cvar_loss = risk_snapshot["cvar_losses"].get(sid, 0.0)
            if (
                cvar_loss > profile.tau
                and self._veto_precursor_streaks.get(sid, 0) >= 2
            ):
                candidates.append((cvar_loss - profile.tau, sid))
        if not candidates:
            return False, None
        candidates.sort(reverse=True)
        return True, candidates[0][1]

    def _compute_lookahead_prediction_accuracy(
        self,
        lookahead_result,
        previous_beliefs: Dict[str, BeliefDistribution],
        stakeholder_messages: Dict[str, str],
    ) -> float:
        response_accuracy = compute_prediction_accuracy(
            lookahead_result.predicted_responses,
            stakeholder_messages,
        )

        delta_scores: List[float] = []
        for (
            stakeholder_id,
            predicted_delta,
        ) in lookahead_result.predicted_belief_deltas.items():
            if (
                stakeholder_id not in previous_beliefs
                or stakeholder_id not in self._beliefs
            ):
                continue
            actual_delta = (
                self._beliefs[stakeholder_id].positive_mass()
                - previous_beliefs[stakeholder_id].positive_mass()
            )
            delta_scores.append(max(0.0, 1.0 - abs(predicted_delta - actual_delta)))

        if delta_scores:
            return float(
                0.5 * response_accuracy + 0.5 * (sum(delta_scores) / len(delta_scores))
            )
        return float(response_accuracy)

    def close(self) -> None:
        return None
