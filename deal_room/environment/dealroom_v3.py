"""
DealRoomV3 environment for DealRoom v3 - OpenEnv wrapper with causal graph inference.
"""

import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..committee.causal_graph import (
    BeliefDistribution,
    compute_engagement_level,
    create_neutral_beliefs,
    propagate_beliefs,
    sample_graph,
)
from ..committee.deliberation_engine import CommitteeDeliberationEngine
from models import DealRoomAction, DealRoomObservation, DealRoomState, SimulationResult
from ..rewards.pareto_efficiency import compute_terminal_reward
from ..rewards.utterance_scorer import UtteranceScorer
from ..stakeholders.archetypes import ARCHETYPE_PROFILES, get_archetype
from ..stakeholders.cvar_preferences import check_veto_trigger, evaluate_deal
from .llm_client import validate_api_keys


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
        veto_warning_threshold_ratio: float = 0.70

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
class ScenarioConfig:
    task_id: str
    max_rounds: int = 10
    seed: Optional[int] = None


class DealRoomV3:
    def __init__(self):
        validate_api_keys()
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
        self._step_count: int = 0
        self._round_number: int = 0
        self._episode_id: str = ""

    def reset(
        self, seed: Optional[int] = None, task_id: str = "aligned", **kwargs
    ) -> DealRoomObservation:
        self._rng = np.random.default_rng(seed)
        self._scenario = ScenarioConfig(task_id=task_id, seed=seed)
        self._episode_id = str(uuid.uuid4())[:8]
        self._step_count = 0
        self._round_number = 0

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
            max_rounds=10,
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
            offer_state={
                "price": None,
                "timeline_weeks": None,
                "security_commitments": None,
                "support_level": None,
                "liability_cap": None,
                "days_to_deadline": 30,
                "event_round": -1,
                "event_triggered": False,
            },
            feasibility_state={
                "is_feasible": False,
                "violations": ["unresolved_constraints"],
            },
            active_blockers=[],
            deal_stage="evaluation",
            rounds_since_last_contact={sid: 0 for sid in STANDARD_STAKEHOLDERS},
            approval_caps={},
            weak_signal_history={sid: [] for sid in STANDARD_STAKEHOLDERS},
            requested_artifacts={sid: [] for sid in STANDARD_STAKEHOLDERS},
        )

        return self._build_observation(vendor_action=None, is_reset=True)

    def step(
        self, action: DealRoomAction
    ) -> Tuple[DealRoomObservation, float, bool, Dict[str, Any]]:
        self._step_count += 1
        self._round_number += 1

        previous_beliefs = {sid: b.copy() for sid, b in self._beliefs.items()}

        for sid in STANDARD_STAKEHOLDERS:
            is_targeted = sid in action.target_ids
            if is_targeted:
                self._beliefs[sid] = self._bayesian_update(
                    self._beliefs[sid], action, sid, is_targeted=True
                )
            else:
                self._beliefs[sid] = self._bayesian_update(
                    self._beliefs[sid], action, sid, is_targeted=False
                )

        deliberation_engine = CommitteeDeliberationEngine(
            graph=self._graph, n_deliberation_steps=3
        )
        deliberation_result = deliberation_engine.run(
            vendor_action=action,
            beliefs_before_action=previous_beliefs,
            beliefs_after_vendor_action=self._beliefs,
            render_summary=True,
        )
        self._beliefs = deliberation_result.updated_beliefs

        noisy_deltas = self._update_noisy_engagement(self._beliefs, previous_beliefs)

        responses = self._generate_stakeholder_responses(action, previous_beliefs)

        reward = self._compute_reward(action, responses)

        done = self._round_number >= self._state.max_rounds

        obs = self._build_observation(vendor_action=action, is_reset=False)

        info = {
            "deliberation_summary": deliberation_result.summary_dialogue,
            "propagation_deltas": deliberation_result.propagation_deltas,
            "noisy_engagement_deltas": noisy_deltas,
        }

        self._state.round_number = self._round_number
        self._state.step_count = self._step_count

        return obs, reward, done, info

    def _bayesian_update(
        self,
        belief: BeliefDistribution,
        action: DealRoomAction,
        stakeholder_id: str,
        is_targeted: bool,
    ) -> BeliefDistribution:
        from ..committee.causal_graph import VENDOR_TYPES
        import math

        likelihoods = {
            "send_document(DPA)_proactive": {
                "competent": 0.85,
                "incompetent": 0.15,
                "trustworthy": 0.80,
                "deceptive": 0.20,
                "aligned": 0.80,
                "misaligned": 0.20,
            },
            "send_document(security_cert)_proactive": {
                "competent": 0.80,
                "incompetent": 0.20,
                "trustworthy": 0.75,
                "deceptive": 0.25,
                "aligned": 0.75,
                "misaligned": 0.25,
            },
            "send_document(roi_model)_to_finance": {
                "competent": 0.75,
                "incompetent": 0.25,
                "trustworthy": 0.60,
                "deceptive": 0.40,
                "aligned": 0.70,
                "misaligned": 0.30,
            },
            "send_document(implementation_timeline)": {
                "competent": 0.70,
                "incompetent": 0.30,
                "trustworthy": 0.60,
                "deceptive": 0.40,
                "aligned": 0.70,
                "misaligned": 0.30,
            },
            "direct_message_role_specific": {
                "competent": 0.70,
                "incompetent": 0.30,
                "trustworthy": 0.65,
                "deceptive": 0.35,
                "aligned": 0.70,
                "misaligned": 0.30,
            },
        }

        best_key = "default"
        for key in likelihoods:
            if key in action.action_type:
                best_key = key
                break

        damping = 1.0 if is_targeted else 0.3
        like = likelihoods.get(
            best_key,
            {
                "competent": 0.5,
                "incompetent": 0.5,
                "trustworthy": 0.5,
                "deceptive": 0.5,
                "aligned": 0.5,
                "misaligned": 0.5,
            },
        )

        new_dist = {}
        for vendor_type, prior_prob in belief.distribution.items():
            likelihood = like.get(vendor_type, 0.5)
            dampened_likelihood = 1.0 + damping * (likelihood - 1.0)
            new_dist[vendor_type] = prior_prob * dampened_likelihood

        total = sum(new_dist.values())
        new_dist = {k: max(0.01, v / total) for k, v in new_dist.items()}

        probs = [new_dist.get(t, 0.01) for t in VENDOR_TYPES]
        entropy = -sum(p * math.log(p, 2) for p in probs if p > 0)
        LOG2_6 = math.log(2, 6)
        confidence = 1.0 - (entropy / LOG2_6 if LOG2_6 > 0 else 0)

        return BeliefDistribution(
            distribution=new_dist,
            stakeholder_role=belief.stakeholder_role,
            confidence=confidence,
            history=belief.history + [(action.action_type, damping)],
        )

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
        self, action: DealRoomAction, responses: Dict[str, str]
    ) -> float:
        total_score = 0.0
        n = 0
        for sid in action.target_ids if action.target_ids else STANDARD_STAKEHOLDERS:
            score, _ = self._utterance_scorer.score(
                action=action,
                obs=self._build_observation(vendor_action=action, is_reset=False),
                stakeholder_id=sid,
                engagement_levels=self._noisy_engagement,
                graph=self._graph,
                lookahead_used=action.lookahead is not None,
                belief_state_hash=str(self._beliefs[sid].distribution),
            )
            total_score += (
                score.goal + score.trust + score.information + score.risk + score.causal
            ) / 5.0
            n += 1

        if n == 0:
            n = 1

        return total_score / n

    def _build_observation(
        self, vendor_action: Optional[DealRoomAction], is_reset: bool
    ) -> DealRoomObservation:
        weak_signals = self._generate_weak_signals()
        veto_precursors = self._compute_veto_precursors()

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

        return DealRoomObservation(
            reward=None,
            metadata={"graph_seed": self._graph.seed if self._graph else None},
            round_number=self._round_number,
            max_rounds=self._state.max_rounds if self._state else 10,
            stakeholders={
                sid: {"role": get_archetype(sid).role if get_archetype(sid) else sid}
                for sid in STANDARD_STAKEHOLDERS
            },
            stakeholder_messages={},
            engagement_level=dict(self._noisy_engagement),
            weak_signals=weak_signals,
            known_constraints=[],
            requested_artifacts={sid: [] for sid in STANDARD_STAKEHOLDERS},
            approval_path_progress={
                sid: {"band": "neutral"} for sid in STANDARD_STAKEHOLDERS
            },
            deal_momentum="stalling",
            deal_stage=self._state.deal_stage if self._state else "evaluation",
            competitor_events=[],
            veto_precursors=veto_precursors,
            scenario_hint=None,
            active_blockers=self._state.active_blockers if self._state else [],
            days_to_deadline=30,
            done=False,
            info={},
            engagement_level_delta=engagement_level_delta.get(
                STANDARD_STAKEHOLDERS[0] if STANDARD_STAKEHOLDERS else "", None
            ),
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

    def _compute_veto_precursors(self) -> Dict[str, str]:
        precursors = {}
        for sid in STANDARD_STAKEHOLDERS:
            profile = get_archetype(sid)
            if profile:
                deal_terms = {"price": 100000, "timeline_weeks": 12}
                _, cvar_loss = evaluate_deal(deal_terms, profile, self._rng)
                if cvar_loss > profile.tau * OBS_CONFIG.veto_warning_threshold_ratio:
                    precursors[sid] = (
                        f"CVaR warning: {cvar_loss:.2f} vs threshold {profile.tau:.2f}"
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
