"""
DealRoom Environment — Main Entry Point
OpenEnv-compliant POMDP for multi-stakeholder enterprise contract negotiation.

All 5 issue fixes applied:
  Issue 1: ClaimsTracker receives individual IDs, expansion centralized here
  Issue 2: Group target belief_deltas uses max() across expanded targets
  Issue 3: Veto risk skips round 0 (opening round guard)
  Issue 4: Stage progression requires STAGE_MIN_ROUNDS
  Issue 5: Momentum is three-state: +1/0/-1
"""

import uuid
import logging
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from models import DealRoomAction, DealRoomObservation, DealRoomState
from server.validator import OutputValidator
from server.grader import CCIGrader
from server.scenarios import SCENARIOS
from server.stakeholders import StakeholderEngine
from server.claims import ClaimsTracker, expand_targets

logger = logging.getLogger(__name__)

STAGE_REGRESSION = {
    "final_approval": "legal_review",
    "legal_review": "negotiation",
    "negotiation": "evaluation",
    "evaluation": "evaluation",
    "closed": "closed",
    "regressed": "evaluation",
}

STAGE_PROGRESSION = {
    "evaluation": "negotiation",
    "negotiation": "legal_review",
    "legal_review": "final_approval",
    "final_approval": "closed",
}

STAGE_MIN_SAT = {
    "evaluation": 0.45,
    "negotiation": 0.50,
    "legal_review": 0.55,
    "final_approval": 0.60,
}

STAGE_MIN_ROUNDS = {
    "evaluation": 2,
    "negotiation": 2,
    "legal_review": 1,
    "final_approval": 1,
}


class DealRoomEnvironment:
    """
    OpenEnv-compliant multi-stakeholder enterprise negotiation environment.

    Usage:
        env = DealRoomEnvironment()
        obs = env.reset(seed=42, task_id="aligned")
        obs, reward, done, info = env.step(action)
        state = env.state  # @property
    """

    def __init__(self):
        self._state = DealRoomState()
        self.stakeholder_engine = StakeholderEngine()
        self.validator = OutputValidator(mode="strict")
        self.claims_tracker = ClaimsTracker()

        self._prev_satisfaction: Dict[str, float] = {}
        self._prev_stage: str = "evaluation"
        self._stage_regressed_this_round: bool = False
        self._precursors_this_round: Dict[str, str] = {}
        self._active_competitor_events: List[str] = []
        self._prev_blocker_count: int = 0
        self._belief_deltas: Dict[str, float] = {}
        self.rng: np.random.Generator = np.random.default_rng(None)

    def reset(
        self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs
    ) -> DealRoomObservation:
        """Reset for new episode. task_id passed via kwargs per OpenEnv spec."""
        task_id = kwargs.get("task_id", "aligned")
        if task_id not in SCENARIOS:
            raise ValueError(
                f"Unknown task_id '{task_id}'. Valid: {list(SCENARIOS.keys())}"
            )

        self.rng = np.random.default_rng(seed)
        scenario = SCENARIOS[task_id]
        ep_id = episode_id or str(uuid.uuid4())[:8]
        sids = list(scenario["initial_beliefs"].keys())

        self._state = DealRoomState(
            episode_id=ep_id,
            task_id=task_id,
            round_number=0,
            max_rounds=scenario["max_rounds"],
            beliefs=deepcopy(scenario["initial_beliefs"]),
            satisfaction=dict(scenario["initial_satisfaction"]),
            trust_floors={k: 0.20 for k in sids},
            permanent_marks={k: [] for k in sids},
            tracked_claims={},
            veto_risk={k: 0.0 for k in sids},
            veto_precursors_fired={k: False for k in sids},
            active_blockers=[],
            deal_stage="evaluation",
            stage_regressions=0,
            validation_failures=0,
            fallback_streak=0,
            scrutiny_mode=False,
            exec_escalation_used=False,
            deal_closed=False,
            deal_failed=False,
            failure_reason="",
            final_terms=None,
            rounds_since_last_contact={k: 0 for k in sids},
        )

        self.stakeholder_engine.reset(self._state, self.rng, scenario)
        self.claims_tracker.reset()

        self._prev_satisfaction = dict(self._state.satisfaction)
        self._prev_stage = "evaluation"
        self._stage_regressed_this_round = False
        self._precursors_this_round = {}
        self._active_competitor_events = []
        self._prev_blocker_count = 0
        self._belief_deltas = {}

        responses = self.stakeholder_engine.generate_opening()
        self._update_active_blockers()

        return self._build_observation(responses, hint=None, is_done=False)

    def step(
        self, action: DealRoomAction
    ) -> Tuple[DealRoomObservation, float, bool, Dict]:
        """
        Process one negotiation turn.
        Returns: (observation, reward, done, info)
        reward is 0.0 during episode; CCI score only at successful terminal step.
        """
        if self._state.deal_closed or self._state.deal_failed:
            obs = self._build_observation({}, hint=None, is_done=True)
            return obs, 0.0, True, {"error": "episode_already_done"}

        parsed_dict, confidence = self.validator.validate(action.message)
        if confidence < 0.4:
            self._state.validation_failures += 1
            self._state.fallback_streak += 1
            if self._state.fallback_streak >= 2:
                self._state.scrutiny_mode = True
        else:
            self._state.fallback_streak = 0

        normalized = DealRoomAction(
            action_type=parsed_dict.get("action_type", action.action_type),
            target=parsed_dict.get("target", action.target),
            message=parsed_dict.get("message", action.message),
            channel=parsed_dict.get("channel", action.channel),
            mode=parsed_dict.get("mode", action.mode),
            documents=action.documents or parsed_dict.get("documents", []),
            proposed_terms=action.proposed_terms,
        )

        expanded_targets = expand_targets(normalized.target)
        for t in expanded_targets:
            if self.claims_tracker.extract_and_track(t, normalized.message):
                self._state.trust_floors[t] = max(
                    0.0, self._state.trust_floors.get(t, 0.20) - 0.12
                )
                self._state.permanent_marks[t].append("contradiction_penalty")

        self._tick_background_beliefs()

        self.stakeholder_engine.apply_action(normalized.dict(), self.rng)
        self._belief_deltas = self.stakeholder_engine.get_belief_deltas()

        self._apply_conditional_shocks()

        self._update_veto_risk(normalized)
        self._evaluate_veto_precursors()

        threshold = SCENARIOS[self._state.task_id]["veto_threshold"]
        for sid, risk in self._state.veto_risk.items():
            if risk > threshold:
                self._state.deal_failed = True
                self._state.failure_reason = f"silent_veto:{sid}"
                obs = self._build_observation(
                    {sid: "We've decided not to move forward at this time."},
                    hint=None,
                    is_done=True,
                )
                obs.info = {
                    "veto_triggered": True,
                    "veto_source": sid,
                    "round_signals": self._build_round_signals(normalized),
                }
                self._state.round_number += 1
                return obs, 0.0, True, obs.info

        self._stage_regressed_this_round = False
        if self._should_regress_stage():
            self._regress_stage()
        if not self._stage_regressed_this_round:
            self._maybe_advance_stage()

        self._prev_blocker_count = len(self._state.active_blockers)
        self._update_active_blockers()
        responses = self.stakeholder_engine.generate_responses(
            normalized.dict(), self._state
        )

        hint = None
        if (
            self._state.task_id == "hostile_acquisition"
            and self._state.round_number == 3
        ):
            hint = SCENARIOS["hostile_acquisition"]["round_3_hint"]

        done, reward = self._check_terminal()

        obs = self._build_observation(responses, hint=hint, is_done=done)
        obs.info = self._build_round_signals(normalized)
        obs.info["validation_confidence"] = confidence

        self._prev_satisfaction = dict(self._state.satisfaction)
        self._prev_stage = self._state.deal_stage
        self._state.round_number += 1

        return obs, reward, done, obs.info

    @property
    def state(self) -> DealRoomState:
        return self._state

    def _tick_background_beliefs(self):
        """
        Stakeholder beliefs drift 2% toward neutral each round independently of agent.
        Models that unaddressed concerns compound over time.
        """
        alpha = 0.02
        for sid, beliefs in self._state.beliefs.items():
            for dim in ["competence", "risk_tolerance", "pricing_rigor"]:
                current = beliefs[dim]
                beliefs[dim] = round(
                    float(np.clip(current + (0.5 - current) * alpha, 0.0, 1.0)), 4
                )

    def _apply_conditional_shocks(self):
        """
        Satisfaction drops ONLY when there is a logical cause.
        No random unconditional drops — that creates unlearnable dynamics.

        Cause A: Competitor event active + agent hasn't contacted this stakeholder for 2+ rounds
        Cause B: Contradiction penalty was just applied this round
        """
        for sid in self._state.satisfaction:
            shock = 0.0

            if (
                self._active_competitor_events
                and self._state.rounds_since_last_contact.get(sid, 0) >= 2
            ):
                shock = 0.07

            marks = self._state.permanent_marks.get(sid, [])
            if marks and marks[-1] == "contradiction_penalty":
                shock = max(shock, 0.10)

            if shock > 0:
                self._state.satisfaction[sid] = max(
                    self._state.trust_floors.get(sid, 0.0),
                    self._state.satisfaction[sid] - shock,
                )

        if self._state.task_id in ("conflicted", "hostile_acquisition"):
            if self.rng.random() < 0.05 and not self._active_competitor_events:
                self._active_competitor_events.append("competitor_demo_scheduled")

    def _update_veto_risk(self, action: DealRoomAction):
        """
        Issue 3 fix: skip round 0 (opening observations only, no agent action taken yet).
        Growth rate is intentional — precursor window gives ~3 rounds of warning.
        """
        if self._state.round_number == 0:
            return

        for sid in self._state.veto_risk:
            sat = self._state.satisfaction.get(sid, 0.5)
            if sat < 0.30:
                self._state.veto_risk[sid] = min(1.0, self._state.veto_risk[sid] + 0.08)
            elif sat < 0.38:
                self._state.veto_risk[sid] = min(1.0, self._state.veto_risk[sid] + 0.04)
            else:
                self._state.veto_risk[sid] = max(0.0, self._state.veto_risk[sid] - 0.02)

        if action.action_type == "exec_escalation":
            if self._state.exec_escalation_used:
                for sid in self._state.veto_risk:
                    self._state.veto_risk[sid] = min(
                        1.0, self._state.veto_risk[sid] + 0.05
                    )
            self._state.exec_escalation_used = True

    def _evaluate_veto_precursors(self):
        """
        Emit ambiguous warning signals when veto risk enters 0.28–0.50 range.
        Each stakeholder gets at most one precursor signal.
        Multiple message templates — seeded variation prevents memorization.
        """
        self._precursors_this_round = {}
        PRECURSOR_MSGS = [
            "{sid} has been unusually brief in recent replies.",
            "{sid} delegated follow-up coordination to their assistant.",
            "Note: {sid}'s calendar shows significant competing priorities this week.",
            "{sid} missed the last two check-in messages.",
        ]
        for sid, risk in self._state.veto_risk.items():
            if 0.28 <= risk <= 0.50 and not self._state.veto_precursors_fired.get(
                sid, False
            ):
                self._state.veto_precursors_fired[sid] = True
                tmpl = PRECURSOR_MSGS[int(self.rng.integers(0, len(PRECURSOR_MSGS)))]
                self._precursors_this_round[sid] = tmpl.replace("{sid}", sid)

    def _should_regress_stage(self) -> bool:
        """Stage regression when new blockers appear at advanced stages, or critical sat drop."""
        if self._state.deal_stage not in ("legal_review", "final_approval"):
            return False
        new_blocker = len(self._state.active_blockers) > self._prev_blocker_count
        critical_drop = any(
            self._state.satisfaction.get(s, 1.0) < 0.30 for s in ("CFO", "Legal")
        )
        return new_blocker or critical_drop

    def _regress_stage(self):
        current = self._state.deal_stage
        self._state.deal_stage = STAGE_REGRESSION.get(current, "evaluation")
        self._state.stage_regressions += 1
        self._stage_regressed_this_round = True

    def _maybe_advance_stage(self):
        """
        Issue 4 fix: require STAGE_MIN_ROUNDS before any stage can advance.
        Prevents document-spam instant progression in early rounds.
        """
        if len(self._state.active_blockers) > 0:
            return
        if not self._state.satisfaction:
            return
        min_sat = min(self._state.satisfaction.values())
        current = self._state.deal_stage
        if current not in STAGE_PROGRESSION:
            return
        threshold = STAGE_MIN_SAT.get(current, 0.45)
        min_rounds = STAGE_MIN_ROUNDS.get(current, 2)
        if min_sat >= threshold and self._state.round_number >= min_rounds:
            next_stage = STAGE_PROGRESSION[current]
            self._state.deal_stage = next_stage
            if next_stage == "closed":
                self._state.deal_closed = True

    def _update_active_blockers(self):
        threshold = SCENARIOS[self._state.task_id]["block_threshold"]
        self._state.active_blockers = [
            sid for sid, sat in self._state.satisfaction.items() if sat < threshold
        ]

    def _check_terminal(self) -> Tuple[bool, float]:
        """
        Returns (done, reward).
        Success → (True, CCI score)
        All failure modes → (True, 0.0)
        Still running → (False, 0.0)
        """
        if self._state.deal_failed:
            return True, 0.0
        if self._state.deal_closed or self._state.deal_stage == "closed":
            self._state.deal_closed = True
            return True, CCIGrader.compute(self._state)
        if self._state.round_number >= self._state.max_rounds:
            return True, 0.0
        if (
            len(self._state.active_blockers) >= 3
            and self._state.deal_stage == "evaluation"
        ):
            self._state.deal_failed = True
            self._state.failure_reason = "mass_blocking"
            return True, 0.0
        return False, 0.0

    def _build_observation(
        self, responses: Dict[str, str], hint: Optional[str], is_done: bool
    ) -> DealRoomObservation:
        """
        Build agent observation. Adds delay and noise per POMDP design.
        engagement_level is 1-step delayed with Gaussian noise — never exact satisfaction.
        """
        engagement = {}
        for sid in self._state.satisfaction:
            prev = self._prev_satisfaction.get(sid, 0.5)
            noise = float(self.rng.normal(0, 0.04))
            engagement[sid] = round(float(np.clip(prev + noise, 0.0, 1.0)), 3)

        n_blockers = len(self._state.active_blockers)
        if n_blockers >= 2 or self._stage_regressed_this_round:
            momentum = "critical"
        elif n_blockers == 0 and not self._stage_regressed_this_round:
            momentum = "progressing"
        else:
            momentum = "stalling"

        days = max(
            0,
            SCENARIOS[self._state.task_id]["days_to_deadline"]
            - self._state.round_number * 3,
        )

        return DealRoomObservation(
            round_number=self._state.round_number,
            max_rounds=self._state.max_rounds,
            stakeholder_messages=responses,
            engagement_level=engagement,
            deal_momentum=momentum,
            deal_stage=self._state.deal_stage,
            competitor_events=list(self._active_competitor_events),
            veto_precursors=dict(self._precursors_this_round),
            scenario_hint=hint,
            active_blockers=list(self._state.active_blockers),
            days_to_deadline=days,
            done=is_done,
            info={},
        )

    def _build_round_signals(self, action: DealRoomAction) -> Dict[str, Any]:
        """
        Dense causal signals for RL credit assignment.
        These are metadata in info — NOT part of the reward.
        Reward remains 0.0 until episode end.

        Issue 2 fix: group targets use max delta across expanded targets.
        Issue 5 fix: three-state momentum +1/0/-1.
        """
        targets = expand_targets(action.target)
        if not targets:
            targets = [action.target]
        target_delta = max(
            (self._belief_deltas.get(t, 0.0) for t in targets), default=0.0
        )

        if self._stage_regressed_this_round:
            momentum_dir = -1
        elif self._state.deal_stage != self._prev_stage:
            momentum_dir = 1
        elif len(self._state.active_blockers) < self._prev_blocker_count:
            momentum_dir = 1
        else:
            momentum_dir = 0

        n_advocates = sum(1 for sat in self._state.satisfaction.values() if sat >= 0.65)

        return {
            "new_advocates": n_advocates,
            "new_blockers": len(self._state.active_blockers),
            "momentum_direction": momentum_dir,
            "backchannel_received": action.channel == "backchannel",
            "belief_deltas": dict(self._belief_deltas),
            "target_responded_positively": target_delta > 0.04,
            "stage_changed": self._state.deal_stage != self._prev_stage,
            "stage": self._state.deal_stage,
            "veto_risk_max": max(self._state.veto_risk.values())
            if self._state.veto_risk
            else 0.0,
        }
