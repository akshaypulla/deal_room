"""
DealRoomEnvironment V2.5.

Deterministic hybrid enterprise negotiation environment with dynamic stakeholders,
hidden constraints, dense milestone rewards, and a strong terminal grader.
"""

from __future__ import annotations

import json
import uuid
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from models import DealRoomAction, DealRoomObservation, DealRoomState
from server.claims import CommitmentLedger
from server.grader import CCIGrader
from server.scenarios import SCENARIOS, expand_targets, generate_episode
from server.semantics import DEFAULT_ANALYZER
from server.stakeholders import StakeholderEngine, approval_band
from server.validator import OutputValidator

STAGE_ORDER = ["evaluation", "negotiation", "legal_review", "final_approval", "closed"]

CONSTRAINT_DISCOVERY_INTENTS = {
    "budget_ceiling": "discover_budget",
    "delivery_window": "discover_timeline",
    "compliance_addendum": "discover_compliance",
    "supplier_process": "share_vendor_packet",
}

CONSTRAINT_ARTIFACTS = {
    "budget_ceiling": {"roi_model"},
    "delivery_window": {"implementation_timeline"},
    "compliance_addendum": {"dpa", "security_cert"},
    "supplier_process": {"vendor_packet", "support_plan"},
}


class DealRoomEnvironment:
    def __init__(self):
        self._state = DealRoomState()
        self.validator = OutputValidator(mode="strict")
        self.commitment_ledger = CommitmentLedger()
        self.semantic_analyzer = DEFAULT_ANALYZER
        self.stakeholder_engine = StakeholderEngine()
        self.rng = np.random.default_rng()
        self._scenario: Dict[str, Any] = {}
        self._prev_bands: Dict[str, str] = {}
        self._round_weak_signals: Dict[str, List[str]] = {}
        self._round_veto_precursors: Dict[str, str] = {}
        self._last_dense_reward: float = 0.0

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs,
    ) -> DealRoomObservation:
        task_id = kwargs.get("task_id", "aligned")
        if task_id not in SCENARIOS:
            raise ValueError(f"Unknown task_id '{task_id}'. Valid: {list(SCENARIOS)}")

        self.rng = np.random.default_rng(seed)
        self._scenario = generate_episode(task_id, seed)
        self.commitment_ledger.reset()

        stakeholders = deepcopy(self._scenario["stakeholders"])
        stakeholder_private = deepcopy(self._scenario["stakeholder_private"])
        hidden_constraints = deepcopy(self._scenario["hidden_constraints"])
        requested_artifacts = deepcopy(self._scenario["requested_artifacts"])
        approval_caps = deepcopy(self._scenario["approval_caps"])

        self._state = DealRoomState(
            episode_id=episode_id or str(uuid.uuid4())[:8],
            step_count=0,
            task_id=task_id,
            round_number=0,
            max_rounds=int(self._scenario["template"]["max_rounds"]),
            stakeholders=stakeholders,
            stakeholder_private=stakeholder_private,
            hidden_constraints=hidden_constraints,
            relationship_edges=deepcopy(self._scenario["relationship_edges"]),
            commitment_ledger=[],
            deferred_effects=[],
            offer_state={
                "price": None,
                "timeline_weeks": None,
                "security_commitments": None,
                "support_level": None,
                "liability_cap": None,
                "days_to_deadline": self._scenario["days_to_deadline"],
                "event_round": self._scenario["event_round"],
                "event_triggered": False,
            },
            feasibility_state={"is_feasible": False, "violations": ["unresolved_constraints"]},
            active_blockers=[],
            deal_stage="evaluation",
            stage_regressions=0,
            rounds_since_last_contact={stakeholder_id: 0 for stakeholder_id in stakeholders},
            approval_caps=approval_caps,
            semantic_threshold_jitter=deepcopy(self._scenario["semantic_threshold_jitter"]),
            weak_signal_history={stakeholder_id: [] for stakeholder_id in stakeholders},
            requested_artifacts=requested_artifacts,
            discovered_constraints=[],
            milestone_flags={},
            external_events=[],
            validation_failures=0,
            malformed_actions=0,
            last_action_error=None,
            deal_closed=False,
            deal_failed=False,
            failure_reason="",
            final_terms=None,
        )

        self.stakeholder_engine.reset(self._state, self.rng)
        self._update_active_blockers()
        self._prev_bands = self._current_bands()
        self._round_weak_signals = self._collect_initial_signals()
        self._round_veto_precursors = {}
        self._last_dense_reward = 0.0

        opening = self.stakeholder_engine.generate_opening()
        return self._build_observation(opening, is_done=False, stage_direction=0)

    def step(self, action: DealRoomAction) -> Tuple[DealRoomObservation, float, bool, Dict[str, Any]]:
        if self._state.deal_closed or self._state.deal_failed:
            observation = self._build_observation({}, is_done=True, stage_direction=0)
            return observation, 0.0, True, {"error": "episode_already_done"}

        available_targets = list(self._state.stakeholders.keys())
        normalized_payload, confidence = self.validator.validate(
            json.dumps(action.model_dump()),
            available_targets=available_targets,
        )
        normalized = DealRoomAction(
            action_type=str(normalized_payload["action_type"]),
            target=str(normalized_payload["target"]),
            target_ids=list(normalized_payload["target_ids"]),
            message=str(normalized_payload["message"]),
            documents=list(normalized_payload["documents"]),
            proposed_terms=normalized_payload["proposed_terms"],
            channel=str(normalized_payload["channel"]),
            mode=str(normalized_payload["mode"]),
        )

        self._state.last_action_error = normalized_payload.get("error")
        if normalized_payload.get("malformed_action"):
            self._state.validation_failures += 1
            self._state.malformed_actions += 1

        if not normalized.target_ids:
            normalized.target_ids = expand_targets(normalized.target, available_targets)
        if not normalized.target_ids and normalized.target == "all":
            normalized.target_ids = available_targets

        analysis = self.semantic_analyzer.analyze(
            normalized.message,
            {
                "documents": normalized.documents,
                "requested_artifacts": self._state.requested_artifacts,
            },
            {stakeholder_id: payload["role"] for stakeholder_id, payload in self._state.stakeholders.items()},
        )

        ledger_result = self.commitment_ledger.ingest(
            normalized.target_ids,
            list(analysis["claim_candidates"]),
            self._state.semantic_threshold_jitter,
        )
        self._state.commitment_ledger = deepcopy(self.commitment_ledger.claims)

        pre_bands = self._current_bands()
        pre_blockers = set(self._state.active_blockers)

        self._apply_contradictions(ledger_result["contradictions"])
        step_details = self.stakeholder_engine.apply_action(normalized.model_dump(), analysis)
        self._tick_contact_counters(normalized.target_ids)
        self._apply_action_terms(normalized, analysis)
        self._trigger_scenario_event_if_needed()
        self._tick_deferred_effects()

        constraint_result = self._update_constraint_visibility_and_resolution(normalized, analysis)
        self._update_feasibility_state()
        stage_direction = self._advance_or_regress_stage(normalized)
        stage_changed = stage_direction != 0
        self._update_active_blockers()
        self._round_weak_signals = self._build_round_signals(constraint_result)
        self._round_veto_precursors = self._build_veto_precursors()

        dense_reward, reward_breakdown = self._compute_dense_reward(
            normalized_payload.get("malformed_action", False),
            constraint_result,
            pre_bands,
            pre_blockers,
            step_details["satisfied_requests"],
            stage_direction,
        )
        self._last_dense_reward = dense_reward

        early_close = self._apply_close_attempt_penalty(normalized)
        done, terminal_reward = self._check_terminal(normalized)
        reward = terminal_reward if done else (0.0 if early_close else dense_reward)

        responses = self.stakeholder_engine.generate_responses(
            normalized.target_ids if normalized.target_ids else available_targets
        )
        observation = self._build_observation(responses, is_done=done, stage_direction=stage_direction)
        info = {
            "dense_reward_breakdown": reward_breakdown,
            "validation_confidence": confidence,
            "semantic_backend": analysis["backend"],
            "constraint_updates": constraint_result,
            "relationship_effects": step_details["propagation"],
            "active_blockers": list(self._state.active_blockers),
            "feasibility": deepcopy(self._state.feasibility_state),
            "approval_bands": self._current_bands(),
            "last_action_error": self._state.last_action_error,
        }
        observation.info = info

        self._prev_bands = self._current_bands()
        self._state.round_number += 1
        self._state.step_count = self._state.round_number
        return observation, reward, done, info

    @property
    def state(self) -> DealRoomState:
        return self._state

    def _apply_contradictions(self, contradictions: List[Dict[str, object]]):
        for contradiction in contradictions:
            stakeholder_id = contradiction["stakeholder_id"]
            private = self._state.stakeholder_private[stakeholder_id]
            private["trust"] = self._clamp(private["trust"] - 0.10)
            private["perceived_fit"] = self._clamp(private["perceived_fit"] - 0.03)
            self._state.approval_caps[stakeholder_id] = min(
                self._state.approval_caps.get(stakeholder_id, 1.0), 0.58
            )
            private["permanent_marks"].append("semantic_contradiction")
            self._state.deferred_effects.append(
                {
                    "type": "resistance_spike",
                    "stakeholder_id": stakeholder_id,
                    "delay": 1,
                    "delta": 0.06,
                }
            )

    def _tick_contact_counters(self, target_ids: List[str]):
        for stakeholder_id in self._state.rounds_since_last_contact:
            if stakeholder_id in target_ids:
                self._state.rounds_since_last_contact[stakeholder_id] = 0
            else:
                self._state.rounds_since_last_contact[stakeholder_id] += 1

    def _apply_action_terms(self, action: DealRoomAction, analysis: Dict[str, object]):
        if action.proposed_terms:
            for key, value in action.proposed_terms.items():
                if key == "security_commitments" and isinstance(value, list):
                    normalized = ",".join(sorted(str(item).lower() for item in value))
                else:
                    normalized = value
                self._state.offer_state[key] = normalized
        for claim in analysis["claim_candidates"]:
            slot = claim["slot"]
            if slot == "price" and self._state.offer_state.get("price") is None:
                self._state.offer_state["price"] = claim["value"]
            elif slot == "timeline_weeks" and self._state.offer_state.get("timeline_weeks") is None:
                self._state.offer_state["timeline_weeks"] = claim["value"]
            elif slot == "security_posture":
                self._state.offer_state["security_commitments"] = str(claim["value"]).lower()
            elif slot == "support_level":
                self._state.offer_state["support_level"] = str(claim["value"]).lower()
            elif slot == "liability":
                self._state.offer_state["liability_cap"] = str(claim["value"]).lower()

        violations = self._check_term_violations()
        if violations and action.proposed_terms:
            for stakeholder_id, private in self._state.stakeholder_private.items():
                if private["mandatory"] or stakeholder_id in action.target_ids:
                    private["trust"] = self._clamp(private["trust"] - 0.08)
                    private["perceived_fit"] = self._clamp(private["perceived_fit"] - 0.06)
                    private["permanent_marks"].append("infeasible_promise")
                    self._state.deferred_effects.append(
                        {
                            "type": "blocker_check",
                            "stakeholder_id": stakeholder_id,
                            "delay": 2,
                            "delta": 0.05,
                        }
                    )

    def _trigger_scenario_event_if_needed(self):
        event_round = self._state.offer_state.get("event_round")
        if (
            event_round is not None
            and not self._state.offer_state.get("event_triggered")
            and self._state.round_number >= event_round
        ):
            self._state.offer_state["event_triggered"] = True
            self._state.external_events.append("authority_shift")
            if "legal_compliance" in self._state.stakeholder_private:
                private = self._state.stakeholder_private["legal_compliance"]
                private["mandatory"] = True
                private["authority"] = min(1.0, private["authority"] + 0.05)
            if "compliance_addendum" in self._state.hidden_constraints:
                self._state.hidden_constraints["compliance_addendum"]["status"] = "hinted"
                self._state.hidden_constraints["compliance_addendum"]["revealed_by"].append("authority_shift")

    def _tick_deferred_effects(self):
        remaining = []
        for effect in self._state.deferred_effects:
            effect["delay"] -= 1
            if effect["delay"] > 0:
                remaining.append(effect)
                continue
            stakeholder_id = effect["stakeholder_id"]
            private = self._state.stakeholder_private.get(stakeholder_id)
            if not private:
                continue
            if effect["type"] == "resistance_spike":
                private["private_resistance"] = self._clamp(private["private_resistance"] + effect["delta"])
            elif effect["type"] == "blocker_check":
                private["private_resistance"] = self._clamp(private["private_resistance"] + effect["delta"])
                private["approval"] = self._clamp(private["approval"] - 0.03)
        self._state.deferred_effects = remaining

    def _update_constraint_visibility_and_resolution(
        self,
        action: DealRoomAction,
        analysis: Dict[str, object],
    ) -> Dict[str, List[str]]:
        hinted = []
        known = []
        resolved = []
        intent_matches = analysis["intent_matches"]
        artifact_matches = set(analysis["artifact_matches"])

        for constraint_id, constraint in self._state.hidden_constraints.items():
            slot = constraint["slot"]
            discovery_intent = CONSTRAINT_DISCOVERY_INTENTS[constraint_id]
            intent_score = float(intent_matches.get(discovery_intent, 0.0))
            artifact_overlap = bool(artifact_matches & CONSTRAINT_ARTIFACTS[constraint_id])
            threshold = 0.74 + float(self._state.semantic_threshold_jitter.get(slot, 0.0))

            if constraint["status"] == "hidden" and (
                intent_score >= max(0.55, threshold - 0.15) or artifact_overlap
            ):
                constraint["status"] = "hinted"
                constraint["revealed_by"].append("semantic_probe")
                hinted.append(constraint_id)

            if constraint["status"] in {"hidden", "hinted"} and (
                intent_score >= threshold or artifact_overlap
            ):
                constraint["status"] = "known"
                constraint["revealed_by"].append("aligned_follow_up")
                if constraint_id not in self._state.discovered_constraints:
                    self._state.discovered_constraints.append(constraint_id)
                known.append(constraint_id)

            if constraint["status"] == "known" and self._constraint_is_resolved(constraint_id, constraint):
                constraint["resolved"] = True
                resolved.append(constraint_id)
            else:
                constraint["resolved"] = False

        return {"hinted": hinted, "known": known, "resolved": resolved}

    def _constraint_is_resolved(self, constraint_id: str, constraint: Dict[str, Any]) -> bool:
        checker = constraint["checker"]
        if constraint_id == "budget_ceiling":
            price = self._state.offer_state.get("price")
            return price is not None and float(price) <= checker["max_price"]
        if constraint_id == "delivery_window":
            timeline = self._state.offer_state.get("timeline_weeks")
            return timeline is not None and float(timeline) <= checker["max_timeline_weeks"]
        if constraint_id == "compliance_addendum":
            commitments = str(self._state.offer_state.get("security_commitments") or "").lower()
            return checker["must_include"] in commitments
        if constraint_id == "supplier_process":
            support = str(self._state.offer_state.get("support_level") or "").lower()
            return checker["must_include"] in support
        return False

    def _check_term_violations(self) -> List[str]:
        violations = []
        for constraint_id, constraint in self._state.hidden_constraints.items():
            checker = constraint["checker"]
            if constraint_id == "budget_ceiling":
                price = self._state.offer_state.get("price")
                if price is not None and float(price) > checker["max_price"]:
                    violations.append("price")
            elif constraint_id == "delivery_window":
                timeline = self._state.offer_state.get("timeline_weeks")
                if timeline is not None and float(timeline) > checker["max_timeline_weeks"]:
                    violations.append("timeline_weeks")
            elif constraint_id == "compliance_addendum":
                commitments = str(self._state.offer_state.get("security_commitments") or "").lower()
                if commitments and checker["must_include"] not in commitments:
                    violations.append("security_commitments")
            elif constraint_id == "supplier_process":
                support = str(self._state.offer_state.get("support_level") or "").lower()
                if support and checker["must_include"] not in support:
                    violations.append("support_level")
        return violations

    def _update_feasibility_state(self):
        unresolved = [
            constraint_id
            for constraint_id, constraint in self._state.hidden_constraints.items()
            if not constraint.get("resolved")
        ]
        violations = self._check_term_violations()
        if unresolved:
            violations = list(dict.fromkeys(violations + ["unresolved_constraints"]))
        self._state.feasibility_state = {
            "is_feasible": not unresolved and not violations,
            "violations": violations,
        }

    def _advance_or_regress_stage(self, action: DealRoomAction) -> int:
        current_index = STAGE_ORDER.index(self._state.deal_stage)
        mandatory_ids = self._mandatory_stakeholders()
        contacted_mandatory = all(
            bool(self._state.stakeholder_private[stakeholder_id]["band_history"])
            for stakeholder_id in mandatory_ids
        )
        mandatory_workable = all(
            self._state.stakeholder_private[stakeholder_id]["approval"] >= 0.62
            and self._state.stakeholder_private[stakeholder_id]["private_resistance"] <= 0.65
            for stakeholder_id in mandatory_ids
        )
        known_constraints = all(
            constraint["status"] == "known" for constraint in self._state.hidden_constraints.values()
        )
        requested_clear = all(
            not self._state.requested_artifacts.get(stakeholder_id)
            for stakeholder_id in mandatory_ids
        )

        target_stage = self._state.deal_stage
        if self._state.deal_stage == "evaluation" and (contacted_mandatory or self._state.discovered_constraints):
            target_stage = "negotiation"
        elif self._state.deal_stage == "negotiation" and contacted_mandatory and known_constraints:
            target_stage = "legal_review"
        elif self._state.deal_stage == "legal_review" and mandatory_workable and requested_clear:
            target_stage = "closed" if self._can_close(action) else "final_approval"
        elif self._state.deal_stage == "final_approval" and self._can_close(action):
            target_stage = "closed"

        if self._state.deal_stage in {"legal_review", "final_approval"} and self._state.active_blockers:
            regressed_to = STAGE_ORDER[max(0, current_index - 1)]
            if regressed_to != self._state.deal_stage:
                self._state.deal_stage = regressed_to
                self._state.stage_regressions += 1
                return -1

        if target_stage != self._state.deal_stage:
            self._state.deal_stage = target_stage
            if target_stage == "closed":
                self._state.deal_closed = True
            return 1
        return 0

    def _build_round_signals(self, constraint_result: Dict[str, List[str]]) -> Dict[str, List[str]]:
        signals = {stakeholder_id: [] for stakeholder_id in self._state.stakeholders}
        for stakeholder_id, private in self._state.stakeholder_private.items():
            if self._state.rounds_since_last_contact.get(stakeholder_id, 0) >= 2:
                signals[stakeholder_id].append("Replies have become noticeably shorter.")
            if private["private_resistance"] > 0.58:
                signals[stakeholder_id].append("Their tone suggests unresolved internal risk.")
            for constraint_id in constraint_result["hinted"]:
                constraint = self._state.hidden_constraints[constraint_id]
                if stakeholder_id not in signals:
                    signals[stakeholder_id] = []
                signals[stakeholder_id].append(str(constraint["weak_signal"]))
        self._state.weak_signal_history = {
            stakeholder_id: list(dict.fromkeys(history + signals.get(stakeholder_id, [])))
            for stakeholder_id, history in self._state.weak_signal_history.items()
        }
        return {key: value for key, value in signals.items() if value}

    def _build_veto_precursors(self) -> Dict[str, str]:
        precursors = {}
        for stakeholder_id, private in self._state.stakeholder_private.items():
            if not private.get("veto_power"):
                continue
            if private["private_resistance"] > 0.58 or private["approval"] < 0.45:
                precursors[stakeholder_id] = (
                    "Internal follow-up has gone quiet and the approval risk around this stakeholder is rising."
                )
        return precursors

    def _compute_dense_reward(
        self,
        malformed_action: bool,
        constraint_result: Dict[str, List[str]],
        pre_bands: Dict[str, str],
        pre_blockers: set[str],
        satisfied_requests: List[Dict[str, str]],
        stage_direction: int,
    ) -> Tuple[float, Dict[str, float]]:
        if malformed_action:
            return 0.0, {"malformed_action": 0.0}

        breakdown: Dict[str, float] = {}
        for constraint_id in constraint_result["hinted"]:
            key = f"hint:{constraint_id}"
            if not self._state.milestone_flags.get(key):
                self._state.milestone_flags[key] = True
                breakdown[key] = 0.03
        for constraint_id in constraint_result["known"]:
            key = f"known:{constraint_id}"
            if not self._state.milestone_flags.get(key):
                self._state.milestone_flags[key] = True
                breakdown[key] = 0.04

        for item in satisfied_requests:
            key = f"artifact:{item['stakeholder_id']}:{item['artifact']}"
            if not self._state.milestone_flags.get(key):
                self._state.milestone_flags[key] = True
                breakdown[key] = 0.03

        current_bands = self._current_bands()
        for stakeholder_id in self._mandatory_stakeholders():
            if self._band_rank(current_bands[stakeholder_id]) > self._band_rank(pre_bands[stakeholder_id]):
                breakdown[f"band:{stakeholder_id}"] = 0.02

        removed_blockers = pre_blockers - set(self._state.active_blockers)
        if removed_blockers:
            breakdown["removed_blocker"] = 0.02
        if stage_direction > 0 and self._state.deal_stage != "closed":
            breakdown["stage_advance"] = 0.03

        reward = min(0.15, round(sum(breakdown.values()), 4))
        return reward, breakdown

    def _apply_close_attempt_penalty(self, action: DealRoomAction) -> bool:
        if action.action_type not in {"group_proposal", "exec_escalation"}:
            return False
        if self._can_close(action):
            return False
        for stakeholder_id in self._mandatory_stakeholders():
            private = self._state.stakeholder_private[stakeholder_id]
            private["trust"] = self._clamp(private["trust"] - 0.06)
            private["permanent_marks"].append("premature_close")
        return True

    def _can_close(self, action: DealRoomAction) -> bool:
        if action.action_type not in {"group_proposal", "exec_escalation"}:
            return False
        if not self._state.feasibility_state.get("is_feasible", False):
            return False
        for stakeholder_id in self._mandatory_stakeholders():
            private = self._state.stakeholder_private[stakeholder_id]
            if private["approval"] < 0.62 or private["private_resistance"] > 0.65:
                return False
        return not self._state.active_blockers

    def _check_terminal(self, action: DealRoomAction) -> Tuple[bool, float]:
        if self._state.deal_closed:
            self._state.final_terms = deepcopy(self._state.offer_state)
            return True, CCIGrader.compute(self._state)
        if any(
            private.get("veto_power") and private["private_resistance"] > 0.72
            for private in self._state.stakeholder_private.values()
        ):
            self._state.deal_failed = True
            self._state.failure_reason = "silent_veto"
            return True, CCIGrader.MIN_SCORE
        if self._state.round_number >= self._state.max_rounds - 1:
            self._state.deal_failed = True
            self._state.failure_reason = "timeout"
            return True, CCIGrader.MIN_SCORE
        return False, 0.0

    def close(self) -> None:
        """No-op close for inference-script compatibility."""
        return None

    def _mandatory_stakeholders(self) -> List[str]:
        return [
            stakeholder_id
            for stakeholder_id, payload in self._state.stakeholder_private.items()
            if payload.get("mandatory")
        ]

    def _update_active_blockers(self):
        self._state.active_blockers = [
            stakeholder_id
            for stakeholder_id, private in self._state.stakeholder_private.items()
            if approval_band(private["approval"], private["private_resistance"]) == "blocker"
        ]

    def _current_bands(self) -> Dict[str, str]:
        return {
            stakeholder_id: approval_band(private["approval"], private["private_resistance"])
            for stakeholder_id, private in self._state.stakeholder_private.items()
        }

    def _build_observation(
        self,
        stakeholder_messages: Dict[str, str],
        is_done: bool,
        stage_direction: int,
    ) -> DealRoomObservation:
        engagement = {}
        approval_progress = {}
        stakeholders = {}
        for stakeholder_id, payload in self._state.stakeholders.items():
            private = self._state.stakeholder_private[stakeholder_id]
            approval = private["approval"]
            engagement[stakeholder_id] = round(
                float(np.clip(approval + self.rng.normal(0.0, 0.025), 0.0, 1.0)),
                3,
            )
            band = approval_band(private["approval"], private["private_resistance"])
            approval_progress[stakeholder_id] = {
                "band": band,
                "mandatory": private["mandatory"],
                "authority": payload["authority"],
            }
            stakeholders[stakeholder_id] = {
                "display_name": payload["display_name"],
                "role": payload["role"],
                "mandatory": private["mandatory"],
                "authority": payload["authority"],
            }

        known_constraints = [
            {
                "id": constraint_id,
                "label": constraint["label"],
                "required_artifact": constraint["required_artifact"],
                "status": constraint["status"],
            }
            for constraint_id, constraint in self._state.hidden_constraints.items()
            if constraint["status"] == "known"
        ]
        if len(self._state.active_blockers) >= 2:
            momentum = "critical"
        elif stage_direction < 0:
            momentum = "critical"
        elif stage_direction > 0 or self._last_dense_reward >= 0.04:
            momentum = "progressing"
        else:
            momentum = "stalling"

        scenario_hint = None
        if "authority_shift" in self._state.external_events:
            scenario_hint = "Authority shifted mid-process and compliance scrutiny has increased."

        return DealRoomObservation(
            round_number=self._state.round_number,
            max_rounds=self._state.max_rounds,
            stakeholders=stakeholders,
            stakeholder_messages=stakeholder_messages,
            engagement_level=engagement,
            weak_signals=deepcopy(self._round_weak_signals),
            known_constraints=known_constraints,
            requested_artifacts=deepcopy(self._state.requested_artifacts),
            approval_path_progress=approval_progress,
            deal_momentum=momentum,
            deal_stage=self._state.deal_stage,
            competitor_events=deepcopy(self._state.external_events),
            veto_precursors=deepcopy(self._round_veto_precursors),
            scenario_hint=scenario_hint,
            active_blockers=list(self._state.active_blockers),
            days_to_deadline=max(
                0,
                int(self._scenario["days_to_deadline"]) - (self._state.round_number * 3),
            ),
            done=is_done,
            info={},
        )

    def _collect_initial_signals(self) -> Dict[str, List[str]]:
        signals = {}
        for constraint in self._state.hidden_constraints.values():
            if self._scenario["observability"] == "high":
                stakeholder_id = self._mandatory_stakeholders()[0]
                signals.setdefault(stakeholder_id, []).append(str(constraint["weak_signal"]))
                constraint["status"] = "hinted"
        return signals

    @staticmethod
    def _band_rank(band: str) -> int:
        return {"blocker": 0, "neutral": 1, "workable": 2, "supporter": 3}[band]

    @staticmethod
    def _clamp(value: float) -> float:
        return round(float(np.clip(value, 0.0, 1.0)), 4)
