import random
import uuid
from typing import Any, Dict, List, Optional

from .email_message import EmailMessage, EmailAttachment
from .inbox import EmailInbox
from .buyer_stakeholder import StatefulArchetypeAgent, ARCHETYPE_PROFILES
from .reward_extractor import RewardExtractor, compute_terminal_reward
from .progress_score import (
    compute_progress_score,
    compute_shaping_reward,
    check_early_stopping,
)
from .anti_gaming import AntiGamingValidator


SCENARIOS = {
    "aligned": {
        "stakeholders": ["Legal", "Finance", "CTO"],
        "initial_concerns": ["liability"],
        "docs_required": ["DPA"],
        "initial_messages": {
            "Legal": "We need to review the liability terms in the agreement.",
            "Finance": "Can you provide ROI analysis for this investment?",
            "CTO": "What integration approach are you proposing?",
        },
    },
    "conflicted": {
        "stakeholders": ["Legal", "Finance", "CTO", "Procurement"],
        "initial_concerns": ["liability", "pricing", "integration"],
        "docs_required": ["DPA", "roi_model"],
        "initial_messages": {
            "Legal": "We have significant concerns about the liability limitations.",
            "Finance": "The pricing structure needs to be justified.",
            "CTO": "Integration complexity is a major concern for our team.",
            "Procurement": "We need to compare this against other vendors.",
        },
    },
    "hostile_acquisition": {
        "stakeholders": ["Legal", "Finance", "CTO", "ExecSponsor"],
        "initial_concerns": ["liability", "pricing", "cultural_fit"],
        "docs_required": ["DPA", "roi_model", "security_cert"],
        "initial_messages": {
            "Legal": "This acquisition timeline is unrealistic given the compliance requirements.",
            "Finance": "The valuation does not reflect our actual worth.",
            "CTO": "Technical integration of these two companies will be extremely challenging.",
            "ExecSponsor": "We are not interested in this acquisition under any terms.",
        },
    },
}

DEAL_STAGE_ORDER = ["initial", "qualification", "discovery", "proposal", "negotiation", "closing", "closed"]
MAX_ROUNDS = 10


class EmailNegotiationCore:
    def __init__(self, scenario_type: str = "aligned", use_buyer_llm: bool = False):
        self.scenario_type = scenario_type
        self.use_buyer_llm = use_buyer_llm
        self._episode_id = ""
        self._step_count = 0
        self._inbox: Optional[EmailInbox] = None
        self._agents: Dict[str, StatefulArchetypeAgent] = {}
        self._validator = AntiGamingValidator()
        self._reward_extractor = RewardExtractor()
        self._previous_progress = 0.0
        self._progress_history: List[float] = []
        self._scenario: Optional[Dict] = None
        self._docs_delivered: List[str] = []
        self._concerns_resolved = 0
        self._deal_stage = "initial"
        self._terminal_outcome: Optional[str] = None
        self._action_history: List[Dict] = []

    def reset(self) -> Dict[str, Any]:
        self._step_count = 0
        self._previous_progress = 0.0
        self._progress_history = []
        self._docs_delivered = []
        self._concerns_resolved = 0
        self._deal_stage = "initial"
        self._terminal_outcome = None
        self._action_history = []
        self._episode_id = str(uuid.uuid4())
        self._validator.reset()

        self._scenario = SCENARIOS.get(self.scenario_type, SCENARIOS["aligned"])

        self._inbox = EmailInbox()
        self._agents = {}
        for sid in self._scenario["stakeholders"]:
            profile = ARCHETYPE_PROFILES.get(sid, {"archetype": "general", "weight": 0.1, "primary_topics": [], "alignment_base": 0.5})
            agent = StatefulArchetypeAgent.from_archetype(sid, profile)
            agent.current_concerns = list(self._scenario["initial_concerns"])
            self._agents[sid] = agent
            self._inbox.add_stakeholder(sid, initial_concerns=list(self._scenario["initial_concerns"]))

        progress = self._compute_progress()
        return {
            "inbox_summary": self._inbox.get_full_summary(
                progress, self._deal_stage, self._scenario["initial_concerns"]
            ),
            "deal_stage": self._deal_stage,
            "progress_score": progress,
            "unresolved_concerns": list(self._scenario["initial_concerns"]),
        }

    def step(self, action_dict: Dict[str, Any]) -> Dict[str, Any]:
        self._step_count += 1

        valid, reason = self._validator.validate_action(action_dict)
        if not valid:
            self._action_history.append(action_dict)
            return self._make_obs(
                reward=-0.1,
                done=False,
                reward_breakdown={"validation_failure": -0.1, "reason": reason},
            )

        target = action_dict["target"]
        self._action_history.append(action_dict)

        if action_dict.get("include_document"):
            self._docs_delivered.append(action_dict["include_document"])

        response_data = self._generate_stakeholder_response(target, action_dict)

        self._add_email_to_inbox(target, action_dict, response_data)

        step_reward, breakdown = self._compute_reward(action_dict, response_data)

        self._update_agents(target, response_data, action_dict)

        progress = self._compute_progress()
        shaping = compute_shaping_reward(progress, self._previous_progress)
        self._previous_progress = progress
        self._progress_history.append(progress)

        self._advance_deal_stage(response_data)

        terminal = self._check_terminal(response_data, step_count=self._step_count)
        if terminal:
            terminal_reward = compute_terminal_reward(terminal)
            step_reward += terminal_reward
            self._terminal_outcome = terminal

        step_reward += shaping

        done = (
            self._terminal_outcome is not None
            or self._step_count >= MAX_ROUNDS
            or check_early_stopping(self._progress_history)
        )

        return self._make_obs(reward=step_reward, done=done, reward_breakdown=breakdown)

    def get_state(self) -> Dict[str, Any]:
        alignment = {sid: agent.alignment_score for sid, agent in self._agents.items()}
        return {
            "episode_id": self._episode_id,
            "scenario_type": self.scenario_type,
            "progress_score": self._previous_progress,
            "docs_delivered": self._docs_delivered,
            "concerns_resolved": self._concerns_resolved,
            "concerns_total": len(self._scenario["initial_concerns"]) if self._scenario else 0,
            "terminal_outcome": self._terminal_outcome,
            "deal_stage": self._deal_stage,
            "alignment_scores": alignment,
        }

    def _generate_stakeholder_response(self, target: str, action_dict: Dict) -> Dict[str, Any]:
        if target not in self._agents:
            return {"reply": "", "concerns_raised": [], "sentiment": "neutral", "terms_mentioned": [], "escalation_detected": False}

        agent = self._agents[target]
        inbox_msgs = [m.body for m in self._inbox.all_messages[-5:]] if self._inbox else []
        return agent.generate_email_response(action_dict, inbox_msgs)

    def _add_email_to_inbox(self, target: str, action_dict: Dict, response_data: Dict) -> None:
        if not self._inbox or target not in self._inbox.stakeholder_inboxes:
            return
        subject_map = {
            "address_concern": "Question about your requirements",
            "offer_document": f"Document: {action_dict.get('include_document', 'Materials')}",
            "make_concession": "Proposed adjustment to terms",
            "escalate_to_exec": "Request for executive engagement",
            "group_proposal": "Comprehensive proposal",
            "walkaway": "Closing the conversation",
        }
        subject = subject_map.get(action_dict.get("intent", ""), "Email from seller")
        body = response_data.get("reply", "")
        msg = EmailMessage(
            from_addr="seller@company.com",
            to_addr=target,
            cc=action_dict.get("cc", []),
            subject=subject,
            body=body,
            document_type=action_dict.get("include_document"),
            attachments=[action_dict["include_document"]] if action_dict.get("include_document") else [],
        )
        self._inbox.deliver_email(msg)

    def _compute_reward(self, action_dict: Dict, response_data: Dict) -> tuple:
        raw_reward, breakdown = self._reward_extractor.extract(response_data, action_dict)
        clipped = self._validator.clip_reward(raw_reward, "dense")
        cta_penalty, _ = self._validator.check_cta_no_response_penalty(
            action_dict, response_data, self._step_count
        )
        total = clipped + cta_penalty
        breakdown["cta_penalty"] = cta_penalty
        return total, breakdown

    def _update_agents(self, target: str, response_data: Dict, action_dict: Dict) -> None:
        if target in self._agents:
            agent = self._agents[target]
            delta = response_data.get("alignment_delta", 0.0)
            agent.alignment_score = max(0.0, min(1.0, agent.alignment_score + delta))
            if response_data.get("concerns_raised"):
                agent.current_concerns.extend(response_data["concerns_raised"])
            if response_data["sentiment"] == "positive" and not response_data["concerns_raised"]:
                self._concerns_resolved += 1
            agent.update_memory(f"Step {self._step_count}: {response_data['sentiment']} response")

    def _compute_progress(self) -> float:
        if not self._agents:
            return 0.0
        alignment = {sid: agent.alignment_score for sid, agent in self._agents.items()}
        docs_required = self._scenario.get("docs_required", []) if self._scenario else []
        concerns_total = len(self._scenario["initial_concerns"]) if self._scenario else 0
        return compute_progress_score(
            alignment_scores=alignment,
            docs_delivered=self._docs_delivered,
            docs_required=docs_required,
            deal_stage=self._deal_stage,
            concerns_resolved=self._concerns_resolved,
            concerns_total=concerns_total,
        )

    def _advance_deal_stage(self, response_data: Dict) -> None:
        sentiment = response_data.get("sentiment", "neutral")
        alignment_avg = sum(a.alignment_score for a in self._agents.values()) / len(self._agents)
        progress = self._previous_progress

        if self._deal_stage == "initial" and self._step_count >= 1:
            self._deal_stage = "qualification"
        elif self._deal_stage == "qualification" and alignment_avg > 0.55:
            self._deal_stage = "discovery"
        elif self._deal_stage == "discovery" and progress > 0.3:
            self._deal_stage = "proposal"
        elif self._deal_stage == "proposal" and progress > 0.5:
            self._deal_stage = "negotiation"
        elif self._deal_stage == "negotiation" and progress > 0.75:
            self._deal_stage = "closing"
        elif self._deal_stage == "closing" and self._concerns_resolved >= len(self._scenario.get("initial_concerns", [])):
            self._deal_stage = "closed"

    def _check_terminal(self, response_data: Dict, step_count: int) -> Optional[str]:
        alignment_avg = sum(a.alignment_score for a in self._agents.values()) / len(self._agents)
        if self._deal_stage == "closed":
            return "deal_closed"
        if any(a.escalation_detected for a in self._agents.values()):
            return "veto"
        if response_data.get("escalation_detected"):
            return "veto"
        if alignment_avg < 0.15:
            return "veto"
        if step_count >= MAX_ROUNDS:
            return "max_rounds"
        return None

    def _make_obs(
        self,
        reward: float,
        done: bool,
        reward_breakdown: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        progress = self._compute_progress()
        unresolved = []
        for agent in self._agents.values():
            unresolved.extend(agent.current_concerns)
        if self._scenario:
            unresolved = [c for c in unresolved if c in self._scenario.get("initial_concerns", [])]

        return {
            "inbox_summary": self._inbox.get_full_summary(progress, self._deal_stage, unresolved) if self._inbox else "",
            "deal_stage": self._deal_stage,
            "progress_score": progress,
            "unresolved_concerns": list(set(unresolved)),
            "reward": reward,
            "done": done,
            "reward_breakdown": reward_breakdown or {},
        }

    def close(self) -> None:
        """No-op for API compatibility with DealRoomTextEnv.close()."""
        pass
