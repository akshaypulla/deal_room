import uuid
import random
from typing import Optional

from openenv.core import Environment

from models import EmailAction, EmailObservation, EmailState


class EmailNegotiationEnvironment(Environment):
    """
    OpenEnv wrapper around EmailNegotiationCore.
    Maintains state across HTTP requests (reset → step → state cycle).
    """

    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(self):
        self._core: Optional[object] = None
        self._episode_id: str = ""
        self._step_count: int = 0
        self._scenario_type: str = "aligned"
        self._progress_history: list = []

    def _init_core(self) -> None:
        from server.email_env.email_negotiation import EmailNegotiationCore
        scenario = random.choice(["aligned", "aligned", "conflicted", "hostile_acquisition"])
        self._scenario_type = scenario
        self._core = EmailNegotiationCore(
            scenario_type=scenario,
            use_buyer_llm=False,
        )
        self._core.reset()
        self._episode_id = str(uuid.uuid4())
        self._step_count = 0
        self._progress_history = []

    def reset(self) -> EmailObservation:
        self._init_core()
        initial_obs = self._core.reset()
        return EmailObservation(
            inbox_summary=initial_obs["inbox_summary"],
            deal_stage=initial_obs["deal_stage"],
            progress_score=initial_obs["progress_score"],
            unresolved_concerns=initial_obs["unresolved_concerns"],
            round_number=0,
            max_rounds=10,
            reward=0.0,
            done=False,
        )

    def step(self, action: EmailAction, timeout_s: Optional[float] = None) -> EmailObservation:
        if self._core is None:
            self._init_core()
        self._step_count += 1

        action_dict = {
            "intent": action.intent,
            "target": action.target,
            "tone": action.tone,
            "cc": action.cc,
            "include_document": action.include_document,
            "concession_term": action.concession_term,
            "concession_value": action.concession_value,
            "urgency": action.urgency,
        }

        result = self._core.step(action_dict)

        return EmailObservation(
            inbox_summary=result["inbox_summary"],
            deal_stage=result["deal_stage"],
            progress_score=result["progress_score"],
            unresolved_concerns=result["unresolved_concerns"],
            round_number=self._step_count,
            max_rounds=10,
            reward=result["reward"],
            done=result["done"],
        )

    @property
    def state(self) -> EmailState:
        if self._core is None:
            return EmailState(
                episode_id=self._episode_id or str(uuid.uuid4()),
                step_count=self._step_count,
                scenario_type=self._scenario_type,
            )
        core_state = self._core.get_state()
        return EmailState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            scenario_type=core_state.get("scenario_type", self._scenario_type),
            progress_score=core_state.get("progress_score", 0.0),
            docs_delivered=core_state.get("docs_delivered", []),
            concerns_resolved=core_state.get("concerns_resolved", 0),
            concerns_total=core_state.get("concerns_total", 0),
            terminal_outcome=core_state.get("terminal_outcome"),
        )

    def close(self) -> None:
        pass