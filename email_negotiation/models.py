from typing import Any, Dict, List, Optional

from openenv.core import Action, Observation, State


class EmailAction(Action):
    intent: str = "address_concern"
    target: str = "Legal"
    tone: str = "formal"
    cc: List[str] = []
    include_document: Optional[str] = None
    concession_term: Optional[str] = None
    concession_value: Optional[float] = None
    urgency: str = "normal"


class EmailObservation(Observation):
    inbox_summary: str = ""
    deal_stage: str = "initial"
    progress_score: float = 0.0
    unresolved_concerns: List[str] = []
    round_number: int = 0
    max_rounds: int = 10
    reward: float = 0.0
    done: bool = False


class EmailState(State):
    episode_id: str = ""
    step_count: int = 0
    scenario_type: str = "aligned"
    progress_score: float = 0.0
    docs_delivered: List[str] = []
    concerns_resolved: int = 0
    concerns_total: int = 0
    terminal_outcome: Optional[str] = None