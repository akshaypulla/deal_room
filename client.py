"""Deal Room Environment client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import DealRoomAction, DealRoomObservation


class DealRoomEnv(EnvClient[DealRoomAction, DealRoomObservation, State]):
    def _step_payload(self, action: DealRoomAction) -> Dict:
        return action.model_dump()

    def _parse_result(self, payload: Dict) -> StepResult[DealRoomObservation]:
        observation = DealRoomObservation(**payload.get("observation", {}))
        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("round_number", 0),
        )
