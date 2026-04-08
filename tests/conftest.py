from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import DealRoomAction
from server.deal_room_environment import DealRoomEnvironment


@pytest.fixture
def env() -> DealRoomEnvironment:
    return DealRoomEnvironment()


@pytest.fixture
def aligned_env(env: DealRoomEnvironment) -> DealRoomEnvironment:
    env.reset(seed=42, task_id="aligned")
    return env


@pytest.fixture
def simple_action() -> DealRoomAction:
    return DealRoomAction(
        action_type="direct_message",
        target="all",
        message="I want to understand the real blocker so I can tailor the proposal responsibly.",
    )
