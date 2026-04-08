import time

from models import DealRoomAction
from server.deal_room_environment import DealRoomEnvironment


def test_reset_performance():
    env = DealRoomEnvironment()
    start = time.perf_counter()
    for idx in range(25):
        env.reset(seed=idx, task_id="aligned")
    assert time.perf_counter() - start < 2.0


def test_step_performance():
    env = DealRoomEnvironment()
    env.reset(seed=42, task_id="aligned")
    target_id = next(iter(env.state.stakeholders))
    action = DealRoomAction(
        action_type="direct_message",
        target=target_id,
        target_ids=[target_id],
        message="Help me understand the real blocker so I can tailor the proposal correctly.",
    )
    start = time.perf_counter()
    for _ in range(25):
        env.step(action)
        if env.state.deal_failed or env.state.deal_closed:
            env.reset(seed=42, task_id="aligned")
    assert time.perf_counter() - start < 2.0
