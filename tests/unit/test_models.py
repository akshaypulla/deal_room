from models import DealRoomAction, DealRoomObservation, DealRoomReward, DealRoomState


def test_action_target_ids_are_deduplicated():
    action = DealRoomAction(target_ids=["finance", "finance", "technical"])
    assert action.target_ids == ["finance", "technical"]


def test_state_is_callable_for_state_and_state_method_compat():
    state = DealRoomState()
    assert state() is state


def test_observation_supports_dynamic_fields():
    observation = DealRoomObservation(
        stakeholders={"finance": {"role": "finance"}},
        weak_signals={"finance": ["Board risk is rising."]},
        known_constraints=[{"id": "budget_ceiling"}],
        requested_artifacts={"finance": ["roi_model"]},
        approval_path_progress={"finance": {"band": "neutral"}},
    )
    assert "finance" in observation.stakeholders
    assert observation.known_constraints[0]["id"] == "budget_ceiling"


def test_reward_model_tracks_value_done_and_info():
    reward = DealRoomReward(value=0.12, done=False, info={"milestone": "hint:budget_ceiling"})
    assert reward.value == 0.12
    assert reward.done is False
    assert reward.info["milestone"] == "hint:budget_ceiling"
