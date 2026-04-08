from models import DealRoomAction
from server.deal_room_environment import DealRoomEnvironment
from server.grader import CCIGrader


def test_reset_creates_dynamic_roster(env: DealRoomEnvironment):
    obs = env.reset(seed=42, task_id="aligned")
    assert 2 <= len(obs.stakeholders) <= 4
    assert obs.deal_stage == "evaluation"


def test_step_returns_dense_reward_in_bounds(aligned_env: DealRoomEnvironment):
    target_id = next(iter(aligned_env.state.stakeholders))
    obs, reward, done, info = aligned_env.step(
        DealRoomAction(
            action_type="direct_message",
            target=target_id,
            target_ids=[target_id],
            message="Help me understand the actual internal approval constraint so I can tailor the proposal.",
        )
    )
    assert 0.0 <= reward <= 0.15
    assert done is False
    assert "dense_reward_breakdown" in info


def test_constraint_can_be_discovered_with_probe_and_artifact(aligned_env: DealRoomEnvironment):
    target_id = next(iter(aligned_env.state.stakeholders))
    aligned_env.step(
        DealRoomAction(
            action_type="direct_message",
            target=target_id,
            target_ids=[target_id],
            message="What budget ceiling or board risk do we need to respect here?",
        )
    )
    obs, _, _, info = aligned_env.step(
        DealRoomAction(
            action_type="send_document",
            target=target_id,
            target_ids=[target_id],
            message="Here is the ROI model with exact payback assumptions.",
            documents=[{"type": "roi_model", "specificity": "high"}],
        )
    )
    discovered = info["constraint_updates"]["hinted"] + info["constraint_updates"]["known"]
    assert discovered
    assert obs.known_constraints or obs.weak_signals


def test_premature_close_applies_penalty(aligned_env: DealRoomEnvironment):
    _, reward, done, _ = aligned_env.step(
        DealRoomAction(
            action_type="group_proposal",
            target="all",
            message="We should sign now.",
        )
    )
    mandatory = [
        payload
        for payload in aligned_env.state.stakeholder_private.values()
        if payload["mandatory"]
    ]
    assert reward == 0.0
    assert done is False
    assert any("premature_close" in payload["permanent_marks"] for payload in mandatory)


def test_feasible_close_returns_terminal_score(env: DealRoomEnvironment):
    env.reset(seed=42, task_id="aligned")
    for stakeholder_id, payload in env.state.stakeholder_private.items():
        payload["approval"] = 0.75
        payload["private_resistance"] = 0.30
        payload["trust"] = 0.80
    for constraint in env.state.hidden_constraints.values():
        constraint["status"] = "known"
        constraint["resolved"] = True
    env.state.requested_artifacts = {stakeholder_id: [] for stakeholder_id in env.state.stakeholders}
    env.state.feasibility_state = {"is_feasible": True, "violations": []}
    env.state.deal_stage = "final_approval"
    obs, reward, done, _ = env.step(
        DealRoomAction(
            action_type="group_proposal",
            target="all",
            message="I believe we are ready to move to final approval.",
            proposed_terms={
                "price": 180000,
                "timeline_weeks": 14,
                "security_commitments": ["gdpr"],
                "support_level": "named_support_lead",
                "liability_cap": "mutual_cap",
            },
        )
    )
    assert done is True
    assert 0.0 < reward < 1.0
    assert obs.done is True


def test_feasible_close_on_final_round_scores_instead_of_timing_out(env: DealRoomEnvironment):
    env.reset(seed=42, task_id="aligned")
    for stakeholder_id, payload in env.state.stakeholder_private.items():
        payload["approval"] = 0.75
        payload["private_resistance"] = 0.30
        payload["trust"] = 0.80
    for constraint in env.state.hidden_constraints.values():
        constraint["status"] = "known"
        constraint["resolved"] = True
    env.state.requested_artifacts = {stakeholder_id: [] for stakeholder_id in env.state.stakeholders}
    env.state.feasibility_state = {"is_feasible": True, "violations": []}
    env.state.deal_stage = "final_approval"
    env.state.round_number = env.state.max_rounds - 1
    obs, reward, done, _ = env.step(
        DealRoomAction(
            action_type="group_proposal",
            target="all",
            message="I believe we are ready to move to final approval.",
            proposed_terms={
                "price": 180000,
                "timeline_weeks": 14,
                "security_commitments": ["gdpr"],
                "support_level": "named_support_lead",
                "liability_cap": "mutual_cap",
            },
        )
    )
    assert done is True
    assert 0.0 < reward < 1.0
    assert obs.done is True


def test_legal_review_can_close_directly_on_final_round_when_ready(env: DealRoomEnvironment):
    env.reset(seed=42, task_id="hostile_acquisition")
    for stakeholder_id, payload in env.state.stakeholder_private.items():
        payload["approval"] = 0.75
        payload["private_resistance"] = 0.30
        payload["trust"] = 0.80
    for constraint in env.state.hidden_constraints.values():
        constraint["status"] = "known"
        constraint["resolved"] = True
    env.state.requested_artifacts = {stakeholder_id: [] for stakeholder_id in env.state.stakeholders}
    env.state.feasibility_state = {"is_feasible": True, "violations": []}
    env.state.active_blockers = []
    env.state.offer_state["event_triggered"] = True
    env.state.deal_stage = "legal_review"
    env.state.round_number = env.state.max_rounds - 1
    obs, reward, done, _ = env.step(
        DealRoomAction(
            action_type="group_proposal",
            target="all",
            message="All blockers are cleared and the terms are feasible; I recommend final approval now.",
            proposed_terms={
                "price": 180000,
                "timeline_weeks": 14,
                "security_commitments": ["gdpr"],
                "support_level": "named_support_lead",
                "liability_cap": "mutual_cap",
            },
        )
    )
    assert done is True
    assert 0.0 < reward < 1.0
    assert obs.done is True


def test_timeout_terminal_score_stays_inside_open_interval(env: DealRoomEnvironment):
    env.reset(seed=42, task_id="aligned")
    env.state.round_number = env.state.max_rounds - 1

    obs, reward, done, _ = env.step(
        DealRoomAction(
            action_type="direct_message",
            target=next(iter(env.state.stakeholders)),
            message="Checking whether timeout terminal scoring stays inside the open interval.",
        )
    )

    assert done is True
    assert reward == CCIGrader.MIN_SCORE
    assert obs.done is True
