from models import DealRoomState
from server.grader import CCIGrader


def make_state(feasible: bool = True) -> DealRoomState:
    return DealRoomState(
        deal_closed=True,
        stakeholders={"finance": {}, "technical": {}},
        stakeholder_private={
            "finance": {
                "trust": 0.8,
                "approval": 0.8,
                "perceived_fit": 0.8,
                "private_resistance": 0.3,
                "mandatory": True,
                "veto_power": True,
                "permanent_marks": [],
            },
            "technical": {
                "trust": 0.75,
                "approval": 0.78,
                "perceived_fit": 0.76,
                "private_resistance": 0.3,
                "mandatory": True,
                "veto_power": False,
                "permanent_marks": [],
            },
        },
        hidden_constraints={"budget_ceiling": {"resolved": feasible}},
        feasibility_state={"is_feasible": feasible, "violations": [] if feasible else ["price"]},
        round_number=4,
        max_rounds=10,
    )


def test_grader_returns_zero_for_infeasible_close():
    assert CCIGrader.compute(make_state(feasible=False)) == 0.0


def test_grader_returns_positive_for_feasible_close():
    assert CCIGrader.compute(make_state(feasible=True)) > 0.0


def test_grader_returns_zero_when_constraint_unresolved():
    state = make_state(feasible=True)
    state.hidden_constraints["budget_ceiling"]["resolved"] = False
    assert CCIGrader.compute(state) == 0.0


def test_grader_penalizes_relationship_damage():
    clean = make_state(feasible=True)
    damaged = make_state(feasible=True)
    damaged.stakeholder_private["finance"]["permanent_marks"] = [
        "premature_close",
        "semantic_contradiction",
    ]
    assert CCIGrader.compute(damaged) < CCIGrader.compute(clean)
