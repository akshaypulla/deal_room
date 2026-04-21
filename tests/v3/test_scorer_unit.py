#!/usr/bin/env python3
"""
Unit tests for world-state-based deterministic utterance scorer.
Tests scoring functions directly without HTTP API.
"""

import sys

sys.path.insert(0, "/app/env")

import numpy as np
from deal_room.rewards.utterance_scorer import (
    UtteranceScorer,
    UtteranceScore,
    compute_prediction_accuracy,
    LOG2_6,
    LOOKAHEAD_COST,
)
from deal_room.committee.causal_graph import BeliefDistribution
from deal_room.stakeholders.cvar_preferences import StakeholderRiskProfile


def make_belief(
    competent=0.2,
    incompetent=0.2,
    trustworthy=0.2,
    deceptive=0.2,
    aligned=0.1,
    misaligned=0.1,
):
    return BeliefDistribution(
        distribution={
            "competent": competent,
            "incompetent": incompetent,
            "trustworthy": trustworthy,
            "deceptive": deceptive,
            "aligned": aligned,
            "misaligned": misaligned,
        },
        stakeholder_role="test",
        confidence=1.0,
    )


def make_profile(lambda_risk=0.5, alpha=0.2, tau=0.8):
    return StakeholderRiskProfile(
        stakeholder_id="test",
        role="test",
        alpha=alpha,
        tau=tau,
        lambda_risk=lambda_risk,
    )


class MockGraph:
    def __init__(self):
        self.nodes = ["A", "B", "C"]
        self.edges = [("A", "B"), ("B", "C")]
        self.authority_weights = {"A": 0.4, "B": 0.3, "C": 0.3}

    def get_outgoing(self, stakeholder):
        result = {}
        for src, dst in self.edges:
            if src == stakeholder:
                result[dst] = 1.0
        return result

    def get_influencers(self, stakeholder):
        result = {}
        for src, dst in self.edges:
            if dst == stakeholder:
                result[src] = 1.0
        return result


def test_lookahead_cost_exactly_0_07():
    assert LOOKAHEAD_COST == 0.07
    print("  ✓ LOOKAHEAD_COST is exactly 0.07")


def test_log2_6_correct():
    expected = np.log2(6)
    assert abs(LOG2_6 - expected) < 1e-6
    print(f"  ✓ LOG2_6 = {LOG2_6:.4f} (correct)")


def test_lookahead_penalty_applied():
    b_before = {"Legal": make_belief(competent=0.3)}
    b_after = {"Legal": make_belief(competent=0.35)}

    scorer = UtteranceScorer()
    score_no_look = scorer.score(
        beliefs_before=b_before,
        beliefs_after=b_after,
        graph=MockGraph(),
        lookahead_used=False,
    )
    score_look = scorer.score(
        beliefs_before=b_before,
        beliefs_after=b_after,
        graph=MockGraph(),
        lookahead_used=True,
    )

    assert score_look.lookahead_used == True
    assert score_look.goal < score_no_look.goal, (
        f"lookahead goal ({score_look.goal}) should be lower than no-look ({score_no_look.goal})"
    )
    diff = score_no_look.goal - score_look.goal
    assert abs(diff - LOOKAHEAD_COST) < 0.001, (
        f"penalty={diff}, expected {LOOKAHEAD_COST}"
    )
    print(f"  ✓ lookahead penalty applied: {diff:.4f} (expected {LOOKAHEAD_COST})")


def test_all_dimensions_bounded_0_1():
    b_before = {"Legal": make_belief(0.3)}
    b_after = {"Legal": make_belief(0.35)}
    graph = MockGraph()
    scorer = UtteranceScorer()

    for _ in range(20):
        score = scorer.score(
            beliefs_before=b_before,
            beliefs_after=b_after,
            graph=graph,
            targeted_ids=["Legal"],
            risk_profiles={"Legal": make_profile(lambda_risk=0.6)},
            deal_terms={"price": 100000, "timeline_weeks": 12},
            authority_weights={"Legal": 0.5},
        )
        for dim in ["goal", "trust", "information", "risk", "causal"]:
            val = getattr(score, dim)
            assert 0.0 <= val <= 1.0, f"{dim}={val} outside [0,1]"
    print("  ✓ all dimensions bounded [0, 1]")


def test_goal_score_increases_with_approval():
    b_before = {
        "Legal": make_belief(competent=0.2, trustworthy=0.2, aligned=0.1),
        "Finance": make_belief(competent=0.2, trustworthy=0.2, aligned=0.1),
    }
    b_after_positive = {
        "Legal": make_belief(competent=0.3, trustworthy=0.3, aligned=0.15),
        "Finance": make_belief(competent=0.3, trustworthy=0.3, aligned=0.15),
    }
    b_after_negative = {
        "Legal": make_belief(competent=0.1, trustworthy=0.1, aligned=0.05),
        "Finance": make_belief(competent=0.1, trustworthy=0.1, aligned=0.05),
    }

    scorer = UtteranceScorer()
    s_pos = scorer.score(
        b_before,
        b_after_positive,
        graph=MockGraph(),
        authority_weights={"Legal": 0.5, "Finance": 0.5},
    )
    s_neg = scorer.score(
        b_before,
        b_after_negative,
        graph=MockGraph(),
        authority_weights={"Legal": 0.5, "Finance": 0.5},
    )

    assert s_pos.goal > s_neg.goal, (
        f"positive approval goal ({s_pos.goal}) should exceed negative ({s_neg.goal})"
    )
    assert s_pos.goal > 0.5 > s_neg.goal, "positive should be >0.5, negative <0.5"
    print(f"  ✓ goal: positive={s_pos.goal:.3f} > negative={s_neg.goal:.3f}")


def test_trust_targeted_delta():
    b_before = {
        "Legal": make_belief(trustworthy=0.2),
        "Finance": make_belief(trustworthy=0.2),
    }
    b_after_targeted = {
        "Legal": make_belief(trustworthy=0.4),
        "Finance": make_belief(trustworthy=0.2),
    }
    b_after_untargeted = {
        "Legal": make_belief(trustworthy=0.2),
        "Finance": make_belief(trustworthy=0.4),
    }

    scorer = UtteranceScorer()
    s = scorer.score(
        b_before, b_after_targeted, graph=MockGraph(), targeted_ids=["Legal"]
    )
    assert s.trust > 0.5, f"targeted trust should increase, got {s.trust}"

    s2 = scorer.score(
        b_before, b_after_untargeted, graph=MockGraph(), targeted_ids=["Legal"]
    )
    assert s2.trust == 0.5, f"untargeted stakeholding should get 0.5, got {s2.trust}"
    print(
        f"  ✓ trust for targeted stakeholder increased: {s.trust:.3f} (untargeted: {s2.trust})"
    )


def test_information_entropy_reduction():
    uniform = make_belief(
        competent=1 / 6,
        incompetent=1 / 6,
        trustworthy=1 / 6,
        deceptive=1 / 6,
        aligned=1 / 6,
        misaligned=1 / 6,
    )
    moderately_certain = make_belief(
        competent=0.35,
        trustworthy=0.25,
        aligned=0.15,
        incompetent=0.1,
        deceptive=0.1,
        misaligned=0.05,
    )
    highly_certain = make_belief(
        competent=0.5,
        trustworthy=0.2,
        aligned=0.1,
        incompetent=0.1,
        deceptive=0.05,
        misaligned=0.05,
    )

    b_before_uniform = {"Legal": uniform}
    b_after_moderate = {"Legal": moderately_certain}
    b_after_high = {"Legal": highly_certain}

    scorer = UtteranceScorer()
    s_moderate = scorer.score(b_before_uniform, b_after_moderate, graph=MockGraph())
    s_high = scorer.score(b_before_uniform, b_after_high, graph=MockGraph())

    assert s_moderate.information > 0.5, (
        f"entropy reduction should give >0.5, got {s_moderate.information}"
    )
    assert s_high.information >= s_moderate.information, (
        f"more entropy reduction should give >= info: high={s_high.information:.3f} >= moderate={s_moderate.information:.3f}"
    )
    print(
        f"  ✓ info: moderate_reduction={s_moderate.information:.3f}, high_reduction={s_high.information:.3f}"
    )


def test_determinism_consistency():
    b_before = {"Legal": make_belief(competent=0.3)}
    b_after = {"Legal": make_belief(competent=0.35)}
    scorer = UtteranceScorer()

    scores = []
    for _ in range(10):
        s = scorer.score(b_before, b_after, graph=MockGraph(), targeted_ids=["Legal"])
        scores.append((s.goal, s.trust, s.information, s.risk, s.causal))

    for i in range(1, len(scores)):
        assert scores[i] == scores[0], (
            f"Scoring non-deterministic: {scores[i]} vs {scores[0]}"
        )
    print(f"  ✓ scoring is deterministic across 10 runs")


def test_prediction_accuracy():
    pred = {"Legal": "send_document", "Finance": "send_document"}
    actual = {"Legal": "send_document", "Finance": "request_info"}
    acc = compute_prediction_accuracy(pred, actual)
    assert 0.0 < acc < 1.0, f"accuracy={acc} should be between 0 and 1"
    assert compute_prediction_accuracy({}, {}) == 0.0
    assert compute_prediction_accuracy({"Legal": "a"}, {"Legal": "a"}) == 1.0
    print(f"  ✓ compute_prediction_accuracy works (partial={acc:.3f})")


def test_scorer_reset():
    b_before = {"Legal": make_belief(competent=0.2)}
    b_after = {"Legal": make_belief(competent=0.35)}
    scorer = UtteranceScorer()

    s1 = scorer.score(b_before, b_after, graph=MockGraph(), targeted_ids=["Legal"])
    s2 = scorer.score(b_before, b_after, graph=MockGraph(), targeted_ids=["Legal"])
    assert s1.trust == s2.trust, "reset should not affect stateless scorer"
    print("  ✓ scorer reset (no state accumulated in stateless scorer)")


def test_risk_cvar_no_profile_returns_0_5():
    b_before = {"Legal": make_belief()}
    b_after = {"Legal": make_belief(competent=0.35)}
    scorer = UtteranceScorer()

    s_no_profile = scorer.score(
        b_before, b_after, graph=MockGraph(), risk_profiles=None
    )
    s_empty = scorer.score(b_before, b_after, graph=MockGraph(), risk_profiles={})
    assert s_no_profile.risk == 0.5, (
        f"no risk profiles should give 0.5, got {s_no_profile.risk}"
    )
    assert s_empty.risk == 0.5, (
        f"empty risk profiles should give 0.5, got {s_empty.risk}"
    )
    print("  ✓ risk falls back to 0.5 when no risk profiles provided")


def test_causal_no_targets_returns_0():
    b_before = {"Legal": make_belief()}
    b_after = {"Legal": make_belief(competent=0.35)}
    scorer = UtteranceScorer()

    s_no_target = scorer.score(b_before, b_after, graph=MockGraph(), targeted_ids=[])
    assert s_no_target.causal == 0.0, (
        f"no targets should give causal=0, got {s_no_target.causal}"
    )
    print("  ✓ causal returns 0.0 when no targeted_ids")


def test_blocker_resolution_affects_goal():
    blockers_before = [
        "Contract terms not finalized",
        "Data processing agreement missing",
    ]
    blockers_after_resolved = []
    blockers_after_same = list(blockers_before)

    b = {"Legal": make_belief(competent=0.3)}
    scorer = UtteranceScorer()

    s_resolved = scorer.score(
        b,
        b,
        graph=MockGraph(),
        active_blockers_before=blockers_before,
        active_blockers_after=blockers_after_resolved,
    )
    s_same = scorer.score(
        b,
        b,
        graph=MockGraph(),
        active_blockers_before=blockers_before,
        active_blockers_after=blockers_after_same,
    )

    assert s_resolved.goal > s_same.goal, (
        f"resolved blockers should give higher goal ({s_resolved.goal}) vs same ({s_same.goal})"
    )
    print(
        f"  ✓ blocker resolution: resolved={s_resolved.goal:.3f} > same={s_same.goal:.3f}"
    )


if __name__ == "__main__":
    print("\n=== World-State Scorer Unit Tests ===\n")
    test_lookahead_cost_exactly_0_07()
    test_log2_6_correct()
    test_lookahead_penalty_applied()
    test_all_dimensions_bounded_0_1()
    test_goal_score_increases_with_approval()
    test_trust_targeted_delta()
    test_information_entropy_reduction()
    test_determinism_consistency()
    test_prediction_accuracy()
    test_scorer_reset()
    test_risk_cvar_no_profile_returns_0_5()
    test_causal_no_targets_returns_0()
    test_blocker_resolution_affects_goal()
    print("\n=== All tests passed ===")
