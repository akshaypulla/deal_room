"""Validate DealRoom v4 belief tracking system."""

import sys

sys.path.insert(0, "/Users/akshaypulla/Documents/deal_room")

from deal_room_v4_clean import (
    DealRoomV4,
    BeliefState,
    belief_update,
    _get_signals,
    propagate_beliefs,
    CausalGraph,
)

print("=" * 70)
print("DEALROOM V4 BELIEF TRACKING VALIDATION")
print("=" * 70)

# Test 1: Three independent dimensions - no cross-contamination
print("\n[TEST 1] Three independent dimensions - no cross-contamination")
print("-" * 70)
b = BeliefState(competence=0.5, trust=0.5, alignment=0.5)
print(f"Initial: competence={b.competence}, trust={b.trust}, alignment={b.alignment}")

# Update trust via DPA action
b.update("trust", +0.40, learning_rate=0.15)
print(
    f"After trust update: competence={b.competence}, trust={b.trust}, alignment={b.alignment}"
)

c_after = b.competence
t_after = b.trust
a_after = b.alignment

# Verify competence and alignment unchanged (should remain 0.5)
comp_unchanged = abs(c_after - 0.5) < 1e-9
align_unchanged = abs(a_after - 0.5) < 1e-9
trust_changed = abs(t_after - 0.5) > 1e-9

print(f"  competence unchanged: {comp_unchanged} (value={c_after})")
print(f"  alignment unchanged: {align_unchanged} (value={a_after})")
print(f"  trust changed: {trust_changed} (value={t_after})")
TEST1_PASS = comp_unchanged and align_unchanged and trust_changed
print(f"RESULT: {'PASS' if TEST1_PASS else 'FAIL'}")

# Test 2: Bayesian update with substring matching
print("\n[TEST 2] Bayesian update with substring matching")
print("-" * 70)

# Test "send_document(DPA)_proactive"
sig1 = _get_signals("send_document(DPA)_proactive", None)
print(f"  'send_document(DPA)_proactive' signals: {sig1}")
dpa_has_trust = sig1.get("trust", 0) > 0
dpa_has_alignment = sig1.get("alignment", 0) > 0
dpa_has_competence = sig1.get("competence", 0) > 0
print(f"    has trust signal: {dpa_has_trust} ({sig1.get('trust', 0)})")
print(f"    has alignment signal: {dpa_has_alignment} ({sig1.get('alignment', 0)})")
print(f"    has competence signal: {dpa_has_competence} ({sig1.get('competence', 0)})")

# Test "send_document(security_cert)_proactive"
sig2 = _get_signals("send_document(security_cert)_proactive", None)
print(f"  'send_document(security_cert)_proactive' signals: {sig2}")
sec_has_trust = sig2.get("trust", 0) > 0
sec_has_competence = sig2.get("competence", 0) > 0
sec_has_alignment = sig2.get("alignment", 0) > 0
print(f"    has trust signal: {sec_has_trust} ({sig2.get('trust', 0)})")
print(f"    has competence signal: {sec_has_competence} ({sig2.get('competence', 0)})")
print(f"    has alignment signal: {sec_has_alignment} ({sig2.get('alignment', 0)})")

# Verify different signals
diff_signals = sig1 != sig2
print(f"  Different signal maps: {diff_signals}")

# DPA focuses on trust+alignment, security_cert focuses on trust+competence
dpa_primary = (
    sig1.get("trust", 0) > sig1.get("competence", 0) and sig1.get("alignment", 0) > 0
)
sec_focus = (
    sig2.get("competence", 0) > sig2.get("alignment", 0) and sig2.get("trust", 0) > 0
)

TEST2_PASS = (
    dpa_has_trust
    and dpa_has_alignment
    and not dpa_has_competence
    and sec_has_competence
    and sec_has_trust
    and not sec_has_alignment
    and diff_signals
)
print(f"RESULT: {'PASS' if TEST2_PASS else 'FAIL'}")

# Test 3: Non-targeted damping - 0.5x strength
print("\n[TEST 3] Non-targeted damping - 0.5x (half-strength)")
print("-" * 70)

# Create initial beliefs
initial_finance = BeliefState(competence=0.5, trust=0.5, alignment=0.5)
initial_legal = BeliefState(competence=0.5, trust=0.5, alignment=0.5)

# Update Finance with roi_model (targeted)
finance_targeted = belief_update(
    initial_finance,
    "send_document(roi_model)_to_finance",
    [{"name": "roi_model"}],
    is_targeted=True,
)
# Update Legal with same action but non-targeted
legal_nontargeted = belief_update(
    initial_legal,
    "send_document(roi_model)_to_finance",
    [{"name": "roi_model"}],
    is_targeted=False,
)

print(
    f"  Finance (targeted): competence={finance_targeted.competence:.6f}, trust={finance_targeted.trust:.6f}"
)
print(
    f"  Legal (non-targeted): competence={legal_nontargeted.competence:.6f}, trust={legal_nontargeted.trust:.6f}"
)

# The non-targeted should get smaller change
finance_delta = finance_targeted.competence - initial_finance.competence
legal_delta = legal_nontargeted.competence - initial_legal.competence

print(f"  Finance competence delta: {finance_delta:.6f}")
print(f"  Legal competence delta: {legal_delta:.6f}")
print(f"  Ratio (Legal/Finance): {legal_delta / finance_delta:.6f} (expected ~0.5)")

# Verify damping is 0.5x
TEST3_PASS = (
    legal_delta > 0
    and finance_delta > 0
    and abs(legal_delta / finance_delta - 0.5) < 0.01
)
print(f"RESULT: {'PASS' if TEST3_PASS else 'FAIL'}")

# Test 4: Damping bound - beliefs never exceed [0.01, 0.99]
print("\n[TEST 4] Damping bound - beliefs never exceed [0.01, 0.99]")
print("-" * 70)

# Test upper bound
b_upper = BeliefState(competence=0.98, trust=0.98, alignment=0.98)
b_upper.update("trust", +2.0, learning_rate=0.15)  # Large positive delta
print(f"  After large positive update: trust={b_upper.trust:.6f}")
upper_bounded = b_upper.trust <= 0.99 + 1e-9

# Test lower bound
b_lower = BeliefState(competence=0.02, trust=0.02, alignment=0.02)
b_lower.update("trust", -2.0, learning_rate=0.15)  # Large negative delta
print(f"  After large negative update: trust={b_lower.trust:.6f}")
lower_bounded = b_lower.trust >= 0.01 - 1e-9

# Test multiple updates don't escape bounds
b_multi = BeliefState(competence=0.5, trust=0.5, alignment=0.5)
for _ in range(20):
    b_multi.update("trust", +0.5, learning_rate=0.15)
print(f"  After 20 positive updates: trust={b_multi.trust:.6f}")
multi_bounded = b_multi.trust <= 0.99 + 1e-9

TEST4_PASS = upper_bounded and lower_bounded and multi_bounded
print(f"  Upper bound maintained: {upper_bounded}")
print(f"  Lower bound maintained: {lower_bounded}")
print(f"  Multiple updates bounded: {multi_bounded}")
print(f"RESULT: {'PASS' if TEST4_PASS else 'FAIL'}")

# Test 5: positive_mass geometric mean
print("\n[TEST 5] positive_mass geometric mean")
print("-" * 70)

# Test with known values
b_test = BeliefState(competence=0.8, trust=0.8, alignment=0.8)
expected_pm = (0.8 * 0.8 * 0.8) ** (1 / 3)
actual_pm = b_test.positive_mass()
print(f"  competence=0.8, trust=0.8, alignment=0.8")
print(f"  Expected: {expected_pm:.6f}, Actual: {actual_pm:.6f}")

# Test asymmetric values
b_asym = BeliefState(competence=0.5, trust=0.75, alignment=1.0)
expected_asym = (0.5 * 0.75 * 1.0) ** (1 / 3)
actual_asym = b_asym.positive_mass()
print(f"  competence=0.5, trust=0.75, alignment=1.0")
print(f"  Expected: {expected_asym:.6f}, Actual: {actual_asym:.6f}")

TEST5_PASS = (
    abs(expected_pm - actual_pm) < 1e-9 and abs(expected_asym - actual_asym) < 1e-9
)
print(f"RESULT: {'PASS' if TEST5_PASS else 'FAIL'}")

# Test 6: Entropy calculation
print("\n[TEST 6] Entropy calculation - H_bin sum")
print("-" * 70)


# Binary entropy function for reference
def binary_entropy(p):
    if p <= 0.01 or p >= 0.99:
        return 0.0
    import math

    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


# Test p=0.5 (max entropy per dimension = 1.0)
b_half = BeliefState(competence=0.5, trust=0.5, alignment=0.5)
expected_entropy = binary_entropy(0.5) * 3
actual_entropy = b_half.entropy()
print(f"  p=0.5 for all dimensions")
print(f"  Expected entropy: {expected_entropy:.6f} (max = 3.0)")
print(f"  Actual entropy: {actual_entropy:.6f}")
half_entropy_correct = abs(actual_entropy - 3.0) < 1e-6

# Test p=0.01 (minimum entropy ≈ 0)
b_min = BeliefState(competence=0.01, trust=0.01, alignment=0.01)
expected_min = 0.0  # Should return 0.0 for p <= 0.01
actual_min = b_min.entropy()
print(f"  p=0.01 for all dimensions")
print(f"  Expected entropy: 0.0 (min)")
print(f"  Actual entropy: {actual_min:.6f}")
min_entropy_correct = abs(actual_min) < 1e-9

# Test p=0.99 (minimum entropy ≈ 0)
b_max = BeliefState(competence=0.99, trust=0.99, alignment=0.99)
expected_max = 0.0  # Should return 0.0 for p >= 0.99
actual_max = b_max.entropy()
print(f"  p=0.99 for all dimensions")
print(f"  Expected entropy: 0.0 (min)")
print(f"  Actual entropy: {actual_max:.6f}")
max_entropy_correct = abs(actual_max) < 1e-9

TEST6_PASS = half_entropy_correct and min_entropy_correct and max_entropy_correct
print(f"  p=0.5 correct: {half_entropy_correct}")
print(f"  p=0.01 correct: {min_entropy_correct}")
print(f"  p=0.99 correct: {max_entropy_correct}")
print(f"RESULT: {'PASS' if TEST6_PASS else 'FAIL'}")

# Test 7: Propagation through graph
print("\n[TEST 7] Propagation through graph - belief changes propagate")
print("-" * 70)

# Create a simple graph with known structure
import numpy as np

# Use fixed seed for reproducibility
rng = np.random.default_rng(42)

# Create test stakeholders
stakeholders = ["A", "B", "C"]
hierarchy = {"A": 3, "B": 3, "C": 3}

# Create graph manually
graph = CausalGraph(
    nodes=stakeholders,
    edges={
        ("A", "B"): 0.8,  # A -> B with weight 0.8
        ("A", "C"): 0.5,  # A -> C with weight 0.5
    },
    authority_weights={"A": 0.33, "B": 0.33, "C": 0.34},
    scenario_type="aligned",
    seed=42,
)

# Initial beliefs
beliefs_before = {
    "A": BeliefState(competence=0.5, trust=0.5, alignment=0.5),
    "B": BeliefState(competence=0.5, trust=0.5, alignment=0.5),
    "C": BeliefState(competence=0.5, trust=0.5, alignment=0.5),
}

# Apply large positive delta to A only (simulating targeted action)
beliefs_after = {
    "A": BeliefState(
        competence=0.5, trust=0.9, alignment=0.5
    ),  # A trust increased significantly
    "B": BeliefState(competence=0.5, trust=0.5, alignment=0.5),
    "C": BeliefState(competence=0.5, trust=0.5, alignment=0.5),
}

print(
    f"  Before propagation: A trust={beliefs_before['A'].trust}, B trust={beliefs_before['B'].trust}, C trust={beliefs_before['C'].trust}"
)
print(
    f"  After targeted action: A trust={beliefs_after['A'].trust}, B trust={beliefs_after['B'].trust}, C trust={beliefs_after['C'].trust}"
)

# Propagate
propagated = propagate_beliefs(graph, beliefs_before, beliefs_after, n_steps=3)

print(
    f"  After propagation: A trust={propagated['A'].trust:.6f}, B trust={propagated['B'].trust:.6f}, C trust={propagated['C'].trust:.6f}"
)

# B and C should have changed due to A's change
b_changed = abs(propagated["B"].trust - 0.5) > 1e-6
c_changed = abs(propagated["C"].trust - 0.5) > 1e-6

# B should change more than C (because A->B weight > A->C weight)
b_delta = abs(propagated["B"].trust - 0.5)
c_delta = abs(propagated["C"].trust - 0.5)
propagation_correct = b_changed and c_changed and b_delta > c_delta

print(f"  B changed: {b_changed} (delta={b_delta:.6f})")
print(f"  C changed: {c_changed} (delta={c_delta:.6f})")
print(f"  B delta > C delta: {b_delta > c_delta}")

TEST7_PASS = propagation_correct
print(f"RESULT: {'PASS' if TEST7_PASS else 'FAIL'}")

# Test 8: No self-loop propagation
print("\n[TEST 8] No self-loop propagation - nodes don't influence themselves")
print("-" * 70)

# Create a graph with self-implied edge (should not matter)
graph_self = CausalGraph(
    nodes=["X", "Y"],
    edges={
        ("X", "Y"): 0.9,
        # No self-loop
    },
    authority_weights={"X": 0.5, "Y": 0.5},
    scenario_type="aligned",
    seed=99,
)

# Initial beliefs - X starts high trust, Y starts low
beliefs_before_x = {
    "X": BeliefState(competence=0.5, trust=0.95, alignment=0.5),  # X high trust
    "Y": BeliefState(competence=0.5, trust=0.1, alignment=0.5),  # Y low trust
}

beliefs_after_x = {
    "X": BeliefState(competence=0.5, trust=0.95, alignment=0.5),  # X unchanged
    "Y": BeliefState(
        competence=0.5, trust=0.9, alignment=0.5
    ),  # Y increased (targeted action)
}

print(
    f"  X: before trust={beliefs_before_x['X'].trust}, after trust={beliefs_after_x['X'].trust}"
)
print(
    f"  Y: before trust={beliefs_before_x['Y'].trust}, after trust={beliefs_after_x['Y'].trust}"
)

# Propagate
propagated_x = propagate_beliefs(
    graph_self, beliefs_before_x, beliefs_after_x, n_steps=3
)

print(
    f"  After propagation: X trust={propagated_x['X'].trust:.6f}, Y trust={propagated_x['Y'].trust:.6f}"
)

# X should NOT change (no self-loop, and Y shouldn't change X because the direction is X->Y not Y->X)
x_unchanged = abs(propagated_x["X"].trust - 0.95) < 1e-6
# Y should change (because Y is the target and the graph is X->Y)
y_changed_from_propagation = (
    abs(propagated_x["Y"].trust - 0.9) > 1e-6
    or abs(propagated_x["Y"].trust - 0.9) < 1e-6
)  # Y might change via propagation too

# Actually we need to check the case where Y is already the target and gets update, but propagation shouldn't make it worse
# The key check: X should not be affected by Y's change in propagation (no reverse direction)

# Create more explicit test: X trusts 0.95, Y trusts 0.5
# Apply action that changes Y significantly
beliefs_before_x2 = {
    "X": BeliefState(competence=0.5, trust=0.95, alignment=0.5),
    "Y": BeliefState(competence=0.5, trust=0.5, alignment=0.5),
}
beliefs_after_x2 = {
    "X": BeliefState(
        competence=0.5, trust=0.95, alignment=0.5
    ),  # X unchanged (not targeted)
    "Y": BeliefState(competence=0.5, trust=0.85, alignment=0.5),  # Y changed (targeted)
}

propagated_x2 = propagate_beliefs(
    graph_self, beliefs_before_x2, beliefs_after_x2, n_steps=3
)

# X should remain 0.95 (no self influence)
x_still_high = abs(propagated_x2["X"].trust - 0.95) < 1e-6
print(f"  X trust unchanged: {x_still_high} (value={propagated_x2['X'].trust:.6f})")

# The point is: X's belief should NOT change just because Y changed
# In the graph structure (X->Y), X influences Y, not the other way
# So even if Y changes due to targeted action, X should not get influenced by Y through propagation

# Check that X remains unchanged after propagation
TEST8_PASS = x_still_high
print(f"RESULT: {'PASS' if TEST8_PASS else 'FAIL'}")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
results = {
    "Test 1 - Independent dimensions": TEST1_PASS,
    "Test 2 - Substring matching": TEST2_PASS,
    "Test 3 - Non-targeted damping (0.5x)": TEST3_PASS,
    "Test 4 - Damping bounds [0.01, 0.99]": TEST4_PASS,
    "Test 5 - positive_mass geometric mean": TEST5_PASS,
    "Test 6 - Entropy calculation": TEST6_PASS,
    "Test 7 - Graph propagation": TEST7_PASS,
    "Test 8 - No self-loop propagation": TEST8_PASS,
}

for test_name, passed in results.items():
    status = "PASS" if passed else "FAIL"
    print(f"  {test_name}: {status}")

all_passed = all(results.values())
print(f"\n{'=' * 70}")
print(f"OVERALL: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
print(f"{'=' * 70}")
