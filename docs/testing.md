# DealRoom Testing Guide

## Test Suite Overview

The DealRoom test suite validates the complete environment including unit tests for individual components, integration tests for environment interactions, end-to-end workflow tests, and performance benchmarks.

**Date:** 2026-04-09  
**Result:** 22/22 tests passed

### Test Categories

| Category | Tests | Passed | Duration |
|----------|-------|--------|----------|
| Unit | 10 | 10 | ~0.45s |
| Integration | 6 | 6 | ~0.62s |
| E2E | 1 | 1 | ~1.23s |
| Performance | 2 | 2 | ~0.14s |
| **Total** | **22** | **22** | **2.54s** |

---

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific category
pytest tests/unit/ -v
pytest tests/integration/ -v
pytest tests/e2e/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=term-missing

# Run specific test file
pytest tests/unit/test_models.py -v

# Run tests matching pattern
pytest tests/ -k "validator" -v
```

---

## Component Coverage

| Component | File | Tests |
|-----------|------|-------|
| Models | `models.py` | 3 |
| Validator | `server/validator.py` | 3 |
| Claims | `server/claims.py` | 2 |
| Grader | `server/grader.py` | 2 |
| Semantics | `server/semantics.py` | 2 |
| Environment | `server/deal_room_environment.py` | 5 |
| Web UI | `server/app.py` | 2 |
| E2E Workflows | `inference.py` | 1 |
| Benchmarking | `test_benchmarking.py` | 2 |

---

## Key Behaviors Validated

### 1. Dynamic Stakeholder Roster
The environment generates a dynamic roster from a role library based on task type:
- **aligned:** 2-3 stakeholders, higher initial trust/approval
- **conflicted:** 3-4 stakeholders, conflicting incentives
- **hostile_acquisition:** 4 stakeholders, authority shifts, low trust

### 2. Hidden Constraints
Constraints start hidden and become known through semantic analysis:
- **budget_ceiling:** Discovered via ROI/timeline mentions
- **delivery_window:** Discovered via implementation artifact sharing
- **compliance_addendum:** Discovered via DPA/security mentions
- **supplier_process:** Discovered via vendor packet mentions

### 3. Dense Reward System
Rewards accumulate for milestone achievements:
- Constraint hint: +0.03
- Constraint known: +0.04
- Artifact satisfied: +0.03
- Band improvement: +0.02
- Blocker removal: +0.02
- Stage advancement: +0.03
- **Maximum per step:** 0.15

### 4. Terminal Grading (CCIGrader)
Final score weighted by:
- **Approval Completeness:** 35%
- **Constraint Satisfaction:** 25%
- **Term Feasibility:** 15%
- **Relationship Durability:** 15%
- **Efficiency:** 10%

### 5. Validator Behavior
- JSON parsing with fallback to heuristic extraction
- Dynamic target resolution with alias support
- Proposed terms filtering
- Confidence scoring (1.0 for JSON, 0.6 for heuristic, 0.0 for unparseable)

---

## Detailed Test Results

### E2E Tests

**test_baseline_runs_aligned** - PASSED (~1.23s)
- End-to-end baseline run of the aligned scenario using a deterministic policy
- Steps: 7, Final Score: 0.90

### Integration Tests

**test_reset_creates_dynamic_roster** - PASSED
- Validates reset generates a dynamic stakeholder roster with correct count and initial stage

**test_step_returns_dense_reward_in_bounds** - PASSED
- Verifies step returns reward within valid bounds [0.0, 0.15]

**test_constraint_can_be_discovered_with_probe_and_artifact** - PASSED
- Tests constraint discovery through semantic analysis

**test_premature_close_applies_penalty** - PASSED
- Verifies early close attempt applies trust penalty with permanent mark

**test_feasible_close_returns_terminal_score** - PASSED
- Validates successful close returns positive terminal score

### Performance Tests

**test_reset_performance** - PASSED
- Validates reset completes within acceptable time (~80ms per operation)

**test_step_performance** - PASSED
- Validates step completes within acceptable time (~80ms per operation)

---

## Test Execution Log

```
tests/e2e/test_workflows.py::test_baseline_runs_aligned PASSED           [  4%]
tests/integration/test_environment.py::test_reset_creates_dynamic_roster PASSED [  9%]
tests/integration/test_environment.py::test_step_returns_dense_reward_in_bounds PASSED [ 13%]
tests/integration/test_environment.py::test_constraint_can_be_discovered_with_probe_and_artifact PASSED [ 18%]
tests/integration/test_environment.py::test_premature_close_applies_penalty PASSED [ 22%]
tests/integration/test_environment.py::test_feasible_close_returns_terminal_score PASSED [ 27%]
tests/integration/test_web_ui.py::test_root_redirects_to_web PASSED      [ 31%]
tests/integration/test_web_ui.py::test_web_page_exposes_playground_and_custom_tabs PASSED [ 36%]
tests/performance/test_benchmarking.py::test_reset_performance PASSED     [ 40%]
tests/performance/test_benchmarking.py::test_step_performance PASSED     [ 45%]
tests/unit/test_claims.py::test_commitment_ledger_flags_numeric_contradiction PASSED [ 50%]
tests/unit/test_claims.py::test_commitment_ledger_trims_history PASSED   [ 54%]
tests/unit/test_grader.py::test_grader_returns_zero_for_infeasible_close PASSED [ 59%]
tests/unit/test_grader.py::test_grader_returns_positive_for_feasible_close PASSED [ 63%]
tests/unit/test_models.py::test_action_target_ids_are_deduplicated PASSED [ 68%]
tests/unit/test_models.py::test_state_is_callable_for_state_and_state_method_compat PASSED [ 72%]
tests/unit/test_models.py::test_observation_supports_dynamic_fields PASSED [ 77%]
tests/unit/test_semantics.py::test_semantic_analyzer_extracts_claims_and_artifacts PASSED [ 81%]
tests/unit/test_semantics.py::test_semantic_analyzer_returns_known_backend PASSED [ 86%]
tests/unit/test_validator.py::test_validator_normalizes_dynamic_targets PASSED [ 90%]
tests/unit/test_validator.py::test_validator_soft_rejects_unknown_target PASSED [ 95%]
tests/unit/test_validator.py::test_validator_filters_proposed_terms PASSED [100%]

22 passed in 2.54s
```