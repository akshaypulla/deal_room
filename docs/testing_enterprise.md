# DealRoom v3 — Enterprise Testing Documentation

**Version:** 1.0.0  
**Basis:** Code-grounded audit. Test functions identified by exact name and file.  
**Scope:** `tests/v3/` for integration tests, `tests/unit/` for unit tests, `tests/e2e/` for E2E, `tests/performance/` for benchmarks.

---

## 1. Test Framework and Setup

### Framework
- **pytest** for test execution
- **requests** library for HTTP API testing
- **Docker** for container-based test execution
- **numpy** for numerical assertions

### Test Directory Structure

```
tests/
├── v3/                          # Primary integration test suite (live server tests)
│   ├── conftest.py              # Shared fixtures (BASE_URL, session helpers, assertions)
│   ├── test_00_environment_setup.py
│   ├── test_01_schema_validation.py
│   ├── test_02_reward_integrity.py
│   ├── test_03_causal_inference.py
│   ├── test_04_cvar_veto.py
│   ├── test_05_episode_isolation.py
│   ├── test_06_probabilistic_signals.py
│   ├── test_07_causal_graph.py   # Unit tests run inside container
│   ├── test_08_cvar_preferences.py
│   ├── test_09_full_episode_e2e.py
│   ├── test_10_training_integration.py
│   ├── test_10_training_infrastructure.py
│   ├── test_11_research_properties.py
│   └── test_scorer_unit.py
├── unit/                         # Unit tests (no server required)
│   ├── test_utterance_scorer.py
│   ├── test_observation_mechanism.py
│   ├── test_deliberation_engine.py
│   ├── test_causal_graph.py
│   ├── test_belief_tracker.py
│   ├── test_cvar_preferences.py
│   ├── test_curriculum.py
│   ├── test_grader.py
│   ├── test_inference.py
│   ├── test_semantics.py
│   ├── test_models.py
│   ├── test_claims.py
│   └── test_validator.py
├── integration/
│   ├── test_end_to_end.py
│   ├── test_web_ui.py
│   ├── test_api_sessions.py
│   └── test_environment.py
├── e2e/
│   └── test_workflows.py
└── performance/
    └── test_benchmarking.py
```

### Test Configuration Files

| File | Purpose |
|------|---------|
| `tests/v3/conftest.py` | Shared helpers: `get_session()`, `make_action()`, `step()`, `get_reward()`, `get_obs()`, `assert_near()`, `assert_in_range()` |
| `.env` | Optional: `MINIMAX_API_KEY`, `OPENAI_API_KEY`, `DEALROOM_BASE_URL`, `DEALROOM_CONTAINER_NAME` |

---

## 2. Test Coverage Mapping

| Component | Test Coverage | Files |
|-----------|--------------|-------|
| Environment reset/step | Integration tests against live server | `test_00_environment_setup.py`, `test_09_full_episode_e2e.py` |
| Reward system | API integration + unit tests | `test_02_reward_integrity.py`, `test_scorer_unit.py` |
| CVaR veto logic | API integration + unit tests | `test_04_cvar_veto.py`, `test_08_cvar_preferences.py` |
| Causal graph | Unit tests (container-side) | `test_07_causal_graph.py`, `test_causal_graph.py` |
| Belief propagation | Unit + integration tests | `test_03_causal_inference.py`, `test_07_causal_graph.py` |
| Belief update (Bayesian) | Unit tests | `test_belief_tracker.py` |
| UtteranceScorer | Unit tests | `test_utterance_scorer.py`, `test_scorer_unit.py` |
| Lookahead mechanism | Integration tests | `test_02_reward_integrity.py`, `test_11_research_properties.py` |
| Training system | Integration tests | `test_10_training_integration.py`, `test_10_training_infrastructure.py` |
| Partial observability | Integration tests | `test_03_causal_inference.py`, `test_06_probabilistic_signals.py` |
| Episode isolation | Integration tests | `test_05_episode_isolation.py` |
| Schema validation | Integration tests | `test_01_schema_validation.py` |

---

## 3. Test Categories

### 3.1 Unit Tests

**Purpose:** Test individual functions/classes in isolation (no server required).

| File | Functions/Classes Tested |
|------|-------------------------|
| `tests/unit/test_utterance_scorer.py` | `UtteranceScorer.score()`, `_score_goal()`, `_score_trust()`, `_score_info()`, `_score_risk()`, `_score_causal()` |
| `tests/unit/test_causal_graph.py` | `sample_graph()`, `propagate_beliefs()`, `apply_positive_delta()`, `get_betweenness_centrality()`, `BeliefDistribution` methods |
| `tests/unit/test_belief_tracker.py` | `bayesian_update()`, `_get_likelihood()` |
| `tests/unit/test_cvar_preferences.py` | `compute_cvar()`, `compute_outcome_distribution()`, `check_veto_trigger()`, `get_observable_signals()` |
| `tests/unit/test_deliberation_engine.py` | `CommitteeDeliberationEngine.run()`, `_generate_summary()` |
| `tests/unit/test_grader.py` | `weighted_reward()` |
| `tests/unit/test_models.py` | Pydantic model validators (`truncate_message`, `normalize_target_ids`, `sync_targets`) |

### 3.2 Integration Tests

**Purpose:** Test module interactions via HTTP API against live server (Docker container).

| File | What It Tests |
|------|---------------|
| `tests/v3/test_00_environment_setup.py` | Environment prerequisites: Python deps, API keys, Docker container, server endpoints |
| `tests/v3/test_01_schema_validation.py` | Action/observation schema adherence |
| `tests/v3/test_02_reward_integrity.py` | Reward integrity, bounds, lookahead cost, determinism, hack resistance |
| `tests/v3/test_03_causal_inference.py` | Belief propagation, cross-stakeholder echoes, engagement noise |
| `tests/v3/test_04_cvar_veto.py` | Veto trigger conditions, precursor warnings, scenario difficulty differentiation |
| `tests/v3/test_05_episode_isolation.py` | Session isolation between episodes |
| `tests/v3/test_06_probabilistic_signals.py` | Weak signals, engagement deltas |
| `tests/v3/test_08_cvar_preferences.py` | CVaR computation, outcome distribution |
| `tests/v3/test_09_full_episode_e2e.py` | Full episode execution, all scenarios, strategy comparison |
| `tests/v3/test_10_training_integration.py` | Training loop execution |
| `tests/v3/test_10_training_infrastructure.py` | Training infrastructure, checkpointing |
| `tests/v3/test_11_research_properties.py` | Research claims validation |
| `tests/v3/test_scorer_unit.py` | `UtteranceScorer` against live environment |

### 3.3 End-to-End Tests

| File | What It Tests |
|------|---------------|
| `tests/e2e/test_workflows.py` | Complete user workflows (multi-step negotiations) |
| `tests/integration/test_web_ui.py` | Gradio web interface interaction |
| `tests/integration/test_end_to_end.py` | Full session lifecycle via API |

### 3.4 Performance Tests

| File | What It Tests |
|------|---------------|
| `tests/performance/test_benchmarking.py` | Throughput, latency benchmarks |

---

## 4. Key Test Cases

### Environment Setup (test_00_environment_setup.py)

| Test | What It Validates | Expected Behavior |
|------|-------------------|------------------|
| `test_0_1_python_deps` | Critical Python packages importable | Exit code 0 if deps present |
| `test_0_2_api_keys_configured` | Optional API keys discoverable | Print status, fail-soft |
| `test_0_3_docker_container_running` | Docker container active | Exit code 1 if not running |
| `test_0_4_server_endpoints_responsive` | `/health`, `/metadata`, `/reset`, `/step` return 200 | Status code assertions |
| `test_0_5_llm_client_imports` | `llm_client` module available | Print MAX_TOKENS, fail-soft |

### Reward Integrity (test_02_reward_integrity.py)

| Test | What It Validates | Expected Behavior |
|------|-------------------|------------------|
| `test_2_1_reward_is_single_float` | Reward is numeric, not dict | `isinstance(reward, float)` |
| `test_2_2_lookahead_cost_is_exactly_007` | Lookahead cost = 0.07 | `abs(goal1 - goal2 - 0.07) < 0.015` |
| `test_2_3_reward_in_range_after_valid_actions` | Reward components ∈ [0, 1] | All 5 dimensions bounded |
| `test_2_4_deterministic_reward_with_seed` | Same seed → same reward | `max(rewards) - min(rewards) < 1e-9` |
| `test_2_5_repeat_same_action_does_not_escalate_reward` | Repeated action doesn't inflate reward | `g3 - g1 <= 0.01` |
| `test_2_6_different_targets_different_causal_scores` | Different targets → different causal scores | At least 2 unique scores |
| `test_2_7_informative_action_outperforms_empty` | Substantive action ≥ empty message | `g_subst >= g_empty - 0.1` |
| `test_2_8_reward_non_trivial_variance` | Reward discriminates across actions | `variance > 0.05` |
| `test_2_9_good_documentation_higher_than_poor` | Good docs score ≥ poor docs | `avg_good >= avg_poor - 0.01` |
| `test_2_10_lookahead_improves_prediction_accuracy` | Lookahead accuracy > 0.60 baseline | `mean_accuracy > 0.60` |

### CVaR Veto (test_04_cvar_veto.py)

| Test | What It Validates | Expected Behavior |
|------|-------------------|------------------|
| `test_4_1_veto_precursor_before_veto` | Veto preceded by precursor | `precursor_seen` must be True if `veto_seen` |
| `test_4_2_aligned_no_early_veto` | Aligned scenario survives first step | `done == False` after step 1 |
| `test_4_3_veto_terminates_episode` | Veto triggers done=True | Terminal contains "veto" |
| `test_4_3_veto_deterministic` | Legal veto fires deterministically at step ≤5 | `terminal_category == "veto"`, `veto_stakeholder == "Legal"` |
| `test_4_4_timeout_terminates` | Max rounds triggers done=True | Terminal is "timeout" |
| `test_4_5_scenario_difficulty_differentiation` | Hostile reaches veto pressure earlier than aligned | `hostile_mean <= aligned_mean` |
| `test_4_6_veto_precursor_is_stakeholder_specific` | Precursors keyed by stakeholder | Set of stakeholders non-empty |
| `test_4_7_cveto_not_just_eu` | CVaR veto fires even if EU positive | Veto fires in hostile scenario |

### Causal Inference (test_03_causal_inference.py)

| Test | What It Validates | Expected Behavior |
|------|-------------------|------------------|
| `test_3_1_targeted_stakeholder_engagement_changes` | Targeted stakeholder shows delta | `delta` is numeric |
| `test_3_2_cross_stakeholder_echoes_detected` | Non-targeted stakeholders receive echoes | Echo rate ≥ 65% |
| `test_3_3_engagement_history_window_slides` | History window size stable | Length constant across steps |
| `test_3_4_engagement_noise_not_cancellable` | Noise sigma > 0 | At least 2/5 deltas non-zero |
| `test_3_5_different_targets_different_patterns` | Finance vs Legal targeting produces different patterns | Distance > 0.001 |
| `test_3_6_weak_signals_for_non_targeted` | Weak signals appear | Detection rate ≥ 20% |
| `test_3_7_echo_content_structure` | Echo entries have required fields | Dict with sender field |
| `test_3_8_causal_signal_responds_to_graph_structure` | Hub (ExecSponsor) has more causal impact than leaf | Hub impact > leaf impact × 1.15 |

### Causal Graph (test_07_causal_graph.py — container unit tests)

| Test | What It Validates | Expected Behavior |
|------|-------------------|------------------|
| `test_7_1_propagation_direction` | Signal flows A→B along edge | B positive_mass increases |
| `test_7_2_signal_carries_through_edge` | B changes when A changes | B positive_mass != before |
| `test_7_3_damping_prevents_runaway` | Beliefs bounded [0,1] after dense propagation | 0 < pm < 1 for all nodes |
| `test_7_4_beliefs_normalized_after_propagation` | Belief distributions sum to 1.0 | `abs(total - 1.0) < 1e-6` |
| `test_7_5_no_self_loops` | No self-referential edges | `get_weight(sid, sid) == 0.0` |
| `test_7_6_exec_sponsor_outgoing_authority` | ExecSponsor has ≥2 outgoing edges | `len(outgoing) >= 2` |
| `test_7_7_hub_centrality_beats_leaf` | Hub betweenness ≥ leaf betweenness | `hub_c >= leaf_c` |
| `test_7_8_graph_identifiability` | All 20 sampled graphs pairwise unique | `distinguishable == total_pairs` |
| `test_7_9_hub_node_has_higher_centrality_impact` | Hub behavioral impact > leaf impact | Mean ratio > 1.3 |

### Full Episode E2E (test_09_full_episode_e2e.py)

| Test | What It Validates | Expected Behavior |
|------|-------------------|------------------|
| `test_9_1_aligned_completes` | Aligned scenario runs to completion | Steps ≥ 1 |
| `test_9_2_conflicted_completes` | Conflicted scenario runs to completion | Steps ≥ 1 |
| `test_9_3_hostile_aggressive_produces_veto_or_timeout` | Hostile terminates via veto or timeout | Terminal in valid set |
| `test_9_4_reward_collected_across_episode` | Rewards recorded each step | `len(rewards) >= 1` |
| `test_9_5_reward_variance_non_trivial` | Reward not constant across episode | `variance > 0.05` |
| `test_9_6_strategy_comparison_possible` | Comprehensive strategy > aggressive | `avg_comp > avg_agg + 0.15` |
| `test_9_7_terminal_outcome_meaningful` | done=True fires at natural endpoint | Non-empty terminal set |
| `test_9_8_multidocument_action_works` | Multi-document action returns reward | Status 200, reward not None |

---

## 5. Execution Process

### Running Tests

**Prerequisites:**
1. Docker container running: `dealroom-v3-test:latest` on port 7860
2. Python dependencies: `requests`, `numpy`, `pytest`

**Commands:**

```bash
# Environment setup validation
python tests/v3/test_00_environment_setup.py

# Full v3 integration suite
python tests/v3/test_02_reward_integrity.py
python tests/v3/test_04_cvar_veto.py
python tests/v3/test_09_full_episode_e2e.py

# Unit tests (no container required)
pytest tests/unit/ -v

# E2E tests
pytest tests/e2e/ -v

# Performance tests
pytest tests/performance/ -v
```

**Container startup (from conftest.py:48-74):**

```bash
docker run --rm -d -p 7860:7860 \
  -e MINIMAX_API_KEY=${MINIMAX_API_KEY} \
  --name dealroom-v3-test dealroom-v3-test:latest
# Wait 15s for startup
```

### Dependencies
- Server must be running (Docker container or direct `uvicorn`)
- `BASE_URL` defaults to `http://127.0.0.1:7860`
- API keys optional (runtime degrades gracefully if absent)

---

## 6. Determinism and Reproducibility

### Seed Usage

| Component | Seed Source | Reproducibility |
|-----------|-------------|-----------------|
| Environment reset | `reset(seed=N)` — user-provided | Full if seed provided |
| Causal graph sampling | `np.random.default_rng(seed)` inside `sample_graph()` | Full for same seed |
| Engagement noise | `np.random.default_rng(seed)` — seeded in `reset()` | Full for same seed |
| CVaR sampling | Internal RNG seeded from `seed * 31 + hash(stakeholder_id)` | Reproducible per step |
| Action normalization | Deterministic (no RNG) | N/A |
| Reward scoring | Deterministic state-based computation | Full |

### Randomness Control

**In tests:**
- `test_2_4_deterministic_reward_with_seed` validates: same seed + same action → same reward (spread < 1e-9)
- `LOOKAHEAD_MIN_ACCURACY = 0.60` threshold for lookahead prediction quality

**Stochastic elements in tests:**
- `test_3_2_cross_stakeholder_echoes_detected` uses 65% threshold (allows 35% stochastic failure)
- `test_3_4_engagement_noise_not_cancellable` requires ≥2/5 non-zero deltas (allows some cancellation)

---

## 7. Test Effectiveness Analysis

### Reward System

| Property | Validation | Status |
|----------|-----------|--------|
| Determinism | `test_2_4_deterministic_reward_with_seed` | Validated — same seed yields identical reward |
| Bounds | `test_2_3_reward_in_range_after_valid_actions` | All 5 dimensions clamped to [0, 1] |
| Lookahead cost | `test_2_2_lookahead_cost_is_exactly_007` | Exactly 0.07 (tolerance 0.015) |
| Hack resistance | `test_2_5_repeat_same_action_does_not_escalate_reward` | Repeated action ≤ +0.01 trend |
| Discrimination | `test_2_8_reward_non_trivial_variance` | Variance > 0.05 across different actions |
| Quality signal | `test_2_9_good_documentation_higher_than_poor` | Good docs ≥ poor docs |

### CVaR / Risk Logic

| Property | Validation | Status |
|----------|-----------|--------|
| Veto conditions | `test_4_3_veto_deterministic` | Legal veto fires deterministically at step ≤5 |
| Precursor warning | `test_4_1_veto_precursor_before_veto` | Veto always preceded by precursor |
| Non-premature veto | `test_4_2_aligned_no_early_veto` | Aligned survives first step |
| Scenario differentiation | `test_4_5_scenario_difficulty_differentiation` | Hostile reaches pressure earlier |
| Stakeholder specificity | `test_4_6_veto_precursor_is_stakeholder_specific` | Precursors keyed by stakeholder ID |
| EU vs CVaR | `test_4_7_cveto_not_just_eu` | Veto fires even if EU positive |

### Causal System

| Property | Validation | Status |
|----------|-----------|--------|
| Propagation direction | `test_7_1_propagation_direction` | Signal flows along edges |
| Signal transmission | `test_7_2_signal_carries_through_edge` | B changes when A changes |
| Damping | `test_7_3_damping_prevents_runaway` | Beliefs bounded after dense graph |
| Normalization | `test_7_4_beliefs_normalized_after_propagation` | Sum = 1.0 |
| No self-loops | `test_7_5_no_self_loops` | Confirmed across scenarios |
| Authority invariant | `test_7_6_exec_sponsor_outgoing_authority` | ExecSponsor always has ≥2 outgoing |
| Hub centrality | `test_7_7_hub_centrality_beats_leaf` | Hub betweenness ≥ leaf |
| Graph identifiability | `test_7_8_graph_identifiability` | All 20 graphs pairwise unique |
| Impact ratio | `test_7_9_hub_node_has_higher_centrality_impact` | Hub/leaf impact ratio > 1.3 |
| Cross-stakeholder echoes | `test_3_2_cross_stakeholder_echoes_detected` | Echo rate ≥ 65% |

### Training System

| Property | Validation | Status |
|----------|-----------|--------|
| Training loop runs | `test_10_training_integration.py` | Full loop executes without crash |
| Infrastructure | `test_10_training_infrastructure.py` | Checkpointing, metrics collection work |
| Lookahead accuracy | `test_2_10_lookahead_improves_prediction_accuracy` | Mean accuracy > 0.60 |
| Strategy comparison | `test_9_6_strategy_comparison_possible` | Comprehensive > aggressive by > 0.15 |

---

## 8. Debugging Support

### Step-by-Step Traceability

Each step returns `info` dict with:
- `reward_components`: 5-dimension breakdown (goal, trust, info, risk, causal)
- `deliberation_summary`: Optional LLM-generated committee discussion
- `propagation_deltas`: Per-stakeholder belief mass changes from propagation
- `noisy_engagement_deltas`: Per-stakeholder noisy engagement changes
- `terminal_reward`: Additional reward at episode termination
- `terminal_outcome`: String describing terminal condition
- `veto_stakeholder`: Which stakeholder triggered veto (if any)
- `lookahead_used`: Boolean
- `lookahead_predicted_deltas`: Predicted belief deltas from lookahead
- `lookahead_predicted_responses`: Predicted stakeholder responses
- `lookahead_cvar_impact`: Predicted CVaR impact
- `prediction_accuracy`: Actual vs predicted accuracy for lookahead

### Failure Point Inspection

| Failure Type | How to Inspect |
|-------------|----------------|
| Veto triggered unexpectedly | Check `info["veto_stakeholder"]`, `info["terminal_outcome"]`, `risk_snapshot["cvar_losses"]` |
| Reward too low | Check `info["reward_components"]` per dimension, `info["propagation_deltas"]` |
| No progress in negotiation | Check `observation["deal_stage"]`, `observation["deal_momentum"]`, `observation["active_blockers"]` |
| Episode not terminating | Check `observation["round_number"]` vs `observation["max_rounds"]` |
| Lookahead failures | Check `info["prediction_accuracy"]`, `info["lookahead_predicted_deltas"]` |

### Test Helper Functions (conftest.py)

```python
get_session(task="aligned", seed=None)       # Returns (requests.Session, session_id)
make_action(session_id, action_type, target_ids, message, documents, lookahead)
step(session, session_id, action, timeout)  # Returns parsed JSON
get_reward(result)                           # Extracts reward float
get_obs(result)                              # Extracts observation dict
assert_near(value, target, tol=0.05)         # Tolerance assertion
assert_in_range(value, lo=0.0, hi=1.0)      # Range assertion
```

---

## 9. Coverage Gaps (Prioritized)

### HIGH Priority

**Missing deterministic test for veto trigger timing**
- No test validates that veto fires at the exact correct step across all three veto-powered stakeholders (Legal, Finance, ExecSponsor)
- `test_4_3_veto_deterministic` only tests Legal with seed=42
- Affects: Correctness of veto mechanism for production use

**Missing direct validation of CVaR computation correctness**
- No unit test directly validates `compute_cvar()` against known expected values
- `test_8_cvar_preferences.py` exists but its assertions are not visible in documentation
- Affects: Risk logic correctness

### MEDIUM Priority

**Lookahead not directly validated end-to-end**
- `test_2_10_lookahead_improves_prediction_accuracy` tests accuracy metric but not whether using lookahead actually improves outcomes
- No test verifies that an agent using lookahead outperforms one not using it
- Affects: Planning reliability, lookahead ROI justification

**Belief update likelihoods not tested against ground truth**
- `ACTION_LIKELIHOODS` dict in `belief_tracker.py` defines behavior but no test validates these likelihoods produce expected belief changes
- Affects: Bayesian update correctness

**Scenario-specific graph generation not tested for edge case seeds**
- Graph identifiability only tested with 20 seeds; no adversarial seed testing
- Some seeds may produce degenerate graphs with no edges
- Affects: Graph robustness across seed space

### LOW Priority

**Web UI interaction tests incomplete**
- `test_web_ui.py` exists but test assertions are not reviewed
- Non-critical since UI is optional wrapper around API

**Training convergence not validated**
- No test checks that `HeuristicPolicyAdapter.update_from_batch()` actually changes behavior meaningfully
- Training metrics collected but convergence not asserted
- Affects: Training effectiveness

**Partial observability edge cases not tested**
- No test for what happens when engagement noise pushes value outside [0, 1] bounds
- `_update_noisy_engagement()` clips values, but clip boundaries not tested under extreme noise
- Affects: Observation robustness

---

## 10. Reliability Assessment

### Strengths

1. **Deterministic reward computation**: Reward scoring is purely state-based with no external randomness, enabling reproducible debugging
2. **Comprehensive veto testing**: Veto mechanism has dedicated test suite covering precursor warnings, deterministic triggers, scenario differentiation, and stakeholder specificity
3. **Causal graph unit tests**: Container-side unit tests validate propagation, normalization, damping, and graph invariants directly without network dependency
4. **Schema validation at API boundary**: Action normalization via `OutputValidator` ensures malformed actions are rejected before entering environment
5. **Lookahead prediction accuracy tracking**: Built-in accuracy metric (`prediction_accuracy`) enables continuous monitoring of lookahead quality
6. **Multi-scenario coverage**: All three scenarios (aligned, conflicted, hostile_acquisition) tested across all critical test files
7. **Strategy comparison baseline**: `test_9_6_strategy_comparison_possible` validates that better strategy yields better outcomes

### Weaknesses

1. **Container dependency for integration tests**: Most v3 tests require live Docker container; cannot run in environments without Docker
2. **Stochastic thresholds in tests**: Some assertions use relaxed thresholds (65% echo rate, 2/5 non-zero deltas) that could mask intermittent failures
3. **No property-based testing**: Tests use fixed seeds and fixed action sequences; no exhaustive property-based validation
4. **API key dependency is fail-soft**: Tests that depend on LLM summaries (`_generate_summary()`) silently skip on key absence, reducing coverage
5. **Training convergence not validated**: No assertions that training actually improves policy performance over baseline
6. **Limited concurrency testing**: Session pool uses threading Lock, but no tests validate behavior under concurrent session modifications

### Overall Reliability Level

**MEDIUM-HIGH**

The test suite provides strong coverage of core mechanisms (reward, veto, causal propagation, belief updates). Deterministic components are well-validated. Stochastic and LLM-dependent paths are tested but with fail-soft fallbacks that reduce coverage when dependencies are absent. The container dependency limits test accessibility but ensures consistent runtime environment. The system is suitable for production use with the identified gaps noted for future improvement.