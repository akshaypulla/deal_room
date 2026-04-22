# DealRoom v3 Testing Specification

## 1. Test Setup

### Frameworks
- **pytest**: Test runner for unit and integration tests
- **requests**: HTTP client for API/integration tests against live server

### Test Structure
```
tests/
├── conftest.py              # Root pytest config + fixtures
├── v3/
│   ├── conftest.py         # V3-specific config (loads .env, Docker helpers)
│   ├── test_00_environment_setup.py    # Section 0: Environment validation
│   ├── test_01_schema_validation.py     # Section 1: Observation schema
│   ├── test_02_reward_integrity.py     # Section 2: Reward integrity
│   ├── test_03_causal_inference.py      # Section 3: Causal signals
│   ├── test_04_cvar_veto.py            # Section 4: Veto mechanism
│   ├── test_05_episode_isolation.py     # Section 5: Session isolation
│   ├── test_06_probabilistic_signals.py # Section 6: Signal generation
│   ├── test_07_causal_graph.py          # Section 7: Graph operations
│   ├── test_08_cvar_preferences.py     # Section 8: CVaR preferences
│   ├── test_09_full_episode_e2e.py      # Section 9: End-to-end episodes
│   ├── test_10_training_infrastructure.py  # Section 10: Training harness
│   ├── test_10_training_integration.py  # Section 10: Training integration
│   ├── test_11_research_properties.py   # Section 11: Research claims
│   └── test_assertion_hygiene.py        # Assertion style checks
├── unit/
│   ├── test_utterance_scorer.py   # UtteranceScorer unit tests
│   ├── test_cvar_preferences.py   # CVaR computation unit tests
│   ├── test_causal_graph.py       # CausalGraph unit tests
│   ├── test_belief_tracker.py     # Bayesian update tests
│   ├── test_deliberation_engine.py
│   ├── test_curriculum.py
│   ├── test_grader.py
│   ├── test_models.py
│   ├── test_semantics.py
│   ├── test_claims.py
│   ├── test_validator.py
│   ├── test_inference.py
│   └── test_observation_mechanism.py
├── integration/
│   ├── test_api_sessions.py    # API session management
│   ├── test_environment.py     # Environment integration
│   ├── test_web_ui.py          # Gradio UI tests
│   └── test_end_to_end.py       # Full pipeline tests
├── e2e/
│   └── test_workflows.py        # Workflow tests
└── performance/
    └── test_benchmarking.py    # Benchmark tests
```

### Key Fixtures (tests/conftest.py)
```python
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
        message="...",
    )
```

### Key Fixtures (tests/v3/conftest.py)
```python
BASE_URL = os.getenv("DEALROOM_BASE_URL", "http://127.0.0.1:7860")
CONTAINER_NAME = os.getenv("DEALROOM_CONTAINER_NAME", "dealroom-v3-test")

def get_session(task="aligned", seed=None) → (requests.Session, session_id)
def make_action(session_id, action_type, target_ids, message, documents, lookahead)
def step(session, session_id, action, timeout=30) → parsed result
def get_reward(result) → float
def get_obs(result) → observation dict
def assert_near(value, target, tol=0.05)
def assert_in_range(value, lo=0.0, hi=1.0)
```

### Test Execution
- **Unit tests**: Run directly against code modules (no server needed)
- **Integration tests**: Require live server at `BASE_URL` (Docker container)
- **V3 tests**: API-level tests requiring `requests` and running server

## 2. Test Coverage

### V3 API Tests (tests/v3/)

#### Section 0: Environment Setup (test_00_environment_setup.py)
| Test | What it validates |
|------|-------------------|
| `test_0_1_python_deps` | `requests`, `numpy` imports available |
| `test_0_2_api_keys_configured` | MiniMax API key detectable if set |
| `test_0_3_docker_container_running` | Docker container running |
| `test_0_4_server_endpoints_responsive` | `/health`, `/metadata`, `/reset`, `/step` return 200 |
| `test_0_5_llm_client_imports` | `llm_client` module importable |

#### Section 1: Schema Validation (test_01_schema_validation.py)
| Test | What it validates |
|------|-------------------|
| `test_1_1_all_required_fields_present` | 18 required fields in observation |
| `test_1_2_no_hidden_fields_exposed` | No internal fields (G, causal_graph, etc.) exposed |
| `test_1_3_field_types_correct` | Types match spec (int, dict, list, bool) |
| `test_1_4_engagement_history_window_size` | History has >= 5 entries |
| `test_1_5_engagement_level_delta_single_float` | delta is numeric, not dict |
| `test_1_6_cross_stakeholder_echoes_is_list` | echoes is list of dicts |
| `test_1_7_stakeholder_messages_populated_after_step` | messages dict populated |
| `test_1_8_action_schema_accepts_lookahead` | Lookahead action accepted |
| `test_1_9_approval_path_progress_structure` | progress dict with 'band' field |
| `test_1_10_deal_stage_valid_transitions` | round_number increments, deal_stage valid |
| `test_1_11_documents_field_format` | documents as `[{name, content}, ...]` |

#### Section 2: Reward Integrity (test_02_reward_integrity.py)
| Test | What it validates |
|------|-------------------|
| `test_2_1_reward_is_single_float` | reward is float in [0, 1], not dict |
| `test_2_2_lookahead_cost_is_exactly_007` | goal cost = 0.07 (LOOKAHEAD_COST) |
| `test_2_3_reward_in_range_after_valid_actions` | all 5 dimensions in [0, 1] |
| `test_2_4_deterministic_reward_with_seed` | same seed → same reward (within 1e-9) |
| `test_2_5_repeat_same_action_does_not_escalate_reward` | repeated action doesn't inflate reward |
| `test_2_6_different_targets_different_causal_scores` | different targets → different causal score |
| `test_2_7_informative_action_outperforms_empty` | substantive action >= empty message reward |
| `test_2_8_reward_non_trivial_variance` | reward spread > 0.05 across actions |
| `test_2_9_good_documentation_higher_than_poor` | good docs >= poor docs reward |
| `test_2_10_lookahead_improves_prediction_accuracy` | lookahead accuracy > 0.60 |

#### Section 3: Causal Inference (test_03_causal_inference.py)
| Test | What it validates |
|------|-------------------|
| `test_3_1_targeted_stakeholder_engagement_changes` | engagement_level_delta is numeric |
| `test_3_2_cross_stakeholder_echoes_detected` | echoes present in >= 65% episodes |
| `test_3_3_engagement_history_window_slides` | history length stable across steps |
| `test_3_4_engagement_noise_not_cancellable` | >= 2 non-zero deltas in 5 steps |
| `test_3_5_different_targets_different_patterns` | Finance vs Legal targeting produces different deltas |
| `test_3_6_weak_signals_for_non_targeted` | weak signals in >= 20% episodes |
| `test_3_7_echo_content_structure` | echo dicts have sender field |
| `test_3_8_causal_signal_responds_to_graph_structure` | hub (ExecSponsor) > leaf (Procurement) impact |

#### Section 4: CVaR Veto (test_04_cvar_veto.py)
| Test | What it validates |
|------|-------------------|
| `test_4_1_veto_precursor_before_veto` | precursor appears before veto |
| `test_4_2_aligned_no_early_veto` | aligned scenario survives first step |
| `test_4_3_veto_terminates_episode` | hostile scenario triggers veto within 20 steps |
| `test_4_3_veto_deterministic` | deterministic Legal veto by step 5 (seed=42) |
| `test_4_4_timeout_terminates` | episode terminates after max_rounds |
| `test_4_5_scenario_difficulty_differentiation` | hostile reaches pressure earlier than aligned |
| `test_4_6_veto_precursor_is_stakeholder_specific` | precursors tied to specific stakeholders |
| `test_4_7_cveto_not_just_eu` | veto fires even when EU may be positive |

#### Section 5: Episode Isolation (test_05_episode_isolation.py)
| Test | What it validates |
|------|-------------------|
| `test_5_1_different_seeds_different_initial_state` | different seeds → different engagement levels |
| `test_5_2_round_number_resets_to_zero` | round_number = 0 after reset |
| `test_5_3_done_false_after_reset` | done=False after reset for all 3 scenarios |
| `test_5_4_engagement_history_initialized` | history has >= 5 entries |
| `test_5_5_round_number_increments_correctly` | round_number increments 0→1→2... |
| `test_5_6_all_three_scenarios_work` | aligned, conflicted, hostile_acquisition all return 200 |
| `test_5_7_same_session_same_state_across_steps` | round_number consistent within session |
| `test_5_8_reset_clears_all_session_state` | reset clears round_number, done, history |

### Unit Tests (tests/unit/)

#### test_utterance_scorer.py
| Class | Tests |
|-------|-------|
| `TestLookaheadCost` | `test_lookahead_cost_value` (== 0.07), `test_lookahead_cost_subtracted_from_goal` |
| `TestScoringDimensions` | `test_all_dimensions_in_range` (20 random inputs, all dims in [0,1]) |
| `TestCausalScoring` | `test_causal_score_deterministic` (same graph+target → identical score) |
| `TestUtteranceScore` | `test_utterance_score_defaults`, `test_weighted_sum`, `test_to_dict` |

#### test_cvar_preferences.py
| Class | Tests |
|-------|-------|
| `TestCVaRComputation` | `test_cvar_veto_fires_despite_positive_expected_utility`, `test_cvar_does_not_fire_with_full_documentation`, `test_cvar_decreases_monotonically_with_documentation`, `test_cvar_computation_on_uniform_distribution`, `test_cvar_handles_empty_outcomes` |
| `TestRiskProfileOrdering` | `test_risk_profile_ordering`, `test_lambda_risk_weight` |
| `TestVetoTrigger` | `test_veto_fires_above_tau`, `test_veto_regardless_of_expected_utility` |
| `TestOutcomeDistribution` | `test_outcome_distribution_sums_to_one`, `test_outcome_distribution_respects_domain` |
| `TestObservableSignals` | `test_observable_signals_for_legal`, `test_observable_signals_risk_tolerance` |
| `TestArchetypes` | `test_all_archetypes_defined`, `test_archetype_values_locked`, `test_archetype_utility_weights` |

#### test_causal_graph.py
| Class | Tests |
|-------|-------|
| `TestCausalGraphConstruction` | `test_sample_graph_returns_valid_graph`, `test_sample_graph_no_self_edges`, `test_weight_range` [0.05, 0.95], `test_authority_invariant` (ExecSponsor >= 1 outgoing), `test_authority_weights_sum_to_one` |
| `TestBeliefDistribution` | `test_positive_mass_calculation`, `test_negative_mass_calculation`, `test_belief_copy_independent`, `test_create_neutral_beliefs` |
| `TestPropagation` | `test_propagation_direction`, `test_damping_prevents_runaway`, `test_apply_belief_delta_positive`, `test_apply_belief_delta_negative`, `test_apply_belief_delta_with_damping` |
| `TestBetweennessCentrality` | `test_centrality_hub_higher_than_leaves`, `test_isolated_node_has_zero_centrality` |
| `TestIdentifiability` | `test_behavioral_signature_distinct`, `test_graph_identifiability_statistical` |
| `TestEngagementLevel` | `test_engagement_level_range` [-1,1], `test_engagement_level_neutral_is_zero` |
| `TestScenarioParams` | `test_aligned_sparser_than_conflicted` |

### Integration Tests (tests/integration/)

#### test_api_sessions.py
- Session creation, reset, step workflows
- Session state isolation

#### test_environment.py
- Environment reset/step behavior
- Full episode execution

#### test_web_ui.py
- Gradio UI loading and interaction

#### test_end_to_end.py
- Complete negotiation workflows

## 3. Test Cases

### Example: test_02_reward_integrity.py::test_2_2_lookahead_cost_is_exactly_007
```python
def test_2_2_lookahead_cost_is_exactly_007():
    # Without lookahead
    r = session.post(f"{BASE_URL}/reset", json={"task_id": "aligned", "seed": 20})
    sid1 = r.json().get("metadata", {}).get("session_id")
    r1 = session.post(f"{BASE_URL}/step", json=make_action(sid1, "direct_message", ["Finance"], "Test message.", [], None))
    goal1 = float(r1.json().get("info", {}).get("reward_components", {}).get("goal", 0))

    # With lookahead
    r = session.post(f"{BASE_URL}/reset", json={"task_id": "aligned", "seed": 20})
    sid2 = r.json().get("metadata", {}).get("session_id")
    r2 = session.post(f"{BASE_URL}/step", json=make_action(sid2, "direct_message", ["Finance"], "Test message.", [], {"depth": 2, "n_hypotheses": 2, "action_draft": ...}))
    goal2 = float(r2.json().get("info", {}).get("reward_components", {}).get("goal", 0))

    diff = goal1 - goal2
    expected_cost = LOOKAHEAD_COST  # 0.07
    assert abs(diff - expected_cost) < 0.015
```

### Example: test_cvar_preferences.py::test_cvar_veto_fires_despite_positive_expected_utility
```python
def test_cvar_veto_fires_despite_positive_expected_utility():
    legal_profile = get_archetype("Legal")
    deal_terms = {"price": 100000, "timeline_weeks": 12, "has_dpa": False, "has_security_cert": False}
    rng = np.random.default_rng(42)
    expected_utility, cvar_loss = evaluate_deal(deal_terms, legal_profile, rng, n_samples=1000)
    veto_triggered = check_veto_trigger(cvar_loss, legal_profile)

    assert expected_utility > 0
    assert veto_triggered  # CVaR > tau even though EU > 0
```

### Example: test_causal_graph.py::test_propagation_direction
```python
def test_propagation_direction():
    graph = CausalGraph(nodes=["A","B","C"], edges={("A","B"): 0.7, ("A","C"): 0.0}, ...)
    beliefs_before = create_neutral_beliefs(["A","B","C"])
    beliefs_after = create_neutral_beliefs(["A","B","C"])
    beliefs_after["A"] = apply_positive_delta(beliefs_before["A"], 0.4)

    result = propagate_beliefs(graph, beliefs_before, beliefs_after, n_steps=3)

    assert result["B"].positive_mass() > beliefs_before["B"].positive_mass() + 0.05
```

## 4. Execution Process

### Running Tests
```bash
# All tests
pytest tests/ -v

# Unit tests only
pytest tests/unit/ -v

# V3 integration tests (requires running server)
pytest tests/v3/ -v

# Specific test file
pytest tests/v3/test_02_reward_integrity.py -v

# Specific test
pytest tests/v3/test_02_reward_integrity.py::test_2_2_lookahead_cost_is_exactly_007 -v
```

### Docker-based V3 Tests
Tests in `tests/v3/` require:
1. Docker container `dealroom-v3-test:latest` running on port 7860
2. `MINIMAX_API_KEY` environment variable (optional, tests degrade gracefully)
3. `.env` file with `DEALROOM_BASE_URL` (defaults to `http://127.0.0.1:7860`)

### Test Helpers (tests/v3/conftest.py)
```python
check_container_running()     # docker ps check
ensure_container()           # Start container if not running
get_session(task, seed)     # Create session + initial observation
make_action(...)            # Build action payload
step(session, sid, action)  # Execute step
get_reward(result)           # Extract reward
get_obs(result)             # Extract observation
assert_near(value, target)  # Tolerance comparison
assert_in_range(value, lo, hi)  # Range check
```

## 5. Observed Gaps

Based on code coverage analysis:

### Missing Unit Tests
1. **`deal_room/environment/lookahead.py`**: No dedicated unit test file; `LookaheadSimulator._simulate_one_hypothesis()`, `_generate_hypotheses()`, `_predict_response_text()` untested
2. **`deal_room/committee/belief_tracker.py`**: No `test_belief_tracker.py` unit test found in unit tests directory (test file exists at `tests/unit/test_belief_tracker.py` per glob)
3. **`deal_room/committee/deliberation_engine.py`**: No unit tests; `CommitteeDeliberationEngine.run()` and `_generate_summary()` untested directly
4. **`deal_room/curriculum/adaptive_generator.py`**: `test_curriculum.py` exists but `analyze_failures()` detection logic and `generate_adaptive_scenario()` not fully covered

### Missing Integration Tests
1. **Training integration**: `test_10_training_integration.py` exists but GRPO training loop not validated end-to-end
2. **Gradio UI**: `test_web_ui.py` exists but UI interactions not fully validated
3. **Session pool pruning**: `_prune_locked()` and `_prune_oldest_locked()` not exercised in tests
4. **Validator normalization edge cases**: `_extract_action_type()`, `_extract_target()` not fully tested

### Not Tested (Test Files Missing)
1. `tests/unit/test_observation_mechanism.py` - referenced but contents unknown
2. `tests/unit/test_deliberation_engine.py` - referenced but contents unknown
3. `tests/unit/test_semantics.py` - semantics module untested
4. `tests/unit/test_claims.py` - claims module untested
5. `tests/unit/test_validator.py` - validator module partially tested
6. `tests/unit/test_grader.py` - grader module untested
7. `tests/unit/test_models.py` - model validation untested
8. `tests/unit/test_inference.py` - inference script untested
9. `tests/e2e/test_workflows.py` - end-to-end workflows untested
10. `tests/performance/test_benchmarking.py` - benchmarking untested

### Environment-Specific Gaps
1. **LLM client error paths**: Error classification in `classify_error()` not unit tested
2. **Retry policy**: `RetryPolicy.compute_backoff()` not tested
3. **Session expiration**: TTL-based session pruning not tested in integration
4. **Multi-session contention**: Concurrent session access not tested
