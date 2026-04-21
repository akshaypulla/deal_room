# DealRoom Testing Documentation

## Test Directory Structure

```
tests/
├── conftest.py                    # Root pytest fixtures (env, aligned_env, simple_action)
├── unit/
│   ├── test_utterance_scorer.py  # Five-dimensional reward scoring
│   ├── test_deliberation_engine.py # Committee deliberation engine
│   ├── test_cvar_preferences.py   # CVaR computation and veto logic
│   ├── test_causal_graph.py       # Causal graph and belief propagation
│   ├── test_grader.py            # Terminal grader
│   ├── test_semantics.py          # Semantic analyzer
│   ├── test_belief_tracker.py    # Bayesian belief updates
│   ├── test_observation_mechanism.py # Observation building
│   ├── test_models.py            # Pydantic model validation
│   ├── test_curriculum.py        # Adaptive curriculum
│   ├── test_claims.py            # Commitment ledger
│   ├── test_validator.py         # Output validator
│   ├── test_inference.py         # Inference script
│   └── test_observation_mechanism.py
├── integration/
│   ├── test_end_to_end.py        # Environment V3 integration tests
│   ├── test_api_sessions.py      # API session management
│   ├── test_web_ui.py           # Web interface tests
│   └── test_environment.py       # Environment interaction tests
├── v3/
│   ├── conftest.py              # V3 test configuration and utilities
│   ├── test_00_environment_setup.py  # Environment setup & API key validation
│   ├── test_01_schema_validation.py   # Observation schema validation
│   ├── test_02_reward_integrity.py     # Reward integrity
│   ├── test_03_causal_inference.py    # Causal inference tests
│   ├── test_04_cvar_veto.py           # CVaR veto behavior
│   ├── test_05_episode_isolation.py  # Episode isolation
│   ├── test_06_probabilistic_signals.py # Signal generation
│   ├── test_07_causal_graph.py        # Causal graph tests
│   ├── test_08_cvar_preferences.py    # CVaR preferences
│   ├── test_09_full_episode_e2e.py   # Full episode E2E
│   ├── test_10_training_infrastructure.py # Training infra
│   ├── test_11_research_properties.py # Research property validation
│   ├── test_scorer_unit.py          # Utterance scorer unit tests
│   ├── test_llm_call.py             # LLM call tests
│   ├── test_llm_scoring.py         # LLM-based scoring tests
│   └── test_deterministic_scoring.py # Deterministic scoring
├── e2e/
│   └── test_workflows.py           # End-to-end workflow tests
├── performance/
│   └── test_benchmarking.py       # Performance benchmarking
└── container_test_api.py         # Container test API helper
```

---

## Root Test Configuration (`tests/conftest.py`)

### Fixtures

**`env`**: Creates a fresh `DealRoomEnvironment()` instance for each test.

**`aligned_env`**: Resets the env fixture with `seed=42, task_id="aligned"`.

**`simple_action`**: A pre-built `DealRoomAction` with:
- `action_type="direct_message"`
- `target="all"`
- `message="I want to understand the real blocker so I can tailor the proposal responsibly."`

---

## V3 Test Configuration (`tests/v3/conftest.py`)

### API Key Validation

`validate_api_keys()` exits with code 1 if `MINIMAX_API_KEY` or `OPENAI_API_KEY` are not set, printing instructions to create/edit `.env` file.

### Container Management

`check_container_running()` uses `docker ps` to verify the test container is running.

`ensure_container()` starts the container if not running via `docker run` with API keys passed as `-e` environment variables. Waits 15 seconds for startup.

### Session Helpers

`get_session(task, seed)`:
1. Creates a `requests.Session`
2. POSTs to `/reset` with task and seed
3. Returns `(session, session_id)`

`make_action(session_id, action_type, target_ids, message, documents, lookahead)`:
Builds a properly formatted action dict with metadata containing session_id.

`step(session, session_id, action, timeout)`:
POSTs action to `/step`, returns parsed JSON response.

`get_reward(result)`:
Extracts reward float from step result dict.

`get_obs(result)`:
Extracts observation dict from step result.

### Assertions

`assert_near(value, target, tol=0.05, msg)`:
Asserts `abs(value - target) <= tol`

`assert_in_range(value, lo=0.0, hi=1.0, msg)`:
Asserts `lo <= value <= hi`

---

## V3 Environment Setup Tests (`tests/v3/test_00_environment_setup.py`)

Validates the complete environment is ready before any tests run.

### Test 0.1: Python Dependencies

Imports `requests` and `numpy`. Optional: `python-dotenv`. Fails if critical deps missing.

### Test 0.2: API Keys Configured

Checks that `MINIMAX_API_KEY` and `OPENAI_API_KEY` are set and not placeholder values. Displays masked key values (first 4 and last 4 chars visible).

### Test 0.3: Docker Container Running

Verifies container with name `CONTAINER_NAME` (default: `dealroom-v3-test`) is running via `docker ps`.

### Test 0.4: Server Endpoints Responsive

Tests the full HTTP lifecycle:
1. `GET /health` → 200
2. `GET /metadata` → 200
3. `POST /reset` with `{"task": "aligned"}` → 200, extracts session_id
4. `POST /step` with a valid action using that session_id → 200

### Test 0.5: LLM Client Imports

Attempts to import `deal_room.environment.llm_client` and checks `MAX_TOKENS`. Prints warning if not available (fine in container where module path differs).

**Run with**: `python tests/v3/test_00_environment_setup.py`

---

## V3 Schema Validation Tests (`tests/v3/test_01_schema_validation.py`)

### Test 1.1: Required Fields Present

POSTs to `/reset`, checks that all 18 required fields exist in the observation:
`round_number`, `max_rounds`, `stakeholders`, `stakeholder_messages`, `engagement_level`, `engagement_level_delta`, `engagement_history`, `weak_signals`, `cross_stakeholder_echoes`, `veto_precursors`, `known_constraints`, `requested_artifacts`, `approval_path_progress`, `deal_momentum`, `deal_stage`, `active_blockers`, `days_to_deadline`, `done`

### Test 1.2: Hidden Fields Not Exposed

Recursive DFS search through the entire observation dict for exact matches of hidden field names. **Exact match only** — "graph" matches only the key "graph", NOT "graph_seed" or "graph_structure".

Hidden fields that must NEVER appear:
```
G, causal_graph, graph, true_beliefs, belief_distributions, belief_state, B_i, V_i,
tau, tau_i, risk_thresholds, cvar_thresholds, edge_weights, w_ij,
deliberation_transcript, deliberation_log, internal_dialogue, u_i, u_ij
```

### Test 1.3: Field Types Correct

After a step, validates:
- `round_number`: int
- `max_rounds`: int
- `stakeholders`: dict
- `engagement_level`: dict
- `engagement_level_delta`: numeric (not dict)
- `engagement_history`: list
- `weak_signals`: dict
- `cross_stakeholder_echoes`: list
- `veto_precursors`: dict
- `deal_momentum`: str
- `deal_stage`: str
- `active_blockers`: list
- `days_to_deadline`: numeric
- `done`: bool

### Test 1.4: Engagement History Window Size

Checks that `engagement_history` has ≥ 5 entries. Each entry must be a dict (stakeholder snapshots).

### Test 1.5: Engagement Level Delta Is Single Float

Asserts that `engagement_level_delta` is NOT a dict (must be a single numeric value per the spec).

### Test 1.6: Cross-Stakeholder Echoes Is List of Dicts

Each echo must be a dict with a sender field ("from", "from_stakeholder", or "sender").

### Test 1.7: Stakeholder Messages Populated After Step

After a `send_document` step, `stakeholder_messages` must be a dict.

### Test 1.8: Action Schema Accepts Lookahead

Posts a step with `lookahead: {depth: 2, n_hypotheses: 2, action_draft: {...}}`. Must return 200 and include reward info.

### Test 1.9: Approval Path Progress Structure

Each stakeholder's progress dict must have a `band` field with value in `["blocker", "neutral", "workable", "supporter"]`.

### Test 1.10: Deal Stage Valid Transitions

Validates initial `deal_stage` is one of `["evaluation", "negotiation", "legal_review", "final_approval", "closed"]`. After one step, `round_number` should be 1.

### Test 1.11: Documents Field Format

A `send_document` action with properly formatted documents must:
- Return 200
- Include a reward in the result

**Run with**: `python tests/v3/test_01_schema_validation.py`

---

## Unit Tests

### Utterance Scorer Tests (`tests/unit/test_utterance_scorer.py`)

#### Lookahead Cost Tests

**test_lookahead_cost_value**: `LOOKAHEAD_COST == 0.07`

**test_lookahead_cost_subtracted**: Same action with and without lookahead differs by exactly 0.07 on goal score.

**test_lookahead_cost_not_below_zero**: Even with base score < 0.07, goal score must be ≥ 0 (clipped).

#### Causal Scoring Tests

**test_causal_score_deterministic**: Same action + same graph returns identical causal score (no randomness).

**test_high_centrality_target_scores_higher**: In a star graph (Finance hub, leaves are Legal/TechLead/Procurement), targeting the hub scores ≥ 0.20 higher than targeting a leaf.

#### Scoring Dimensions Tests

**test_all_dimensions_in_range**: For 20 random inputs, all five dimensions (goal, trust, info, risk, causal) are in [0.0, 1.0].

#### Caching Tests

**test_cache_returns_same_score**: With `CACHE.clear()`, calling score twice with identical inputs returns identical goal scores. Second call has `cache_hit=True`.

**test_cache_hit_rate_after_repeated_calls**: After 10 calls with 3 unique messages, cache hit rate > 0.

#### Prediction Accuracy Tests

**test_prediction_accuracy_not_in_reward**: `prediction_accuracy` is computed but NOT added to any reward dimension.

**test_compute_prediction_accuracy**: Exact match → 1.0, partial match → [0, 1), empty → 0.0.

#### Heuristic Score Tests

**test_goal_heuristic_blocks_resolution**: Observation with blockers scores lower than without.

**test_trust_heuristic_role_keywords**: Role-specific message scores higher for that role than a generic message.

**test_information_heuristic_questions**: Questions score higher than statements.

**test_risk_heuristic_veto_precursors**: With veto precursors present, risk score is lower.

#### UtteranceScore Default Tests

**test_utterance_score_defaults**: All fields default to 0.0, prediction_accuracy=None, lookahead_used=False.

---

### Deliberation Engine Tests (`tests/unit/test_deliberation_engine.py`)

**test_deliberation_result_structure**: `DeliberationResult` has `updated_beliefs`, `summary_dialogue`, `propagation_deltas`.

**test_deliberation_steps_per_scenario**: aligned=3, conflicted=3, hostile_acquisition=4.

**test_deliberation_updates_beliefs**: After deliberation, all 3 stakeholders have updated beliefs.

**test_deliberation_propagation_deltas_recorded**: propagation_deltas contains entries for both A and B.

**test_deliberation_no_summary_when_no_targets**: With empty target_ids and render_summary=True, summary is None or empty.

**test_deliberation_pure_python_layer1**: `propagate_beliefs()` runs without LLM calls (just the propagation math).

**test_layer2_returns_string_or_empty**: Returns `DeliberationResult` with summary being string or None (LLM call may fail).

**test_single_step_deliberation**: With n_deliberation_steps=1, still produces valid results.

**test_many_steps_damping**: With 10 steps, damping prevents beliefs from going out of (0, 1) range.

**test_propagation_follows_graph_structure**: In A→B→C chain, B changes more than C (directly connected to A).

**test_no_edges_no_propagation**: With no edges, only targeted stakeholder changes.

---

### CVaR Preferences Tests (`tests/unit/test_cvar_preferences.py`)

#### Core CVaR Tests

**test_cvar_veto_fires_despite_positive_expected_utility**: The central research claim. For Legal with no DPA/security cert:
- expected_utility > 0
- check_veto_trigger returns True

This validates the "silent veto" behavior where CVaR > tau despite positive E[u].

**test_cvar_does_not_fire_with_full_documentation**: CVaR with full docs (DPA + security cert) is significantly lower than without docs (reduction > 0.05).

**test_cvar_decreases_monotonically_with_documentation**: Adding DPA alone doesn't increase CVaR.

**test_cvar_computation_on_uniform_distribution**: CVaR at alpha=0.50 < CVaR at alpha=0.95 for uniform distribution (lower percentile = higher tail expectation).

**test_cvar_handles_empty_outcomes**: `compute_cvar(np.array([]), alpha=0.95)` returns 0.0.

#### Risk Profile Ordering Tests

**test_risk_profile_ordering**: For the same deal terms without DPA, CVaR ordering should be: Legal ≥ Procurement ≥ Finance ≥ Operations ≥ TechLead ≥ ExecSponsor (with 0.7× tolerance for stochastic results).

**test_lambda_risk_weight**: Legal (lambda=0.70) should have lower or equal deal quality score than TechLead (lambda=0.30) due to higher CVaR penalty.

#### Veto Trigger Tests

**test_veto_fires_above_tau**: CVaR 0.20 > tau 0.15 → True; CVaR at tau 0.15 → False; CVaR 0.10 < tau → False.

**test_veto_regardless_of_expected_utility**: Veto fires based on CVaR alone, not E[u].

#### Outcome Distribution Tests

**test_outcome_distribution_sums_to_one**: Mean outcome is in [0, 1] (it's normalized samples, not probabilities summing to 1).

**test_outcome_distribution_respects_domain**: Compliance and cost profiles produce different outcome distributions, both with values in [0, 1.5].

#### Observable Signals Tests

**test_observable_signals_for_legal**: Legal with no DPA has compliance_concern signal > 0.5. With DPA, compliance_concern < without DPA.

**test_observable_signals_risk_tolerance**: `risk_tolerance = 1 - lambda_risk`. Legal < TechLead (higher lambda = lower tolerance).

#### Archetype Tests

**test_all_archetypes_defined**: All 6 archetypes exist with correct IDs.

**test_archetype_values_locked**: Legal alpha=0.95, tau=0.10, lambda=0.70; Finance alpha=0.90, tau=0.15, lambda=0.50; TechLead alpha=0.80, tau=0.25, lambda=0.30.

**test_archetype_utility_weights**: Legal has compliance_coverage > 0.3; Finance has roi_clarity > 0.3.

---

### Causal Graph Tests (`tests/unit/test_causal_graph.py`)

#### Graph Construction Tests

**test_sample_graph_returns_valid_graph**: Returns valid CausalGraph with correct scenario_type, nodes, edges dict, authority_weights dict.

**test_sample_graph_no_self_edges**: For all scenarios and seeds, no edge has source == destination.

**test_weight_range**: All edge weights are in [0.05, 0.95].

**test_authority_invariant**: ExecSponsor has at least 1 outgoing edge in every scenario type.

**test_authority_weights_sum_to_one**: authority_weights sum to 1.0 exactly.

#### Belief Distribution Tests

**test_positive_mass_calculation**: For distribution {competent:0.3, trustworthy:0.2, aligned:0.1, ...}, positive_mass = 0.6.

**test_negative_mass_calculation**: negative_mass = 0.4 for the same distribution.

**test_belief_copy_independent**: Copy doesn't affect original.

**test_create_neutral_beliefs**: Uniform distribution over 6 vendor types (1/6 each).

#### Propagation Tests

**test_propagation_direction**: In graph A→B with A→C weight 0, positive delta to A propagates positively to B.

**test_damping_prevents_runaway**: In fully connected graph with 5 steps, all beliefs stay in (0, 1).

**test_apply_belief_delta_positive**: Positive delta increases positive_mass, decreases negative_mass, total still sums to 1.0.

**test_apply_belief_delta_negative**: Negative delta has the opposite effect.

**test_apply_belief_delta_with_damping**: Higher damping = larger effective delta (less reduction).

#### Betweenness Centrality Tests

**test_centrality_hub_higher_than_leaves**: In a hub-spoke graph, hub centrality ≥ leaf centrality.

**test_isolated_node_has_zero_centrality**: Node C in A→B (C isolated) has centrality 0.0.

#### Identifiability Tests

**test_behavioral_signature_distinct**: Targeting different stakeholders in the same graph produces L1 distance > 0.02 between signatures.

**test_graph_identifiability_statistical**: Across 5 graphs, targeting same stakeholder produces distinguishable signatures (statistical test with tolerance).

#### Engagement Level Tests

**test_engagement_level_range**: Result is in [-1, 1].

**test_engagement_level_neutral_is_zero**: Uniform distribution gives engagement near 0.

#### Scenario Param Tests

**test_aligned_sparser_than_conflicted**: aligned graph has ≤ edges than conflicted graph.

---

### Grader Tests (`tests/unit/test_grader.py`)

**test_grader_returns_zero_for_infeasible_close**: `feasibility_state.is_feasible=False` → MIN_SCORE (0.01).

**test_grader_returns_positive_for_feasible_close**: `feasibility_state.is_feasible=True` and all constraints resolved → score in (0, 1).

**test_grader_returns_zero_when_constraint_unresolved**: Any unresolved constraint → MIN_SCORE.

**test_grader_score_is_strictly_inside_open_interval**: Both feasible and infeasible scores are in (0.0, 1.0), not at boundaries.

**test_grader_penalizes_relationship_damage**: State with permanent_marks (premature_close, semantic_contradiction) scores lower than clean state.

---

### Semantic Analyzer Tests (`tests/unit/test_semantics.py`)

**test_semantic_analyzer_extracts_claims_and_artifacts**: Message "Here is our ROI model. The price is 180000 and the rollout is 14 weeks with GDPR controls." with `documents: [{"type": "roi_model"}]`:
- Extracts "price" and "timeline_weeks" slots
- Matches "roi_model" artifact

**test_semantic_analyzer_returns_known_backend**: Backend is one of {"embedding", "tfidf", "lexical"}.

---

## Integration Tests

### End-to-End Environment Tests (`tests/integration/test_end_to_end.py`)

#### Full Episode Tests

**test_full_episode_runs_without_crash**: reset + 5 steps with valid obs/reward/done/info types.

**test_reward_vector_all_five_dimensions**: reward ≥ 0 (scalar in V3 implementation).

**test_environment_reset_produces_valid_observation**: round_number=0, max_rounds=10, done=False, stakeholders dict non-empty.

#### Veto Handling Tests

**test_veto_flag_possible**: In hostile_acquisition scenario, episode completes (either veto or max rounds).

#### Lookahead Tests

**test_lookahead_action_does_not_advance_state**: An action with lookahead has lookahead stored but doesn't change step behavior (documents current expected behavior).

**test_lookahead_reduces_goal_score**: Same action with and without lookahead should differ by 0.07 (behavior documented, direct check not possible with scalar reward).

#### Multiple Scenario Tests

**test_aligned_scenario_runs**: Completes without crash.

**test_conflicted_scenario_runs**: Completes without crash.

**test_hostile_acquisition_scenario_runs**: Completes without crash.

#### Action Type Tests

**test_send_document_action**: send_document with documents list processes correctly.

**test_multiple_targets**: group_proposal with multiple target_ids processes correctly.

#### Reward Integrity Tests

**test_reward_non_negative**: Reward is a valid numeric type.

**test_info_dict_contains_debug_info**: Info dict is a dict (contains deliberation info in V3).

---

## V3 Full Episode E2E Tests (`tests/v3/test_09_full_episode_e2e.py`)

### Helper: `run_episode(scenario, strategy, max_steps, seed)`

Three built-in strategies:
- **NEUTRAL**: Direct message, send_document (DPA, timeline, ROI), direct message cycles
- **AGGRESSIVE**: Repeated exec_escalation × 8
- **COMPREHENSIVE**: Multi-document send_documents covering all stakeholders

Returns: `(steps, terminal_outcome, all_rewards, final_observation)`

### Test 9.1: Aligned Completes

`run_episode("aligned", "neutral")` completes with ≥1 steps.

### Test 9.2: Conflicted Completes

`run_episode("conflicted", "comprehensive")` completes with ≥1 steps.

### Test 9.3: Hostile + Aggressive Produces Veto or Timeout

`run_episode("hostile_acquisition", "aggressive")` terminal must be one of: "veto", "timeout", error codes. NOT crash.

### Test 9.4: Reward Collected Across Episode

Collects rewards throughout the episode. At least 1 reward entry.

### Test 9.5: Reward Variance Non-Trivial

Reward range (max - min) across episode is non-zero.

### Test 9.6: Strategy Comparison Possible

Runs comprehensive (3 episodes) and aggressive (3 episodes) on conflicted scenario. Computes average reward per strategy. Shows that comparison data was collected.

### Test 9.7: Terminal Outcome Meaningful

Runs all combinations of (aligned, conflicted, hostile_acquisition) × (neutral, comprehensive, aggressive). Collects non-empty terminal outcomes. At least one non-"unknown" terminal seen.

### Test 9.8: Multi-Document Action Works

A single `send_document` action with 3 documents (DPA, security_cert, liability_terms) returns 200 with a reward.

---

## Additional V3 Tests

### Test 02: Reward Integrity (`tests/v3/test_02_reward_integrity.py`)

Tests that rewards are consistent and non-gameable.

### Test 03: Causal Inference (`tests/v3/test_03_causal_inference.py`)

Tests the causal graph inference properties.

### Test 04: CVaR Veto (`tests/v3/test_04_cvar_veto.py`)

Tests that CVaR veto triggers correctly under various conditions.

### Test 05: Episode Isolation (`tests/v3/test_05_episode_isolation.py`)

Tests that episodes don't bleed state between sessions.

### Test 06: Probabilistic Signals (`tests/v3/test_06_probabilistic_signals.py`)

Tests that weak signals and cross-stakeholder echoes are generated correctly.

### Test 07: Causal Graph (`tests/v3/test_07_causal_graph.py`)

V3-specific causal graph tests.

### Test 08: CVaR Preferences (`tests/v3/test_08_cvar_preferences.py`)

V3-specific CVaR preference tests.

### Test 10: Training Infrastructure (`tests/v3/test_10_training_infrastructure.py`)

Tests the GRPO trainer and curriculum generator.

### Test 11: Research Properties (`tests/v3/test_11_research_properties.py`)

Tests the core research claims about the environment.

---

## Performance Tests (`tests/performance/test_benchmarking.py`)

Benchmarks environment performance metrics.

---

## E2E Workflow Tests (`tests/e2e/test_workflows.py`)

End-to-end workflow tests for complete deal closing scenarios.

---

## Running Tests

### Run all unit tests
```bash
pytest tests/unit/ -v
```

### Run all integration tests
```bash
pytest tests/integration/ -v
```

### Run all V3 tests (requires container)
```bash
pytest tests/v3/ -v
```

### Run V3 environment setup first
```bash
python tests/v3/test_00_environment_setup.py
python tests/v3/test_01_schema_validation.py
python tests/v3/test_09_full_episode_e2e.py
```

### Run with specific test file
```bash
pytest tests/unit/test_utterance_scorer.py -v
pytest tests/unit/test_cvar_preferences.py -v
pytest tests/unit/test_causal_graph.py -v
```

### Run calibration
```bash
python calibrate.py
```

---

## Key Test Design Principles

1. **Deterministic scoring**: The utterance scorer and causal graph use no randomness — same inputs always produce same outputs.

2. **Research property validation**: Tests verify the core research claims (silent veto despite positive E[u], behavioral signatures are distinguishable, lookahead cost is exactly 0.07).

3. **Schema isolation**: V3 tests verify that internal state (causal graph, beliefs, CVaR parameters) is never exposed to the agent.

4. **Container-native testing**: V3 integration tests are designed to run against a Docker container with API keys, not against local code.

5. **Stochastic tolerance**: CVaR-based tests allow for statistical variation (e.g., 0.7× tolerance on profile ordering) since they use Monte Carlo sampling.
