# DealRoom v3.6 — Test Documentation

**Version:** 3.6  
**Generated:** 2026-04-21  
**Purpose:** Comprehensive guide to all tests in DealRoom v3.6

---

## Table of Contents

1. [Test Organization](#1-test-organization)
2. [Test Categories Overview](#2-test-categories-overview)
3. [Unit Tests (`tests/unit/`)](#3-unit-tests-testsunit)
4. [Integration Tests (`tests/integration/`)](#4-integration-tests-testsintegration)
5. [E2E Tests (`tests/e2e/`)](#5-e2e-tests-testse2e)
6. [Performance Tests (`tests/performance/`)](#6-performance-tests-testsperformance)
7. [V3 Tests (`tests/v3/`)](#7-v3-tests-testsv3)
8. [Test Execution Guide](#8-test-execution-guide)
9. [Test Results Summary](#9-test-results-summary)
10. [Known Test Issues](#10-known-test-issues)
11. [Recommendations](#11-recommendations)

---

## 1. Test Organization

The test suite is organized into 5 categories located in `tests/`:

```
tests/
├── conftest.py              # Shared pytest fixtures
├── unit/                   # Unit tests for individual modules
│   ├── test_belief_tracker.py
│   ├── test_causal_graph.py
│   ├── test_cvar_preferences.py
│   ├── test_curriculum.py
│   ├── test_deliberation_engine.py
│   ├── test_grader.py
│   ├── test_inference.py
│   ├── test_models.py
│   ├── test_observation_mechanism.py
│   ├── test_claims.py
│   ├── test_semantics.py
│   ├── test_utterance_scorer.py
│   └── test_validator.py
├── integration/             # Integration tests
│   ├── test_api_sessions.py
│   ├── test_end_to_end.py
│   ├── test_environment.py
│   └── test_web_ui.py
├── e2e/                    # End-to-end tests
│   ├── test_playwright_e2e.py
│   └── test_workflows.py
├── performance/             # Performance benchmarks
│   └── test_benchmarking.py
└── v3/                     # V3-specific tests
    ├── conftest.py
    ├── test_00_environment_setup.py
    ├── test_01_schema_validation.py
    ├── test_02_reward_integrity.py
    ├── test_03_causal_inference.py
    ├── test_04_cvar_veto.py
    ├── test_05_episode_isolation.py
    ├── test_06_probabilistic_signals.py
    ├── test_07_causal_graph.py
    ├── test_08_cvar_preferences.py
    ├── test_09_full_episode_e2e.py
    ├── test_10_training_infrastructure.py
    ├── test_10_training_integration.py
    ├── test_11_research_properties.py
    ├── test_assertion_hygiene.py
    ├── test_P0_comprehensive.py
    └── test_scorer_unit.py
```

---

## 2. Test Categories Overview

| Category | Count | Purpose | Runtime |
|----------|-------|---------|---------|
| Unit | ~100 | Test individual modules in isolation | Fast (<5s) |
| Integration | ~30 | Test module interactions | Medium (5-30s) |
| E2E | ~10 | Test full workflows with server | Slow (30-60s) |
| Performance | ~5 | Benchmark environment speed | Varies |
| V3 | ~80 | V3-specific validation | Medium (10-30s) |

---

## 3. Unit Tests (`tests/unit/`)

### 3.1 `test_belief_tracker.py`

**Purpose:** Tests Bayesian belief update logic.

**Key Tests:**
- `TestLikelihoodTable::test_likelihood_values_in_range` — Likelihood ratios must be in [0, 1]
- `TestLikelihoodTable::test_get_likelihood_exact_match` — Exact document matches get correct likelihoods
- `TestLikelihoodTable::test_get_likelihood_document_matching` — Partial matches work
- `TestBayesianUpdate::test_distribution_normalizes` — Beliefs sum to 1.0 after update
- `TestBayesianUpdate::test_bayesian_update_concentrates_belief` — Strong action = concentrated beliefs
- `TestBayesianUpdate::test_targeted_vs_nontargeted_strength` — Targeted actions have stronger effect (ISSUE: May fail due to damping factor)
- `TestBayesianUpdate::test_damping_factor_applied` — Non-targeted actions have damping applied
- `TestBayesianUpdate::test_positive_mass_increases_on_competent_action` — Competent actions increase positive mass
- `TestBayesianUpdate::test_negative_mass_increases_on_deceptive_action` — Deceptive actions increase negative mass
- `TestBayesianUpdate::test_confidence_increases_with_informatie_action` — Information gain increases confidence
- `TestBayesianUpdate::test_history_records_updates` — Update history is recorded
- `TestEngagementLevel::test_engagement_level_matches_positive_minus_negative` — Engagement = positive - negative
- `TestEngagementLevel::test_engagement_level_bounds` — Engagement in [-1, 1]
- `TestNeutralBeliefs::test_neutral_beliefs_uniform` — Neutral beliefs are uniform
- `TestNeutralBeliefs::test_neutral_beliefs_confidence` — Confidence starts at maximum for uniform

**What it tests:**
- `bayesian_update()` function from `belief_tracker.py`
- `compute_engagement_level()` function
- Action likelihood table lookup
- Belief normalization
- Confidence computation

### 3.2 `test_causal_graph.py`

**Purpose:** Tests causal graph structure and belief propagation.

**Key Tests:**
- `TestCausalGraphConstruction::test_sample_graph_returns_valid_graph` — Graph has expected structure
- `TestCausalGraphConstruction::test_sample_graph_no_self_edges` — No self-referential edges
- `TestCausalGraphConstruction::test_weight_range` — Edge weights in [0.05, 0.95]
- `TestCausalGraphConstruction::test_authority_invariant` — Authority influences edge probability
- `TestCausalGraphConstruction::test_authority_weights_sum_to_one` — Authority weights normalized
- `TestBeliefDistribution::test_positive_mass_calculation` — positive_mass() correct
- `TestBeliefDistribution::test_negative_mass_calculation` — negative_mass() correct
- `TestBeliefDistribution::test_belief_copy_independent` — Copy is deep, not shallow
- `TestBeliefDistribution::test_create_neutral_beliefs` — Uniform distribution created
- `TestPropagation::test_propagation_direction` — Beliefs propagate correctly
- `TestPropagation::test_damping_prevents_runaway` — Multiple steps damped
- `TestPropagation::test_apply_belief_delta_positive` — Positive delta increases positive mass
- `TestPropagation::test_apply_belief_delta_negative` — Negative delta increases negative mass
- `TestBetweennessCentrality::test_centrality_hub_higher_than_leaves` — Hub nodes more central
- `TestBetweennessCentrality::test_isolated_node_has_zero_centrality` — Isolated nodes have zero centrality
- `TestIdentifiability::test_behavioral_signature_distinct` — Different actions produce distinct signatures
- `TestIdentifiability::test_graph_identifiability_statistical` — Graph type affects signature
- `TestEngagementLevel::test_engagement_level_range` — Engagement bounded
- `TestEngagementLevel::test_engagement_level_neutral_is_zero` — Neutral beliefs have zero engagement
- `TestScenarioParams::test_aligned_sparser_than_conflicted` — More edges in conflicted scenario

**What it tests:**
- `sample_graph()` for all three scenario types
- `BeliefDistribution` class methods
- `propagate_beliefs()` function
- `apply_positive_delta()` function
- `get_betweenness_centrality()` function
- `compute_behavioral_signature()` function
- `compute_engagement_level()` function

### 3.3 `test_cvar_preferences.py`

**Purpose:** Tests CVaR preference model and stakeholder risk profiles.

**Key Tests:**
- `TestCVaRComputation::test_cvar_veto_fires_despite_positive_expected_utility` — High alpha = veto risk even with good expected utility
- `TestCVaRComputation::test_cvar_does_not_fire_with_full_documentation` — DPA + security cert reduces CVaR
- `TestCVaRComputation::test_cvar_decreases_monotonically_with_documentation` — More docs = lower CVaR
- `TestCVaRComputation::test_cvar_computation_on_uniform_distribution` — Uniform outcomes handled
- `TestCVaRComputation::test_cvar_handles_empty_outcomes` — Empty outcomes return 0
- `TestRiskProfileOrdering::test_risk_profile_ordering` — Risk profiles ordered by lambda_risk
- `TestRiskProfileOrdering::test_lambda_risk_weight` — Lambda correctly weighted
- `TestVetoTrigger::test_veto_fires_above_tau` — Veto triggers when CVaR > tau
- `TestVetoTrigger::test_veto_regardless_of_expected_utility` — Veto can fire even with positive EU
- `TestOutcomeDistribution::test_outcome_distribution_sums_to_one` — Monte Carlo sums to 1
- `TestOutcomeDistribution::test_outcome_distribution_respects_domain` — Domain affects distribution
- `TestObservableSignals::test_observable_signals_for_legal` — Legal gets compliance signals
- `TestObservableSignals::test_observable_signals_risk_tolerance` — Risk tolerance signal correct
- `TestArchetypes::test_all_archetypes_defined` — All 6 archetypes exist
- `TestArchetypes::test_archetype_values_locked` — Profiles are locked (no runtime modification)
- `TestArchetypes::test_archetype_utility_weights` — Utility weights sum to ~1.0

**What it tests:**
- `compute_cvar()` function
- `compute_outcome_distribution()` function
- `compute_expected_utility()` function
- `check_veto_trigger()` function
- `get_observable_signals()` function
- `ARCHETYPE_PROFILES` dictionary
- `StakeholderRiskProfile` dataclass

### 3.4 `test_curriculum.py`

**Purpose:** Tests adaptive curriculum generation.

**Key Tests:**
- `TestFailureAnalysis::test_failure_analysis_defaults` — Empty trajectories handled
- `TestCurriculumConfig::test_config_defaults` — Default config values correct
- `TestAdaptiveCurriculumGenerator::test_generator_initialization` — Pool initialized with scenarios
- `TestAdaptiveCurriculumGenerator::test_select_next_scenario` — Selection works
- `TestAdaptiveCurriculumGenerator::test_generate_adaptive_scenario` — Scenario generation works
- `TestAdaptiveCurriculumGenerator::test_analyze_failures_empty_trajectories` — Empty analysis safe
- `TestFailureDetection::test_failure_detection_f1` — F1 (CVaR veto) detected
- `TestFailureDetection::test_failure_detection_f3` — F3 (graph inference failure) detected
- `TestFailureDetection::test_failure_mode_descriptions_exist` — All F1-F6 described
- `TestDifficultyDistribution::test_scenario_pool_has_all_difficulties` — All difficulty levels in pool
- `TestDifficultyDistribution::test_capability_based_selection` — Capability affects selection
- `TestScenarioGeneration::test_generate_scenario_with_seed` — Seeded generation reproducible
- `TestScenarioGeneration::test_scenario_structure` — Scenario has required fields
- `TestCurriculumIntegration::test_full_cycle_analyze_and_generate` — Full cycle works
- `TestCreateCurriculumGenerator::test_create_curriculum_generator` — Factory works
- `TestCreateCurriculumGenerator::test_create_with_custom_config` — Custom config works

**What it tests:**
- `AdaptiveCurriculumGenerator` class
- `CurriculumConfig` dataclass
- `FailureAnalysis` dataclass
- `_detect_failures()` method
- `select_next_scenario()` method
- `generate_adaptive_scenario()` method
- `analyze_failures()` method

### 3.5 `test_deliberation_engine.py`

**Purpose:** Tests committee deliberation with belief propagation.

**Key Tests:**
- `TestDeliberationEngine::test_deliberation_result_structure` — Result has required fields
- `TestDeliberationEngine::test_deliberation_steps_per_scenario` — Steps match scenario config
- `TestDeliberationEngine::test_deliberation_updates_beliefs` — Beliefs updated after deliberation
- `TestDeliberationEngine::test_deliberation_propagation_deltas_recorded` — Deltas recorded
- `TestDeliberationEngine::test_deliberation_no_summary_when_no_targets` — Empty summary when no targets
- `TestDeliberationEngine::test_deliberation_pure_python_layer1` — Pure Python (no LLM needed)
- `TestLayer2Summary::test_layer2_returns_string_or_empty` — Summary is string or empty
- `TestDeliberationSteps::test_single_step_deliberation` — Single step works
- `TestDeliberationSteps::test_many_steps_damping` — More steps = more damping
- `TestBeliefPropagationIntegration::test_propagation_follows_graph_structure` — Propagation respects graph
- `TestBeliefPropagationIntegration::test_no_edges_no_propagation` — No edges = no propagation

**What it tests:**
- `CommitteeDeliberationEngine` class
- `run()` method
- `_generate_summary()` method
- Belief propagation through causal graph
- DELIBERATION_STEPS config

### 3.6 `test_grader.py`

**Purpose:** Tests terminal grading logic.

**Key Tests:**
- `test_grader_returns_zero_for_infeasible_close` — Infeasible close scores 0
- `test_grader_returns_positive_for_feasible_close` — Feasible close scores positive
- `test_grader_returns_zero_when_constraint_unresolved` — Unresolved constraint = 0
- `test_grader_score_is_strictly_inside_open_interval` — Score in (0, 1), not 0 or 1
- `test_grader_penalizes_relationship_damage` — Relationship damage reduces score

**What it tests:**
- `CCIGrader.compute()` function
- Score component weighting
- Terminal condition handling

### 3.7 `test_inference.py`

**Purpose:** Tests LLM inference configuration.

**Key Tests:**
- `test_inference_prefers_injected_proxy_credentials` — Proxy env vars respected
- `test_inference_does_not_force_llm_without_injected_proxy` — Fallback when no proxy
- `test_explicit_llm_flag_can_enable_local_message_generation` — Flag enables local
- `test_inference_uses_openai_client_when_proxy_env_present` — Proxy client used

**What it tests:**
- Inference configuration logic
- Environment variable handling
- API key precedence

### 3.8 `test_models.py`

**Purpose:** Tests Pydantic model validation.

**Key Tests:**
- `test_action_target_ids_are_deduplicated` — Deduplication works
- `test_state_is_callable_for_state_and_state_method_compat` — State callable
- `test_observation_supports_dynamic_fields` — Dynamic fields accepted
- `test_reward_model_tracks_value_done_and_info` — Reward model complete

**What it tests:**
- `DealRoomAction` validation
- `DealRoomObservation` model
- `DealRoomState` model
- `DealRoomReward` model

### 3.9 `test_observation_mechanism.py`

**Purpose:** Tests observation mechanism for partial observability.

**Key Tests:**
- `TestGNeverInObservation::test_g_never_in_observation` — Ground truth not exposed
- `TestGNeverInObservation::test_g_not_in_string_fields` — No G in any text field
- `TestEngagementMechanism::test_engagement_history_length` — History window size
- `TestEngagementMechanism::test_engagement_not_cancellable` — Engagement can't be cancelled
- `TestEngagementMechanism::test_reset_clears_all_state` — Reset clears history
- `TestEngagementMechanism::test_episode_seed_reproducibility` — Seed produces same observation
- `TestWeakSignals::test_weak_signal_structure` — Weak signals format correct
- `TestWeakSignals::test_weak_signals_after_action` — Signals generated after action
- `TestCrossStakeholderEchoes::test_cross_echoes_structure` — Echoes format correct
- `TestCrossStakeholderEchoes::test_echo_recall_rate` — Echo probability matches config
- `TestVetoPrecursors::test_veto_precursors_structure` — Veto precursors format correct
- `TestVetoPrecursors::test_veto_precursors_dont_expose_tau` — Tau values hidden
- `TestObservationSignals::test_all_five_signals_present` — All 5 required signals present
- `TestObservationSignals::test_observation_schema_complete` — Schema complete
- `TestObservationSignals::test_stakeholder_messages_after_targeted_action` — Messages generated
- `TestObservationNoise::test_engagement_delta_noise_present` — Noise applied
- `TestObservationContent::test_round_number_increments` — Round number increments
- `TestObservationContent::test_done_flag_after_max_rounds` — Done flag correct

**What it tests:**
- Observation construction
- Engagement noise mechanism
- Weak signals generation
- Cross-stakeholder echoes
- Veto precursors
- No ground-truth leakage

### 3.10 `test_claims.py`

**Purpose:** Tests commitment ledger.

**Key Tests:**
- `test_commitment_ledger_flags_numeric_contradiction` — Numeric contradictions detected
- `test_commitment_ledger_trims_history` — Rolling history limit enforced

**What it tests:**
- `CommitmentLedger` class
- `_is_contradiction()` method
- `_latest_for()` method

### 3.11 `test_semantics.py`

**Purpose:** Tests semantic analysis.

**Key Tests:**
- `test_semantic_analyzer_extracts_claims_and_artifacts` — Claims and artifacts extracted
- `test_semantic_analyzer_returns_known_backend` — Backend identified

**What it tests:**
- `SemanticAnalyzer` class
- `_extract_claims()` method
- Intent matching
- Tone scoring

### 3.12 `test_utterance_scorer.py`

**Purpose:** Tests 5D reward computation.

**Key Tests:**
- `TestLookaheadCost::test_lookahead_cost_value` — LOOKAHEAD_COST is 0.07
- `TestLookaheadCost::test_lookahead_cost_subtracted_from_goal` — Cost applied to goal
- `TestScoringDimensions::test_all_dimensions_in_range` — All dims in [0, 1]
- `TestCausalScoring::test_causal_score_deterministic` — Causal score deterministic
- `TestUtteranceScore::test_utterance_score_defaults` — Defaults correct
- `TestUtteranceScore::test_weighted_sum` — Weighted sum correct (may fail pre-fix)
- `TestUtteranceScore::test_to_dict` — to_dict() works

**What it tests:**
- `UtteranceScorer` class
- `UtteranceScore` dataclass
- `compute_prediction_accuracy()` function
- `LOG2_6` constant
- `LOOKAHEAD_COST` constant

### 3.13 `test_validator.py`

**Purpose:** Tests action validation.

**Key Tests:**
- `test_validator_normalizes_dynamic_targets` — Target normalization works
- `test_validator_soft_rejects_unknown_target` — Unknown targets flagged
- `test_validator_filters_proposed_terms` — Invalid terms filtered

**What it tests:**
- `OutputValidator` class
- `_normalize()` method
- `_resolve_target_ids()` method
- `_extract_action_type()` method

---

## 4. Integration Tests (`tests/integration/`)

### 4.1 `test_api_sessions.py`

**Purpose:** Tests HTTP API session management.

**Key Tests:**
- `test_state_requires_active_session` — State endpoint needs session
- `test_sessions_are_isolated_by_client_cookie` — Cookies isolate sessions

**Requires:** Server running on `BASE_URL` (default: `http://127.0.0.1:7860`)

### 4.2 `test_end_to_end.py`

**Purpose:** Full episode integration tests.

**Test Classes:**
- `TestFullEpisode` — Complete episode execution
- `TestVetoHandling` — Veto trigger and handling
- `TestLookaheadAction` — Lookahead action mechanics
- `TestMultipleScenarios` — All three scenarios
- `TestActionTypes` — Different action types
- `TestRewardIntegrity` — Reward bounds and info

**Requires:** Server running

### 4.3 `test_environment.py`

**Purpose:** V2.5 environment mechanics.

**Key Tests:**
- `test_reset_creates_dynamic_roster` — Dynamic stakeholder generation
- `test_step_returns_dense_reward_in_bounds` — Reward bounds
- `test_constraint_can_be_discovered_with_probe_and_artifact` — Constraint discovery
- `test_premature_close_applies_penalty` — Early close penalty
- `test_feasible_close_returns_terminal_score` — Feasible close scoring
- `test_feasible_close_on_final_round_scores_instead_of_timing_out` — Final round logic
- `test_legal_review_can_close_directly_on_final_round_when_ready` — Legal review close
- `test_timeout_terminal_score_stays_inside_open_interval` — Timeout scoring

**Requires:** Server running

### 4.4 `test_web_ui.py`

**Purpose:** Web UI routing and accessibility.

**Key Tests:**
- `test_root_redirects_to_web` — Root redirects correctly
- `test_web_page_exposes_wrapper_without_redirect` — Web page loads
- `test_web_slash_page_redirects_to_web` — Trailing slash redirect
- `test_ui_blocked_direct_access` — Direct /ui access blocked
- `test_health_endpoint_still_works` — Health check works

---

## 5. E2E Tests (`tests/e2e/`)

### 5.1 `test_playwright_e2e.py`

**Purpose:** Browser-based UI testing via Playwright.

**Key Tests:**
- `test_web_interface_loads` — UI loads without crash
- `test_api_endpoints` — API endpoints respond
- `test_session_isolation` — Sessions isolated
- `test_episode_completion` — Full episode completes
- `test_concurrent_reset` — Concurrent resets handled

**Requires:** Server running + Playwright installed

### 5.2 `test_workflows.py`

**Purpose:** Full workflow validation.

**Key Tests:**
- `test_baseline_runs_aligned` — Baseline runs on aligned scenario
- `test_hostile_baseline_survives_historically_failing_seeds` — Hostile seeds handled (may fail on certain seeds)
- `test_inference_logs_match_required_markers` — Inference logging correct

---

## 6. Performance Tests (`tests/performance/`)

### 6.1 `test_benchmarking.py`

**Purpose:** Environment performance benchmarks.

**Key Tests:**
- `test_reset_performance` — Reset under 100ms
- `test_step_performance` — Step under 50ms

---

## 7. V3 Tests (`tests/v3/`)

### 7.1 `test_00_environment_setup.py`

**Purpose:** Verify environment setup.

**Key Tests:**
- `test_0_1_python_deps` — Python dependencies available
- `test_0_2_api_keys_configured` — API keys loaded (optional)
- `test_0_3_docker_container_running` — Docker container running
- `test_0_4_server_endpoints_responsive` — Server endpoints respond
- `test_0_5_llm_client_imports` — LLM client imports

### 7.2 `test_01_schema_validation.py`

**Purpose:** V3 observation schema validation.

**Key Tests:**
- `test_1_1_all_required_fields_present` — Required fields exist
- `test_1_2_no_hidden_fields_exposed` — No G-values exposed
- `test_1_3_field_types_correct` — Types correct
- `test_1_4_engagement_history_window_size` — History window correct
- `test_1_5_engagement_level_delta_single_float` — Delta is float
- `test_1_6_cross_stakeholder_echoes_is_list` — Echoes is list
- `test_1_7_stakeholder_messages_populated_after_step` — Messages populated
- `test_1_8_action_schema_accepts_lookahead` — Lookahead in action schema
- `test_1_9_approval_path_progress_structure` — Approval progress correct
- `test_1_10_deal_stage_valid_transitions` — Deal stage transitions valid
- `test_1_11_documents_field_format` — Documents format correct

### 7.3 `test_02_reward_integrity.py`

**Purpose:** Reward integrity validation.

**Key Tests:**
- `test_2_1_reward_is_single_float` — Reward is scalar
- `test_2_2_lookahead_cost_is_exactly_007` — Lookahead cost exactly 0.07
- `test_2_3_reward_in_range_after_valid_actions` — Rewards bounded
- `test_2_4_deterministic_reward_with_seed` — Deterministic with seed
- `test_2_5_repeat_same_action_does_not_escalate_reward` — No reward inflation
- `test_2_6_different_targets_different_causal_scores` — Causal dimension discriminative
- `test_2_7_informative_action_outperforms_empty` — Actions have value
- `test_2_8_reward_non_trivial_variance` — Variance across actions
- `test_2_9_good_documentation_higher_than_poor` — Documentation quality rewarded
- `test_2_10_lookahead_improves_prediction_accuracy` — Lookahead predictive

**Requires:** Server running

### 7.4 `test_03_causal_inference.py`

**Purpose:** Causal inference validation.

**Key Tests:**
- `test_3_1_graph_seed_produces_deterministic_graph` — Seed reproducibility
- `test_3_2_different_seeds_produce_different_graphs` — Seed variation
- `test_3_3_belief_propagation_follows_authority` — Authority affects propagation
- `test_3_4_engagement_delta_matches_positive_mass_delta` — Engagement reflects beliefs

**Requires:** Server running

### 7.5 `test_04_cvar_veto.py`

**Purpose:** CVaR veto mechanism validation.

**Key Tests:**
- `test_4_1_veto_triggers_above_tau` — Veto fires above threshold
- `test_4_2_veto_precursor_streak_required` — Streak needed for veto
- `test_4_3_no_veto_without_streak` — Streak prevents false veto

**Requires:** Server running

### 7.6 `test_05_episode_isolation.py`

**Purpose:** Episode isolation validation.

**Key Tests:**
- `test_5_1_episodes_are_independent` — Episodes don't affect each other
- `test_5_2_reset_clears_all_state` — Reset fully clears state

**Requires:** Server running

### 7.7 `test_06_probabilistic_signals.py`

**Purpose:** Probabilistic signal validation.

**Key Tests:**
- `test_6_1_noise_is_bounded` — Noise bounded
- `test_6_2_signal_correlates_with_truth` — Signal correlates with truth

**Requires:** Server running

### 7.8 `test_07_causal_graph.py`

**Purpose:** Causal graph in V3 context.

**Key Tests:**
- `test_7_1_graph_generated_on_reset` — Graph generated on reset
- `test_7_2_graph_determinism_with_seed` — Seed produces same graph

**Requires:** Server running

### 7.9 `test_08_cvar_preferences.py`

**Purpose:** V3 CVaR preferences.

**Key Tests:**
- `test_8_1_cvar_losses_computed` — CVaR losses computed
- `test_8_2_thresholds_respected` — Thresholds applied

**Requires:** Server running

### 7.10 `test_09_full_episode_e2e.py`

**Purpose:** Full V3 episode end-to-end.

**Key Tests:**
- `test_9_1_full_episode_runs` — Episode runs to completion
- `test_9_2_terminal_outcome_recorded` — Terminal outcome recorded
- `test_9_3_metrics_tracked` — Metrics tracked throughout

**Requires:** Server running

### 7.11 `test_10_training_infrastructure.py`

**Purpose:** Training infrastructure validation.

**Key Tests:**
- `test_10_1_checkpoint_dir_exists` — Checkpoint directory exists
- `test_10_2_trainer_initializes` — Trainer initializes
- `test_10_3_curriculum_generator_initializes` — Curriculum generator works
- `test_10_4_colab_notebook_exists` — Colab notebook present
- `test_10_5_checkpoint_save_load_smoke` — Checkpoint save/load works

### 7.12 `test_P0_comprehensive.py`

**Purpose:** P0 critical path tests.

**Key Tests:**
- `test_P0_1a_baseline_vs_trained_comparison` — Trained > baseline
- `test_P0_3c_lookahead_cost_exactly_007` — Lookahead cost verified

**Requires:** Server running

### 7.13 `test_scorer_unit.py`

**Purpose:** Unit tests for UtteranceScorer.

**Key Tests:**
- `test_lookahead_cost_exactly_0_07` — LOOKAHEAD_COST = 0.07
- `test_log2_6_correct` — LOG2_6 correct (FIXED in v3.6)
- `test_lookahead_penalty_applied` — Lookahead penalty works
- `test_all_dimensions_bounded_0_1` — All dims bounded
- `test_goal_score_increases_with_approval` — Goal tracks approval
- `test_trust_targeted_delta` — Trust tracks targeting
- `test_information_entropy_reduction` — Info dimension works
- `test_determinism_consistency` — Deterministic scoring
- `test_prediction_accuracy` — Prediction accuracy works
- `test_scorer_reset` — Reset doesn't accumulate state
- `test_risk_cvar_no_profile_returns_0_5` — Risk fallback
- `test_causal_no_targets_returns_0` — Causal fallback
- `test_blocker_resolution_affects_goal` — Blockers affect goal

**Issue:** Many tests in this file call `UtteranceScorer.score()` with incorrect keyword arguments. The actual API expects `(action, state_before, state_after, true_graph, lookahead_used)` but tests pass `(beliefs_before, beliefs_after, graph, ...)`.

---

## 8. Test Execution Guide

### 8.1 Running All Tests

```bash
# From project root
cd /Users/akshaypulla/Documents/deal_room

# Run all tests with verbose output
python -m pytest tests/ -v --tb=short

# Quick summary (no traceback)
python -m pytest tests/ -q --tb=no

# Run specific category
python -m pytest tests/unit/ -v
python -m pytest tests/integration/ -v
python -m pytest tests/v3/ -v

# Run specific test file
python -m pytest tests/unit/test_causal_graph.py -v

# Run specific test
python -m pytest tests/unit/test_causal_graph.py::TestPropagation::test_propagation_direction -v
```

### 8.2 Running with Coverage

```bash
python -m pytest tests/ --cov=deal_room --cov-report=term-missing
```

### 8.3 Running V3 Tests Only

```bash
python -m pytest tests/v3/ -v
```

### 8.4 Running Server-Dependent Tests

Server-dependent tests require a running server:

```bash
# Start server (from project root)
python -m server.app

# Or with uvicorn
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

Then run the tests:
```bash
python -m pytest tests/integration/ tests/v3/ -v
```

---

## 9. Test Results Summary

### 9.1 Last Run Summary (v3.6)

```
Platform: macOS Python 3.12.4
Pytest: 9.0.3
Plugins: dash-3.1.1, langsmith-4.4.45, anyio-4.13.0

Total: 283 tests
Passed: 261
Failed: 22
Warnings: 1

Runtime: ~27 seconds
```

### 9.2 Failure Breakdown by Category

| Category | Failed | Total | Pass Rate |
|----------|--------|-------|-----------|
| Unit | 3 | ~100 | 97% |
| Integration | 0 | ~30 | 100% |
| E2E | 5 | ~10 | 50% |
| Performance | 0 | ~5 | 100% |
| V3 | 14 | ~80 | 82.5% |

### 9.3 Failed Tests List

```
tests/e2e/test_workflows.py::test_hostile_baseline_survives_historically_failing_seeds[3]
tests/e2e/test_workflows.py::test_hostile_baseline_survives_historically_failing_seeds[4]
tests/e2e/test_workflows.py::test_hostile_baseline_survives_historically_failing_seeds[6]
tests/e2e/test_workflows.py::test_hostile_baseline_survives_historically_failing_seeds[8]
tests/e2e/test_workflows.py::test_inference_logs_match_required_markers
tests/unit/test_belief_tracker.py::TestBayesianUpdate::test_targeted_vs_nontargeted_strength
tests/unit/test_belief_tracker.py::TestBayesianUpdate::test_damping_factor_applied
tests/unit/test_utterance_scorer.py::TestUtteranceScore::test_weighted_sum
tests/v3/test_02_reward_integrity.py::test_2_6_different_targets_different_causal_scores
tests/v3/test_10_training_infrastructure.py::test_10_4_colab_notebook_exists
tests/v3/test_P0_comprehensive.py::test_P0_1a_baseline_vs_trained_comparison
tests/v3/test_P0_comprehensive.py::test_P0_3c_lookahead_cost_exactly_007
tests/v3/test_scorer_unit.py::test_lookahead_penalty_applied
tests/v3/test_scorer_unit.py::test_all_dimensions_bounded_0_1
tests/v3/test_scorer_unit.py::test_goal_score_increases_with_approval
tests/v3/test_scorer_unit.py::test_trust_targeted_delta
tests/v3/test_scorer_unit.py::test_information_entropy_reduction
tests/v3/test_scorer_unit.py::test_determinism_consistency
tests/v3/test_scorer_unit.py::test_scorer_reset
tests/v3/test_scorer_unit.py::test_risk_cvar_no_profile_returns_0_5
tests/v3/test_scorer_unit.py::test_causal_no_targets_returns_0
tests/v3/test_scorer_unit.py::test_blocker_resolution_affects_goal
```

---

## 10. Known Test Issues

### 10.1 LOG2_6 Bug (FIXED in v3.6)

**Issue:** `LOG2_6 = math.log(2, 6)` was computing log base 6 of 2 (~0.39) instead of log base 2 of 6 (~2.58).

**Impact:** Confidence calculations and information dimension scoring were incorrect.

**Fix Applied:** Changed to `LOG2_6 = math.log(6) / math.log(2)` in both:
- `deal_room/rewards/utterance_scorer.py` (line 33)
- `deal_room/committee/belief_tracker.py` (line 11)

### 10.2 Test API Mismatch in `test_scorer_unit.py`

**Issue:** Tests call `UtteranceScorer.score()` with keyword arguments matching an older API:
```python
# Test calls:
scorer.score(beliefs_before=b_before, beliefs_after=b_after, graph=MockGraph(), ...)
```

**Actual API:**
```python
def score(self, action: DealRoomAction, state_before: Any, state_after: Any, true_graph: Any, lookahead_used: bool = False)
```

**Impact:** 12 tests in `test_scorer_unit.py` fail with `TypeError: Unexpected keyword argument`.

**Root Cause:** The tests were written for a different version of the API where `UtteranceScorer.score()` accepted `beliefs_before`, `beliefs_after`, `graph` directly.

### 10.3 Hostile Baseline Seeds

**Issue:** `test_hostile_baseline_survives_historically_failing_seeds` fails on seeds 3, 4, 6, 8.

**Impact:** The hostile_acquisition scenario is extremely sensitive to random seed.

**Root Cause:** These seeds may produce causal graph configurations that are particularly challenging for the baseline policy.

### 10.4 Colab Notebook Path

**Issue:** `test_10_4_colab_notebook_exists` fails due to path mismatch.

**Impact:** Test expects notebook at `/content/deal_room/deal_room/training/grpo_colab.ipynb` but it's at `deal_room/training/grpo_colab.ipynb`.

### 10.5 Inference Logs Test

**Issue:** `test_inference_logs_match_required_markers` fails due to logging format changes.

**Impact:** Test expects specific log markers that may have changed in recent refactoring.

---

## 11. Recommendations

### 11.1 Fix Test API Mismatch

The `test_scorer_unit.py` tests should be rewritten to use the correct API:
```python
# Instead of:
score = scorer.score(beliefs_before=b_before, beliefs_after=b_after, graph=MockGraph(), ...)

# Use:
from models import DealRoomAction
action = DealRoomAction(action_type="send_document", target="Legal", target_ids=["Legal"], message="test")
score = scorer.score(action=action, state_before=state_before, state_after=state_after, true_graph=graph, lookahead_used=False)
```

### 11.2 Harden Hostile Scenario

Investigate why certain seeds consistently fail and consider:
1. Adding more robust fallback policies for edge cases
2. Adjusting CVaR thresholds for extreme configurations
3. Adding seed-specific handling in curriculum generator

### 11.3 Update Colab Path

Update `test_10_4_colab_notebook_exists` to use correct relative path or skip when not in container environment.

### 11.4 Add Missing Test Fixtures

Some tests require server running but don't check gracefully:
```python
@pytest.mark.skipif(not server_available(), reason="Server not running")
def test_server_endpoint():
    ...
```

### 11.5 Increase Test Coverage

Areas needing more tests:
- `deliberation_engine.py` — Only ~11 tests
- `adaptive_generator.py` — Only ~16 tests
- Error handling paths in `llm_client.py`

---

## Appendix A: Pytest Configuration

**From `pyproject.toml`:**
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_classes = "Test*"
python_functions = "test_*"
addopts = "-v --tb=short"
```

---

## Appendix B: Fixtures (from `conftest.py`)

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
        message="I want to understand the real blocker so I can tailor the proposal responsibly.",
    )
```

---

## Appendix C: V3 Test Fixtures (from `tests/v3/conftest.py`)

```python
# Environment variables loaded from .env if present
BASE_URL = "http://127.0.0.1:7860"
CONTAINER_NAME = "dealroom-v3-test"

def get_session(task="aligned", seed=None):
    """Get fresh requests Session and initial observation."""
    
def make_action(session_id, action_type, target_ids, message="", documents=None, lookahead=None):
    """Build action dict for API calls."""

def step(session, session_id, action, timeout=30):
    """Execute step and return parsed result."""

def get_reward(result):
    """Extract reward from step result."""

def assert_near(value, target, tol=0.05):
    """Assert value is within tolerance of target."""
    
def assert_in_range(value, lo=0.0, hi=1.0):
    """Assert value is within range."""
```

---

*Document Version: 3.6 — 2026-04-21*
