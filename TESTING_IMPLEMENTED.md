# DealRoom v3 Testing Documentation

## What is being tested?

The DealRoom v3 test suite validates that the environment is implementation-correct according to the research specification. Tests are organized into sections (0-11) and cover environment setup, schema validation, reward integrity, causal inference, CVaR veto mechanism, episode isolation, probabilistic signals, causal graph unit tests, CVaR preferences unit tests, full episode end-to-end tests, training infrastructure, and research properties validation.

## Why this testing approach?

The tests verify both that the system works correctly (functionality) and that it cannot be gamed or exploited (reward integrity). The v3 test suite runs in two modes:
1. **HTTP API tests** (tests 0-5, 9, 11): Run against the live FastAPI server via HTTP requests on `http://127.0.0.1:7860`.
2. **Container unit tests** (tests 6-8, 10, scorer unit tests): Run inside the Docker container directly importing the Python modules.

This split ensures both the API boundary and internal logic are validated.

## Test Suite Structure

### Section 0: Environment Setup (`test_00_environment_setup.py`)

**What**: Validates that the runtime environment is properly configured before any tests run.

**Why**: Tests dependencies, API keys, Docker container status, and server endpoints.

**Tests**:

- `test_0_1_python_deps()`: Verifies `requests` and `numpy` are importable. Exit code 1 if any critical dependency is missing. python-dotenv is optional.

- `test_0_2_api_keys_configured()`: Checks MINIMAX_API_KEY and OPENAI_API_KEY. Does NOT fail if missing—runtime degrades gracefully with LLM features becoming no-ops. Prints configured/unconfigured status.

- `test_0_3_docker_container_running()`: Runs `docker ps --filter name=dealroom-v3-test -q`. Exits with code 1 if container not running. Prints startup command if missing.

- `test_0_4_server_endpoints_responsive()`: Tests HTTP endpoints:
  - `GET /health` → 200
  - `GET /metadata` → 200
  - `POST /reset` with `task_id=aligned` → 200, validates session_id in response
  - `POST /step` with valid action → 200
  All requests use 10-30s timeout.

- `test_0_5_llm_client_imports()`: Attempts import of `deal_room.environment.llm_client.validate_api_keys` and `MAX_TOKENS`. Prints warning if import fails (acceptable if running outside container).

**Status**: WORKING. All endpoint tests pass against a running container. API key test is informational only.

---

### Section 1: Schema Validation (`test_01_schema_validation.py`)

**What**: Validates the observation schema has exactly 18 required fields, no hidden fields are exposed, field types are correct, and the action schema accepts all valid action types including lookahead.

**Why**: The agent interface depends on these field contracts. Schema violations would cause agent code to fail or expose internal research state.

**Tests**:

- `test_1_1_all_required_fields_present()`: POST /reset, checks all 18 fields are present: `round_number`, `max_rounds`, `stakeholders`, `stakeholder_messages`, `engagement_level`, `engagement_level_delta`, `engagement_history`, `weak_signals`, `cross_stakeholder_echoes`, `veto_precursors`, `known_constraints`, `requested_artifacts`, `approval_path_progress`, `deal_momentum`, `deal_stage`, `active_blockers`, `days_to_deadline`, `done`.

- `test_1_2_no_hidden_fields_exposed()`: POST /reset with `task_id=hostile_acquisition`, recursively searches entire observation for hidden fields: `G`, `causal_graph`, `graph`, `true_beliefs`, `belief_distributions`, `belief_state`, `B_i`, `V_i`, `tau`, `tau_i`, `risk_thresholds`, `cvar_thresholds`, `edge_weights`, `w_ij`, `deliberation_transcript`, `deliberation_log`, `internal_dialogue`, `u_i`, `u_ij`. Key constraint: `"graph"` matches only key `"graph"`, NOT `"graph_seed"` or `"graph_structure"` (exact key matching).

- `test_1_3_field_types_correct()`: After /reset and /step, validates:
  - `round_number`: int
  - `max_rounds`: int
  - `stakeholders`: dict
  - `engagement_level`: dict
  - `engagement_level_delta`: numeric (int or float, NOT dict)
  - `engagement_history`: list
  - `weak_signals`: dict
  - `cross_stakeholder_echoes`: list
  - `veto_precursors`: dict
  - `deal_momentum`: str
  - `deal_stage`: str
  - `active_blockers`: list
  - `days_to_deadline`: numeric
  - `done`: bool

- `test_1_4_engagement_history_window_size()`: After /reset with `task_id=conflicted`, validates `engagement_history` has ≥5 entries, each entry is a dict (stakeholder snapshot).

- `test_1_5_engagement_level_delta_single_float()`: After step, validates `engagement_level_delta` is NOT a dict (this was a discovered bug where the field was incorrectly shaped). Must be a single numeric value.

- `test_1_6_cross_stakeholder_echoes_is_list()`: After reset, validates `cross_stakeholder_echoes` is a list of dicts with `from`/`from_stakeholder`/`sender`/`source` field present.

- `test_1_7_stakeholder_messages_populated_after_step()`: After step with `action_type=send_document` targeting Finance with DPA document, validates `stakeholder_messages` is a dict (not empty).

- `test_1_8_action_schema_accepts_lookahead()`: After step with lookahead nested action draft, validates:
  - HTTP 200 response (schema accepted)
  - Response contains reward info
  - Lookahead diagnostics returned in info

- `test_1_9_approval_path_progress_structure()`: After reset, validates `approval_path_progress` is dict with stakeholder keys, each containing `band` field with value in `["blocker", "neutral", "workable", "supporter"]`.

- `test_1_10_deal_stage_valid_transitions()`: After reset and step, validates `deal_stage` is in `["evaluation", "negotiation", "legal_review", "final_approval", "closed"]` and `round_number` increments correctly.

- `test_1_11_documents_field_format()`: After step with `send_document` action including DPA and security_cert documents, validates HTTP 200 and reward is returned.

**Status**: WORKING. All 11/11 schema checks pass. Critical: test_1_5 specifically validates the fix for engagement_level_delta being a dict instead of a float.

---

### Section 2: Reward Integrity (`test_02_reward_integrity.py`)

**What**: Validates that the reward mechanism is unhackable—rewards are single floats, all dimensions stay in [0,1], lookahead cost is exactly 0.07, repeated actions don't inflate reward, and different targets produce different causal scores.

**Why**: If rewards can be gamed, the RL training is meaningless. These tests verify the core reward specification.

**Tests**:

- `test_2_1_reward_is_single_float()`: After step, reward must be numeric, NOT a dict, in [0, 1].

- `test_2_2_lookahead_cost_is_exactly_007()`: With same seed (20), takes same action with and without lookahead. Goal dimension without lookahead minus goal dimension with lookahead should be exactly 0.07 (tolerance 0.015). Also validates `lookahead_predicted_deltas` present in info. **This is the exact cost test**—the 0.07 value is hardcoded in `LOOKAHEAD_COST` constant.

- `test_2_3_reward_in_range_after_valid_actions()`: After 4 different actions (direct_message, send_document with DPA, send_document with timeline, direct_message to Procurement), validates:
  - Reward is finite (not NaN)
  - All reward_components values are in [0.0, 1.0]

- `test_2_4_deterministic_reward_with_seed()`: Three trials with different seeds (100, 111, 122), same action. Prints reward variance. **Note**: This test does NOT assert low variance—it just prints. This is a limitation.

- `test_2_5_repeat_same_action_does_not_escalate_reward()`: Takes same action (direct_message to Finance) 3 times. Prints g1, g2, g3 and trend. Does NOT assert trend is negative—prints only. **Note**: This is a limitation; the test should verify reward doesn't systematically increase.

- `test_2_6_different_targets_different_causal_scores()`: Targets Finance, Legal, TechLead each 3 times with send_document. Expects at least 2 distinct score values across targets. This validates the causal dimension is discriminative based on target graph position.

- `test_2_7_informative_action_outperforms_empty()`: Empty message vs substantive message with ROI document. Substantive should score >= empty - 0.1. **Note**: Tolerance of 0.1 means this is a weak test.

- `test_2_8_reward_non_trivial_variance()`: 5 different actions, expects reward variance > 0.01 across actions. If variance is too low, grader is not discriminative.

- `test_2_9_good_documentation_higher_than_poor()`: Poor docs (minimal content) vs good docs (DPA + security cert) across 3 seeds each. Average good should >= average poor - 0.05.

**Status**: MOSTLY WORKING with limitations. test_2_4 and test_2_5 do not assert—they just print. The lookahead cost test (2_2) is precise. test_2_6 (different targets different scores) is a strong test for causal discriminativeness.

---

### Section 3: Causal Inference Signal (`test_03_causal_inference.py`)

**What**: Validates that the causal graph produces detectable signals: engagement changes in targeted stakeholders, cross-stakeholder echoes, engagement history sliding window, noise cannot be cancelled, different targets produce different patterns, weak signals appear, echo structure is correct, and causal signal responds to graph structure.

**Why**: The causal graph is the core research mechanism. If its signals aren't detectable, the whole environment is broken.

**Tests**:

- `test_3_1_targeted_stakeholder_engagement_changes()`: After targeted send_document to Finance, `engagement_level_delta` must be numeric (not None, not dict).

- `test_3_2_cross_stakeholder_echoes_detected()`: 8 episodes targeting Finance with ROI model. Expects echoes detected in ≥30% of episodes. This validates `echo_recall_probability=0.70` is active.

- `test_3_3_engagement_history_window_slides()`: After 3 steps, history window length must remain constant (not grow, not shrink). This validates the sliding window implementation.

- `test_3_4_engagement_noise_not_cancellable()`: 5 steps with different messages. Expects ≥2 non-zero deltas out of 5. If all near zero, noise may be disabled. This is a key anti-gaming test.

- `test_3_5_different_targets_different_patterns()`: 4 Finance targeting episodes vs 4 Legal targeting episodes. Expects mean delta distance > 0.001 between groups. Different targets should produce measurably different propagation patterns.

- `test_3_6_weak_signals_for_non_targeted()`: 12 episodes targeting Finance. Expects weak signals detected in ≥20% of episodes. This validates the weak signal mechanism is active.

- `test_3_7_echo_content_structure()`: After targeted action, echoes must be list of dicts with sender field (`from`/`from_stakeholder`/`sender`/`source`).

- `test_3_8_causal_signal_responds_to_graph_structure()`: 4 episodes targeting ExecSponsor (authority hub) vs 4 targeting TechLead (leaf). Prints hub and leaf delta variance. **Note**: This test does NOT assert hub vs leaf difference—it just prints variance. This is a limitation.

**Status**: WORKING for most tests. test_3_8 is informational only (no assertion). The 30% echo rate, 20% weak signal rate, and noise cancellation tests are all meaningful validations.

---

### Section 4: CVaR Veto Mechanism (`test_04_cvar_veto.py`)

**What**: Validates that veto precursors fire before veto, aligned scenario doesn't veto immediately, veto terminates episode, timeout terminates episode, hostile scenario reaches veto pressure faster than aligned, veto precursors are stakeholder-specific, and CVaR fires even when EU is positive.

**Why**: CVaR veto is the core research claim. If veto doesn't work correctly, the entire tail-risk mechanism is broken.

**Tests**:

- `test_4_1_veto_precursor_before_veto()`: In hostile_acquisition scenario with aggressive action (exec_escalation to ExecSponsor), up to 18 steps. If veto fires, asserts precursor was seen first. If veto doesn't fire in 18 steps, asserts precursors were seen. **Issue**: Stochastic—may not trigger veto in given seed.

- `test_4_2_aligned_no_early_veto()`: In aligned scenario with friendly message to Finance, after first step `done=False`. Asserts aligned scenario does NOT veto immediately.

- `test_4_3_veto_terminates_episode()`: In hostile_acquisition with aggressive action, up to 20 steps. If veto fires, prints terminal outcome. **Issue**: Stochastic—may not trigger in 20 steps.

- `test_4_4_timeout_terminates()`: In aligned scenario, runs max_rounds + 2 steps. Asserts episode terminates at natural endpoint. This always works since max rounds is deterministic.

- `test_4_5_scenario_difficulty_differentiation()`: hostile_acquisition vs aligned, each 3 trials. Asserts hostile reaches veto pressure (precursors or veto) at round ≤ aligned's first pressure round. **Note**: If neither triggers, both get n+1 and test passes vacuously.

- `test_4_6_veto_precursor_is_stakeholder_specific()`: In hostile_acquisition with aggressive action, up to 15 steps. Tracks which stakeholders get precursor warnings. Prints stakeholder IDs if precursors seen. **Issue**: Informational only—no assertion on specificity.

- `test_4_7_cveto_not_just_eu()`: In hostile_acquisition, up to 18 steps. If veto fires, prints confirmation. **Issue**: Stochastic, informational only.

**Status**: PARTIALLY WORKING. test_4_2 and test_4_4 are deterministic and always pass. test_4_1, 4_3, 4_5 are stochastic. test_4_6 and 4_7 are informational only. The veto mechanism exists and fires correctly when conditions are met, but the tests cannot guarantee triggering in a fixed number of steps due to randomness.

---

### Section 5: Episode Isolation (`test_05_episode_isolation.py`)

**What**: Validates that different seeds produce different initial states, round_number resets to 0, done=False after reset, engagement_history is initialized, round_number increments correctly, all three scenarios work without crash, session state is consistent across steps, and reset clears all session state.

**Why**: Episode isolation ensures training episodes don't leak state between each other. If seeds produced identical states, the RL algorithm would learn deterministic patterns rather than generalizable behavior.

**Tests**:

- `test_5_1_different_seeds_different_initial_state()`: Resets with seed 42 and 43. Sums absolute differences in engagement_level values. Asserts diff > 0.001. **This is a critical test** verifying G is resampled on each reset.

- `test_5_2_round_number_resets_to_zero()`: After 3 steps, reset with new seed. Asserts round_number = 0.

- `test_5_3_done_false_after_reset()`: All three scenarios (aligned, conflicted, hostile_acquisition) validated: done=False after reset.

- `test_5_4_engagement_history_initialized()`: After reset, history has ≥5 entries.

- `test_5_5_round_number_increments_correctly()`: 5 steps, asserts round_number = 1, 2, 3, 4, 5 respectively.

- `test_5_6_all_three_scenarios_work()`: Each scenario takes one step without crash (HTTP 200).

- `test_5_7_same_session_same_state_across_steps()`: After 2 steps, round_number of step 2 = round_number of step 1 + 1.

- `test_5_8_reset_clears_all_session_state()`: After 3 steps, reset with new seed. Asserts round_number=0, done=False, history reinitialized.

**Status**: WORKING. All 8/8 tests are deterministic and always pass. test_5_1 is the critical graph resampling validation.

---

### Section 6: Probabilistic Signals (`test_06_probabilistic_signals.py`) — Container Only

**What**: Validates weak_signals and cross_stakeholder_echoes fields exist, echo structure is correct, weak signals populated after action, echo firing rate is non-zero, weak signal values are list[str] tags, and echo_recall_probability is configured at 0.70.

**Why**: These are the partial observability mechanisms. The agent doesn't see the true causal graph—only these proxy signals.

**Tests** (all run inside container via direct Python imports):

- `test_6_1_weak_signals_field_exists()`: Creates DealRoomV3, reset(), asserts `obs.weak_signals` attribute exists.

- `test_6_2_cross_stakeholder_echoes_exists()`: Same, asserts `obs.cross_stakeholder_echoes` exists.

- `test_6_3_echo_structure()`: Calls `env._generate_cross_stakeholder_echoes(action)`, asserts returns list of dicts with sender field.

- `test_6_4_weak_signals_populated_after_action()`: After step with send_document(DPA), asserts `obs2.weak_signals` is dict (populated).

- `test_6_5_echo_firing_rate_nonzero()`: 20 episodes, each creates fresh env, takes step with send_document(DPA) to Finance. Asserts echoes fired in >0% of episodes (rate > 0). The hard assertion is `rate > 0` (not `> 30%`), so even rare firing passes.

- `test_6_6_weak_signal_threshold_respected()`: Validates OBS_CONFIG is initialized, weak_signal_hard_threshold exists, and weak signal values are list[str] tags (not numeric).

- `test_6_7_echo_recall_probability_configured()`: OBS_CONFIG.echo_recall_probability should be 0.70 (or 0 < prob < 1).

**Status**: WORKING. All 7/7 tests pass. test_6_5's assertion is weak (just `> 0` rather than `> 0.30`), but the mechanism is validated.

---

### Section 7: Causal Graph Unit Tests (`test_07_causal_graph.py`) — Container Only

**What**: Validates propagation direction follows graph edges, signal carries through edges, damping prevents runaway amplification, beliefs stay normalized after propagation, no self-loops exist, ExecSponsor has outgoing edges (authority invariant), hub node has highest betweenness centrality, and all sampled graphs produce unique behavioral signatures.

**Why**: The causal graph is the mathematical backbone of the environment. If graph operations are broken, nothing else works.

**Tests** (all run inside container):

- `test_7_1_propagation_direction()`: Simple 3-node graph with edge A→B (weight 0.7). A gets +0.4 delta. Asserts B's positive_mass increases > 0.05 and C (no edge from A) is unaffected (< 0.03 change). **Strong test**: Verifies directionality.

- `test_7_2_signal_carries_through_edge()`: 2-node graph A→B (weight 0.8). A gets +0.2 delta. Asserts B's belief changes. **Strong test**.

- `test_7_3_damping_prevents_runaway()`: 5-node complete graph (25 edges), A gets +0.5 delta, 5 propagation steps. Asserts all beliefs stay in (0, 1). This tests the 0.85^step damping factor works.

- `test_7_4_beliefs_normalized_after_propagation()`: Sampled graph with 5 stakeholders, propagation. Asserts all belief distributions sum to 1.0 ± 1e-6.

- `test_7_5_no_self_loops()`: Sampled graphs (10 seeds × 3 scenarios). Asserts for all stakeholders: `g.get_weight(sid, sid) == 0.0`.

- `test_7_6_exec_sponsor_outgoing_authority()`: 10 seeds × 3 scenarios. Asserts ExecSponsor has ≥ 2 outgoing edges with weight > 0.1 in every case. **Authority invariant test**.

- `test_7_7_hub_centrality_beats_leaf()`: Star graph Hub→{A,B,C,D}. Asserts Hub's betweenness ≥ max leaf betweenness.

- `test_7_8_graph_identifiability()`: 20 sampled graphs, each with behavioral signature (Finance targeting, 0.4 delta, 3 steps). Asserts ALL 190 pairs are distinguishable (no two graphs produce identical signatures). **This is the strongest test**: It validates that the graph structure is causally meaningful and no two graphs are behaviorally equivalent. Takes ~60 seconds.

**Status**: WORKING. All 8/8 tests pass. test_7_8 is the most important—graph identifiability confirms the causal mechanism is not degenerate.

---

### Section 8: CVaR Preferences Unit Tests (`test_08_cvar_preferences.py`) — Container Only

**What**: Validates the CORE research claim (EU>0 yet CVaR>tau → veto fires), good documentation yields lower CVaR than poor documentation, CVaR formula is mathematically correct, risk tolerance ordering (Legal < Finance < ExecSponsor), aggressive timeline increases TechLead CVaR, and CVaR is coherent (≤ max outcome).

**Why**: These are the foundational mathematical claims of the research. If CVaR doesn't work as specified, the entire veto mechanism is invalid.

**Tests**:

- `test_8_1_core_claim()`: Creates Legal profile (tau=0.10), terms with no DPA, no security cert, liability_cap=0.2. Evaluates deal. Asserts:
  - EU > 0 (deal has positive expected value)
  - CVaR loss > tau (Legal's tolerance exceeded)
  - CVaR loss > EU (tail risk dominates expected value)
  
  **This is the core research claim validation**. It directly tests that CVaR can veto a deal that EU would approve.

- `test_8_2_good_docs_lower_cvar_than_poor()`: Poor terms (no DPA, no cert, liability=0.2) vs good terms (DPA+cert, liability=1.0). Asserts good CVaR < poor CVaR.

- `test_8_3_cvar_formula_correct()`: Tests:
  - CVaR(95) of [1.0×5, 0.5×5] is between 0.5 and 1.0
  - CVaR(90) of [0.8×10] < CVaR(90) of [0.2×10]
  
  **Mathematical validation** of the CVaR formula.

- `test_8_4_tau_ordering()`: Asserts Legal.tau (0.10) < Finance.tau (0.15) < ExecSponsor.tau (0.40). The ordering reflects real organizational risk tolerance.

- `test_8_5_aggressive_timeline_higher_cvar()`: TechLead profile, 4-week timeline vs 16-week timeline (all else equal). Asserts aggressive CVaR > reasonable CVaR.

- `test_8_6_cvar_subadditivity_sanity()`: CVaR(95) of [0.3, 0.4, 0.5, 0.6, 0.9] ≤ max(0.9). This is the coherence axiom: CVaR should not exceed the worst-case outcome.

**Status**: WORKING. All 6/6 tests pass. test_8_1 is the most important—it directly validates the core research claim that CVaR-based veto differs from EU-based rejection.

---

### Section 9: Full Episode End-to-End (`test_09_full_episode_e2e.py`)

**What**: Validates all three scenarios complete without crash, hostile+aggressive produces veto or timeout (not crash), reward entries are collected throughout episode, reward variance is non-trivial, strategy comparison is possible, terminal outcome is meaningful, and multi-document actions work.

**Why**: Integration testing. Individual components can work in isolation but fail when composed.

**Tests**:

- `test_9_1_aligned_completes()`: aligned + neutral strategy, up to 20 steps. Asserts steps ≥ 1. Prints terminal outcome.

- `test_9_2_conflicted_completes()`: conflicted + comprehensive strategy, up to 20 steps. Asserts steps ≥ 1. Prints terminal outcome.

- `test_9_3_hostile_aggressive_produces_veto_or_timeout()`: hostile_acquisition + aggressive (exec_escalation repeated 8×), up to 15 steps. Valid terminals: veto, timeout, or HTTP errors (which indicate server issues). **Issue**: Stochastic—may timeout instead of vetoing.

- `test_9_4_reward_collected_across_episode()`: conflicted + comprehensive, 15 steps. Asserts ≥1 reward entries collected. Prints reward trajectory.

- `test_9_5_reward_variance_non_trivial()`: aligned + comprehensive, 12 steps. If ≥2 rewards, asserts max - min > 0.01. **Issue**: weak threshold.

- `test_9_6_strategy_comparison_possible()`: 3 comprehensive runs vs 3 aggressive runs. Prints average scores. **Issue**: No assertion—just prints.

- `test_9_7_terminal_outcome_meaningful()`: 9 combinations (3 scenarios × 3 strategies), up to 15 steps each. Asserts terminal outcome is non-empty and not "unknown". Prints all unique terminals seen.

- `test_9_8_multidocument_action_works()`: One step with send_document containing 3 documents (DPA, security_cert, liability_terms) to Legal. Asserts HTTP 200 and reward returned.

**Status**: MOSTLY WORKING. test_9_1, 9_2, 9_4, 9_7, 9_8 are deterministic. test_9_3 is stochastic. test_9_5 has weak threshold. test_9_6 is informational only.

---

### Section 10: Training Infrastructure (`test_10_training_infrastructure.py`) — Container Only

**What**: Validates GRPOTrainer imports, TrainingMetrics has all required fields, AdaptiveCurriculumGenerator imports, grpo_colab.ipynb exists, GRPOTrainer class is properly defined, and checkpoint save/load works (if PyTorch available).

**Why**: The training harness must be importable and functional for RL training to be possible.

**Tests**:

- `test_10_1_grpo_trainer_imports()`: Imports `deal_room.training.grpo_trainer`, asserts `GRPOTrainer` and `TrainingMetrics` exist.

- `test_10_2_training_metrics_fields()`: Checks TrainingMetrics has: goal_reward, trust_reward, info_reward, risk_reward, causal_reward, lookahead_usage_rate. All 6 must be present.

- `test_10_3_curriculum_generator_imports()`: Imports `deal_room.curriculum.adaptive_generator`, asserts `AdaptiveCurriculumGenerator` exists.

- `test_10_4_colab_notebook_exists()`: Checks `/app/env/deal_room/training/grpo_colab.ipynb` or `/app/env/grpo_colab.ipynb`. Notebook must have ≥5 cells with "cells" key.

- `test_10_5_training_loop_smoke_test()`: Asserts GRPOTrainer has `__init__`. Prints confirmation.

- `test_10_6_checkpoint_save_load_smoke()`: Tries importing torch. Prints PyTorch availability status. No actual save/load test runs.

**Status**: WORKING for imports. test_10_6 is a placeholder—no actual checkpoint I/O is tested.

---

### Section 11: Research Properties (`test_11_research_properties.py`)

**What**: Validates all 12 research properties that define the specification. This is the master validation suite.

**Why**: These are the top-level claims that must all be satisfied for the environment to be considered research-ready.

**Tests**:

- P1 (G is hidden): Asserts none of 7 hidden fields appear in observation.

- P2 (Episode reset regenerates G): Different seeds (100 vs 200) → engagement_level diff > 0.001. **Graph resampling validated**.

- P3 (CVaR veto despite positive EU): Direct import of cvar_preferences, evaluates deal with Legal profile (tau=0.10, no DPA/cert). Asserts EU > 0 AND CVaR > tau. **Core claim validated**.

- P4 (Five reward dims discriminative): 5 steps with same action, prints reward variance. **Informational only**—no assertion on discriminativeness.

- P5 (Lookahead cost exactly 0.07): Same as test_2_2 but with tighter naming. Asserts |diff - 0.07| < 0.015.

- P6 (Engagement noise not cancellable): 5 steps, ≥2 non-zero deltas. Same as test_3_4.

- P7 (Cross-stakeholder echoes present): After reset+step with send_document to Finance, asserts "cross_stakeholder_echoes" in observation.

- P8 (Weak signals present): After reset with conflicted, asserts "weak_signals" in observation.

- P9 (r^causal varies with target): Finance vs Legal vs TechLead targeting, asserts causal mechanism active (echoes ≥ 0, weak_signals ≥ 0, some delta ≠ 0). **But does NOT assert different targets produce different scores**—just that something is non-zero.

- P10 (Every reset different G): 5 resets with different seeds, asserts ≥2 unique engagement_level signatures. **Graph uniqueness validated**.

- P11 (Full episode no crash): 8 steps each in all 3 scenarios, asserts HTTP 200 throughout.

- P12 (Training loop imports): Imports GRPOTrainer and AdaptiveCurriculumGenerator. Asserts not None.

**Status**: 11/12 are functional. P4 and P9 are weaker than they should be (informational rather than asserting discriminativeness). P3 is the most important—it validates the core CVaR claim mathematically.

---

### Scorer Unit Tests (`test_scorer_unit.py`)

**What**: Direct unit tests for UtteranceScorer without HTTP API. Tests lookahead cost is exactly 0.07, LOG2_6 constant is correct, lookahead penalty is applied, all dimensions bounded [0,1], goal score increases with approval, trust responds to targeting, information rewards entropy reduction, scoring is deterministic, prediction accuracy works, scorer is stateless, risk falls back to 0.5 without profiles, causal returns 0 with no targets, and blocker resolution affects goal.

**Why**: These test the scorer in isolation, ensuring no HTTP or environment noise affects scoring.

**Tests**:

- `test_lookahead_cost_exactly_0_07()`: Asserts `LOOKAHEAD_COST == 0.07` (exact equality, not approximation).

- `test_log2_6_correct()`: Asserts LOG2_6 ≈ 2.585 (math.log(2, 6)).

- `test_lookahead_penalty_applied()`: Two identical scoring calls, one with lookahead_used=True. Asserts goal_with_lookahead < goal_without AND diff ≈ 0.07.

- `test_all_dimensions_bounded_0_1()`: 20 random belief pairs, all 5 dimensions must be in [0, 1].

- `test_goal_score_increases_with_approval()`: Positive belief delta vs negative belief delta. Asserts positive_goal > negative_goal AND positive_goal > 0.5 > negative_goal.

- `test_trust_targeted_delta()`: Targeted vs untargeted belief changes. Asserts targeted trust > 0.5 AND untargeted trust == 0.5.

- `test_information_entropy_reduction()`: Uniform beliefs → moderately certain → highly certain. Asserts info_score(high) >= info_score(moderate) > 0.5.

- `test_determinism_consistency()`: 10 identical scoring calls. Asserts all 10 results are identical.

- `test_prediction_accuracy()`: Jaccard test cases: empty-empty → 0.0, {Legal: a} → {Legal: a} → 1.0, partial overlap → between 0 and 1.

- `test_scorer_reset()`: Two identical scoring calls. Asserts results identical (scorer is stateless).

- `test_risk_cvar_no_profile_returns_0_5()`: risk_profiles=None and risk_profiles={} both yield risk == 0.5.

- `test_causal_no_targets_returns_0()`: targeted_ids=[] yields causal == 0.0.

- `test_blocker_resolution_affects_goal()`: With blockers resolved vs same blockers. Asserts resolved_goal > same_goal.

**Status**: WORKING. All 13/13 tests pass. These are the most precise tests of the scoring mechanism.

---

## Summary: What is Working vs What is Not

### WORKING (High Confidence)

| Component | Test Coverage | Notes |
|---|---|---|
| Observation schema (18 fields) | test_1_1 to test_1_11 | All 11 pass. engagement_level_delta is correctly a float. |
| Hidden fields not exposed | test_1_2 | G and all internal fields hidden. |
| Causal graph propagation | test_7_1 to test_7_7 | All 7 pass. Directionality, damping, normalization, no self-loops, authority invariant. |
| Graph identifiability | test_7_8 | 20/20 unique signatures. Critical validation. |
| CVaR formula correctness | test_8_3 | Mathematical formula validated. |
| CVaR core claim (EU>0 yet CVaR>tau) | test_8_1 | Core research claim validated. |
| Tau ordering (Legal<Finance<Exec) | test_8_4 | Validated. |
| Reward dimensions bounded [0,1] | test_scorer_unit | All 13 scorer tests pass. |
| Lookahead cost exactly 0.07 | test_2_2, test_scorer | Exact 0.07 validated. |
| Deterministic scoring | test_scorer_unit | 10-run consistency validated. |
| Episode isolation (different seeds) | test_5_1 | Graph resampling validated. |
| Round number reset/increment | test_5_2, test_5_5 | Correct. |
| Terminal reward computation | test_9_7 | Terminal outcomes meaningful. |
| Training infrastructure imports | test_10_1 to test_10_3 | GRPOTrainer, TrainingMetrics, curriculum all import. |
| Environment setup (Python deps, Docker) | test_0_1, test_0_3 | Container and deps validated. |

### WORKING WITH LIMITATIONS

| Component | Issue | Test |
|---|---|---|
| Determinism across seeds | test_2_4 prints but doesn't assert | Should assert low variance |
| Repeat action doesn't inflate reward | test_2_5 prints but doesn't assert | Should assert trend ≤ 0 |
| Causal signal responds to graph | test_3_8 prints but doesn't assert hub vs leaf | Should assert hub delta variance > leaf |
| Strategy comparison | test_9_6 prints but doesn't assert | Should assert comp > agg |
| Echo rate 30% | test_3_2 asserts ≥30%, but test_6_5 only asserts >0% | test_6_5 is too weak |

### STOCHASTIC / NOT FULLY VALIDATED

| Component | Issue |
|---|---|
| CVaR veto triggering | test_4_1, 4_3, 4_5 are stochastic—may not trigger in fixed steps |
| Veto precursor stakeholder-specificity | test_4_6 is informational only |
| Different targets produce different scores | test_2_6 is good, but P9 in test_11 doesn't assert discriminativeness |
| Hostile scenario difficulty | test_4_5 can pass vacuously if neither triggers |
| Lookahead prediction accuracy | Only tested in scorer unit (test_scorer) but not validated against actual environment |

### NOT TESTED

| Component | Gap |
|---|---|
| Checkpoint save/load | test_10_6 is a placeholder |
| Full GRPO training loop | Smoke test only—no actual gradient update |
| Adaptive curriculum selection | Imports validated but selection logic not tested |
| LLM deliberation summary quality | Existence validated but content not validated |
| Gradio web UI | Not tested |
| Session timeout behavior | Not tested |
| Concurrent sessions | Not tested |

---

## Key Test Constants

```python
LOOKAHEAD_COST = 0.07  # Exact, not approximate
OBS_CONFIG.echo_recall_probability = 0.70  # 70% echo rate
OBS_CONFIG.weak_signal_hard_threshold = 0.12  # Hard threshold
OBS_CONFIG.engagement_noise_sigma = 0.03  # Noise sigma
VETO_WARNING_THRESHOLD_RATIO = 0.70  # Precursor fires at 70% of tau
LOG2_6 = math.log(2, 6)  # ~2.585, max entropy for 6 types
DAMPING_FACTOR = 0.85  # Per propagation step

REWARD_WEIGHTS = {"goal": 0.30, "trust": 0.25, "info": 0.20, "risk": 0.15, "causal": 0.10}
TERMINAL_REWARDS = {"deal_closed": 1.0, "veto": -1.0, "max_rounds": -0.8, "stage_regression": -0.3, "impasse": -0.5}
```

## How to Run Tests

```bash
# HTTP API tests (requires container running)
python tests/v3/test_00_environment_setup.py
python tests/v3/test_01_schema_validation.py
python tests/v3/test_02_reward_integrity.py
python tests/v3/test_03_causal_inference.py
python tests/v3/test_04_cvar_veto.py
python tests/v3/test_05_episode_isolation.py
python tests/v3/test_09_full_episode_e2e.py
python tests/v3/test_11_research_properties.py

# Container unit tests (run inside container)
docker exec dealroom-v3-test python /app/env/tests/v3/test_06_probabilistic_signals.py
docker exec dealroom-v3-test python /app/env/tests/v3/test_07_causal_graph.py
docker exec dealroom-v3-test python /app/env/tests/v3/test_08_cvar_preferences.py
docker exec dealroom-v3-test python /app/env/tests/v3/test_10_training_infrastructure.py
docker exec dealroom-v3-test python /app/env/tests/v3/test_scorer_unit.py
```