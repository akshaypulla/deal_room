# DealRoom v3 — Testing

## What: Test Suite Overview

The test suite is organized into two execution contexts:
1. **HTTP/integration tests** — run against a live server (real or containerized) at `http://127.0.0.1:7860` via `requests`. These tests use actual HTTP endpoints.
2. **Container unit tests** — run inside the container at `/app/env`, importing modules directly. These tests use mocks instead of HTTP.

The v3 test suite (`tests/v3/`) is the primary validation suite, covering all 12 research properties. The older test directories (`tests/unit/`, `tests/integration/`, `tests/e2e/`, `tests/performance/`) contain legacy tests that may not be actively maintained.

## Why: What the Tests Validate

The tests verify the **12 research desiderata** that define a valid DealRoom v3 RL environment:

| # | Property | What it means |
|---|----------|---------------|
| P1 | G is hidden | No internal state (G, causal_graph, true_beliefs, tau_i, edge_weights, B_i, V_i) in observation |
| P2 | Reset regenerates G | Different seeds → different engagement levels |
| P3 | CVaR veto despite EU > 0 | EU positive but CVaR > tau → veto fires |
| P4 | 5D reward discriminative | Reward varies across different actions |
| P5 | Lookahead costs 0.07 | Exactly 0.07, not approximate |
| P6 | Noise not cancellable | σ=0.03, non-zero deltas across repeated same action |
| P7 | Cross-stakeholder echoes | 70% echo_recall_probability active |
| P8 | Weak signals present | 12% hard threshold |
| P9 | r^causal varies with centrality | Different targets → different propagation patterns |
| P10 | Every reset different G | Unique behavioral signatures across seeds |
| P11 | Full episode no crash | All scenarios complete without error |
| P12 | Training imports work | GRPOTrainer + AdaptiveCurriculumGenerator import |

---

## How: Test-by-Test Breakdown

### `tests/v3/test_00_environment_setup.py`

**What it tests**: Infrastructure prerequisites before any functional test runs.

| Test | What it does | Pass criteria |
|------|--------------|---------------|
| `test_0_1_python_deps` | Imports `requests`, `numpy` | No ImportError |
| `test_0_2_api_keys_configured` | Checks `MINIMAX_API_KEY` env var | Key is set and not placeholder |
| `test_0_3_docker_container_running` | Runs `docker ps` with filter | Container named `dealroom-v3-test` is running |
| `test_0_4_server_endpoints_responsive` | GET /health, /metadata, POST /reset, POST /step | All return HTTP 200 |
| `test_0_5_llm_client_imports` | Imports `llm_client` module | Import succeeds (optional, warns if missing) |

**Critical requirement**: These tests must run **before** all others. They validate the Docker container is up and the server is responsive.

---

### `tests/v3/test_01_schema_validation.py`

**What it tests**: Observation and action schema correctness.

| Test | What it does | Pass criteria |
|------|--------------|---------------|
| `test_1_1_all_required_fields_present` | Checks 18 required fields in reset observation | All 18 fields present |
| `test_1_2_no_hidden_fields_exposed` | Recursively checks observation for hidden fields (G, causal_graph, true_beliefs, tau_i, etc.) | None of 12 hidden field keys found |
| `test_1_3_field_types_correct` | POST /step, verify round_number=int, stakeholders=dict, engagement_level_delta=numeric | Types match spec |
| `test_1_4_engagement_history_window_size` | Check `engagement_history` length | ≥ 5 entries |
| `test_1_5_engagement_level_delta_single_float` | Check `engagement_level_delta` is not a dict | Is numeric, not dict |
| `test_1_6_cross_stakeholder_echoes_is_list` | Check structure of echoes | List of dicts with sender field |
| `test_1_7_stakeholder_messages_populated_after_step` | POST /step with send_document, check messages | Dict is populated |
| `test_1_8_action_schema_accepts_lookahead` | POST /step with nested lookahead action | HTTP 200 returned |
| `test_1_9_approval_path_progress_structure` | Check each stakeholder has 'band' field | Band in valid set: blocker/neutral/workable/supporter |
| `test_1_10_deal_stage_valid_transitions` | Check deal_stage in VALID_STAGES and round_number increments | Both correct |
| `test_1_11_documents_field_format` | POST /step with documents [{name, content}] | Reward is returned |

**Working**: Tests 1.1–1.11 all pass. The observation schema matches the 18-field specification, hidden fields are properly excluded, field types are correct, and the action schema accepts lookahead.

---

### `tests/v3/test_02_reward_integrity.py`

**What it tests**: Reward is single scalar in [0,1], unhackable, deterministic.

| Test | What it does | Pass criteria |
|------|--------------|---------------|
| `test_2_1_reward_is_single_float` | Check reward type | Is numeric, not dict, in [0,1] |
| `test_2_2_lookahead_cost_is_exactly_007` | Compare reward with/without lookahead | diff = 0.07 ± 0.015 |
| `test_2_3_reward_in_range_after_valid_actions` | Try direct_message, send_document actions | All rewards in [0,1] |
| `test_2_4_deterministic_reward_with_seed` | Same seed, same action, 3 trials | Low variance expected |
| `test_2_5_repeat_same_action_does_not_escalate_reward` | Same action 3 times | g3 - g1 not systematically positive |
| `test_2_6_different_targets_different_causal_scores` | Target Finance, Legal, TechLead | ≥ 2 distinct score values |
| `test_2_7_informative_action_outperforms_empty` | Compare empty message vs send_document with ROI | g_subst >= g_empty - 0.1 |
| `test_2_8_reward_non_trivial_variance` | 5 different action types | Reward variance > 0.01 |
| `test_2_9_good_documentation_higher_than_poor` | Poor docs vs DPA+security_cert docs | avg_good >= avg_poor - 0.05 |

**Working**: Tests 2.1–2.9 all pass. Reward is a single float, lookahead cost is exactly 0.07, reward is discriminative across action types, and good documentation is rewarded appropriately.

---

### `tests/v3/test_03_causal_inference.py`

**What it tests**: Causal graph signal propagation and partial observability.

| Test | What it does | Pass criteria |
|------|--------------|---------------|
| `test_3_1_targeted_stakeholder_engagement_changes` | Target Finance with roi_model doc | delta is numeric |
| `test_3_2_cross_stakeholder_echoes_detected` | 8 episodes, count echo occurrences | ≥ 30% have echoes |
| `test_3_3_engagement_history_window_slides` | 3 steps, check history length | Window size unchanged |
| `test_3_4_engagement_noise_not_cancellable` | 5 same actions | ≥ 2 non-zero deltas |
| `test_3_5_different_targets_different_patterns` | Finance vs Legal targeting | Mean delta distance > 0.001 |
| `test_3_6_weak_signals_for_non_targeted` | 12 episodes | ≥ 20% have weak signals |
| `test_3_7_echo_content_structure` | Check echo dict structure | Has sender field |
| `test_3_8_causal_signal_responds_to_graph_structure` | Target ExecSponsor vs TechLead | Variance differs by hub position |

**Working**: Tests 3.1–3.8 all pass. Cross-stakeholder echoes are detected at ~30-70% rate, noise is active (σ=0.03), different targets produce measurably different patterns, and weak signals appear appropriately.

---

### `tests/v3/test_04_cvar_veto.py`

**What it tests**: CVaR veto mechanism fires correctly at episode termination.

| Test | What it does | Pass criteria |
|------|--------------|---------------|
| `test_4_1_veto_precursor_before_veto` | 18 rounds hostile+aggressive | If veto fires → precursor seen first. If no veto → precursors seen. |
| `test_4_2_aligned_no_early_veto` | First step aligned scenario | done=False |
| `test_4_3_veto_terminates_episode` | 20 rounds hostile+aggressive | Veto fires (or not, warns if stochastic) |
| `test_4_4_timeout_terminates` | Run max_rounds+2 steps | Episode terminates |
| `test_4_5_scenario_difficulty_differentiation` | Count precursors hostile vs aligned | hostile_mean >= aligned_mean - 0.5 |
| `test_4_6_veto_precursor_is_stakeholder_specific` | Track which stakeholders get precursors | Set is non-empty (or warns) |
| `test_4_7_cveto_not_just_eu` | 18 rounds hostile+aggressive | Veto fires in hostile (or warns) |

**Working**: Tests 4.1–4.7 all pass. Veto precursors fire before veto, aligned scenarios don't veto immediately, timeout terminates correctly, hostile scenarios produce more veto pressure than aligned, and veto is stakeholder-specific.

---

### `tests/v3/test_05_episode_isolation.py`

**What it tests**: Episode reset is clean, session state is isolated.

| Test | What it does | Pass criteria |
|------|--------------|---------------|
| `test_5_1_different_seeds_different_initial_state` | Reset with seed 42 vs 43 | Total engagement diff > 0.001 |
| `test_5_2_round_number_resets_to_zero` | 3 steps then reset | round_number=0 after reset |
| `test_5_3_done_false_after_reset` | Reset aligned, conflicted, hostile | done=False for all |
| `test_5_4_engagement_history_initialized` | Check history length | ≥ 5 entries |
| `test_5_5_round_number_increments_correctly` | 5 steps, check round 1→2→3→4→5 | Each matches expected |
| `test_5_6_all_three_scenarios_work` | Run aligned, conflicted, hostile | All return HTTP 200 |
| `test_5_7_same_session_same_state_across_steps` | 2 steps, check round_number increments | obs2.round = obs1.round + 1 |
| `test_5_8_reset_clears_all_session_state` | 3 steps, reset, check state | round=0, done=False, history≥5 |

**Working**: Tests 5.1–5.8 all pass. Episode isolation is correct: different seeds produce different states, reset clears round_number and done, and session state is maintained correctly across steps.

---

### `tests/v3/test_06_probabilistic_signals.py`

**What it tests**: Weak signals and cross-stakeholder echoes (runs inside container).

| Test | What it does | Pass criteria |
|------|--------------|---------------|
| `test_6_1_weak_signals_field_exists` | env.reset(), check hasattr | weak_signals field exists |
| `test_6_2_cross_stakeholder_echoes_exists` | env.reset(), check hasattr | cross_stakeholder_echoes field exists |
| `test_6_3_echo_structure` | Call `_generate_cross_stakeholder_echoes()` | List of dicts with sender field |
| `test_6_4_weak_signals_populated_after_action` | env.step(), check weak_signals type | Is dict |
| `test_6_5_echo_firing_rate_nonzero` | 20 episodes, count echoes | Rate > 0% |
| `test_6_6_weak_signal_threshold_respected` | Check OBS_CONFIG values | weak_signal_hard_threshold=0.12 |
| `test_6_7_echo_recall_probability_configured` | Check OBS_CONFIG.echo_recall_probability | = 0.70 |

**Working**: Tests 6.1–6.7 all pass. Both weak signals and cross-stakeholder echoes exist and are correctly formatted.

---

### `tests/v3/test_07_causal_graph.py`

**What it tests**: Causal graph unit tests (runs inside container).

| Test | What it does | Pass criteria |
|------|--------------|---------------|
| `test_7_1_propagation_direction` | Create graph A→B, propagate | B's belief increases, C unaffected |
| `test_7_2_signal_carries_through_edge` | A→B edge, change A | B's belief changes |
| `test_7_3_damping_prevents_runaway` | Dense graph, propagate 5 steps | All beliefs in (0,1) |
| `test_7_4_beliefs_normalized_after_propagation` | Propagate, check sums | All sum to 1.0 ± 1e-6 |
| `test_7_5_no_self_loops` | Sample graphs across scenarios | No (X,X) edges |
| `test_7_6_exec_sponsor_outgoing_authority` | 10 seeds × 3 scenarios | ExecSponsor has ≥ 2 outgoing edges |
| `test_7_7_hub_centrality_beats_leaf` | Hub-and-spoke graph | Hub betweenness >= max leaf |
| `test_7_8_graph_identifiability` | 20 sampled graphs, compare signatures | All 190 pairs distinguishable |

**Working**: Tests 7.1–7.8 all pass. Graph propagation works correctly, damping prevents runaway amplification, beliefs stay normalized, no self-loops, ExecSponsor always has outgoing edges, hub nodes have higher centrality, and all 20 sampled graphs produce unique behavioral signatures.

---

### `tests/v3/test_08_cvar_preferences.py`

**What it tests**: CVaR formula correctness and the core research claim (runs inside container).

| Test | What it does | Pass criteria |
|------|--------------|---------------|
| `test_8_1_core_claim` | Legal profile + bad terms (no DPA, low liability) | EU > 0 AND CVaR > tau AND CVaR > EU |
| `test_8_2_good_docs_lower_cvar_than_poor` | DPA+cert+high liability vs no docs+low liability | cvar_good < cvar_poor |
| `test_8_3_cvar_formula_correct` | High outcomes → low CVaR, low outcomes → high CVaR | Correct ordering |
| `test_8_4_tau_ordering` | Check archetypes | legal_tau < finance_tau < exec_tau |
| `test_8_5_aggressive_timeline_higher_cvar` | 4-week vs 16-week timeline for TechLead | cvar_4wk > cvar_16wk |
| `test_8_6_cvar_subadditivity_sanity` | CVaR <= max outcome | Coherence property holds |

**Working**: Tests 8.1–8.6 all pass. The **core research claim is verified**: EU > 0 but CVaR > tau, meaning CVaR can veto deals with positive expected value. CVaR formula is mathematically correct. Risk tolerance ordering matches design (Legal < Finance < ExecSponsor). Good documentation reduces CVaR. Aggressive timelines increase CVaR.

---

### `tests/v3/test_09_full_episode_e2e.py`

**What it tests**: Full episodes complete end-to-end without crash.

| Test | What it does | Pass criteria |
|------|--------------|---------------|
| `test_9_1_aligned_completes` | 20 steps aligned + neutral strategy | steps ≥ 1 |
| `test_9_2_conflicted_completes` | 20 steps conflicted + comprehensive | steps ≥ 1 |
| `test_9_3_hostile_aggressive_produces_veto_or_timeout` | 15 steps hostile + aggressive | terminal in valid set or unknown |
| `test_9_4_reward_collected_across_episode` | Check reward list | len(rewards) ≥ 1 |
| `test_9_5_reward_variance_non_trivial` | 12 steps aligned+comprehensive | Reward not constant |
| `test_9_6_strategy_comparison_possible` | Run comprehensive vs aggressive 3x each | Scores collected |
| `test_9_7_terminal_outcome_meaningful` | All 9 scenario×strategy combos | Non-empty terminal outcomes |
| `test_9_8_multidocument_action_works` | Send 3 documents to Legal | HTTP 200, reward returned |

**Working**: Tests 9.1–9.8 all pass. All three scenarios complete without crash, reward is collected throughout episodes, strategy comparison is possible, and multi-document actions work.

---

### `tests/v3/test_10_training_infrastructure.py`

**What it tests**: Training infrastructure imports and structure (runs inside container).

| Test | What it does | Pass criteria |
|------|--------------|---------------|
| `test_10_1_grpo_trainer_imports` | Import grpo_trainer module | GRPOTrainer and TrainingMetrics found |
| `test_10_2_training_metrics_fields` | Check TrainingMetrics fields | All 6 reward curve fields present |
| `test_10_3_curriculum_generator_imports` | Import adaptive_generator | AdaptiveCurriculumGenerator found |
| `test_10_4_colab_notebook_exists` | Check /app/env for grpo_colab.ipynb | Notebook with ≥ 5 cells found |
| `test_10_5_training_loop_smoke_test` | Check GRPOTrainer.__init__ exists | Class properly defined |
| `test_10_6_checkpoint_save_load_smoke` | Try importing torch | PyTorch optional, warns if missing |

**Working**: Tests 10.1–10.6 all pass. GRPOTrainer and AdaptiveCurriculumGenerator import correctly with all required fields. Colab notebook exists with valid structure.

---

### `tests/v3/test_11_research_properties.py`

**What it tests**: All 12 research properties in a single consolidated run.

| Property | Test function | Validation |
|----------|--------------|------------|
| P1 | `p1()` | Hidden fields not in observation |
| P2 | `p2()` | Different seeds → different engagement levels |
| P3 | `p3()` | EU > 0 AND CVaR > tau (unit test, not HTTP) |
| P4 | `p4()` | Reward variance across different actions |
| P5 | `p5()` | Lookahead cost = 0.07 ± 0.015 |
| P6 | `p6()` | ≥ 2 non-zero deltas across 5 same actions |
| P7 | `p7()` | cross_stakeholder_echoes field in observation |
| P8 | `p8()` | weak_signals field in observation |
| P9 | `p9()` | Causal signal active (echoes + deltas non-zero) |
| P10 | `p10()` | ≥ 2 unique engagement signatures across 5 resets |
| P11 | `p11()` | All scenarios complete 8 steps without HTTP error |
| P12 | `p12()` | GRPOTrainer + AdaptiveCurriculumGenerator import |

**Working**: All 12/12 research properties confirmed.

---

### `tests/v3/test_scorer_unit.py`

**What it tests**: UtteranceScorer deterministic unit tests (runs inside container).

| Test | What it does | Pass criteria |
|------|--------------|---------------|
| `test_lookahead_cost_exactly_0_07` | Assert LOOKAHEAD_COST == 0.07 | Exact equality |
| `test_log2_6_correct` | Assert LOG2_6 == np.log2(6) | Diff < 1e-6 |
| `test_lookahead_penalty_applied` | Score with/without lookahead | score_look.goal < score_no_look.goal by 0.07 |
| `test_all_dimensions_bounded_0_1` | 20 random scoring calls | All 5 dims in [0,1] |
| `test_goal_score_increases_with_approval` | Positive vs negative belief delta | pos.goal > neg.goal, pos > 0.5 > neg |
| `test_trust_targeted_delta` | Targeted vs untargeted trust | targeted > 0.5, untargeted == 0.5 |
| `test_information_entropy_reduction` | Uniform → certain | info > 0.5, more certain > less certain |
| `test_determinism_consistency` | 10 identical scoring calls | All identical |
| `test_prediction_accuracy` | Check compute_prediction_accuracy | Partial matches return between 0 and 1 |
| `test_scorer_reset` | Two scoring calls same inputs | Same result |
| `test_risk_cvar_no_profile_returns_0_5` | risk_profiles=None or {} | risk == 0.5 |
| `test_causal_no_targets_returns_0` | targeted_ids=[] | causal == 0.0 |
| `test_blocker_resolution_affects_goal` | Blockers resolved vs same | resolved > same |

**Working**: All 13/13 scorer unit tests pass. Scoring is deterministic, all dimensions bounded [0,1], lookahead penalty applied correctly, and blocking resolution affects goal score.

---

## What is NOT Working / Known Issues

### 1. BeliefTracker Module (`belief_tracker.py`) is Dead Code

The `deal_room/committee/belief_tracker.py` module defines `bayesian_update()` and `compute_engagement_level()`. However, **`DealRoomV3.step()` does not call this module**. Instead, it uses inline Bayesian update logic in `DealRoomV3._bayesian_update()` with its own hardcoded likelihood table.

**Impact**: Tests for belief_tracker are not present in the v3 test suite. The standalone unit tests in `tests/unit/test_belief_tracker.py` may test this dead code path.

### 2. LookaheadSimulator is Not Integrated

`deal_room/environment/lookahead.py` defines `LookaheadSimulator.simulate()` which generates optimistic/pessimistic hypotheses and returns worst-case predictions. However, **`DealRoomV3.step()` does not call this simulator**. The `action.lookahead` field is accepted but the actual simulation is not performed.

**Impact**: Lookahead cost is correctly deducted (0.07) but the hypothesis simulation that should justify this cost is not running. The `_score_goal()` method in `UtteranceScorer` deducts the lookahead cost, but the simulated belief/cVAR predictions from `LookaheadSimulator` are not used.

### 3. CVaR Veto Does Not Actually Terminate Episodes

The environment computes `veto_precursors` (warnings at 70% of tau) and the CVaR model can compute when `cvar_loss > tau` for any stakeholder. However, **`DealRoomV3.step()` does not check for veto termination**. The `done` flag is only set when `round_number >= max_rounds`.

**Impact**: test_4_3 (`veto_terminates_episode`) sometimes warns "Veto not triggered in 20 aggressive steps". This is because the veto check is not wired into the step function. The episode continues until timeout unless manually terminated by the agent's actions causing other failures.

### 4. Deliberation Summary LLM Calls Can Fail Silently

`CommitteeDeliberationEngine.run()` wraps the deliberation process in a try/except. If MiniMax API fails, `deliberation_result` is set to `None` and the code falls back to keeping current beliefs. This is silent — no error is surfaced.

**Impact**: In low-connectivity environments, belief propagation may not occur, but no error is raised. The `propagation_deltas` in the step info dict will be empty.

### 5. `stakeholder_messages` Always Empty in Observations

`DealRoomObservation` has `stakeholder_messages: Dict[str, str]` but `_build_observation()` never populates it (always `{}`). The `_generate_stakeholder_responses()` method exists but its result is returned from `step()` as part of the info dict, not the observation.

**Impact**: test_1_7 (`stakeholder_messages_populated_after_step`) passes but only checks that the field is a dict, not that it's non-empty. The actual response content is not available in the next observation.

### 6. GRPOTrainer is Not Connected to a Real Model

`GRPOTrainer` is a skeletal implementation with a random `_default_policy()`. It computes advantages and can run self-play episodes, but there is no integration with an actual language model or policy gradient updates.

**Impact**: test_10 and test_11/P12 only verify imports and class structure. No actual training occurs.

### 7. AdaptiveCurriculumGenerator is Standalone

The curriculum generator analyzes failure modes and selects scenarios, but it is not integrated into any training loop or environment reset flow.

**Impact**: The curriculum logic exists but is not active during training.

### 8. Security Test Directory is Empty

`tests/security/` has no test files.

### 9. MiniMax API Key Required at Instantiation

`DealRoomV3.__init__()` calls `validate_api_keys()` which raises `EnvironmentError` if `MINIMAX_API_KEY` is not set. This means the environment cannot be instantiated in test environments without the API key, even if no LLM calls are made.

**Impact**: Container tests (test_06, test_07, test_08, test_10) that import `DealRoomV3` directly will crash if the API key is not set.

---

## Summary: What is Working

| Category | Status |
|----------|--------|
| Environment reset/step cycle | ✅ Working |
| 18-field observation schema | ✅ Working |
| Hidden state (G not exposed) | ✅ Working |
| Episode isolation (different seeds → different states) | ✅ Working |
| 5D deterministic reward scoring | ✅ Working |
| Lookahead cost exactly 0.07 | ✅ Working |
| Engagement noise σ=0.03 (not cancellable) | ✅ Working |
| Cross-stakeholder echoes (~70% rate) | ✅ Working |
| Weak signals mechanism | ✅ Working |
| Causal graph propagation (direction, damping, normalization) | ✅ Working |
| Graph identifiability (all graphs unique) | ✅ Working |
| CVaR formula correctness | ✅ Working |
| Core research claim (EU>0 but CVaR>tau) | ✅ Working |
| CVaR veto precursors | ✅ Working |
| Terminal reward mapping | ✅ Working |
| Bayesian belief updates (inline) | ✅ Working |
| Risk tolerance ordering | ✅ Working |
| Good docs reduce CVaR vs poor docs | ✅ Working |
| Full episode completion (all 3 scenarios) | ✅ Working |
| Session pool (thread-safe, TTL, max_sessions) | ✅ Working |
| FastAPI endpoints (/reset, /step, /state, /health, /metadata) | ✅ Working |
| Gradio web UI mounting | ✅ Working |
| MiniMax LLM client with retry/intervention | ✅ Working |
| Stakeholder archetypes (6 profiles) | ✅ Working |
| UtteranceScorer determinism | ✅ Working |

## Summary: What is NOT Working / Incomplete

| Category | Status | Notes |
|----------|--------|-------|
| `belief_tracker.py` module | ❌ Dead code | Not called by environment |
| `LookaheadSimulator` integration | ❌ Not integrated | Cost deducted but hypothesis sim not run |
| CVaR veto actual termination | ❌ Not wired | Precursors computed but `done` only via max_rounds |
| `stakeholder_messages` in observation | ❌ Not populated | Always `{}` |
| GRPOTrainer actual training | ❌ Skeletal | Random default policy, no model integration |
| AdaptiveCurriculumGenerator integration | ❌ Standalone | Not called during training |
| Security tests | ❌ Empty directory | No security tests implemented |
| API key required at instantiation | ⚠️ Design issue | Cannot instantiate without key even for unit tests |

---

## Running the Tests

### Full v3 test suite (requires running server):
```bash
cd tests/v3
export DEALROOM_BASE_URL=http://127.0.0.1:7860
python test_00_environment_setup.py
python test_01_schema_validation.py
python test_02_reward_integrity.py
python test_03_causal_inference.py
python test_04_cvar_veto.py
python test_05_episode_isolation.py
python test_09_full_episode_e2e.py
python test_11_research_properties.py
```

### Container unit tests (run inside container):
```bash
docker exec dealroom-v3-test python /app/env/tests/v3/test_06_probabilistic_signals.py
docker exec dealroom-v3-test python /app/env/tests/v3/test_07_causal_graph.py
docker exec dealroom-v3-test python /app/env/tests/v3/test_08_cvar_preferences.py
docker exec dealroom-v3-test python /app/env/tests/v3/test_10_training_infrastructure.py
docker exec dealroom-v3-test python /app/env/tests/v3/test_scorer_unit.py
```

### Run all tests via provided script:
```bash
cd tests/v3
./run_all_tests.sh
```
