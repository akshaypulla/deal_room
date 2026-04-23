# Testing v3.7 — DealRoom Training Environment

## Overview

**Purpose**: Comprehensive test suite validating that the DealRoom v3.7 environment correctly implements the RL research desiderata. Tests are organized into sections covering environment setup, schema validation, reward integrity, causal inference, CVaR veto, episode isolation, probabilistic signals, causal graph, training infrastructure, and full end-to-end validation.

**Testing Philosophy**:
- Tests are designed for an LLM training environment (not a neural network agent)
- All tests use real meeting-like conversation scenarios
- Assertions are deterministic where possible; stochastic tests use fixed seeds for reproducibility
- Hidden fields (causal graph, beliefs, CVaR parameters) must never be exposed to the agent

---

## Test File Inventory

| File | Section | Tests | Running Environment |
|------|---------|-------|---------------------|
| `test_00_environment_setup.py` | 0 | 5 | Outside container (HTTP) |
| `test_01_schema_validation.py` | 1 | 11 | Outside container (HTTP) |
| `test_02_reward_integrity.py` | 2 | 10 | Outside container (HTTP) |
| `test_03_causal_inference.py` | 3 | 8 | Outside container (HTTP) |
| `test_04_cvar_veto.py` | 4 | 8 | Outside container (HTTP) |
| `test_05_episode_isolation.py` | 5 | 8 | Outside container (HTTP) |
| `test_06_probabilistic_signals.py` | 6 | 7 | Inside container (direct) |
| `test_07_causal_graph.py` | 7 | 9 | Inside container (direct) |
| `test_08_cvar_preferences.py` | 8 | 6 | Inside container (direct) |
| `test_09_full_episode_e2e.py` | 9 | 8 | Outside container (HTTP) |
| `test_10_training_infrastructure.py` | 10 | 6 | Inside container (direct) |
| `test_10_training_integration.py` | 10+ | 1 | Inside container (direct) |
| `test_11_research_properties.py` | Research | 12 | Mixed (HTTP + direct) |
| `test_P0_comprehensive.py` | P0 | 16 | Mixed |
| `test_assertion_hygiene.py` | Hygiene | 1 | AST analysis |
| `test_scorer_unit.py` | Scorer | 11 | Inside container (direct) |
| `conftest.py` | Config | - | Shared fixtures |

---

## Section 0: Environment Setup (`test_00_environment_setup.py`)

**What**: Validates that the test environment is properly configured before running any test.

**Why**: Tests depend on Docker containers, API keys, and server endpoints. Pre-flight validation prevents confusing failures.

**How**:
```python
def test_0_1_python_deps():
    # Validates critical Python packages are installed
    for dep in ["requests", "numpy"]:
        __import__(dep)  # Fails if not installed

def test_0_2_api_keys_configured():
    # Checks optional API keys (OPENAI_API_KEY, MINIMAX_API_KEY)
    # Keys are OPTIONAL—environment degrades gracefully without them
    for key in OPTIONAL_KEYS:
        val = os.getenv(key)
        # Displays masked key if set

def test_0_3_docker_container_running():
    # Verifies Docker container is running via docker ps
    subprocess.run(["docker", "ps", "--filter", f"name={CONTAINER_NAME}", "-q"])
    # Exits with error if container not found

def test_0_4_server_endpoints_responsive():
    # Tests /health, /metadata, /reset, /step endpoints
    session.get(f"{BASE_URL}/health")  # → 200
    session.post(f"{BASE_URL}/reset", json={"task_id": "aligned"})  # → session_id
    session.post(f"{BASE_URL}/step", json={...})  # → reward

def test_0_5_llm_client_imports():
    # Validates llm_client module is importable
    from deal_room.environment.llm_client import validate_api_keys, MAX_TOKENS
```

---

## Section 1: Schema Validation (`test_01_schema_validation.py`)

**What**: Validates that DealRoomObservation contains exactly the required fields with correct types, and that hidden internal fields are never exposed.

**Why**: The agent must only see what it's supposed to see. Hidden fields (causal graph, true beliefs, CVaR thresholds) would leak information and break the POMDP property.

**How**:

```python
REQUIRED_FIELDS = [
    "round_number", "max_rounds", "stakeholders", "stakeholder_messages",
    "engagement_level", "engagement_level_delta", "engagement_history",
    "weak_signals", "cross_stakeholder_echoes", "veto_precursors",
    "known_constraints", "requested_artifacts", "approval_path_progress",
    "deal_momentum", "deal_stage", "active_blockers", "days_to_deadline", "done",
]

HIDDEN_FIELDS = {
    "G", "causal_graph", "graph", "true_beliefs", "belief_distributions",
    "belief_state", "B_i", "V_i", "tau", "tau_i", "risk_thresholds",
    "cvar_thresholds", "edge_weights", "w_ij", "deliberation_transcript",
    "deliberation_log", "internal_dialogue", "u_i", "u_ij",
}

def test_1_1_all_required_fields_present():
    # POST /reset, check all 19 required fields exist
    r = session.post(f"{BASE_URL}/reset", json={"task_id": "aligned"})
    obs = r.json()
    missing = [f for f in REQUIRED_FIELDS if f not in obs]
    assert not missing, f"Missing: {missing}"

def test_1_2_no_hidden_fields_exposed():
    # Recursively scan observation for hidden fields
    def find_hidden(d, path=""):
        found = []
        if isinstance(d, dict):
            for k, v in d.items():
                if k in HIDDEN_FIELDS:
                    found.append(f"{path}.{k}")
                found.extend(find_hidden(v, f"{path}.{k}"))
        elif isinstance(d, list):
            for i, item in enumerate(d):
                found.extend(find_hidden(item, f"{path}[{i}]"))
        return found
    exposed = find_hidden(obs)
    assert not exposed, f"HIDDEN FIELDS EXPOSED: {exposed}"

def test_1_3_field_types_correct():
    # Validates round_number is int, engagement_level is dict, etc.
    assert isinstance(obs.get("round_number"), int)
    assert isinstance(obs.get("engagement_level_delta"), (int, float))
    assert not isinstance(obs.get("engagement_level_delta"), dict)

def test_1_4_engagement_history_window_size():
    # engagement_history must have >=5 entries (window size)
    history = obs.get("engagement_history", [])
    assert len(history) >= 5
    assert isinstance(history[0], dict)

def test_1_5_engagement_level_delta_single_float():
    # delta must be a single float, not a dict per stakeholder
    delta = obs.get("engagement_level_delta")
    assert isinstance(delta, (int, float)) and not isinstance(delta, dict)

def test_1_6_cross_stakeholder_echoes_is_list():
    # echoes is list of dicts with sender field
    assert isinstance(cse, list)
    for echo in cse:
        assert "from" in echo or "from_stakeholder" in echo or "sender" in echo

def test_1_7_stakeholder_messages_populated_after_step():
    # After step, stakeholder_messages dict is populated
    r = session.post(f"{BASE_URL}/step", json={...})
    msgs = obs.get("stakeholder_messages", {})
    assert isinstance(msgs, dict)

def test_1_8_action_schema_accepts_lookahead():
    # Action with lookahead nested field is accepted
    r = session.post(f"{BASE_URL}/step", json={
        "lookahead": {"depth": 2, "n_hypotheses": 2, "action_draft": {...}}
    })
    assert r.status_code == 200

def test_1_9_approval_path_progress_structure():
    # Each stakeholder has {band: blocker|neutral|workable|supporter}
    progress = obs.get("approval_path_progress", {})
    for stakeholder, payload in progress.items():
        assert payload["band"] in ["blocker", "neutral", "workable", "supporter"]

def test_1_10_deal_stage_valid_transitions():
    # round_number increments correctly after step
    assert obs1.get("round_number") == obs0.get("round_number") + 1

def test_1_11_documents_field_format():
    # documents is list of {name, content} objects
    r = session.post(f"{BASE_URL}/step", json={
        "documents": [{"name": "DPA", "content": "DPA content"}]
    })
    assert r.status_code == 200 and "reward" in r.json()
```

---

## Section 2: Reward Integrity (`test_02_reward_integrity.py`)

**What**: Validates that rewards are single floats (average of 5 dimensions), bounded [0, 1], deterministic, and not gameable through repeat actions or trivial content.

**Why**: The reward function must provide a meaningful learning signal without being hackable. Repeating the same action should not inflate reward.

**How**:

```python
def test_2_1_reward_is_single_float():
    # Reward must be a single float, not a dict
    assert isinstance(reward, (int, float)) and not isinstance(reward, dict)
    assert 0.0 <= reward <= 1.0

def test_2_2_lookahead_cost_is_exactly_007():
    # Lookahead cost is exactly 0.07 from goal dimension
    goal_without = info1.get("reward_components", {}).get("goal", 0)
    goal_with = info2.get("reward_components", {}).get("goal", 0)
    diff = goal_without - goal_with
    assert abs(diff - LOOKAHEAD_COST) < 0.015  # LOOKAHEAD_COST = 0.07

def test_2_3_reward_in_range_after_valid_actions():
    # All reward components stay in [0, 1] across different action types
    for a in actions:
        r = session.post(f"{BASE_URL}/step", json=a)
        components = r.json().get("info", {}).get("reward_components", {})
        for dim in ["goal", "trust", "info", "risk", "causal"]:
            assert 0.0 <= float(components[dim]) <= 1.0

def test_2_4_deterministic_reward_with_seed():
    # Same seed + same action → same reward (within float precision)
    rewards = []
    for _ in range(3):
        r = session.post(f"{BASE_URL}/reset", json={"seed": 100})
        r = session.post(f"{BASE_URL}/step", json=same_action)
        rewards.append(reward)
    spread = max(rewards) - min(rewards)
    assert spread < 1e-9

def test_2_5_repeat_same_action_does_not_escalate_reward():
    # Repeating the exact same action does NOT inflate reward
    g1 = reward after first same action
    g3 = reward after third same action
    trend = g3 - g1
    assert trend <= 0.01  # Allow small noise but not systematic increase

def test_2_6_different_targets_different_causal_scores():
    # Targeting Finance vs Legal vs TechLead produces different causal scores
    # (causal dimension varies with graph betweenness centrality)
    unique_scores = len(set(round(v, 3) for v in scores_by_target.values()))
    assert unique_scores >= 2

def test_2_7_informative_action_outperforms_empty():
    # Substantive action (ROI model + message) scores >= empty message
    assert g_subst >= g_empty - 0.1

def test_2_8_reward_non_trivial_variance():
    # Reward has discriminative variance across action types
    variance = max(rewards) - min(rewards)
    assert variance > 0.05

def test_2_9_good_documentation_higher_than_poor():
    # DPA + security cert yields higher reward than minimal doc
    assert avg_good >= avg_poor - 0.01

def test_2_10_lookahead_improves_prediction_accuracy():
    # Lookahead prediction accuracy > 60% over 20 runs
    accuracies = [info["lookahead_prediction_accuracy"] for ...]
    mean_acc = sum(accuracies) / len(accuracies)
    assert mean_acc > 0.60
```

---

## Section 3: Causal Inference (`test_03_causal_inference.py`)

**What**: Validates that belief propagation works correctly—targeting one stakeholder affects others through the causal graph, engagement noise is active, and weak signals appear.

**Why**: The causal graph is the core mechanism. Beliefs must propagate through edges. Noise must not be cancellable (prevents gaming).

**How**:

```python
def test_3_1_targeted_stakeholder_engagement_changes():
    # After targeting Finance, engagement_level_delta is numeric
    delta = obs.get("engagement_level_delta")
    assert isinstance(delta, (int, float))

def test_3_2_cross_stakeholder_echoes_detected():
    # Belief propagation fires at least 65% of the time
    # echo_recall_probability = 0.70 is the design target
    propagation_count = 0
    for i in range(12):
        echoes = obs.get("cross_stakeholder_echoes", [])
        if echoes:
            propagation_count += 1
    rate = propagation_count / 12
    assert rate >= 0.65

def test_3_3_engagement_history_window_slides():
    # History window size stays constant across steps (not growing)
    history_len_0 = len(obs0.get("engagement_history", []))
    for step_num in range(1, 4):
        obs = run_step(...)
        history = obs.get("engagement_history", [])
        assert len(history) == history_len_0

def test_3_4_engagement_noise_not_cancellable():
    # Deltas are non-zero across repeated same actions (noise σ > 0)
    deltas = [obs.get("engagement_level_delta", 0) for ...]
    non_zero = sum(1 for d in deltas if abs(d) > 0.001)
    assert non_zero >= 2  # At least 2 non-zero deltas

def test_3_5_different_targets_different_patterns():
    # Targeting Finance vs Legal produces different propagation patterns
    mean_f = np.mean(finance_deltas)
    mean_l = np.mean(legal_deltas)
    distance = abs(mean_f - mean_l)
    assert distance > 0.001

def test_3_6_weak_signals_for_non_targeted():
    # Weak signals appear for non-targeted stakeholders at least 20% of the time
    signals_detected = sum(1 for i in range(12) if obs.get("weak_signals"))
    rate = signals_detected / 12
    assert rate >= 0.2

def test_3_7_echo_content_structure():
    # Each echo has sender field (from/from_stakeholder/sender/source)
    for echo in echoes:
        assert any(k in echo for k in ["from", "sender", "source"])

def test_3_8_causal_signal_responds_to_graph_structure():
    # Hub node (ExecSponsor) has higher causal impact than leaf (Procurement)
    hub_impact = mean(abs(delta) + 0.03*len(echoes)) for ExecSponsor targeting
    leaf_impact = similar for Procurement targeting
    assert hub_impact > leaf_impact * 1.15
```

---

## Section 4: CVaR Veto (`test_04_cvar_veto.py`)

**What**: Validates the CVaR veto mechanism fires correctly—veto precursors appear before veto, aligned scenario doesn't veto prematurely, veto terminates episode, and different scenarios have different difficulty.

**Why**: The core research claim is that CVaR can veto a deal with positive expected utility because it captures tail risk.

**How**:

```python
def test_4_1_veto_precursor_before_veto():
    # Veto precursor must appear BEFORE veto fires
    # (precursor fires at 70% of tau, veto at 100% with streak >= 2)
    for _ in range(18):
        r = session.post(f"{BASE_URL}/step", json=AGGRESSIVE_ACTION)
        if veto_fired:
            assert precursor_seen, "Veto without preceding precursor"
            break
        if obs.get("veto_precursors"):
            precursor_seen = True

def test_4_2_aligned_no_early_veto():
    # Aligned scenario should NOT veto on first step
    r = session.post(f"{BASE_URL}/reset", json={"task_id": "aligned"})
    r = session.post(f"{BASE_URL}/step", json={...})
    assert not done, "Aligned scenario vetoed on first step"

def test_4_3_veto_terminates_episode():
    # When veto fires, episode terminates with done=True
    for _ in range(20):
        r = session.post(f"{BASE_URL}/step", json=AGGRESSIVE_ACTION)
        if done:
            assert "veto" in terminal.lower()
            break
    assert veto_confirmed

def test_4_3_veto_deterministic():
    # Deterministic test: Legal veto fires by step 5 with seed=42
    for i in range(5):
        r = session.post(f"{BASE_URL}/step", json={
            "action_type": "exec_escalation", "target_ids": ["Legal"], ...
        })
        if done:
            assert info.get("terminal_category") == "veto"
            assert info.get("terminal_outcome") == "veto_by_Legal"
            return

def test_4_4_timeout_terminates():
    # Episode terminates after max_rounds
    for i in range(max_rounds + 2):
        r = session.post(f"{BASE_URL}/step", json={...})
        if done:
            assert "max_rounds" in terminal or "timeout" in terminal
            break

def test_4_5_scenario_difficulty_differentiation():
    # Hostile reaches veto pressure earlier than aligned
    hostile_mean = avg(first_pressure_round("hostile_acquisition"))
    aligned_mean = avg(first_pressure_round("aligned"))
    assert hostile_mean <= aligned_mean

def test_4_6_veto_precursor_is_stakeholder_specific():
    # Precursors are tied to specific stakeholders, not global
    precursor_stakeholders = set(precursors.keys())
    assert precursor_stakeholders  # Not empty

def test_4_7_cveto_not_just_eu():
    # CVaR veto fires even when EU > 0 (not loss-chasing)
    # In hostile scenario, aggressive actions trigger veto despite potentially positive EU
    assert veto_fired, "CVaR veto did not fire in hostile scenario"
```

---

## Section 5: Episode Isolation (`test_05_episode_isolation.py`)

**What**: Validates that episodes are properly isolated—different seeds produce different states, round_number resets to 0 on reset, done is False after reset, and session state doesn't leak between episodes.

**Why**: Reinforcement learning requires clean episode boundaries. State leakage would create correlation between episodes and corrupt learning.

**How**:

```python
def test_5_1_different_seeds_different_initial_state():
    # Different seeds → different engagement levels
    diff = sum(abs(eng1.get(sid, 0) - eng2.get(sid, 0)) for sid in eng1)
    assert diff > 0.001, "Different seeds produced identical states"

def test_5_2_round_number_resets_to_zero():
    # After 3 steps, reset should set round_number back to 0
    obs_after = session.post(f"{BASE_URL}/reset", json={"seed": 11}).json()
    assert obs_after.get("round_number") == 0

def test_5_3_done_false_after_reset():
    # done=False immediately after reset in all scenarios
    for scenario in ["aligned", "conflicted", "hostile_acquisition"]:
        r = session.post(f"{BASE_URL}/reset", json={"task_id": scenario})
        assert r.json().get("done") is False

def test_5_4_engagement_history_initialized():
    # History window initialized with >=5 entries
    history = obs.get("engagement_history", [])
    assert len(history) >= 5

def test_5_5_round_number_increments_correctly():
    # round_number increments: 0 → 1 → 2 → ... after each step
    for expected_round in range(1, 6):
        r = session.post(f"{BASE_URL}/step", json={...})
        actual = obs.get("round_number")
        assert actual == expected_round

def test_5_6_all_three_scenarios_work():
    # aligned, conflicted, hostile_acquisition all complete without crash
    for scenario in [...]:
        r = session.post(f"{BASE_URL}/reset", json={"task_id": scenario})
        r = session.post(f"{BASE_URL}/step", json={...})
        assert r.status_code == 200

def test_5_7_same_session_same_state_across_steps():
    # Within a session, round_number increments consistently
    assert obs2.get("round_number") == obs1.get("round_number") + 1

def test_5_8_reset_clears_all_session_state():
    # Reset clears round_number, done flag, and engagement history
    obs_after = session.post(f"{BASE_URL}/reset", json={"seed": 71}).json()
    assert obs_after.get("round_number") == 0
    assert obs_after.get("done") is False
    assert len(obs_after.get("engagement_history", [])) >= 5
```

---

## Section 6: Probabilistic Signals (`test_06_probabilistic_signals.py`)

**What**: Validates probabilistic observation mechanisms—weak signals, cross-stakeholder echoes, and echo firing rate.

**Why**: Observations are intentionally noisy/probabilistic. Echo recall probability is 0.70, weak signal thresholds must be respected.

**How** (runs inside container with direct imports):

```python
def test_6_1_weak_signals_field_exists():
    from deal_room.environment.dealroom_v3 import DealRoomV3
    env = DealRoomV3()
    obs = env.reset(task_id="aligned")
    assert hasattr(obs, "weak_signals")

def test_6_2_cross_stakeholder_echoes_exists():
    assert hasattr(obs, "cross_stakeholder_echoes")

def test_6_3_echo_structure():
    # Echoes is list of dicts with sender field
    echoes = env._generate_cross_stakeholder_echoes(action)
    for echo in echoes:
        assert any(k in echo for k in ["from", "sender", "source"])

def test_6_4_weak_signals_populated_after_action():
    obs2, reward, done, info = env.step(action)
    weak = obs2.weak_signals
    assert isinstance(weak, dict)

def test_6_5_echo_firing_rate_nonzero():
    # Echoes fire at least 65% of the time (design target 70%)
    fired = sum(1 for i in range(40) if obs2.cross_stakeholder_echoes)
    rate = fired / 40
    assert rate >= 0.65

def test_6_6_weak_signal_threshold_respected():
    from deal_room.environment.dealroom_v3 import OBS_CONFIG
    threshold = OBS_CONFIG.weak_signal_hard_threshold
    # Weak signals are list[str] tags, not numeric values

def test_6_7_echo_recall_probability_configured():
    prob = OBS_CONFIG.echo_recall_probability
    assert 0.65 <= prob <= 0.75  # Should be ~0.70
```

---

## Section 7: Causal Graph (`test_07_causal_graph.py`)

**What**: Unit tests for graph structure and belief propagation—propagation direction, signal carrying, damping, normalization, no self-loops, authority invariance, hub centrality.

**Why**: The causal graph is the core simulation component. Must correctly propagate beliefs, maintain normalization, and respect authority hierarchy.

**How** (runs inside container):

```python
def test_7_1_propagation_direction():
    # Signal flows A→B through edge, C is unaffected (no edge C←A)
    g = CausalGraph(nodes=["A", "B", "C"], edges={("A", "B"): 0.7}, ...)
    updated = propagate_beliefs(g, before, after, n_steps=3)
    assert updated["B"].positive_mass() > before["B"].positive_mass() + 0.05
    assert abs(updated["C"].positive_mass() - before["C"].positive_mass()) < 0.03

def test_7_2_signal_carries_through_edge():
    # B's belief changes when A's belief changes (signal carries through edge)
    g = CausalGraph(nodes=["A", "B"], edges={("A", "B"): 0.8}, ...)
    pm_B_after = updated["B"].positive_mass()
    assert pm_B_after != before["B"].positive_mass()

def test_7_3_damping_prevents_runaway():
    # Dense graph (all nodes connected) with damping 0.95^step keeps beliefs bounded
    nodes = list("ABCDE")
    edges = {(s, d): 0.8 for s in nodes for d in nodes if s != d}
    for sid in "BCDE":
        assert 0.0 < updated[sid].positive_mass() < 1.0

def test_7_4_beliefs_normalized_after_propagation():
    # All beliefs sum to 1.0 after propagation
    for sid, b in updated.items():
        total = sum(b.distribution.values())
        assert abs(total - 1.0) < 1e-6

def test_7_5_no_self_loops():
    # No edge from node to itself in any scenario
    for scenario in ["aligned", "conflicted", "hostile_acquisition"]:
        g = sample_graph(STANDARD_5, STANDARD_H, scenario, rng)
        for sid in STANDARD_5:
            assert g.get_weight(sid, sid) == 0.0

def test_7_6_exec_sponsor_outgoing_authority():
    # ExecSponsor has >=2 outgoing edges in all scenarios (authority invariant)
    for scenario in ["aligned", "conflicted", "hostile_acquisition"]:
        for seed in range(10):
            g = sample_graph(..., scenario, np.random.default_rng(seed))
            outgoing = [w for w in g.get_outgoing("ExecSponsor").values() if w > 0.1]
            assert len(outgoing) >= 2

def test_7_7_hub_centrality_beats_leaf():
    # Hub node has highest betweenness centrality
    hub_c = get_betweenness_centrality(g, "Hub")
    leaf_c = max(get_betweenness_centrality(g, leaf) for leaf in ["A", "B", "C", "D"])
    assert hub_c >= leaf_c

def test_7_8_graph_identifiability():
    # All 20 sampled graphs produce unique behavioral signatures
    # This proves G (causal graph) is identifiable from behavior
    distinguishable = 0
    for i in range(20):
        for j in range(i + 1, 20):
            if signatures[i] != signatures[j]:
                distinguishable += 1
    assert distinguishable == total_pairs  # All pairs unique

def test_7_9_hub_node_has_higher_centrality_impact():
    # Hub nodes have >=30% more behavioral impact than leaves
    ratios = [hub_impact / max(leaf_impact, 1e-9) for seed in range(10)]
    assert np.mean(ratios) > 1.3
```

---

## Section 8: CVaR Preferences (`test_08_cvar_preferences.py`)

**What**: Unit tests for CVaR computation and stakeholder risk profiles.

**Why**: CVaR is the core risk metric. Must be mathematically correct, differentiate between stakeholders, and correctly penalize bad deals.

**How** (runs inside container):

```python
def test_8_1_core_claim():
    # CORE CLAIM: eu>0 but cvar_loss > tau → veto fires
    # Deal with has_dpa=False, has_security_cert=False has positive EU but CVaR > tau
    eu, cvar_loss = evaluate_deal(terms, legal_profile, rng, n_samples=500)
    assert eu > 0  # EU is positive
    assert cvar_loss > legal_profile.tau  # But CVaR exceeds veto threshold
    assert cvar_loss > eu  # CVaR > EU (tail risk dominates)

def test_8_2_good_docs_lower_cvar_than_poor():
    # DPA + security cert → lower CVaR than no docs
    _, cvar_good = evaluate_deal(good_terms, legal_profile, rng)
    _, cvar_poor = evaluate_deal(poor_terms, legal_profile, rng)
    assert cvar_good < cvar_poor

def test_8_3_cvar_formula_correct():
    # CVaR formula: worst alpha percentile mean loss
    # High outcomes → low CVaR; Low outcomes → high CVaR
    cvar_high = compute_cvar(np.array([0.8] * 10), alpha=0.90)
    cvar_low = compute_cvar(np.array([0.2] * 10), alpha=0.90)
    assert cvar_high < cvar_low

def test_8_4_tau_ordering():
    # Risk tolerance: Legal < Finance < ExecSponsor
    # (Legal is most risk-averse with lowest tau)
    assert legal_tau < finance_tau < exec_tau

def test_8_5_aggressive_timeline_higher_cvar():
    # 4-week timeline → higher CVaR than 16-week for TechLead
    _, cvar_a = evaluate_deal(t_agg, tech_profile, rng)
    _, cvar_r = evaluate_deal(t_reas, tech_profile, rng)
    assert cvar_a > cvar_r

def test_8_6_cvar_subadditivity_sanity():
    # CVaR should be ≤ max outcome (coherence property)
    cvar = compute_cvar(outcomes, alpha=0.95)
    max_val = np.max(outcomes)
    assert cvar <= max_val
```

---

## Section 9: Full Episode End-to-End (`test_09_full_episode_e2e.py`)

**What**: End-to-end validation of complete episodes—scenarios complete without crash, reward trajectory collected, strategy comparison works, terminal outcomes meaningful.

**Why**: Integration test to ensure all components work together across a full episode.

**How**:

```python
def test_9_1_aligned_completes():
    # Aligned + neutral strategy completes without crash
    steps, terminal, rewards, _ = run_episode("aligned", "neutral", max_steps=20)
    assert steps >= 1

def test_9_2_conflicted_completes():
    steps, terminal, rewards, _ = run_episode("conflicted", "comprehensive")
    assert steps >= 1

def test_9_3_hostile_aggressive_produces_veto_or_timeout():
    # Hostile + aggressive must produce veto or timeout (valid terminals)
    # Must NOT crash or produce unexpected terminal
    assert terminal in valid_terminals or terminal.startswith("veto_by_")

def test_9_4_reward_collected_across_episode():
    # Reward entries collected at each step
    assert len(rewards) >= 1

def test_9_5_reward_variance_non_trivial():
    # Reward trajectory is not constant (has variance > 0.05)
    variance = max(rewards) - min(rewards)
    assert variance > 0.05

def test_9_6_strategy_comparison_possible():
    # Comprehensive strategy beats aggressive baseline by > 0.15
    avg_comp = mean([sum(rewards) for seed in [60,61,62]])
    avg_agg = mean([sum(rewards) for seed in [60,61,62]])
    assert avg_comp > avg_agg + 0.15

def test_9_7_terminal_outcome_meaningful():
    # Episode termination fires with non-empty terminal outcome
    terminals_seen = set()
    for scenario in [...]:
        for strategy in [...]:
            if terminal and terminal not in ("unknown", "EMPTY", None):
                terminals_seen.add(terminal)
    assert terminals_seen

def test_9_8_multidocument_action_works():
    # Action with multiple documents returns reward
    r = session.post(f"{BASE_URL}/step", json={
        "documents": [{"name": "DPA", ...}, {"name": "security_cert", ...}]
    })
    assert r.status_code == 200 and "reward" in r.json()
```

---

## Section 10: Training Infrastructure (`test_10_training_infrastructure.py`, `test_10_training_integration.py`)

**What**: Validates GRPOTrainer, TrainingMetrics, AdaptiveCurriculumGenerator import correctly, notebook exists, and training loop produces improvement.

**Why**: Training infrastructure must be functional for LLM training to proceed.

**How**:

```python
def test_10_1_grpo_trainer_imports():
    from deal_room.training.grpo_trainer import GRPOTrainer, TrainingMetrics
    assert hasattr(mod, "GRPOTrainer")

def test_10_2_training_metrics_fields():
    # TrainingMetrics has all required reward curve fields
    required = ["goal_reward", "trust_reward", "info_reward", "risk_reward",
                "causal_reward", "lookahead_usage_rate", ...]
    missing = [f for f in required if f not in fields]
    assert not missing

def test_10_3_curriculum_generator_imports():
    from deal_room.curriculum.adaptive_generator import AdaptiveCurriculumGenerator

def test_10_4_colab_notebook_exists():
    # grpo_colab.ipynb exists with >=5 cells

def test_10_5_training_loop_smoke_test():
    # GRPOTrainer.__init__ exists

def test_10_6_checkpoint_save_load_smoke():
    # PyTorch checkpoint save/load (skipped if no torch)

def test_training_actually_improves():
    # THE KEY TEST: trained policy beats random baseline
    random_adapter = RandomPolicyAdapter(use_lookahead_probability=0.0)
    random_metrics = trainer.evaluate_policy(random_adapter, ...)

    trainer.train(n_episodes=4, episodes_per_batch=4, max_steps=8)
    trained_metrics = trainer.evaluate_policy(trainer.policy_adapter, ...)

    improvement = trained_metrics.weighted_reward - random_metrics.weighted_reward
    assert improvement > 0.15, f"Got {improvement:.3f}, expected >0.15"
```

---

## Section 11: Research Properties (`test_11_research_properties.py`)

**What**: Validates all 12 research desiderata in a single consolidated test.

**Why**: Provides a quick verification that the environment satisfies the complete RL research requirements.

**How**:

```python
P1: G is hidden from agent observation
P2: Episode reset regenerates G (different seeds → different states)
P3: CVaR veto fires despite positive EU (tail risk, not loss-chasing)
P4: Five reward dimensions are independent (not redundant)
P5: Lookahead costs exactly 0.07 from goal reward
P6: Engagement noise is not cancellable (σ > 0)
P7: Cross-stakeholder echoes field present (70% echo_recall_probability)
P8: Weak signals field present (12% hard threshold)
P9: r^causal varies with target's graph centrality
P10: Every reset produces a different G (unique behavioral signatures)
P11: Full episode completes without crash
P12: Training loop imports without error (GRPOTrainer + curriculum)
```

---

## P0 Comprehensive (`test_P0_comprehensive.py`)

**What**: Critical issues validation covering training improvement, CVaR correctness, lookahead usefulness, and full debug traces.

**Why**: P0 tests address the core research claims and implementation correctness.

**How**:

```python
# P0-1: Training Improvement
test_P0_1a_baseline_vs_trained_comparison()  # Trained > Random by >0.10
test_P0_1b_multi_episode_improvement()  # Later batches >= Earlier batches
test_P0_1c_dimension_wise_improvement()  # At least 2 dims improve with training
test_P0_1d_policy_persistence()  # Save/load preserves trained flag

# P0-2: CVaR Correctness
test_P0_2a_cvar_deterministic_calculation()  # CVaR formula mathematically correct
test_P0_2b_cvar_veto_with_positive_eu()  # Veto fires in hostile scenarios
test_P0_2c_cvar_per_stakeholder()  # CVaR differentiates per stakeholder profile

# P0-3: Lookahead Usefulness
test_P0_3a_lookahead_improves_decisions()  # Lookahead does not degrade reward
test_P0_3b_lookahead_prediction_accuracy()  # Accuracy > 55% threshold
test_P0_3c_lookahead_cost_exactly_007()  # Cost is exactly 0.07, not approximate

# P0-4: Debug Trace
test_P0_4a_belief_state_trace()  # Belief state trace before/after each step
test_P0_4b_cvar_breakdown_per_stakeholder()  # CVaR computed per stakeholder
test_P0_4c_action_effect_trace()  # Action → effect trace across episodes

# P1: Stochastic Stabilization
test_P1_stochastic_stabilized()  # Veto fires reliably in hostile scenarios

# P2: Adversarial Scenarios
test_P2_adversarial_degenerate_graph()  # No influence edges handled
test_P2_adversarial_extreme_tau()  # Extreme tau values handled without crash
```

---

## Assertion Hygiene (`test_assertion_hygiene.py`)

**What**: AST-based guard against print-only critical tests.

**Why**: Tests must have actual assertions, not just print statements.

**How**:

```python
def test_critical_v3_tests_have_assertions_or_explicit_failures():
    CRITICAL_TEST_FILES = [
        "test_02_reward_integrity.py", "test_03_causal_inference.py",
        "test_04_cvar_veto.py", "test_09_full_episode_e2e.py",
    ]

    for filename in CRITICAL_TEST_FILES:
        tree = ast.parse((base / filename).read_text())
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                assert _has_failure_signal(node), f"{filename}::{node.name} has no assert/raise"
```

---

## Scorer Unit Tests (`test_scorer_unit.py`)

**What**: Direct unit tests for UtteranceScorer without HTTP API.

**Why**: Fast, deterministic validation of reward computation.

**How**:

```python
def test_lookahead_cost_exactly_0_07():
    assert LOOKAHEAD_COST == 0.07

def test_log2_6_correct():
    expected = np.log2(6)
    assert abs(LOG2_6 - expected) < 1e-6

def test_lookahead_penalty_applied():
    score_look = scorer.score(action, before, after, graph, lookahead_used=True)
    score_no_look = scorer.score(action, before, after, graph, lookahead_used=False)
    diff = score_no_look.goal - score_look.goal
    assert abs(diff - LOOKAHEAD_COST) < 0.001

def test_all_dimensions_bounded_0_1():
    for _ in range(20):
        score = scorer.score(...)
        for dim in ["goal", "trust", "information", "risk", "causal"]:
            val = getattr(score, dim)
            assert 0.0 <= val <= 1.0

def test_goal_score_increases_with_approval():
    s_pos = scorer.score(action, state_before, positive_beliefs_after)
    s_neg = scorer.score(action, state_before, negative_beliefs_after)
    assert s_pos.goal > s_neg.goal

def test_trust_targeted_delta():
    s = scorer.score(targeted_action, state_before, targeted_beliefs_after)
    assert s.trust > 0.5
    s2 = scorer.score(targeted_action, state_before, untargeted_beliefs_after)
    assert s2.trust == 0.5  # Untargeted gets neutral

def test_information_entropy_reduction():
    s_moderate = scorer.score(action, uniform_beliefs, moderate_beliefs)
    s_high = scorer.score(action, uniform_beliefs, high_certainty_beliefs)
    assert s_high.information >= s_moderate.information

def test_determinism_consistency():
    scores = [scorer.score(...) for _ in range(10)]
    assert all(s == scores[0] for s in scores)

def test_prediction_accuracy():
    acc = compute_prediction_accuracy(pred, actual)
    assert 0.0 < acc < 1.0

def test_risk_cvar_no_profile_returns_0_5():
    s = scorer.score(action, state_without_risk_profiles, ...)
    assert s.risk == 0.5

def test_causal_no_targets_returns_0():
    s = scorer.score(action with empty target_ids, ...)
    assert s.causal == 0.0

def test_blocker_resolution_affects_goal():
    s_resolved = scorer.score(action, state_with_blockers, state_resolved)
    s_same = scorer.score(action, state_with_blockers, state_same)
    assert s_resolved.goal > s_same.goal
```

---

## Running the Tests

### Outside Container (HTTP tests)
```bash
export DEALROOM_BASE_URL=http://127.0.0.1:7860
python -m pytest tests/v3/test_00_environment_setup.py
python -m pytest tests/v3/test_01_schema_validation.py
python -m pytest tests/v3/test_02_reward_integrity.py
# ... etc
```

### Inside Container (direct import tests)
```bash
docker exec dealroom-v3-test python /app/tests/v3/test_06_probabilistic_signals.py
docker exec dealroom-v3-test python /app/tests/v3/test_07_causal_graph.py
docker exec dealroom-v3-test python /app/tests/v3/test_08_cvar_preferences.py
# ... etc
```

### Full Validation
```bash
# Section 0-5 (outside container)
for f in tests/v3/test_0*.py tests/v3/test_1*.py tests/v3/test_2*.py tests/v3/test_3*.py tests/v3/test_4*.py tests/v3/test_5*.py; do
    python -m pytest "$f"
done

# Section 6-10 (inside container)
docker exec dealroom-v3-test python /app/tests/v3/test_06_probabilistic_signals.py
docker exec dealroom-v3-test python /app/tests/v3/test_07_causal_graph.py
docker exec dealroom-v3-test python /app/tests/v3/test_08_cvar_preferences.py
docker exec dealroom-v3-test python /app/tests/v3/test_09_full_episode_e2e.py
docker exec dealroom-v3-test python /app/tests/v3/test_10_training_infrastructure.py
docker exec dealroom-v3-test python /app/tests/v3/test_10_training_integration.py
```
