# DealRoom v3 Testing Documentation

## Overview

This document describes the comprehensive testing strategy for DealRoom v3, covering unit tests, integration tests, end-to-end tests, and performance benchmarks. The test suite ensures reliability of the RL environment, committee decision-making, reward computation, and training pipeline.

## Test Organization

```
tests/
├── v3/                    # Core environment tests (v3 release)
│   ├── conftest.py        # Shared fixtures
│   ├── test_00_environment_setup.py
│   ├── test_01_schema_validation.py
│   ├── test_02_reward_integrity.py
│   ├── test_03_causal_inference.py
│   ├── test_04_cvar_veto.py
│   ├── test_05_episode_isolation.py
│   ├── test_06_probabilistic_signals.py
│   ├── test_07_causal_graph.py
│   ├── test_08_cvar_preferences.py
│   ├── test_09_full_episode_e2e.py
│   ├── test_10_training_integration.py
│   ├── test_10_training_infrastructure.py
│   ├── test_11_research_properties.py
│   ├── test_scorer_unit.py
│   └── test_assertion_hygiene.py
├── unit/                  # Unit tests for components
│   ├── test_deliberation_engine.py
│   └── test_curriculum.py
├── integration/           # Integration tests
│   └── test_environment.py
├── e2e/                   # End-to-end workflow tests
│   └── test_workflows.py
└── performance/           # Performance benchmarks
```

---

## Core Environment Tests (tests/v3/)

### Test 00: Environment Setup

**File:** `test_00_environment_setup.py`

Validates basic environment initialization and lifecycle:

| Test | Description | Expected Outcome |
|------|-------------|------------------|
| `test_env_creation` | Environment instantiates correctly | Environment object created |
| `test_reset_returns_observation` | Reset produces valid observation | Observation dict with required keys |
| `test_action_space_bounds` | Actions within valid range | No out-of-bounds errors |
| `test_observation_space` | Observation shape valid | Correct dtype and dimensions |

**Fixtures (conftest.py):**
```python
@pytest.fixture
def env():
    return DealRoomV3(config=default_config)

@pytest.fixture
def scenario():
    return ScenarioConfig(stakeholders=[...], constraints=[...])
```

---

### Test 01: Schema Validation

**File:** `test_01_schema_validation.py`

Ensures all data structures conform to schemas:

**Validates:**
- `AgentAction` schema compliance
- `Observation` schema compliance
- `StakeholderState` schema compliance
- `CommitteeDecision` schema compliance

**Test Cases:**
- Action with missing required field → ValidationError
- Observation with extra field → Accepted (lenient parsing)
- Stakeholder with invalid archetype → ValueError

---

### Test 02: Reward Integrity

**File:** `test_02_reward_integrity.py`

Verifies reward computation correctness:

| Test | Description |
|------|-------------|
| `test_scorer_deterministic` | Same input produces same output |
| `test_scorer_bounded` | Scores within [0, 1] range |
| `test_weighted_combination` | Weighted sum matches formula |
| `test_pareto_efficiency` | Pareto checker identifies improvements |

**Scoring Dimensions Verified:**
- Goal alignment score computation
- Trust building score updates
- Information gain calculation
- Risk management penalty
- Causal coherence factor

---

### Test 03: Causal Inference

**File:** `test_03_causal_inference.py`

Tests causal reasoning capabilities:

**Validates:**
- Causal graph structure (DAG property)
- Conditional probability computation
- Influence propagation through graph
- Counterfactual reasoning support

**Key Tests:**
```python
def test_causal_dag_property():
    """Graph must not contain cycles"""
    graph = CausalGraph()
    assert graph.is_acyclic()

def test_influence_propagation():
    """Evidence flows through causal edges"""
    result = graph.propagate(evidence={'price': 0.8})
    assert 'outcomes' in result
```

---

### Test 04: CVaR Veto

**File:** `test_04_cvar_veto.py`

Tests risk-sensitive decision making:

| Test | Description |
|------|-------------|
| `test_cvar_below_threshold_triggers_veto` | CVaR < threshold blocks outcome |
| `test_cvar_above_threshold_allows_progression` | CVaR ≥ threshold permits continue |
| `test_veto_preserves_worst_case` | Veto activates on tail risk |
| `test_veto_counting` | Multiple vetoes tracked correctly |

**CVaR Threshold Tests:**
```python
def test_veto_on_tail_risk():
    outcomes = generate_tail_risk_scenario()
    cvar = compute_cvar(outcomes, alpha=0.1)
    assert cvar < VETO_THRESHOLD
    assert veto_triggered == True
```

---

### Test 05: Episode Isolation

**File:** `test_05_episode_isolation.py`

Ensures episodes don't share state:

**Validates:**
- State reset clears all intermediate data
- No cross-contamination between episodes
- Random state properly seeded per episode
- Belief states independent per scenario

**Test Strategy:**
```python
def test_no_state_leakage():
    env1 = DealRoomV3()
    env2 = DealRoomV3()
    
    # Run episode 1
    env1.reset()
    for _ in range(10):
        env1.step(random_action())
    
    # Episode 2 should not see episode 1 data
    env2.reset()
    initial_state = env2.get_observation()
    assert initial_state['history_length'] == 0
```

---

### Test 06: Probabilistic Signals

**File:** `test_06_probabilistic_signals.py`

Tests belief propagation with uncertainty:

**Validates:**
- Bayesian update formulas
- Probability distribution maintenance
- Uncertainty quantification
- Evidence combination rules

**Test Cases:**
```python
def test_belief_update_formula():
    prior = Beta(2, 2)
    likelihood = 0.7
    posterior = update(prior, likelihood)
    
    # Check posterior mean is between prior and likelihood
    assert prior.mean() <= posterior.mean() <= likelihood
```

---

### Test 07: Causal Graph

**File:** `test_07_causal_graph.py`

Comprehensive causal structure tests:

| Test | Description |
|------|-------------|
| `test_node_addition` | Nodes added with correct attributes |
| `test_edge_creation` | Edges have proper causal strength |
| `test_dag_validation` | No cycles in graph |
| `test_influence_calculation` | Impact scores computed correctly |
| `test_graph_serializable` | Graph can be serialized/deserialized |

---

### Test 08: CVaR Preferences

**File:** `test_08_cvar_preferences.py`

Tests preference modeling with CVaR:

**Validates:**
- Worst-case scenario identification
- Risk-adjusted utility computation
- Preference consistency across episodes
- Stakeholder archetype differentiation

---

### Test 09: Full Episode E2E

**File:** `test_09_full_episode_e2e.py`

End-to-end negotiation episode test:

**Episode Flow:**
1. Environment reset with scenario
2. Agent selects actions through full episode
3. Committee deliberates each turn
4. Rewards computed at termination
5. Outcome quality assessed

**Success Criteria:**
- Episode completes without crashes
- Final reward is valid number
- Negotiation outcome is consistent with actions

---

### Test 10: Training Integration

**File:** `test_10_training_integration.py`

Validates GRPO trainer integration:

| Test | Description |
|------|-------------|
| `test_policy_adapter_registry` | All adapters registered |
| `test_training_loop_step` | Single training step executes |
| `test_advantage_estimation` | Advantage computed correctly |
| `test_checkpoint_save_load` | Training state preserved |

**Integration Points:**
- Environment ↔ Trainer communication
- Policy ↔ Adapters interface
- Reward signal → Loss computation

---

### Test 10b: Training Infrastructure

**File:** `test_10_training_infrastructure.py`

Tests training system infrastructure:

- Distributed training support
- Resource allocation
- Log aggregation
- Metric collection

---

### Test 11: Research Properties

**File:** `test_11_research_properties.py`

Validates research-grade properties:

**Properties Verified:**
- **Identifiability**: Causal effects can be estimated
- **Consistency**: Estimator converges to true value
- **Unbiasedness**: No systematic error in rewards
- **Variance bounds**: Confidence intervals valid

---

### Scorer Unit Tests

**File:** `test_scorer_unit.py`

Isolated utterance scorer tests:

| Test | Description |
|------|-------------|
| `test_goal_dimension` | Goal scoring in isolation |
| `test_trust_dimension` | Trust scoring in isolation |
| `test_info_dimension` | Information gain scoring |
| `test_risk_dimension` | Risk penalty scoring |
| `test_causal_dimension` | Causal coherence scoring |
| `test_combined_score` | Weighted combination verified |

---

### Assertion Hygiene

**File:** `test_assertion_hygiene.py`

Ensures test assertions are robust:

- No silent passes (all tests assert something)
- Meaningful assertion messages
- Proper exception handling
- Timeout configurations

---

## Unit Tests (tests/unit/)

### Deliberation Engine Tests

**File:** `test_deliberation_engine.py`

Tests committee decision-making logic:

```python
def test_deliberation_cycle():
    """Single deliberation produces decision"""
    engine = DeliberationEngine(committee)
    decision = engine.deliberate(state)
    
    assert decision is not None
    assert hasattr(decision, 'action')
    assert hasattr(decision, 'confidence')

def test_veto_integration():
    """Veto correctly prevents low-CVaR outcomes"""
    decision = engine.deliberate(low_cvar_state)
    assert decision.veto_triggered == True
```

---

### Curriculum Tests

**File:** `test_curriculum.py`

Tests adaptive curriculum generation:

| Test | Description |
|------|-------------|
| `test_difficulty_progression` | Difficulty increases over time |
| `test_performance_feedback` | Performance affects difficulty |
| `test_domain_randomization` | Varied scenarios generated |

---

## Integration Tests (tests/integration/)

### Environment Integration

**File:** `test_environment.py`

Tests component interactions:

**Validates:**
- Environment ↔ Stakeholder communication
- Committee ↔ Belief tracker sync
- Reward ↔ Scorer pipeline
- Server ↔ Environment gateway

---

## E2E Tests (tests/e2e/)

### Workflow Tests

**File:** `test_workflows.py`

Complete workflow validation:

```python
def test_full_negotiation_workflow():
    """Complete negotiation from start to finish"""
    # 1. Initialize environment
    env = DealRoomV3()
    
    # 2. Load scenario
    scenario = load_scenario('standard_negotiation')
    
    # 3. Run episode
    for turn in range(max_turns):
        action = agent.select_action(env.get_observation())
        obs, reward, done, info = env.step(action)
        if done:
            break
    
    # 4. Verify outcome
    assert env.negotiation_outcome is not None
    assert len(env.conversation_history) > 0
```

---

## Performance Tests (tests/performance/)

Benchmarks for scalability:

| Benchmark | Target | Measurement |
|-----------|--------|-------------|
| **Episode throughput** | > 100 eps/sec | Episodes per second |
| **Memory per episode** | < 500 MB | Peak memory usage |
| **Inference latency** | < 50 ms | Action selection time |
| **Reward computation** | < 10 ms | Score calculation time |

---

## Running Tests

### Run All Tests
```bash
pytest tests/ -v
```

### Run Specific Test File
```bash
pytest tests/v3/test_02_reward_integrity.py -v
```

### Run with Coverage
```bash
pytest tests/ --cov=deal_room --cov-report=html
```

### Run Performance Tests
```bash
pytest tests/performance/ --benchmark-json=results.json
```

---

## Continuous Integration

Tests run on every pull request:

1. **Unit tests** - Must pass on all platforms
2. **Integration tests** - Must pass before merge
3. **E2E tests** - Run nightly on main branch
4. **Performance tests** - Baseline comparison on PR

---

## Test Fixtures (conftest.py)

Shared fixtures for consistent test setup:

```python
@pytest.fixture
def mock_stakeholders():
    """Provide mock stakeholder set"""
    return [
        Stakeholder(archetype=Archetype.LEGAL),
        Stakeholder(archetype=Archetype.FINANCE),
    ]

@pytest.fixture
def sample_scenario():
    """Provide standard test scenario"""
    return ScenarioConfig(
        stakeholders=[...],
        constraints={...},
        success_criteria=...
    )
```

---

## Mock Strategy

| Component | Mock Approach |
|-----------|---------------|
| LLM Client | Mock responses with controlled text |
| Environment | Seeded random for reproducibility |
| Stakeholders | Pre-programmed response patterns |
| Committee | Deterministic deliberation results |

---

## Test Data Management

- **Fixtures**: Use `conftest.py` for shared data
- **Factories**: Use factory functions for test objects
- **Seeding**: Always seed random for reproducibility
- **Cleanup**: Use `yield` fixtures for resource cleanup

---

## Debugging Failed Tests

1. **Check seed**: Ensure random is seeded
2. **Inspect state**: Use `pytest.set_trace()` or `breakpoint()`
3. **Isolate**: Run single test to rule out interaction effects
4. **Check mocks**: Verify mock objects behave as expected

---

## Coverage Requirements

| Component | Minimum Coverage |
|-----------|-------------------|
| Environment | 90% |
| Committee | 85% |
| Rewards | 90% |
| Training | 80% |
| Server | 75% |
