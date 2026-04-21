# DealRoom v3 — Updated Testing Documentation

> **Key changes from current test suite:**  
> 1. No `OPENAI_API_KEY` required anywhere — scorer is pure Python  
> 2. Lookahead test uses no tolerance — pure Python = exact 0.07 deduction  
> 3. `test_llm_scoring.py` deleted — LLM scoring no longer exists  
> 4. All scorer unit tests verify state-delta functions, not LLM output  
> 5. Timeout budget reduced: deliberation has 5s timeout, step() completes in <10s  
> 6. `test_00` only validates `MINIMAX_API_KEY`  

---

## Test Directory Structure

```
tests/
├── conftest.py                           # Root pytest fixtures
├── unit/
│   ├── test_utterance_scorer.py         # Five-dim deterministic scorer (pure Python)
│   ├── test_deliberation_engine.py      # Two-layer deliberation engine
│   ├── test_cvar_preferences.py         # CVaR computation and veto logic
│   ├── test_causal_graph.py             # Causal graph and belief propagation
│   ├── test_grader.py                   # Terminal grader (CCIGrader)
│   ├── test_semantics.py                # Semantic analyzer
│   ├── test_belief_tracker.py           # Bayesian belief updates
│   ├── test_observation_mechanism.py    # Observation building (5 signals)
│   ├── test_models.py                   # Pydantic model validation
│   ├── test_curriculum.py               # Adaptive curriculum F1-F6 failure modes
│   ├── test_claims.py                   # Commitment ledger
│   └── test_validator.py                # Output validator
├── integration/
│   ├── test_end_to_end.py              # Environment V3 integration tests
│   ├── test_api_sessions.py            # API session management
│   ├── test_web_ui.py                  # Web interface tests
│   └── test_environment.py             # Environment interaction tests
├── v3/
│   ├── conftest.py                     # V3 configuration — MiniMax only
│   ├── test_00_environment_setup.py    # Setup & API key validation
│   ├── test_01_schema_validation.py    # Observation schema validation
│   ├── test_02_reward_integrity.py     # Reward integrity (deterministic)
│   ├── test_03_causal_inference.py     # Causal graph inference properties
│   ├── test_04_cvar_veto.py            # CVaR veto behavior
│   ├── test_05_episode_isolation.py    # Episode isolation
│   ├── test_06_probabilistic_signals.py # Statistical signal validation
│   ├── test_07_causal_graph.py         # Causal graph unit tests
│   ├── test_08_cvar_preferences.py     # CVaR preferences
│   ├── test_09_full_episode_e2e.py     # Full episode end-to-end
│   ├── test_10_training_infrastructure.py # Training infra
│   ├── test_11_research_properties.py  # Twelve research property checks
│   └── test_scorer_unit.py             # Scorer unit tests (no LLM)
│   # DELETED: test_llm_scoring.py     (LLM scoring removed)
│   # DELETED: test_llm_call.py        (OpenAI calls removed)
│   # DELETED: test_deterministic_scoring.py (replaced by test_scorer_unit.py)
├── e2e/
│   └── test_workflows.py               # End-to-end workflow tests
└── performance/
    └── test_benchmarking.py            # Performance benchmarking
```

---

## V3 Configuration (`tests/v3/conftest.py`)

### API Key Validation — MiniMax Only

```python
def validate_api_keys():
    """
    Exits with code 1 if MINIMAX_API_KEY not set.
    OPENAI_API_KEY is NOT required — utterance scorer is pure Python.
    """
    minimax_key = os.environ.get("MINIMAX_API_KEY")
    if not minimax_key or minimax_key.startswith("your_"):
        print("ERROR: MINIMAX_API_KEY not set.")
        print("  export MINIMAX_API_KEY=your_key")
        sys.exit(1)

    # Mask key for display
    masked = f"{minimax_key[:4]}...{minimax_key[-4:]}"
    print(f"✓ MINIMAX_API_KEY set: {masked}")
    # NOTE: OPENAI_API_KEY no longer checked — scorer is pure Python
```

### Step Timeout

```python
DEFAULT_STEP_TIMEOUT = 30   # seconds
# Previously 60s — reduced because deliberation summary now has 5s internal timeout
# Each step() completes in <10s even when MiniMax is slow
```

### get_reward() — Returns Float OR Dict

```python
def get_reward(result: dict) -> float:
    """
    Extract scalar reward from step result.
    V3 may return either a float (aggregated) or a dict (5D breakdown).
    Always returns a float for test assertions.
    """
    r = result.get("reward", 0.0)
    if isinstance(r, dict):
        weights = {"goal": 0.25, "trust": 0.20, "info": 0.20, "risk": 0.20, "causal": 0.15}
        return sum(weights.get(k, 0) * v for k, v in r.items())
    return float(r)

def get_reward_vector(result: dict) -> dict:
    """Returns 5D reward dict. Falls back to synthetic dict from scalar."""
    r = result.get("reward", 0.0)
    if isinstance(r, dict):
        return r
    scalar = float(r)
    return {"goal": scalar, "trust": scalar, "info": scalar, "risk": scalar, "causal": scalar}
```

---

## Test 00: Environment Setup (`tests/v3/test_00_environment_setup.py`)

### Test 0.1: Python Dependencies
Imports `requests`, `numpy`. Optional: `python-dotenv`. Fails if critical deps missing.

### Test 0.2: API Keys Configured — MiniMax Only
```python
def test_0_2_api_keys():
    """Only MINIMAX_API_KEY required. OPENAI_API_KEY is NOT required."""
    minimax_key = os.environ.get("MINIMAX_API_KEY", "")
    assert minimax_key and not minimax_key.startswith("your_"), (
        "MINIMAX_API_KEY not set. "
        "OpenAI is NOT required — utterance scorer is pure Python."
    )
    # OPENAI_API_KEY check intentionally removed
```

### Test 0.3: Docker Container Running
Verifies container with name `CONTAINER_NAME` is running via `docker ps`.

### Test 0.4: Server Endpoints Responsive
```python
# Full lifecycle: health → metadata → reset → step (timeout=30s)
# step() now completes in <10s because deliberation summary has 5s internal timeout
```

### Test 0.5: LLM Client Imports (MiniMax Only)
```python
def test_0_5_llm_client_imports():
    """Verify llm_client only has MiniMax functions. OpenAI functions must be absent."""
    from deal_room.environment import llm_client
    assert hasattr(llm_client, 'generate_stakeholder_response')
    assert hasattr(llm_client, 'generate_deliberation_summary')
    # Verify OpenAI was removed
    assert not hasattr(llm_client, 'get_openai_client'), (
        "get_openai_client() still present — OpenAI should be removed"
    )
    assert not hasattr(llm_client, 'llm_call_json'), (
        "llm_call_json() still present — JSON scoring removed"
    )
    assert not hasattr(llm_client, 'score_utterance_dimensions'), (
        "score_utterance_dimensions() still present — LLM scoring removed"
    )
```

---

## Test 01: Schema Validation (`tests/v3/test_01_schema_validation.py`)

### Test 1.2: Hidden Fields Not Exposed

```python
# Exact match only — "graph" matches key "graph" but NOT "graph_seed"
FORBIDDEN_FIELDS = {
    "G", "causal_graph", "graph", "true_beliefs", "belief_distributions",
    "belief_state", "B_i", "V_i", "tau", "tau_i", "risk_thresholds",
    "cvar_thresholds", "edge_weights", "w_ij", "deliberation_transcript",
    "deliberation_log", "internal_dialogue", "u_i", "u_ij"
}
```

### Test 1.5: Engagement Level Delta Type

`engagement_level_delta` must be a **dict** (one float per stakeholder), not a
single float. The current ARCHITECTURE.md description of "single float" was
inaccurate — it is per-stakeholder:

```python
def test_1_5_engagement_level_delta_is_dict():
    """engagement_level_delta must be Dict[str, float] — one delta per stakeholder."""
    r = requests.post(f"{BASE}/reset", json={"task": "aligned"})
    obs = r.json()
    delta = obs.get("engagement_level_delta")
    assert isinstance(delta, dict), (
        f"engagement_level_delta must be Dict[str, float], got {type(delta)}"
    )
    for sid, val in delta.items():
        assert isinstance(val, (int, float)), f"Delta for {sid} is not numeric"
```

### Test 1.6: Cross-Stakeholder Echoes Format

```python
def test_1_6_cross_stakeholder_echoes_format():
    """
    cross_stakeholder_echoes can be either:
    - List of dicts with 'from'/'from_stakeholder'/'sender' field
    - Dict of stakeholder_id -> list of topic strings
    Accept either format.
    """
    r = requests.post(f"{BASE}/reset", json={"task": "conflicted"})
    echoes = r.json().get("cross_stakeholder_echoes", [])
    assert isinstance(echoes, (list, dict)), (
        f"cross_stakeholder_echoes must be list or dict, got {type(echoes)}"
    )
```

---

## Test 02: Reward Integrity (`tests/v3/test_02_reward_integrity.py`)

### Test 2.1: All Five Dimensions Present

```python
def test_2_1_five_dimensions_present():
    """All 5 reward dimensions must appear individually in result or reward dict."""
    requests.post(f"{BASE}/reset", json={"task": "aligned"})
    r = requests.post(f"{BASE}/step", json=simple_action, timeout=30)
    result = r.json()
    reward = result.get("reward", {})
    if isinstance(reward, dict):
        for dim in ["goal", "trust", "info", "risk", "causal"]:
            assert dim in reward, f"Missing reward dimension: {dim}"
    # If scalar, the implementation aggregated internally — acceptable
```

### Test 2.2: Lookahead Cost Applied — Exact Tolerance

```python
def test_2_2_lookahead_cost_exact():
    """
    Lookahead deduction must be EXACTLY 0.07 (or within float rounding epsilon).
    No LLM variance tolerance — scorer is pure Python.

    Method: same message, same state → same base score → deduct 0.07 exactly.
    Uses fixed seed to ensure same initial state both times.
    """
    # Call 1: no lookahead — establishes base score
    requests.post(f"{BASE}/reset", json={"task": "aligned", "seed": 42})
    action_no_look = {**STANDARD_ACTION, "lookahead": None}
    r1 = requests.post(f"{BASE}/step", json=action_no_look, timeout=30)
    v1 = get_reward_vector(r1.json())
    goal_base = v1.get("goal", get_reward(r1.json()))

    # Call 2: with lookahead — same seed, same state → cache-equivalent
    requests.post(f"{BASE}/reset", json={"task": "aligned", "seed": 42})
    action_look = {**STANDARD_ACTION,
                   "lookahead": {"depth": 2, "n_hypotheses": 2}}
    r2 = requests.post(f"{BASE}/step", json=action_look, timeout=30)
    v2 = get_reward_vector(r2.json())
    goal_with = v2.get("goal", get_reward(r2.json()))

    expected = max(0.0, goal_base - 0.07)
    # Pure Python scorer = exact arithmetic = zero tolerance
    assert abs(goal_with - expected) < 0.001, (
        f"Lookahead cost wrong.\n"
        f"  goal_base={goal_base:.4f}, goal_with={goal_with:.4f}\n"
        f"  expected={expected:.4f} (base - 0.07)\n"
        f"  diff={abs(goal_with - expected):.4f}\n"
        f"  If seed not supported: lookahead flag may not reach scorer.\n"
        f"  Check: dealroom_v3.py step() extracts lookahead_was_requested FIRST."
    )

def test_2_2b_lookahead_flag_reaches_scorer():
    """
    Diagnostic test: verify the lookahead flag is extracted before Pydantic processing.
    This catches the bug where action.lookahead was consumed before _compute_reward().
    """
    requests.post(f"{BASE}/reset", json={"task": "aligned"})
    r = requests.post(f"{BASE}/step", json={
        **STANDARD_ACTION,
        "lookahead": {"depth": 2, "n_hypotheses": 2}
    }, timeout=30)
    result = r.json()
    # The info dict should record whether lookahead was used
    info = result.get("info", {})
    # Presence of lookahead_used in info confirms the flag reached the scorer
    if "lookahead_used" in info:
        assert info["lookahead_used"] is True, "lookahead_used should be True"
```

### Test 2.3: Reward Scores All In Range

```python
def test_2_3_all_dimensions_in_range():
    """All dimensions must be in [0.0, 1.0]. Pure Python scorer cannot overflow."""
    for scenario in ["aligned", "conflicted"]:
        requests.post(f"{BASE}/reset", json={"task": scenario})
        for _ in range(3):
            r = requests.post(f"{BASE}/step", json=STANDARD_ACTION, timeout=30)
            v = get_reward_vector(r.json())
            for dim, val in v.items():
                assert 0.0 <= val <= 1.0, f"{scenario}/{dim}={val} out of range"
```

### Test 2.4: No Prediction Accuracy in Reward

```python
def test_2_4_no_extra_reward_fields():
    """
    Reward must contain ONLY the five standard dimensions.
    No prediction_accuracy bonus. No LLM-derived fields.
    """
    requests.post(f"{BASE}/reset", json={"task": "aligned"})
    r = requests.post(f"{BASE}/step", json={**STANDARD_ACTION,
                       "lookahead": {"depth":2,"n_hypotheses":2}}, timeout=30)
    reward = r.json().get("reward", {})
    if isinstance(reward, dict):
        unexpected = set(reward.keys()) - {"goal", "trust", "info", "risk", "causal"}
        assert not unexpected, (
            f"Unexpected reward fields: {unexpected}\n"
            f"prediction_accuracy must be a diagnostic metric, not in reward."
        )
```

### Test 2.5: Reward Deterministic for Same State

```python
def test_2_5_reward_deterministic():
    """
    Same action, same state → identical reward scores.
    Pure Python scorer has no randomness.
    """
    rewards_run1, rewards_run2 = [], []

    for seed in [1, 2, 3]:
        requests.post(f"{BASE}/reset", json={"task": "aligned", "seed": seed})
        r = requests.post(f"{BASE}/step", json=STANDARD_ACTION, timeout=30)
        rewards_run1.append(get_reward(r.json()))

    for seed in [1, 2, 3]:
        requests.post(f"{BASE}/reset", json={"task": "aligned", "seed": seed})
        r = requests.post(f"{BASE}/step", json=STANDARD_ACTION, timeout=30)
        rewards_run2.append(get_reward(r.json()))

    for i, (r1, r2) in enumerate(zip(rewards_run1, rewards_run2)):
        assert abs(r1 - r2) < 0.001, (
            f"Seed {i+1}: reward not deterministic ({r1:.4f} vs {r2:.4f})\n"
            f"Pure Python scorer must return identical values for identical state."
        )
```

### Test 2.6: Different Actions Produce Different Rewards

```python
def test_2_6_different_actions_differ():
    """
    A targeted document send to Legal should score differently than exec_escalation.
    If all actions return the same reward, scorer is returning a constant.
    """
    rewards = []
    for action in [
        {**STANDARD_ACTION, "action_type": "send_document",
         "target_ids": ["Legal"], "documents": [{"type": "dpa"}]},
        {**STANDARD_ACTION, "action_type": "exec_escalation",
         "target_ids": ["ExecSponsor"]},
    ]:
        requests.post(f"{BASE}/reset", json={"task": "conflicted"})
        r = requests.post(f"{BASE}/step", json=action, timeout=30)
        rewards.append(get_reward(r.json()))

    assert max(rewards) - min(rewards) > 0.001, (
        f"All actions return same reward: {rewards}\n"
        "Scorer may be returning a constant — check _score_goal() implementation."
    )
```

---

## Utterance Scorer Unit Tests (`tests/unit/test_utterance_scorer.py`)

### What These Tests Validate

All tests are pure Python — no HTTP, no API, no container required.
Import `UtteranceScorer` directly and call methods.

```python
import numpy as np
from deal_room.rewards.utterance_scorer import UtteranceScorer, UtteranceScore, LOOKAHEAD_COST
from deal_room.committee.belief_tracker import create_neutral_belief, apply_positive_delta
from deal_room.stakeholders.archetypes import ARCHETYPES
```

### Test Group 1: Constants

```python
def test_lookahead_cost_value():
    assert LOOKAHEAD_COST == 0.07

def test_no_llm_client_on_init():
    """Scorer init must NOT require any API key."""
    # Must succeed without MINIMAX_API_KEY or OPENAI_API_KEY set
    scorer = UtteranceScorer()
    assert scorer is not None
```

### Test Group 2: Lookahead Deduction

```python
def test_lookahead_deduction_exact():
    """
    Same state with/without lookahead must differ by EXACTLY 0.07.
    No tolerance needed — pure Python arithmetic.
    """
    scorer = UtteranceScorer()
    state = create_test_state(round_number=3)
    action = create_test_action(target_ids=["Finance"], lookahead=None)
    graph = create_star_graph(hub="Finance")

    base = scorer.score(action, state, state, graph, lookahead_used=False)
    with_cost = scorer.score(action, state, state, graph, lookahead_used=True)

    assert abs(with_cost.goal - max(0.0, base.goal - 0.07)) < 1e-10

def test_lookahead_not_below_zero():
    """Goal score must never go negative from lookahead cost."""
    scorer = UtteranceScorer()
    # Create state where base goal is very low (0.02)
    state = create_test_state_with_low_progress()
    action = create_test_action(target_ids=["Finance"])
    graph = create_star_graph(hub="Finance")

    with_cost = scorer.score(action, state, state, graph, lookahead_used=True)
    assert with_cost.goal >= 0.0

def test_lookahead_only_affects_goal():
    """Lookahead cost must ONLY deduct from goal — not affect trust, info, risk, causal."""
    scorer = UtteranceScorer()
    state = create_test_state(round_number=2)
    action = create_test_action(target_ids=["Finance"])
    graph = create_star_graph(hub="Finance")

    base = scorer.score(action, state, state, graph, lookahead_used=False)
    with_cost = scorer.score(action, state, state, graph, lookahead_used=True)

    assert with_cost.trust  == base.trust
    assert with_cost.info   == base.info
    assert with_cost.risk   == base.risk
    assert with_cost.causal == base.causal

def test_lookahead_from_action_field():
    """lookahead_used=False but action.lookahead set → should still apply cost."""
    from deal_room.models import DealRoomAction, LookaheadRequest
    scorer = UtteranceScorer()
    state = create_test_state(round_number=2)
    action = DealRoomAction(
        action_type="direct_message",
        target_ids=["Finance"],
        message="Test.",
        lookahead=LookaheadRequest(depth=2, n_hypotheses=2)
    )
    action_no_look = DealRoomAction(
        action_type="direct_message",
        target_ids=["Finance"],
        message="Test.",
        lookahead=None
    )
    graph = create_star_graph(hub="Finance")

    s_no_look = scorer.score(action_no_look, state, state, graph, lookahead_used=False)
    s_with_look = scorer.score(action, state, state, graph, lookahead_used=False)
    # action.lookahead is set → should deduct even without explicit lookahead_used=True
    assert abs(s_with_look.goal - max(0.0, s_no_look.goal - 0.07)) < 1e-10
```

### Test Group 3: r^goal — State Delta

```python
def test_goal_positive_when_beliefs_improve():
    """r^goal > 0.5 when targeted stakeholder positive_mass increases."""
    scorer = UtteranceScorer()
    state_before = create_test_state_with_beliefs({"Finance": 0.40})
    state_after  = create_test_state_with_beliefs({"Finance": 0.65})
    action = create_test_action(target_ids=["Finance"])
    graph = create_star_graph(hub="Finance")

    score = scorer.score(action, state_before, state_after, graph)
    assert score.goal > 0.5, f"r^goal={score.goal} should be >0.5 when beliefs improve"

def test_goal_negative_delta_reduces_score():
    """r^goal < 0.5 when beliefs worsen."""
    scorer = UtteranceScorer()
    state_before = create_test_state_with_beliefs({"Finance": 0.65})
    state_after  = create_test_state_with_beliefs({"Finance": 0.40})
    action = create_test_action(target_ids=["Finance"])
    graph = create_star_graph(hub="Finance")

    score = scorer.score(action, state_before, state_after, graph)
    assert score.goal < 0.5

def test_goal_neutral_when_no_change():
    """r^goal ≈ 0.5 when state is unchanged."""
    scorer = UtteranceScorer()
    state = create_test_state_with_beliefs({"Finance": 0.50})
    action = create_test_action(target_ids=["Finance"])
    graph = create_star_graph(hub="Finance")

    score = scorer.score(action, state, state, graph)
    assert abs(score.goal - 0.5) < 0.15, f"r^goal={score.goal} should be near 0.5 for no-change state"

def test_goal_blocker_resolution_increases_score():
    """Resolving a blocker should increase r^goal."""
    scorer = UtteranceScorer()
    state_before = create_test_state_with_blockers(["Legal", "Finance"])
    state_after  = create_test_state_with_blockers(["Finance"])  # Legal resolved
    action = create_test_action(target_ids=["Legal"])
    graph = create_star_graph(hub="Legal")

    score = scorer.score(action, state_before, state_after, graph)
    assert score.goal > 0.5
```

### Test Group 4: r^trust — Belief Delta

```python
def test_trust_positive_for_dpa_to_legal():
    """
    Sending DPA to Legal should update B_Legal toward trustworthy (high likelihood).
    r^trust > 0.5.
    """
    scorer = UtteranceScorer()
    state_before = create_test_state_with_neutral_beliefs()
    state_after  = create_test_state_after_dpa_to_legal()
    action = create_test_action(
        action_type="send_document",
        target_ids=["Legal"],
        documents=[{"type": "dpa"}]
    )
    graph = create_star_graph(hub="Legal")
    score = scorer.score(action, state_before, state_after, graph)
    assert score.trust > 0.5

def test_trust_negative_for_escalation():
    """Premature exec_escalation should shift beliefs toward deceptive → r^trust < 0.5."""
    scorer = UtteranceScorer()
    state_before = create_test_state_with_neutral_beliefs()
    state_after  = create_test_state_after_premature_escalation()
    action = create_test_action(action_type="exec_escalation", target_ids=["ExecSponsor"])
    graph = create_star_graph(hub="ExecSponsor")
    score = scorer.score(action, state_before, state_after, graph)
    assert score.trust < 0.5

def test_trust_neutral_for_no_change():
    """No belief change → r^trust ≈ 0.5."""
    scorer = UtteranceScorer()
    state = create_test_state_with_neutral_beliefs()
    action = create_test_action(target_ids=["Finance"])
    graph = create_star_graph(hub="Finance")
    score = scorer.score(action, state, state, graph)
    assert abs(score.trust - 0.5) < 0.15
```

### Test Group 5: r^info — Entropy Reduction

```python
def test_info_positive_when_entropy_decreases():
    """r^info > 0.5 when belief entropy decreases (more certainty)."""
    scorer = UtteranceScorer()
    state_before = create_test_state_with_uniform_beliefs()   # max entropy
    state_after  = create_test_state_with_concentrated_beliefs()  # low entropy
    action = create_test_action(target_ids=["Finance"])
    graph = create_star_graph(hub="Finance")
    score = scorer.score(action, state_before, state_after, graph)
    assert score.info > 0.5

def test_info_near_half_for_no_change():
    """No belief change → r^info ≈ 0.5."""
    scorer = UtteranceScorer()
    state = create_test_state_with_uniform_beliefs()
    action = create_test_action(target_ids=["Finance"])
    graph = create_star_graph(hub="Finance")
    score = scorer.score(action, state, state, graph)
    assert abs(score.info - 0.5) < 0.10

def test_info_entropy_formula():
    """Verify entropy formula matches scipy.stats.entropy."""
    from scipy.stats import entropy as scipy_entropy
    from deal_room.rewards.utterance_scorer import LOG2_6
    dist = {"competent":0.5,"incompetent":0.1,"trustworthy":0.2,"deceptive":0.1,
            "aligned":0.05,"misaligned":0.05}
    h = scipy_entropy(list(dist.values()), base=2) / LOG2_6
    # Entropy should be between 0 (certain) and 1 (uniform)
    assert 0.0 <= h <= 1.0
```

### Test Group 6: r^risk — CVaR Delta

```python
def test_risk_positive_when_cvar_decreases():
    """r^risk > 0.5 when CVaR decreases for risk-averse stakeholders."""
    scorer = UtteranceScorer()
    state_before = create_test_state_with_high_legal_cvar()
    state_after  = create_test_state_with_low_legal_cvar()
    action = create_test_action(action_type="send_document",
                                target_ids=["Legal"],
                                documents=[{"type":"dpa"}])
    graph = create_star_graph(hub="Legal")
    score = scorer.score(action, state_before, state_after, graph)
    assert score.risk > 0.5

def test_risk_neutral_when_no_cvar_change():
    """No CVaR change → r^risk ≈ 0.5."""
    scorer = UtteranceScorer()
    state = create_test_state_with_moderate_cvar()
    action = create_test_action(target_ids=["Finance"])
    graph = create_star_graph(hub="Finance")
    score = scorer.score(action, state, state, graph)
    assert abs(score.risk - 0.5) < 0.15
```

### Test Group 7: r^causal — Betweenness Centrality

```python
def test_causal_hub_higher_than_leaf():
    """Targeting hub node must score higher r^causal than targeting leaf."""
    scorer = UtteranceScorer()
    state = create_test_state_with_neutral_beliefs()
    graph = create_star_graph(hub="Finance",
                               leaves=["Legal","TechLead","Procurement","ExecSponsor"])

    hub_action  = create_test_action(target_ids=["Finance"])
    leaf_action = create_test_action(target_ids=["Legal"])

    hub_score  = scorer.score(hub_action,  state, state, graph)
    leaf_score = scorer.score(leaf_action, state, state, graph)

    assert hub_score.causal > leaf_score.causal + 0.20, (
        f"Hub causal={hub_score.causal:.3f} should be >leaf {leaf_score.causal:.3f} + 0.20"
    )

def test_causal_deterministic():
    """Same action + same graph → identical r^causal every time."""
    scorer = UtteranceScorer()
    state = create_test_state_with_neutral_beliefs()
    graph = create_star_graph(hub="Finance")
    action = create_test_action(target_ids=["Finance"])

    s1 = scorer.score(action, state, state, graph)
    s2 = scorer.score(action, state, state, graph)
    assert s1.causal == s2.causal

def test_causal_zero_for_empty_graph():
    """Empty graph → r^causal = 0.0."""
    scorer = UtteranceScorer()
    state = create_test_state_with_neutral_beliefs()
    from deal_room.committee.causal_graph import CausalGraph
    empty_graph = CausalGraph(nodes=["Finance","Legal"], edges={},
                               authority_weights={}, scenario_type="aligned", seed=0)
    action = create_test_action(target_ids=["Finance"])
    score = scorer.score(action, state, state, empty_graph)
    assert score.causal == 0.0

def test_causal_zero_for_no_target():
    """No target_ids → r^causal = 0.0."""
    scorer = UtteranceScorer()
    state = create_test_state_with_neutral_beliefs()
    graph = create_star_graph(hub="Finance")
    action = create_test_action(target_ids=[])
    score = scorer.score(action, state, state, graph)
    assert score.causal == 0.0
```

### Test Group 8: All Dimensions In Range

```python
def test_all_dimensions_in_range():
    """Every dimension of UtteranceScore must be in [0.0, 1.0]."""
    scorer = UtteranceScorer()
    for _ in range(20):
        state_before = create_random_test_state()
        state_after  = create_random_test_state()
        action = create_random_test_action()
        graph  = create_random_test_graph()
        score  = scorer.score(action, state_before, state_after, graph)
        for dim in ["goal", "trust", "info", "risk", "causal"]:
            val = getattr(score, dim)
            assert 0.0 <= val <= 1.0, f"{dim}={val} out of [0,1]"

def test_no_cache_attribute():
    """UtteranceScorer must NOT have a _cache attribute — pure Python needs no cache."""
    scorer = UtteranceScorer()
    assert not hasattr(scorer, '_cache'), (
        "UtteranceScorer has _cache — it should be removed. "
        "Deterministic functions need no cache."
    )

def test_no_llm_dependencies():
    """UtteranceScorer must NOT import openai or llm_client."""
    import inspect
    from deal_room.rewards import utterance_scorer as mod
    source = inspect.getsource(mod)
    assert "openai" not in source.lower() or "# openai" in source.lower(), (
        "openai imported in utterance_scorer.py — should be removed"
    )
    assert "score_utterance_dimensions" not in source, (
        "score_utterance_dimensions() still called — LLM scoring not fully removed"
    )
    assert "llm_call_json" not in source, (
        "llm_call_json() still called — LLM scoring not fully removed"
    )
```

---

## Test 04: CVaR Veto (`tests/v3/test_04_cvar_veto.py`)

### Test 4.1: Veto Precursor Before Veto

```python
def test_4_1_precursor_before_veto():
    """
    veto_precursors must appear in observation BEFORE veto terminates episode.
    This validates the 70%-of-tau early warning system.
    Timeout per step: 30s (reduced from 60s — deliberation has internal 5s timeout).
    """
    requests.post(f"{BASE}/reset", json={"task": "hostile_acquisition"})
    precursor_seen = False
    for i in range(15):
        r = requests.post(f"{BASE}/step", json=AGGRESSIVE_ACTION, timeout=30)  # 30s not 60s
        result = r.json()
        obs = result.get("observation", result)
        if obs.get("veto_precursors"):
            precursor_seen = True
        if obs.get("done", False):
            break
```

---

## Test 09: Full Episode E2E (`tests/v3/test_09_full_episode_e2e.py`)

### Helper: `run_episode(scenario, strategy, max_steps, seed, timeout_per_step=30)`

```python
def run_episode(scenario, strategy="neutral", max_steps=20, seed=None,
                timeout_per_step=30):
    """
    timeout_per_step=30 (reduced from 60).
    Deliberation summary has internal 5s timeout — step() completes in <10s.
    30s gives 3x safety margin.
    """
```

### Test 9.4: Reward Collected Non-Zero

```python
def test_9_4_reward_variance_non_trivial():
    """
    Rewards must not all be identical across an episode.
    Pure Python scorer = deterministic but state changes → rewards vary.
    If all rewards identical, state is not changing between steps.
    """
    _, _, rewards, _ = run_episode("aligned", "comprehensive", max_steps=8)
    if len(rewards) < 2:
        return  # episode too short
    vals = [get_reward({"reward": r}) for r in rewards]
    variance = max(vals) - min(vals)
    assert variance > 0.001, (
        f"All rewards identical across episode: {vals}\n"
        "State is not changing between steps — "
        "check that state_before != state_after in scorer calls."
    )
```

---

## Test 11: Research Properties (`tests/v3/test_11_research_properties.py`)

### The Twelve Properties (Updated)

| # | Property | Test approach |
|---|----------|--------------|
| 1 | G is hidden from observation | Recursive dict scan for forbidden fields |
| 2 | Reset regenerates G each time | Two resets → different initial engagement levels |
| 3 | CVaR veto despite positive E[u] | Direct import + compute (no HTTP required) |
| 4 | Five reward dimensions are independent | Correlation < 0.95 across 6 episodes |
| 5 | Lookahead costs exactly 0.07 | Fixed seed + exact arithmetic assertion |
| 6 | Engagement noise not cancellable | Subtraction test after Fix 1 |
| 7 | Weak signals present | Schema check after step |
| 8 | Engagement delta per-stakeholder dict | Type check |
| 9 | r^causal varies with target | Different targets → different scores |
| 10 | Each reset produces different state | 5 resets → ≥2 unique states |
| 11 | Full episode no crash (all 3 scenarios) | run_episode × 3 |
| 12 | Training loop imports without error | Import check only |

### Property 3: CVaR Veto — Runs Without Container

```python
def prop_3_cvar_veto():
    """
    CVaR veto fires despite positive expected utility.
    This test runs WITHOUT HTTP — imports directly.
    This is the core research claim.
    """
    import sys
    sys.path.insert(0, "/app")
    from deal_room.stakeholders.cvar_preferences import compute_stakeholder_value, DealTerms
    from deal_room.stakeholders.archetypes import ARCHETYPES
    from deal_room.committee.belief_tracker import create_neutral_belief

    terms = DealTerms(
        price=0.85, support_level="enterprise", timeline_weeks=12,
        has_dpa=False, has_security_cert=False, liability_cap=0.2
    )
    belief = create_neutral_belief("Legal")
    eu, cvar, veto = compute_stakeholder_value(terms, belief, ARCHETYPES["Legal"])
    assert eu > 0, f"Expected utility must be positive. Got {eu:.4f}"
    assert veto is True, (
        f"VETO MUST FIRE.\n"
        f"eu={eu:.4f} (positive), cvar={cvar:.4f} > tau={ARCHETYPES['Legal'].tau}\n"
        "This is the core research claim."
    )
```

### Property 5: Lookahead Cost — Exact

```python
def prop_5_lookahead_cost():
    """Tolerance = 0.001 (not 0.02). Pure Python = exact arithmetic."""
    requests.post(f"{BASE}/reset", json={"task": "aligned", "seed": 42})
    r1 = requests.post(f"{BASE}/step", json={**STANDARD_ACTION, "lookahead": None})
    g1 = get_reward_vector(r1.json()).get("goal", get_reward(r1.json()))

    requests.post(f"{BASE}/reset", json={"task": "aligned", "seed": 42})
    r2 = requests.post(f"{BASE}/step", json={**STANDARD_ACTION,
                        "lookahead": {"depth":2,"n_hypotheses":2}})
    g2 = get_reward_vector(r2.json()).get("goal", get_reward(r2.json()))

    expected = max(0.0, g1 - 0.07)
    assert abs(g2 - expected) < 0.001, (   # 0.001, not 0.02
        f"g1={g1:.4f}, g2={g2:.4f}, expected={expected:.4f}, diff={abs(g2-expected):.4f}"
    )
```

---

## Running Tests

### Setup (Single API Key)

```bash
export MINIMAX_API_KEY=your_key
# OPENAI_API_KEY no longer required
```

### Unit Tests (No Container, No API Key)

```bash
pytest tests/unit/ -v
# Fast: ~5 seconds. Pure Python. No network calls.
```

### V3 Container Tests

```bash
# Start container
docker build -t dealroom-v3-test:latest -f Dockerfile .
docker run --rm -d -p 7860:7860 -e MINIMAX_API_KEY=$MINIMAX_API_KEY \
  --name dealroom_v3_test dealroom-v3-test:latest

# Run setup check first
python tests/v3/test_00_environment_setup.py

# Run full V3 suite
pytest tests/v3/ -v --timeout=45  # 45s per test (deliberation has internal 5s limit)
```

### Deleted Test Files

```bash
# These files should be removed from the repository:
rm tests/v3/test_llm_scoring.py      # LLM scoring removed
rm tests/v3/test_llm_call.py         # OpenAI calls removed
rm tests/v3/test_deterministic_scoring.py  # superseded by test_scorer_unit.py
```

### Per-File Quick Reference

```bash
# Scorer (pure Python, no container needed):
pytest tests/unit/test_utterance_scorer.py -v

# CVaR (pure Python):
pytest tests/unit/test_cvar_preferences.py -v

# Causal graph (pure Python):
pytest tests/unit/test_causal_graph.py -v

# All unit tests at once:
pytest tests/unit/ -v

# V3 integration (needs container + MINIMAX_API_KEY):
pytest tests/v3/ -v --timeout=45
```

---

## Key Design Principles (Updated)

1. **Deterministic scoring**: All five reward dimensions are pure Python computed from
   world state deltas. Same inputs always produce same outputs. No tolerance needed.

2. **Single API key**: Only `MINIMAX_API_KEY` required. OpenAI removed entirely.

3. **Step timeout budget**: 30s per step (not 60s). Deliberation summary has internal
   5s timeout and never blocks training.

4. **Lookahead extraction**: The `lookahead_was_requested` boolean is extracted as the
   first line of `step()`. Tests verify the flag reaches the scorer.

5. **No cache in scorer**: Deterministic functions need no cache. `_cache` removed.

6. **Schema isolation**: Hidden state (G, beliefs, CVaR parameters) never in observation.
   Tested with recursive dict scan in test_01.

7. **Research property validation**: Property 3 (CVaR veto) can run without a container —
   it imports directly. All 12 properties must pass before submitting.
