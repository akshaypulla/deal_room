# DealRoom v3 — Enterprise Architecture Documentation

**Version:** 1.0.0  
**Basis:** Code-grounded audit. No speculation. No documentation leakage.  
**Source:** All statements traceable to functions, classes, or execution flow in source code.

---

## 1. System Overview

### Purpose
DealRoom v3 is a **multi-stakeholder negotiation environment** implemented as an OpenEnv-compatible Gym-style interface. The system simulates a vendor negotiating procurement approval across six committee stakeholders (Legal, Finance, TechLead, Procurement, Operations, ExecSponsor) with belief propagation, CVaR-based veto risk, and lookahead planning.

### Primary Runtime Interface
- **HTTP API** via FastAPI (`server/app.py`)
- **Reset endpoint:** `POST /reset` → initializes session, returns `DealRoomObservation`
- **Step endpoint:** `POST /step` → accepts `DealRoomAction`, returns `(observation, reward, done, info)`
- **State endpoint:** `GET /state` → returns current `DealRoomState`
- **Web interface:** Gradio UI mounted at `/__gradio_ui__/`, accessible via `/web`

### Supporting Interface
- **Direct Python environment** (`deal_room.environment.dealroom_v3.DealRoomV3`) for training/integration without HTTP

---

## 2. Entry Points and Execution Flow

### API Entry Point

```
POST /reset
→ server/app.py:reset()
  → DealRoomSessionPool.reset()
    → dealroom_v3.py:DealRoomV3.reset()
```

```
POST /step
→ server/app.py:step()
  → _normalize_http_action()
  → DealRoomSessionPool.step()
    → dealroom_v3.py:DealRoomV3.step()
```

### Core Execution Loop (DealRoomV3.step)

```
dealroom_v3.py:326-466
DealRoomV3.step(action: DealRoomAction)
├── _normalize_action()                        [line 332]
├── _run_lookahead() if lookahead requested    [line 334]
├── _apply_action_to_offer_state()             [line 349]
├── _update_deal_stage()                       [line 350]
├── bayesian_update() for each stakeholder     [line 354-362]
│   → dealroom_v3.py:354-362
│   → belief_tracker.py:bayesian_update()       [line 106-134]
├── CommitteeDeliberationEngine.run()           [line 365-375]
│   → deliberation_engine.py:46-78
│   → causal_graph.py:propagate_beliefs()       [line 202-233]
├── _update_noisy_engagement()                 [line 386]
├── _generate_stakeholder_responses()          [line 388]
├── _compute_reward()                           [line 390-394]
│   → utterance_scorer.py:UtteranceScorer.score() [line 102-145]
├── _evaluate_committee_risk()                 [line 396]
│   → cvar_preferences.py:compute_outcome_distribution() [line 24-70]
│   → cvar_preferences.py:compute_cvar()       [line 73-93]
├── _compute_veto_precursors()                 [line 397]
├── _check_for_veto()                           [line 399]
│   → veto if cvar_loss > tau AND precursor_streak >= 2
├── compute_terminal_reward() if done          [line 406-416]
│   → pareto_efficiency.py:compute_terminal_reward() [line 39-65]
└── _build_observation()                        [line 456-631]
    → DealRoomObservation (returned to caller)
```

### Training Entry Point

```
grpo_trainer.py:GRPOTrainer.run_self_play_episode()
├── env.reset()
├── adapter.act(observation) → DealRoomAction
├── env.step(action) → (obs, reward, done, info)
├── trajectory.observations.append()
├── trajectory.actions.append()
└── returns EpisodeTrajectory
```

---

## 3. Component Inventory

| Component | File Path | Type | Responsibility |
|-----------|-----------|------|----------------|
| FastAPI Server | `server/app.py` | module | HTTP routing, session cookie management, action normalization |
| Session Pool | `server/session_pool.py` | class `DealRoomSessionPool` | Per-session environment instance management, session TTL pruning |
| Environment | `deal_room/environment/dealroom_v3.py` | class `DealRoomV3` | Core reset/step logic, belief state, offer state, veto evaluation |
| Action Models | `models.py` | pydantic classes `DealRoomAction`, `DealRoomObservation`, `DealRoomState` | Action/observation schema, validation |
| Belief Update | `deal_room/committee/belief_tracker.py` | function `bayesian_update()` | Bayesian belief update given vendor action |
| Causal Graph | `deal_room/committee/causal_graph.py` | class `CausalGraph`, function `propagate_beliefs()` | Graph structure, belief propagation across stakeholder network |
| Deliberation | `deal_room/committee/deliberation_engine.py` | class `CommitteeDeliberationEngine` | Runs propagation, generates optional LLM summary |
| Utterance Scorer | `deal_room/rewards/utterance_scorer.py` | class `UtteranceScorer` | 5-dimension reward scoring (goal, trust, info, risk, causal) |
| CVaR Preferences | `deal_room/stakeholders/cvar_preferences.py` | functions `compute_cvar()`, `compute_outcome_distribution()` | Risk profile evaluation, CVaR computation |
| Stakeholder Archetypes | `deal_room/stakeholders/archetypes.py` | dict `ARCHETYPE_PROFILES`, function `get_archetype()` | Per-stakeholder risk parameters (alpha, tau, lambda_risk, veto_power) |
| Lookahead Simulator | `deal_room/environment/lookahead.py` | class `LookaheadSimulator` | Mental state hypothesis simulation, worst-case CVaR planning |
| Terminal Reward | `deal_room/rewards/pareto_efficiency.py` | function `compute_terminal_reward()` | Determines terminal outcome and reward at episode end |
| GRPO Trainer | `deal_room/training/grpo_trainer.py` | class `GRPOTrainer` | Self-play training loop, policy adapters, metrics computation |
| Adaptive Curriculum | `deal_room/curriculum/adaptive_generator.py` | class `AdaptiveCurriculumGenerator` | Scenario difficulty adaptation based on failure analysis |
| Output Validator | `server/validator.py` | class `OutputValidator` | Action normalization for HTTP API |

---

## 4. Data Flow

### Action → Reward Data Path

```
[Vendor Action]
    ↓
_normalize_action()          [dealroom_v3.py:733-762]
    ↓ (normalized action)
bayesian_update()            [belief_tracker.py:106-134]
    ↓ (updated beliefs per stakeholder)
CommitteeDeliberationEngine.run() [deliberation_engine.py:46-78]
    → propagate_beliefs()    [causal_graph.py:202-233]
    ↓ (propagated beliefs)
_evaluate_committee_risk()   [dealroom_v3.py:853-888]
    → compute_outcome_distribution() [cvar_preferences.py:24-70]
    → compute_cvar()         [cvar_preferences.py:73-93]
    ↓ (cvar_losses, all_utilities, thresholds)
_check_for_veto()            [dealroom_v3.py:897-909]
    → veto if (cvar_loss > tau AND precursor_streak >= 2)
    ↓
_utterance_scorer.score()    [utterance_scorer.py:102-145]
    ↓
UtteranceScore (goal, trust, info, risk, causal)
    ↓
weighted_sum(REWARD_WEIGHTS) → scalar reward ∈ [0,1]
    ↓
DealRoomObservation (returned with reward, info)
```

### Key Data Objects

**DealRoomAction** (`models.py:13-41`)
- `action_type`: str (e.g., "direct_message", "send_document", "concession", "exec_escalation", "walkaway_signal")
- `target_ids`: List[str] — targeted stakeholders
- `message`: str (truncated to 1200 chars)
- `documents`: List[Dict] — e.g., [{"name": "DPA", "content": "..."}]
- `proposed_terms`: Optional[Dict] — e.g., {"liability_cap": 1500000}
- `lookahead`: Optional[LookaheadRequest] — lookahead planning request

**DealRoomObservation** (`models.py:43-66`)
- `engagement_level`: Dict[str, float] — per-stakeholder noisy engagement
- `engagement_level_delta`: float — primary target delta
- `weak_signals`: Dict[str, List[str]] — e.g., {"Finance": ["declining_engagement"]}
- `veto_precursors`: Dict[str, str] — stakeholder → warning message
- `cross_stakeholder_echoes`: List[Dict] — e.g., [{"from": "Finance", "to": "Legal", "content": "cross_reference"}]
- `deal_stage`: str ("evaluation" / "negotiation" / "final_review")
- `deal_momentum`: str ("stalling" / "progressing" / "fragile" / "critical")
- `active_blockers`: List[str]
- `done`: bool
- `info`: Dict — contains reward_components, deliberation_summary, lookahead diagnostics

**DealRoomState** (`models.py:75-113`)
- `offer_state`: Dict — current terms (price, timeline_weeks, liability_cap, has_dpa, etc.)
- `active_blockers`: List[str]
- `deal_stage`: str
- `round_number`: int
- `terminal_outcome`: str
- `veto_stakeholder`: Optional[str]
- `stage_regressions`: int

---

## 5. State Management

### State Variables (DealRoomV3 instance)

| Variable | Type | Location | Description |
|----------|------|----------|-------------|
| `_rng` | `np.random.Generator` | `dealroom_v3.py:187` | Central RNG, seeded per episode |
| `_scenario` | `Optional[ScenarioConfig]` | `dealroom_v3.py:188` | Task ID and seed |
| `_state` | `Optional[DealRoomState]` | `dealroom_v3.py:189` | Main state object |
| `_graph` | `Optional[CausalGraph]` | `dealroom_v3.py:190` | Sampled graph for current episode |
| `_beliefs` | `Dict[str, BeliefDistribution]` | `dealroom_v3.py:191` | Per-stakeholder belief distributions |
| `_noisy_engagement` | `Dict[str, float]` | `dealroom_v3.py:192` | Per-stakeholder engagement with noise |
| `_engagement_history` | `Dict[str, List[float]]` | `dealroom_v3.py:193` | Sliding window (default 5 steps) |
| `_utterance_scorer` | `UtteranceScorer` | `dealroom_v3.py:194` | Reward computation engine |
| `_lookahead_simulator` | `Optional[LookaheadSimulator]` | `dealroom_v3.py:195` | Planning simulator |
| `_step_count` | int | `dealroom_v3.py:196` | Total steps |
| `_round_number` | int | `dealroom_v3.py:197` | Incremented each step |
| `_veto_precursor_streaks` | `Dict[str, int]` | `dealroom_v3.py:199` | Per-stakeholder consecutive precursor count |

### State Lifecycle

```
reset(seed, task_id)
  → _rng = np.random.default_rng(seed)
  → _graph = sample_graph()     [causal_graph.py:80-128]
  → _beliefs = {sid: BeliefDistribution(...)}
  → _state = DealRoomState(...)
  → _veto_precursor_streaks = {sid: 0 for sid in STANDARD_STAKEHOLDERS}
  → returns DealRoomObservation (is_reset=True)

step(action)
  → increments _step_count, _round_number
  → modifies _state.offer_state, _state.deal_stage, _state.active_blockers
  → updates _beliefs (bayesian_update + propagate_beliefs)
  → updates _noisy_engagement, _engagement_history
  → sets _state.terminal_outcome, _state.veto_stakeholder on done
  → returns (observation, reward, done, info)
```

---

## 6. Runtime Behavior

### 6.1 Determinism vs Stochasticity

**Deterministic components:**
- `DealRoomV3.reset()` with fixed seed → identical graph structure, initial beliefs
- `UtteranceScorer.score()` — purely state-based, no RNG
- `bayesian_update()` — purely mathematical, no RNG
- `propagate_beliefs()` — purely graph-based, no RNG
- `compute_cvar()` — purely mathematical, no RNG (uses pre-seeded RNG internally for sampling)
- `compute_terminal_reward()` — purely logical

**Stochastic components:**
- `sample_graph()` — uses RNG to decide edge existence and weights
- `_update_noisy_engagement()` — adds Gaussian noise `N(0, σ=0.03)`
- `_generate_cross_stakeholder_echoes()` — stochastic recall with probability `echo_recall_probability=0.70`
- `LookaheadSimulator._simulate_one_hypothesis()` — stochastic response scoring

**Partial observability mechanisms:**
- Engagement observation is noisy: `engagement_noise_sigma = 0.03`
- Weak signals are generated from noisy engagement, not true beliefs
- Cross-stakeholder echoes are stochastic (70% probability per non-targeted stakeholder)
- True belief distributions are hidden; only engagement levels are observable

### 6.2 Core Dynamics

**Belief update (bayesian_update)** — `belief_tracker.py:106-134`:
- Given action type and documents, selects likelihood from `ACTION_LIKELIHOODS` dict
- Damping: `1.0` if targeted, `0.3` if not targeted
- Updates distribution, recomputes entropy-based confidence

**Belief propagation (propagate_beliefs)** — `causal_graph.py:202-233`:
- N steps (default 3, 4 for hostile_acquisition)
- Per step: each node receives weighted delta from influencers
- Damping: `0.85 ** step` — reduces amplification in later steps
- Prevents runaway belief values via normalization

**Reward computation (UtteranceScorer.score)** — `utterance_scorer.py:102-145`:
- `goal`: weighted approval delta + blocker resolution score + veto headroom improvement
- `trust`: targeted stakeholder positive mass delta + trustworthy mass delta
- `info`: mean entropy reduction across all stakeholders
- `risk`: mean CVaR improvement for risk-averse stakeholders (lambda_risk > 0.30)
- `causal`: betweenness centrality of primary target node / max possible
- Weights: `{"goal": 0.25, "trust": 0.20, "info": 0.20, "risk": 0.20, "causal": 0.15}`
- If lookahead used: goal score reduced by `LOOKAHEAD_COST = 0.07`

**CVaR computation (compute_cvar)** — `cvar_preferences.py:73-93`:
- Takes alpha (tail cutoff percentile) and outcome samples
- Computes mean of tail losses below `(1 - alpha)` quantile
- Used as risk measure for veto evaluation

**Veto trigger conditions** — `dealroom_v3.py:897-909`:
1. `cvar_loss > tau` (stakeholder-specific threshold)
2. `precursor_streak >= 2` (consecutive rounds with precursor warning)
3. Stakeholder must have `veto_power = True` (Legal, Finance, ExecSponsor)

**Termination conditions** — `dealroom_v3.py:399-401`:
- Veto triggered: `done = True`, `terminal_outcome = f"veto_by_{veto_stakeholder}"`
- `max_rounds` reached (default 10): `done = True`, `terminal_outcome = "max_rounds_no_deal"`

---

## 7. Critical Execution Paths

### Path 1: Normal Step (no lookahead)

```
DealRoomV3.step()
→ _normalize_action()
→ bayesian_update()         ×6 stakeholders
→ CommitteeDeliberationEngine.run()
  → propagate_beliefs()
→ _update_noisy_engagement()
→ _generate_stakeholder_responses()
→ _compute_reward()
  → UtteranceScorer.score() → 5-dim weighted sum
→ _evaluate_committee_risk()
  → compute_outcome_distribution() ×6 ×500 samples
  → compute_cvar()          ×6
→ _compute_veto_precursors()
→ _check_for_veto()
→ _build_observation()
→ return (obs, reward, done, info)
```

### Path 2: Lookahead Step

```
DealRoomV3.step()
→ _normalize_action()
→ _run_lookahead()          [lookahead.py:45-76]
  → LookaheadSimulator.simulate()
  → generates optimistic/pessimistic hypotheses
  → simulates n_hypotheses × depth
  → returns SimulationResult (worst-case)
→ (normal path continues, but goal reduced by 0.07)
```

### Path 3: Terminal Step

```
DealRoomV3.step()
→ ... (same as normal path)
→ done = True
→ compute_terminal_reward()  [pareto_efficiency.py:39-65]
  → veto: reward = -1.0, outcome = "veto_by_{stakeholder}"
  → max_rounds + pareto: reward = 0.0, outcome = "max_rounds_pareto"
  → max_rounds no deal: reward = 0.0, outcome = "max_rounds_no_deal"
  → stage_regression: reward = -0.5 * min(3, regressions)
  → impasse: reward = -0.75, outcome = "impasse"
→ reward += terminal_reward
→ _build_observation()
```

---

## 8. Complexity and Risk Areas

### High Complexity

**CVaR computation** — `cvar_preferences.py:24-93`
- Called 6× per step (once per stakeholder)
- Each call: 500 Monte Carlo samples → sort → compute tail mean
- Scenario multiplier applied: aligned=0.12, conflicted=0.22, hostile_acquisition=0.42
- Confidence factor modifies loss: `1.0 - (0.25 * clip(positive_mass, 0, 1))`

**Lookahead simulation** — `lookahead.py:45-76`
- Generates 2 hypotheses (optimistic/pessimistic belief distributions)
- Simulates each with `depth` recursive depth
- Returns worst-case (min predicted goal delta) for robustness

**Causal graph sampling** — `causal_graph.py:80-128`
- Edge probability varies by scenario type and authority level
- Authority nodes (level ≥ 4) always have outgoing edges
- Intra-cluster boost, cross-cluster penalty applied stochastically

### Risk Areas

**Tight coupling:** `DealRoomV3._compute_reward()` calls `UtteranceScorer.score()` which internally calls `self._compute_cvar()` (which calls `compute_outcome_distribution()` with 500 samples). This creates a slow inner loop that cannot be parallelized or cached externally.

**Veto precursor streak counter** — `_veto_precursor_streaks` is a simple dict without atomic updates. In concurrent session scenarios, thread safety is handled at `DealRoomSessionPool` level (Lock), but the streak counter itself is not protected.

**Observation noise sigma** — `engagement_noise_sigma = 0.03` is hardcoded in `_init_obs_config()`. This value is not configurable via environment variables or constructor parameters.

**Graph identifiability** — The system relies on each sampled graph producing a unique behavioral signature. If two different graphs produce identical signatures, the causal reasoning capability degrades. This is validated by `test_7_8_graph_identifiability` (20 graphs, all pairwise distinguishable).

---

## 9. External Dependencies and Integrations

### Direct Imports (runtime)

| Library | Purpose | Location |
|---------|---------|----------|
| `numpy` | RNG, array operations, random sampling | Throughout |
| `fastapi` | HTTP server | `server/app.py` |
| `uvicorn` | ASGI server runner | `server/app.py:340,347` |
| `pydantic` | Data validation (DealRoomAction, DealRoomObservation, DealRoomState) | `models.py` |
| `gradio` | Web UI framework (optional) | `server/app.py:237-307` |

### Optional External Services

| Service | Purpose | Fallback |
|---------|---------|----------|
| `MINIMAX_API_KEY` | LLM deliberation summaries | Silently returns empty string on failure |
| `OPENAI_API_KEY` | Alternative LLM calls | Silently returns empty string on failure |

### Environment Variables

| Variable | Default | Effect |
|----------|---------|--------|
| `DEALROOM_BASE_URL` | `http://127.0.0.1:7860` | Test client target |
| `DEALROOM_ENV_PATH` | `/app/env` | Python path for container imports |
| `DEALROOM_CONTAINER_NAME` | `dealroom-v3-test` | Docker container identification |
| `ENABLE_WEB_INTERFACE` | `true` | Whether to mount Gradio UI |
| `PORT` | `7860` | Server port |
| `DEALROOM_ENABLE_LLM_SUMMARY` | `false` | Whether to render deliberation summaries |

---

## 10. Constraints and Constants

### Reward Weights (deal_room/environment/constants.py:3-9)

```python
REWARD_WEIGHTS = {
    "goal": 0.25,
    "trust": 0.20,
    "info": 0.20,
    "risk": 0.20,
    "causal": 0.15,
}
```

### Terminal Rewards (constants.py:11-17)

```python
TERMINAL_REWARDS = {
    "deal_closed": 1.0,
    "veto": -1.0,
    "max_rounds": 0.0,
    "stage_regression": -0.5,
    "impasse": -0.75,
}
```

### Other Constants (constants.py:19-21)

```python
DEFAULT_MAX_ROUNDS = 10
SUMMARY_TIMEOUT_SECONDS = 5.0
VETO_WARNING_THRESHOLD_RATIO = 0.70  # 70% of tau triggers precursor
```

### Observation Configuration (dealroom_v3.py:41-55)

```python
@dataclass
class ObservationConfig:
    engagement_noise_sigma: float = 0.03
    echo_recall_probability: float = 0.70
    weak_signal_hard_threshold: float = 0.12
    weak_signal_soft_lower: float = 0.08
    weak_signal_soft_probability: float = 0.70
    reference_injection_threshold: float = 0.10
    minimax_base_reference_target: float = 0.60
    engagement_history_window: int = 5
```

### Stakeholder Archetypes — Key Parameters

| Stakeholder | alpha | tau | lambda_risk | veto_power |
|-------------|-------|-----|-------------|------------|
| Legal | 0.95 | 0.10 | 0.70 | Yes |
| Finance | 0.90 | 0.15 | 0.50 | Yes |
| TechLead | 0.80 | 0.25 | 0.30 | No |
| Procurement | 0.85 | 0.20 | 0.45 | No |
| Operations | 0.80 | 0.30 | 0.35 | No |
| ExecSponsor | 0.70 | 0.40 | 0.25 | Yes |

### Causal Graph Scenario Parameters (causal_graph.py:20-45)

| Scenario | base_edge_prob | intra_cluster_boost | cross_cluster_penalty | authority_edge_prob |
|----------|----------------|---------------------|----------------------|---------------------|
| aligned | 0.30 | 0.40 | 0.20 | 0.85 |
| conflicted | 0.45 | 0.50 | 0.15 | 0.80 |
| hostile_acquisition | 0.60 | 0.25 | 0.05 | 0.65 |

### Lookahead Cost (utterance_scorer.py:34)

```python
LOOKAHEAD_COST = 0.07
```