# DealRoom v3 Architecture

## 1. System Overview

DealRoom v3 is a research-grade OpenEnv RL environment for enterprise B2B negotiation. The system simulates a multi-stakeholder buying committee (Legal, Finance, TechLead, Procurement, Operations, ExecSponsor) that evaluates vendor proposals through sequential interactions.

The environment implements:
- **Causal graph inference**: Committee members influence each other's perceptions through a sampled directed graph
- **CVaR-based veto mechanism**: Risk-averse stakeholders can block deals even when expected utility is positive
- **Belief propagation**: Actions update vendor perception across stakeholders via deliberation
- **Lookahead simulation**: Optional action draft preview for robustness testing
- **5-dimensional reward**: goal, trust, info, risk, causal dimensions

## 2. Execution Flow

### Entry Point: `server/app.py`

```
uvicorn.run("server.app:app", host="0.0.0.0", port=port)
```

FastAPI server with endpoints:
- `POST /reset` → `DealRoomSessionPool.reset()` → `DealRoomV3.reset()`
- `POST /step` → `DealRoomSessionPool.step()` → `DealRoomV3.step()`
- `GET /state` → `DealRoomSessionPool.state()`
- `GET /health` → returns status

### Environment Flow: `deal_room/environment/dealroom_v3.py`

**`DealRoomV3.reset(seed, task_id)`**:
1. Initialize RNG with seed
2. Sample CausalGraph via `sample_graph()`
3. Initialize BeliefDistribution for each stakeholder
4. Set initial noisy engagement levels
5. Return initial `DealRoomObservation`

**`DealRoomV3.step(action: DealRoomAction)`**:
1. Normalize action via `_normalize_action()`
2. Run lookahead if `action.lookahead` is set via `_run_lookahead()`
3. Apply action to offer state via `_apply_action_to_offer_state()`
4. Update deal stage via `_update_deal_stage()`
5. Update beliefs for all stakeholders via `bayesian_update()`
6. Run committee deliberation via `CommitteeDeliberationEngine.run()`
7. Update noisy engagement via `_update_noisy_engagement()`
8. Generate stakeholder responses via `_generate_stakeholder_responses()`
9. Compute reward via `_compute_reward()`
10. Evaluate committee risk via `_evaluate_committee_risk()`
11. Check for veto via `_check_for_veto()`
12. Return `(observation, reward, done, info)`

### Deliberation Flow: `deal_room/committee/deliberation_engine.py`

**`CommitteeDeliberationEngine.run()`**:
1. Propagate beliefs via `propagate_beliefs()` (n_steps iterations)
2. Generate LLM summary via `generate_deliberation_summary()` (if enabled)
3. Return `DeliberationResult(updated_beliefs, summary, propagation_deltas)`

### Causal Graph: `deal_room/committee/causal_graph.py`

**`sample_graph()`**:
- Creates directed graph with edges based on scenario parameters
- Authority nodes (level >= 4) always have outgoing edges
- Intra-cluster edges boosted, cross-cluster penalized
- Returns `CausalGraph(nodes, edges, authority_weights, scenario_type, seed)`

**`propagate_beliefs()`**:
- Multi-step belief propagation with damping (0.85^step)
- Each step computes weighted delta from influencers
- Applies `_apply_belief_delta()` to update belief distributions

### Reward Computation: `deal_room/rewards/utterance_scorer.py`

**`UtteranceScorer.score()`** computes 5 dimensions:
- **goal**: approval_delta * authority + blocker resolution + veto headroom
- **trust**: positive_mass_delta + trustworthy_mass_delta for targeted
- **info**: entropy reduction via `_score_info()`
- **risk**: CVaR improvement for risk-averse stakeholders
- **causal**: betweenness centrality of targeted node

Weighted sum with `REWARD_WEIGHTS`:
```python
REWARD_WEIGHTS = {
    "goal": 0.25,
    "trust": 0.20,
    "info": 0.20,
    "risk": 0.20,
    "causal": 0.15,
}
```

### CVaR Veto: `deal_room/environment/dealroom_v3.py`

**`_evaluate_committee_risk()`**:
- Samples 500 outcomes via `compute_outcome_distribution()`
- Computes CVaR at stakeholder alpha via `compute_cvar()`
- Returns `{"all_utilities", "cvar_losses", "thresholds"}`

**`_check_for_veto()`**:
- Veto fires if `cvar_loss > tau` AND precursor streak >= 2
- Returns `(bool, Optional[str])` (veto_triggered, veto_stakeholder)

### Lookahead: `deal_room/environment/lookahead.py`

**`LookaheadSimulator.simulate()`**:
1. Generate optimistic/pessimistic hypotheses
2. Simulate each via `_simulate_one_hypothesis()`
3. Return worst-case result (min predicted_goal_delta)
4. Cost = `LOOKAHEAD_COST = 0.07` deducted from goal

## 3. Components

### Core Environment
| Component | File | Responsibility |
|-----------|------|----------------|
| `DealRoomV3` | `deal_room/environment/dealroom_v3.py` | OpenEnv wrapper, reset/step, state management |
| `StateSnapshot` | `deal_room/environment/dealroom_v3.py` | Immutable state capture for reward computation |
| `ScenarioConfig` | `deal_room/environment/dealroom_v3.py` | Task configuration dataclass |
| `LookaheadSimulator` | `deal_room/environment/lookahead.py` | Mental state hypothesis simulation |

### Committee Dynamics
| Component | File | Responsibility |
|-----------|------|----------------|
| `CausalGraph` | `deal_room/committee/causal_graph.py` | Graph structure with nodes, edges, authority weights |
| `BeliefDistribution` | `deal_room/committee/causal_graph.py` | Vendor perception per stakeholder |
| `CommitteeDeliberationEngine` | `deal_room/committee/deliberation_engine.py` | Belief propagation + LLM summary generation |
| `bayesian_update()` | `deal_room/committee/belief_tracker.py` | Belief update given action |

### Rewards
| Component | File | Responsibility |
|-----------|------|----------------|
| `UtteranceScorer` | `deal_room/rewards/utterance_scorer.py` | 5-dimension world-state scoring |
| `UtteranceScore` | `deal_room/rewards/utterance_scorer.py` | Score dataclass with weighted_sum() |
| `compute_terminal_reward()` | `deal_room/rewards/pareto_efficiency.py` | Terminal reward based on outcome type |

### Stakeholders
| Component | File | Responsibility |
|-----------|------|----------------|
| `ARCHETYPE_PROFILES` | `deal_room/stakeholders/archetypes.py` | 6 stakeholder risk profiles |
| `StakeholderRiskProfile` | `deal_room/stakeholders/cvar_preferences.py` | CVaR parameters per stakeholder |
| `compute_outcome_distribution()` | `deal_room/stakeholders/cvar_preferences.py` | Monte Carlo outcome sampling |
| `compute_cvar()` | `deal_room/stakeholders/cvar_preferences.py` | CVaR computation at alpha |

### Server
| Component | File | Responsibility |
|-----------|------|----------------|
| `app` | `server/app.py` | FastAPI application, CORS, Gradio mounting |
| `DealRoomSessionPool` | `server/session_pool.py` | Session-scoped environment pool |
| `OutputValidator` | `server/validator.py` | Action normalization and validation |

### Training
| Component | File | Responsibility |
|-----------|------|----------------|
| `GRPOTrainer` | `deal_room/training/grpo_trainer.py` | GRPO training loop, self-play |
| `EpisodeTrajectory` | `deal_room/training/grpo_trainer.py` | Trajectory dataclass |
| `PolicyAdapter` | `deal_room/training/grpo_trainer.py` | Protocol for policy implementations |
| `RandomPolicyAdapter` | `deal_room/training/grpo_trainer.py` | Random baseline policy |
| `HeuristicPolicyAdapter` | `deal_room/training/grpo_trainer.py` | Rule-based heuristic policy |
| `AdaptiveCurriculumGenerator` | `deal_room/curriculum/adaptive_generator.py` | Scenario difficulty selection |

### External Integrations
| Component | File | Responsibility |
|-----------|------|----------------|
| `llm_call_text()` | `deal_room/environment/llm_client.py` | MiniMax API via curl, retry with backoff |
| `generate_stakeholder_response()` | `deal_room/environment/llm_client.py` | LLM text generation for responses |
| `generate_deliberation_summary()` | `deal_room/environment/llm_client.py` | LLM deliberation summary |

## 4. Data Flow

### Action Flow
```
Client → DealRoomAction → _normalize_http_action() → _normalize_action()
                                            ↓
                                    OutputValidator.validate()
                                            ↓
                              DealRoomSessionPool.step()
                                            ↓
                               DealRoomV3.step(action)
         ├── _normalize_action() → canonical targets
         ├── _run_lookahead() → LookaheadSimulator.simulate()
         ├── _apply_action_to_offer_state() → updates offer_state
         ├── _update_deal_stage() → evaluation/negotiation/final_review
         ├── bayesian_update() per stakeholder → beliefs
         ├── CommitteeDeliberationEngine.run() → propagate_beliefs()
         ├── _update_noisy_engagement() → adds observation noise
         ├── _generate_stakeholder_responses() → per-stakeholder text
         ├── _compute_reward() → UtteranceScorer.score()
         ├── _evaluate_committee_risk() → CVaR per stakeholder
         ├── _check_for_veto() → veto triggered?
         └── _build_observation() → DealRoomObservation
```

### Belief State
```
Initial: _get_initial_beliefs(task_id, stakeholder_id)
              ↓
    bayesian_update(action) → dampening (1.0 targeted, 0.3 non-targeted)
              ↓
    propagate_beliefs() → multi-step weighted averaging
              ↓
    deliberation_result.updated_beliefs → stored in self._beliefs
```

### Reward Flow
```
UtteranceScorer.score(state_before, state_after)
     ├── _score_goal() → authority-weighted approval delta + blocker delta + veto headroom
     ├── _score_trust() → positive_mass_delta + trustworthy_delta for target
     ├── _score_info() → entropy reduction / LOG2_6
     ├── _score_risk() → CVaR improvement ratio
     └── _score_causal() → betweenness centrality / max_possible
     
weighted_sum(REWARD_WEIGHTS) → scalar reward
```

## 5. APIs / Interfaces

### HTTP Endpoints (server/app.py)
| Endpoint | Method | Request | Response |
|----------|--------|---------|----------|
| `/health` | GET | - | `{"status": "ok", "service": "deal-room", "tasks": [...]}` |
| `/metadata` | GET | - | `{"name": "deal-room", "version": "1.0.0", "tasks": [...]}` |
| `/reset` | POST | `ResetRequest(task_id, seed, episode_id)` | `DealRoomObservation` |
| `/step` | POST | `DealRoomAction` | `{"observation", "reward", "done", "info"}` |
| `/state` | GET | `session_id` query param | `DealRoomState` |

### Data Models (models.py)
```python
DealRoomAction:
  - action_type: str  # direct_message, send_document, group_proposal, etc.
  - target: str       # "all" or stakeholder ID
  - target_ids: List[str]
  - message: str       # truncated to 1200 chars
  - documents: List[Dict[str, str]]
  - proposed_terms: Optional[Dict[str, Any]]
  - channel: str      # default "formal"
  - mode: str         # default "async_email"
  - lookahead: Optional[LookaheadRequest]

DealRoomObservation:
  - reward: Optional[float]
  - metadata: Dict[str, Any]
  - round_number: int
  - max_rounds: int
  - stakeholders: Dict[str, Dict[str, Any]]
  - stakeholder_messages: Dict[str, str]
  - engagement_level: Dict[str, float]
  - weak_signals: Dict[str, List[str]]
  - veto_precursors: Dict[str, str]
  - active_blockers: List[str]
  - deal_momentum: str  # stalling/progressing/fragile/critical
  - deal_stage: str     # evaluation/negotiation/final_review
  - done: bool

DealRoomState:
  - episode_id, step_count, task_id, round_number, max_rounds
  - stakeholders, stakeholder_private, hidden_constraints
  - offer_state, feasibility_state
  - active_blockers, deal_stage, deal_momentum
  - terminal_outcome, veto_stakeholder, deal_failed
```

### Session Pool (server/session_pool.py)
```python
DealRoomSessionPool:
  - reset(task_id, seed, session_id) → (session_id, obs, state)
  - step(session_id, action) → (obs, reward, done, info, state)
  - state(session_id) → DealRoomState
  - has_session(session_id) → bool
```

## 6. State Management

### Environment State (DealRoomV3)
```python
self._rng: np.random.Generator          # seeded RNG
self._scenario: ScenarioConfig           # task_id, max_rounds, seed
self._state: DealRoomState              # current state
self._graph: CausalGraph                # sampled committee graph
self._beliefs: Dict[str, BeliefDistribution]  # vendor perception per stakeholder
self._noisy_engagement: Dict[str, float]  # noisy engagement per stakeholder
self._engagement_history: Dict[str, List[float]]  # sliding window (5 entries)
self._utterance_scorer: UtteranceScorer
self._lookahead_simulator: LookaheadSimulator
self._step_count, self._round_number: int
self._veto_precursor_streaks: Dict[str, int]
```

### Session State (DealRoomSessionPool)
- One `DealRoomV3` instance per session
- Sessions expire after `ttl_seconds` (default 6 hours)
- Max 128 sessions, oldest pruned on overflow

## 7. External Integrations

### MiniMax LLM (deal_room/environment/llm_client.py)
- **API**: MiniMax M2.5 via Anthropic-compatible `/v1/messages` endpoint
- **Auth**: `MINIMAX_API_KEY` environment variable
- **Base URL**: `MINIMAX_BASE_URL` env var, default `https://api.minimax.chat/v1`
- **Functions**:
  - `generate_stakeholder_response(prompt, context)` → optional text
  - `generate_deliberation_summary(prompt, context, timeout)` → optional text
- **Retry**: Exponential backoff with jitter, max 3 auto-retries
- **Error handling**: Auto-recoverable errors retried, auth errors pause for input
- **Fallback**: Returns `None` if API key not set or call fails

### OpenEnv Compatibility (client.py)
- `DealRoomEnv` implements `EnvClient[DealRoomAction, DealRoomObservation, State]`
- Used for inference script compatibility

## 8. Constants

From `deal_room/environment/constants.py`:
```python
REWARD_WEIGHTS = {"goal": 0.25, "trust": 0.20, "info": 0.20, "risk": 0.20, "causal": 0.15}
TERMINAL_REWARDS = {"deal_closed": 1.0, "veto": -1.0, "max_rounds": 0.0, "stage_regression": -0.5, "impasse": -0.75}
DEFAULT_MAX_ROUNDS = 10
SUMMARY_TIMEOUT_SECONDS = 5.0
VETO_WARNING_THRESHOLD_RATIO = 0.70
LOOKAHEAD_COST = 0.07
```

## 9. Scenario Types

From `deal_room/environment/dealroom_v3.py`:
| Scenario | Initial Beliefs | Description |
|----------|-----------------|-------------|
| `aligned` | default cluster | Cooperative, low internal resistance |
| `conflicted` | cost/risk/impl clusters | Competing interests, 2-4 stakeholders |
| `hostile_acquisition` | hostile cluster | Compressed timeline, authority shift event |

From `server/scenarios.py`:
| Scenario | Max Rounds | Days to Deadline | Constraints |
|----------|------------|------------------|-------------|
| `aligned` | 8 | 45 | 1 from pool |
| `conflicted` | 10 | 32 | 1-2 from pool |
| `hostile_acquisition` | 10 | 22 | 2 + authority_shift event |
