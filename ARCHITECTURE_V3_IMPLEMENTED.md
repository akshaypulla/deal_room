# DealRoom v3 — Architecture

## What: Overview

DealRoom v3 is a multi-stakeholder enterprise deal negotiation environment designed for Reinforcement Learning (RL) research. It simulates a vendor negotiating with a committee of 6 stakeholders (Legal, Finance, TechLead, Procurement, Operations, ExecSponsor) across three scenario types: `aligned`, `conflicted`, and `hostile_acquisition`.

The system is built as an OpenEnv-compatible environment with a FastAPI HTTP wrapper, a causal graph inference mechanism for committee dynamics, CVaR-based veto detection, and a 5-dimensional world-state-based reward system.

## Why: Design Rationale

### OpenEnv Compatibility
The environment conforms to the OpenEnv API (`reset()` → observation, `step(action)` → observation, reward, done, info). This allows integration with existing RL training pipelines and evaluation frameworks without modification.

### Causal Graph Hidden from Agent
The causal graph `G` (committee influence relationships, edge weights, authority hierarchy) is generated at reset but **never exposed** in the observation. The agent must infer `G` indirectly through noisy engagement signals and cross-stakeholder echoes. This is a core research property: the agent operates under partial observability.

### CVaR-Based Veto
Traditional RL environments use expected utility (EU). DealRoom v3 implements Conditional Value at Risk (CVaR) veto — a stakeholder can kill a deal even when EU > 0 if the tail risk (worst-case outcomes) exceeds their personal threshold `tau`. This is the **core research claim**: CVaR veto fires despite positive expected utility.

### 5-Dimensional Reward
The single scalar reward is a weighted sum of 5 dimensions computed deterministically from world-state deltas (no LLM involved in scoring):
- **goal**: weighted approval delta + blocker resolution + veto headroom
- **trust**: positive mass delta on targeted stakeholder + trustworthy mass delta
- **info**: entropy reduction across belief distributions
- **risk**: CVaR improvement for risk-averse stakeholders
- **causal**: betweenness centrality of targeted node in `G`

### Lookahead Mechanism
The agent can request a lookahead simulation at cost 0.07. This generates pessimistic worst-case hypotheses about what would happen if the action were taken, allowing the agent to reason about robustness before committing.

---

## How: Module-by-Module Breakdown

### `deal_room/committee/causal_graph.py`

**What it does**: Defines the `CausalGraph` structure and `BeliefDistribution` for committee dynamics.

**Key components**:
- `VENDOR_TYPES = ["competent", "incompetent", "trustworthy", "deceptive", "aligned", "misaligned"]` — 6-type belief space
- `SCENARIO_PARAMS` — scenario-specific edge probability parameters (`aligned`, `conflicted`, `hostile_acquisition`)
- `FUNCTIONAL_CLUSTERS` — stakeholders grouped by function: cost (Finance, Procurement), risk (Legal, Compliance), implementation (TechLead, Operations), authority (ExecSponsor)
- `sample_graph()` — generates a random causal graph given a stakeholder set, authority hierarchy, scenario type, and RNG. Authority nodes (authority level ≥ 4) always have outgoing edges. Same-cluster edges get a boost; cross-cluster edges get a penalty.
- `BeliefDistribution` dataclass — holds a 6-type probability distribution, confidence score, and history. Methods: `positive_mass()`, `negative_mass()`, `to_natural_language()`
- `propagate_beliefs()` — iterative belief propagation through graph edges with damping (0.85^step). Non-targeted stakeholders receive belief updates from their influencers weighted by edge strength
- `apply_positive_delta()` — shifts probability mass from negative types to positive types
- `get_betweenness_centrality()` — measures how often a node lies on shortest paths between other nodes (BFS-based)
- `compute_behavioral_signature()` — generates a dict of engagement deltas for all non-targeted stakeholders when a specific stakeholder receives a belief delta

**Key invariant**: `ExecSponsor` always has ≥ 2 outgoing edges across all scenarios and seeds.

---

### `deal_room/committee/belief_tracker.py`

**What it does**: Implements Bayesian belief updates when a vendor takes an action.

**Key components**:
- `ACTION_LIKELIHOODS` — hardcoded likelihood tables mapping action types (e.g., `send_document(DPA)_proactive`) to probability ratios for each vendor type
- `_get_likelihood()` — matches an action to its likelihood table by substring matching on action type and document names
- `bayesian_update()` — performs Bayesian probability update: `new_dist[vendor_type] = prior_prob * dampened_likelihood`. Damping factor is 1.0 for targeted, 0.3 for non-targeted. Normalizes to sum=1. Computes entropy-based confidence.
- `compute_engagement_level()` — returns `positive_mass() - negative_mass()`

**Note**: This module is defined but appears to be **not used** in the main environment. The environment uses inline Bayesian update logic in `DealRoomV3._bayesian_update()` with its own hardcoded likelihoods.

---

### `deal_room/committee/deliberation_engine.py`

**What it does**: Orchestrates belief propagation after a vendor action and optionally generates a MiniMax LLM summary.

**Key components**:
- `CommitteeDeliberationEngine` class — holds a `CausalGraph` and number of deliberation steps
- `run()` — calls `propagate_beliefs()` then optionally calls `_generate_summary()`
- `_generate_summary()` — builds a prompt describing belief shifts for the targeted stakeholder and other deltas, calls MiniMax LLM to produce a 2-4 sentence deliberation dialogue
- `DeliberationResult` dataclass — `updated_beliefs`, `summary_dialogue`, `propagation_deltas`

The deliberation engine is called inside `DealRoomV3.step()` wrapped in a try/except — if it fails, it silently falls back to keeping current beliefs.

---

### `deal_room/environment/dealroom_v3.py`

**What it does**: The main OpenEnv environment. Implements `reset()` and `step()`.

**Key components**:
- `STANDARD_STAKEHOLDERS` and `STANDARD_HIERARCHY` — fixed 6-stakeholder committee with authority levels
- `INITIAL_BELIEFS` — scenario-specific initial belief distributions (e.g., `hostile_acquisition` starts with higher negative mass)
- `_get_initial_beliefs()` — returns the appropriate belief distribution based on scenario and stakeholder
- `REWARD_WEIGHTS = {"goal": 0.25, "trust": 0.20, "info": 0.20, "risk": 0.20, "causal": 0.15}`
- `OBS_CONFIG` — `ObservationConfig` with noise sigma (0.03), echo recall probability (0.70), weak signal thresholds, engagement history window (5)
- `StateSnapshot` — captures pre-action state for reward computation
- `DealRoomV3.reset()` — reseeds RNG, samples new causal graph, initializes beliefs, noisy engagement levels, engagement history, and full `DealRoomState`. Returns `DealRoomObservation`.
- `DealRoomV3.step()` — the core action processing:
  1. Increments round number
  2. Copies previous beliefs
  3. Performs Bayesian update for each stakeholder (targeted gets damping=1.0, non-targeted gets 0.3)
  4. Runs deliberation engine for belief propagation
  5. Updates noisy engagement with Gaussian noise (σ=0.03)
  6. Generates stakeholder responses (targeted get detailed responses, non-targeted get brief notes)
  7. Computes reward via `UtteranceScorer`
  8. Returns observation, reward, done flag, info dict
- `_bayesian_update()` — inline Bayesian update with hardcoded likelihoods for 5 action types
- `_generate_stakeholder_responses()` — generates response text based on belief state
- `_update_noisy_engagement()` — adds Gaussian noise to true engagement deltas, maintains sliding window history
- `_compute_reward()` — creates `StateSnapshot`, calls `UtteranceScorer.score()`, returns weighted sum
- `_build_observation()` — assembles `DealRoomObservation` with weak signals, veto precursors, engagement deltas, cross-stakeholder echoes
- `_generate_weak_signals()` — generates signal tags (`high_engagement`, `low_engagement`, `improving_engagement`, `declining_engagement`, `high_uncertainty`) based on noisy engagement level and delta
- `_compute_veto_precursors()` — checks if any stakeholder's CVaR loss exceeds 70% of their tau threshold
- `_generate_cross_stakeholder_echoes()` — for each non-targeted stakeholder, randomly (70% probability) generates a cross-reference echo from the targeted stakeholder

---

### `deal_room/environment/llm_client.py`

**What it does**: Thin wrapper around MiniMax M2.5 API via curl. Handles retries, rate limits, auth errors, and interactive pause prompts.

**Key components**:
- `LLMErrorType` enum — 14 error types (network timeout, DNS failure, rate limit, quota exceeded, auth invalid, etc.)
- `LLMError` dataclass — error type, message, API, status code, retry-after, raw exception, timestamp
- `classify_error()` — maps exceptions to `LLMError` types based on string matching and status codes
- `RetryPolicy` dataclass — `max_auto_retries=3`, `base_backoff_sec=1.0`, `backoff_factor=2.0`, `jitter_fraction=0.2`, `rate_limit_default_wait=60.0`, `rate_limit_max_wait=300.0`
- `llm_call_text()` — main API: JSON-encodes prompt, calls `curl` subprocess to `https://api.minimax.chat/v1/messages` with Bearer token. Parses `content[].text` from response. Auto-retries on auto-recoverable errors. Rate limits wait with countdown. Auth errors and quota exceeded require manual intervention.
- `generate_stakeholder_response()` — calls `llm_call_text` with `temperature=0.7`, `max_tokens=200`, `allow_skip=True`
- `generate_deliberation_summary()` — calls `llm_call_text` with `temperature=0.8`, `max_tokens=220`, `allow_skip=True`, `timeout=5.0`
- `validate_api_keys()` — raises `EnvironmentError` if `MINIMAX_API_KEY` not set
- `STATS` — global `LLMCallStats` tracking calls, successes, auto-retries, interventions, skips, and errors by type
- `_interactive_pause()` — prints colored error info to terminal and prompts user for choice: `c` (continue/retry), `w <N>` (wait N seconds), `r` (retry immediately), `s` (skip), `e` (exit)

**Critical note**: API key validation happens at `DealRoomV3.__init__()` time. If `MINIMAX_API_KEY` is not set, the environment **crashes on instantiation**, not on first LLM call.

---

### `deal_room/environment/lookahead.py`

**What it does**: Implements the lookahead simulation mechanism for robust action planning.

**Key components**:
- `LOOKAHEAD_COST = 0.07` — fixed cost deducted from goal score when lookahead is used
- `LookaheadSimulator` class — takes an RNG, simulates action outcomes across multiple hypotheses
- `simulate()` — generates optimistic and pessimistic belief distribution hypotheses for the targeted stakeholder, simulates each, returns worst-case result
- `_generate_hypotheses()` — creates two hypotheses: one shifting mass toward positive types (+0.15 competent, +0.10 trustworthy/aligned), one shifting toward negative types
- `_simulate_one_hypothesis()` — simulates response score, belief delta, CVaR impact, graph info gain for one hypothesis

**Note**: `LookaheadSimulator` in `lookahead.py` is separate from and not directly integrated into `DealRoomV3.step()`. The environment handles lookahead via the `action.lookahead` field passed to `step()`, but the actual simulation logic in `lookahead.py` appears to be defined but not called from the environment's step function.

---

### `deal_room/rewards/utterance_scorer.py`

**What it does**: World-state-based deterministic scoring. Computes 5 reward dimensions from state deltas without LLM calls.

**Key components**:
- `LOG2_6 = math.log(2, 6)` — maximum entropy for 6-type distribution
- `LOOKAHEAD_COST = 0.07` — re-exported here
- `UtteranceScore` dataclass — 5 float fields: goal, trust, info, risk, causal. Method: `weighted_sum(weights)` and `to_dict()`
- `UtteranceScorer.score()` — main entry point. Takes action, state_before, state_after, true_graph, lookahead_used flag. Calls 5 scoring submethods.
- `_score_goal()` — weighted approval delta (authority-weighted belief shift) + blocker resolution bonus - new blocker penalty + veto headroom improvement for lambda_risk > 0.40 stakeholders
- `_score_trust()` — for targeted stakeholders: `0.6 * positive_mass_delta + 0.4 * trustworthy_mass_delta`, clipped to [0, 1]
- `_score_info()` — mean entropy reduction across all stakeholders normalized by LOG2_6
- `_score_risk()` — mean CVaR improvement ratio for stakeholders with lambda_risk > 0.30
- `_score_causal()` — betweenness centrality of first targeted node divided by max possible
- `_compute_cvar()` — samples 500 outcomes from `compute_outcome_distribution()`, computes CVaR at alpha

**All dimensions are bounded [0, 1]. Scoring is deterministic (no randomness).**

---

### `deal_room/rewards/pareto_efficiency.py`

**What it does**: Computes terminal rewards at episode end.

**Key components**:
- `TERMINAL_REWARDS = {"deal_closed": 1.0, "veto": -1.0, "max_rounds": 0.0, "stage_regression": -0.5, "impasse": -0.75}`
- `check_pareto_optimality()` — checks if at least one stakeholder is not dominated (i.e., has no other stakeholder with ≥ utility and ≤ CVaR loss)
- `compute_terminal_reward()` — returns (reward, terminal_outcome_string) based on deal closed, veto triggered, max rounds reached, stage regressions
- `get_pareto_frontier_stakeholders()` — returns list of non-dominated stakeholder IDs

---

### `deal_room/stakeholders/archetypes.py`

**What it does**: Defines the 6 stakeholder risk profiles.

**Key components**:
- `ARCHETYPE_PROFILES` dict — keyed by stakeholder ID: Legal, Finance, TechLead, Procurement, Operations, ExecSponsor
- Each `StakeholderRiskProfile` has: `stakeholder_id`, `role`, `alpha` (tail quantile), `tau` (veto threshold), `lambda_risk` (risk aversion weight), `utility_weights` (5 domain-specific weights), `uncertainty_domains` (risk categories)

**Risk tolerance ordering (tau)**: Legal (0.10) < Finance (0.15) < TechLead (0.25) < Procurement (0.20) < Operations (0.30) < ExecSponsor (0.40)

**CVaR alpha ordering**: ExecSponsor (0.70) < TechLead (0.80) < Procurement (0.85) < Finance (0.90) < Legal (0.95) — Legal is most sensitive to tail risk

---

### `deal_room/stakeholders/cvar_preferences.py`

**What it does**: CVaR computation and deal evaluation.

**Key components**:
- `compute_outcome_distribution()` — Monte Carlo simulation. Base success probability depends on domain (0.80 for compliance, 0.70 for cost, 0.75 for implementation). Adjusts for `has_dpa`, `has_security_cert`, `liability_cap`. Generates `n_samples` outcomes in [0, 1].
- `compute_cvar()` — sorts outcomes, takes the worst `(1-alpha)*100%`, returns mean of tail losses
- `compute_expected_utility()` — mean of outcomes
- `evaluate_deal()` — returns (expected_utility, cvar_loss) tuple
- `check_veto_trigger()` — returns `cvar_loss > tau`
- `get_observable_signals()` — maps domain concerns to signal names and base probabilities
- `compute_deal_quality_score()` — `(1 - lambda_risk) * EU - lambda_risk * CVaR`, clipped to [0, 1]

---

### `deal_room/curriculum/adaptive_generator.py`

**What it does**: Adaptive curriculum generator for training.

**Key components**:
- `FAILURE_MODE_DESCRIPTIONS` — F1 through F6: CVaR veto, trust collapse, failed graph inference, timeout, reward hacking, authority shift blindness
- `CurriculumConfig` — `analysis_batch_size=10`, `easy_ratio=0.20`, `frontier_ratio=0.60`, `hard_ratio=0.20`, `max_graph_variation=0.3`
- `AdaptiveCurriculumGenerator` — maintains a `_scenario_pool` of 15 base configs (5 seeds × 3 scenarios), analyzes failure modes from trajectories, selects next scenario based on estimated agent capability
- `analyze_failures()` — detects failures from trajectories, counts failure modes, estimates capability from recent rewards
- `select_next_scenario()` — capability < 0.3: random. capability < 0.5: easy. capability < 0.75: frontier. else: hard.
- `generate_adaptive_scenario()` — applies failure-mode-specific modifications (e.g., if F1 > 30%, reduce CVaR tension)

**Note**: The curriculum generator is defined but not integrated into the main environment or training loop. It's a standalone module.

---

### `deal_room/training/grpo_trainer.py`

**What it does**: Group Relative Policy Optimization (GRPO) trainer for multi-dimensional reward training.

**Key components**:
- `REWARD_WEIGHTS` — same 5-dim weights as environment
- `TrainingMetrics` dataclass — goal/trust/info/risk/causal rewards, lookahead_usage_rate, prediction_accuracy, total_reward
- `EpisodeTrajectory` dataclass — observations, actions, rewards (5D per step), lookahead_used flags, prediction_accuracies
- `GRPOTrainer` class — model_id (default Qwen/Qwen2.5-3B-Instruct), learning_rate, grpo_clip, entropy_coef, reward_weights
- `compute_group_relative_advantage()` — aggregates 5D rewards to scalar, computes z-score advantage against group mean/std
- `run_self_play_episode()` — runs one episode with either provided policy_fn or `_default_policy` (random stakeholder targeting with generic messages)
- `_default_policy()` — random stakeholder selection, random message from 4 predefined options
- `compute_training_metrics()` — aggregates across trajectories
- `run_training_loop()` — runs n_episodes, each with episodes_per_batch parallel episodes, prints metrics every 10 episodes

**Note**: GRPOTrainer is defined but not connected to an actual model. The `_default_policy` is a random baseline. Actual model training would require integrating with a model backend.

---

### `models.py`

**What it does**: Pydantic models for actions, observations, and state.

**Key components**:
- `DealRoomAction` — `metadata`, `action_type` (default "direct_message"), `target` (default "all"), `target_ids` (list), `message` (truncated to 1200 chars), `documents` (list of {name, content}), `proposed_terms`, `channel`, `mode`, `lookahead` (optional `LookaheadRequest`). Validators: truncate message, normalize target_ids, sync target with target_ids.
- `DealRoomObservation` — 18 required fields: round_number, max_rounds, stakeholders, stakeholder_messages, engagement_level, weak_signals, known_constraints, requested_artifacts, approval_path_progress, deal_momentum, deal_stage, competitor_events, veto_precursors, scenario_hint, active_blockers, days_to_deadline, done, info, engagement_level_delta, engagement_history, cross_stakeholder_echoes. reward=None always (reward comes from step return, not observation).
- `DealRoomReward` — `value: float`, `done: bool`, `info: Dict`
- `DealRoomState` — full internal state including private stakeholder state, hidden_constraints, relationship_edges, commitment_ledger, offer_state, feasibility_state, active_blockers, deal_stage, stage_regressions, validation_failures, deal_closed/deal_failed flags, etc.
- `LookaheadRequest` — `action_draft: DealRoomAction`, `n_hypotheses: int = 2`, `depth: int = 2`
- `SimulationResult` — `predicted_responses`, `predicted_belief_deltas`, `cvar_impact`, `graph_information_gain`, `cost: float = 0.07`

---

### `server/app.py`

**What it does**: FastAPI HTTP server. Thin wrapper — zero business logic.

**Key components**:
- `DealRoomSessionPool` — manages per-session `DealRoomV3` instances
- `GET /` and `GET /web` — returns HTML iframe pointing to `/__gradio_ui__/`
- `GET /health` — returns `{"status": "ok", "service": "deal-room", "tasks": [...]}`
- `GET /metadata` — returns name, version, tasks
- `POST /reset` — calls pool.reset(), returns observation, sets session cookie
- `POST /step` — calls pool.step(), returns {observation, reward, done, info}
- `GET /state` — returns full DealRoomState for session
- `_resolve_session_id()` — checks explicit_id → metadata → headers → query → cookies
- `_setup_gradio_ui()` — tries to mount standalone Gradio app from `server.gradio_standalone`, falls back to OpenEnv Gradio UI with custom DealRoom tab. If Gradio unavailable, returns 503.

**Session management**: Cookie-based session ID with 6-hour TTL, max 128 sessions, auto-prune on access.

---

### `server/session_pool.py`

**What it does**: Thread-safe session pool for DealRoomV3 instances.

**Key components**:
- `SessionEntry` — holds env instance and last_access timestamp
- `DealRoomSessionPool` — max_sessions=128, ttl_seconds=21600 (6 hours)
- `reset()` — creates new env if session doesn't exist, calls env.reset(), returns (session_id, obs, state)
- `step()` — delegates to entry.env.step(), returns (obs, reward, done, info, state)
- `state()` — returns entry.env._state
- `has_session()` — checks if session_id exists
- `_prune_locked()` — removes expired sessions on every reset
- `_prune_oldest_locked()` — removes oldest session if over max_sessions

---

## Data Flow

```
HTTP Request (reset/step)
    ↓
server/app.py路由
    ↓
DealRoomSessionPool (session lookup/creation)
    ↓
DealRoomV3.reset() or DealRoomV3.step()
    ↓
├── sample_graph() → CausalGraph G (hidden)
├── create_neutral_beliefs() / _get_initial_beliefs() → BeliefDistribution per stakeholder
├── CommitteeDeliberationEngine.run() → propagate_beliefs() → BeliefDistribution updates
├── _update_noisy_engagement() → noisy engagement + sliding window
├── UtteranceScorer.score() → 5D reward → weighted sum
└── _build_observation() → DealRoomObservation (G NOT included)
    ↓
HTTP Response {observation, reward, done, info}
```

## Key Research Properties Implemented

1. **G is hidden** — causal graph never appears in observation
2. **Episode reset regenerates G** — different seeds → different graphs → different initial states
3. **CVaR veto despite positive EU** — Legal's tau=0.10 can fire veto when EU > 0
4. **5D reward discriminative** — dimensions capture different aspects of deal progress
5. **Lookahead costs exactly 0.07** — hardcoded, not approximate
6. **Engagement noise σ=0.03** — cannot be cancelled by repeating same action
7. **Cross-stakeholder echoes** — 70% probability echo_recall
8. **Weak signals** — hard threshold 0.12, soft lower 0.08
9. **r^causal varies with graph centrality** — betweenness-based
10. **Graph identifiability** — 20 sampled graphs all produce unique behavioral signatures
