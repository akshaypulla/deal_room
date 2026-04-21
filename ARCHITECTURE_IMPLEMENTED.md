# DealRoom v3 Architecture

## What is DealRoom v3?

DealRoom v3 is a multi-stakeholder procurement negotiation environment designed for reinforcement learning research. It simulates a corporate vendor selection process where an AI agent (the "vendor") must navigate approval from a committee of stakeholders (Legal, Finance, TechLead, Procurement, Operations, ExecSponsor) to close a deal.

The environment is built as an OpenEnv-compatible Gymnasium-style environment with a FastAPI server wrapper and a Gradio web interface.

## Why this architecture?

### Core Research Goals

1. **Causal Graph Hidden from Agent**: The environment contains a causal graph `G` that governs how stakeholder beliefs propagate. This graph is never exposed to the agent—it is regenerated on each episode reset and influences which stakeholders receive "echoes" when a targeted stakeholder is addressed.

2. **CVaR-based Veto Mechanism**: Unlike standard expected utility (EU) based systems, DealRoom v3 uses Conditional Value at Risk (CVaR) with stakeholder-specific risk profiles (alpha, tau, lambda_risk) to determine veto triggers. A deal can have positive EU but still be vetoed because CVaR captures tail risk that EU misses.

3. **Five-Dimensional Reward Signal**: The reward is decomposed into five independent dimensions—goal, trust, information, risk, causal—each computed from world-state deltas rather than message content. This prevents reward hacking via empty messages or repetition.

4. **Lookahead Mechanism with Cost**: The agent can request lookahead simulations to preview action outcomes. Using lookahead costs exactly 0.07 from the goal dimension, creating a trade-off between exploration and exploitation.

5. **Partial Observability via Weak Signals and Echoes**: The agent does not see the true causal graph. Instead, it receives weak signals (engagement-based tags) and cross-stakeholder echoes (indicating which stakeholders "heard about" the action), allowing it to infer graph structure over time.

### Architectural Layers

```
┌─────────────────────────────────────────────────────────────┐
│                     HTTP API Layer (FastAPI)                │
│                  server/app.py + session_pool.py            │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│               OpenEnv Wrapper / Session Management          │
│                   DealRoomV3 (environment/dealroom_v3.py)   │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐  ┌────────────────────┐  ┌─────────────────┐
│  Committee    │  │   Environment      │  │   Rewards       │
│  Module       │  │   Module           │  │   Module        │
│               │  │                   │  │                 │
│ - causal_graph│  │ - dealroom_v3.py  │  │ -utterance_scorer│
│ - belief_tracker│ │ - lookahead.py   │  │ -pareto_efficiency│
│ - deliberation_engine│- llm_client.py │  │                 │
│               │  │ - constants.py   │  │                 │
└───────────────┘  └────────────────────┘  └─────────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                Stakeholders Module                          │
│                   archetypes.py + cvar_preferences.py       │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                Training Module                              │
│               grpo_trainer.py + curriculum/adaptive_generator.py │
└─────────────────────────────────────────────────────────────┘
```

## Module-by-Module Detailed Explanation

### 1. `models.py` — Data Models

**What**: Pydantic models defining the interface between agent and environment.

**Why**: Ensures type safety and validation at the API boundary. All actions and observations are validated before reaching the environment logic.

**How**:
- `DealRoomAction`: Action schema with `action_type` (direct_message, send_document, group_proposal, backchannel, exec_escalation), `target_ids`, `message` (truncated to 1200 chars), `documents` (list of {name, content} dicts), `proposed_terms`, and optional `lookahead` (LookaheadRequest).
- `DealRoomObservation`: 18 required fields plus metadata. Key fields: `engagement_level` (dict of stakeholder floats), `engagement_level_delta` (single float, NOT a dict), `engagement_history` (list of 5 stakeholder snapshots), `weak_signals` (dict of stakeholder → list of tags), `cross_stakeholder_echoes` (list of {from, to, content} dicts), `veto_precursors` (dict of stakeholder → warning message).
- `DealRoomState`: Internal state tracking episode progress, offer state, commitment ledger, feasibility state, veto tracking, and terminal outcomes.
- `LookaheadRequest`: Encapsulates a draft action and simulation parameters (n_hypotheses=2, depth=2).
- `SimulationResult`: Lookahead output containing predicted responses, belief deltas, and cvar_impact.

**Critical Implementation Detail**: `engagement_level_delta` must be a single float, not a dict. This was a discovered bug in earlier versions where the field was incorrectly shaped.

### 2. `deal_room/environment/dealroom_v3.py` — Core Environment

**What**: The main `DealRoomV3` class implementing the OpenEnv `reset()` and `step()` interface.

**Why**: This is the heart of the environment. It orchestrates belief updates, deliberation propagation, reward computation, veto checking, and observation construction.

**How**:

**Initialization and Reset**:
- `reset(seed, task_id)` initializes a fresh RNG, generates a causal graph via `sample_graph()` using the task_id scenario parameters, initializes belief distributions per stakeholder via `_get_initial_beliefs()`, sets noisy engagement levels (with σ=0.03 noise), and builds the initial `DealRoomState`.
- Three scenarios: `aligned` (cooperative, high intra-cluster edges), `conflicted` (competing interests, mixed clusters), `hostile_acquisition` (adversarial, many edges but high cross-cluster penalty).
- `episode_id` is a UUID fragment, `round_number` starts at 0, `_step_count` at 0.

**Step Function**:
1. Normalize action target IDs (case-insensitive canonical mapping).
2. Run lookahead simulation if requested, storing results for reward accuracy computation.
3. Apply action to offer state (price adjustments based on action type and document names).
4. Update deal stage based on round number (evaluation < 3, negotiation 3-6, final_review ≥ 6).
5. For each stakeholder, run `bayesian_update()` to update beliefs based on action type and targeting.
6. Run `CommitteeDeliberationEngine` for belief propagation through the causal graph.
7. Update noisy engagement levels (additive noise cannot be cancelled).
8. Generate stakeholder responses (targeted vs non-targeted).
9. Compute reward via `UtteranceScorer.score()`.
10. Evaluate committee risk to get CVaR losses per stakeholder.
11. Check for veto trigger (requires 2 consecutive precursor rounds above tau threshold).
12. If done, compute terminal reward via `compute_terminal_reward()`.
13. Build and return `DealRoomObservation`.

**Key Internal Methods**:
- `_evaluate_committee_risk()`: Uses `compute_outcome_distribution()` (500 samples) per stakeholder, multiplies CVaR by scenario multiplier and confidence factor.
- `_compute_veto_precursors()`: Flags stakeholders whose cvar_loss exceeds 70% of tau (warning) or exceeds tau (critical).
- `_update_veto_precursor_streaks()`: Tracks consecutive precursor rounds for each stakeholder.
- `_check_for_veto()`: Returns (True, stakeholder_id) if cvar_loss > tau for 2+ consecutive rounds.
- `_build_observation()`: Constructs the 18-field observation, generates weak signals and cross-stakeholder echoes.

**Observation Generation**:
- Weak signals: Based on noisy engagement level (>0.7 = high_engagement, <0.3 = low_engagement), engagement delta (>0.1 = improving, < -0.1 = declining), and belief confidence (<0.4 = high_uncertainty).
- Cross-stakeholder echoes: Each non-targeted stakeholder has a 70% chance of receiving an echo (configured via `echo_recall_probability`).
- `engagement_history`: Rolling window of 5 entries per stakeholder.

### 3. `deal_room/committee/causal_graph.py` — Causal Graph Structure

**What**: Graph data structures and functions for belief propagation simulation.

**Why**: The causal graph is the mechanism by which actions on one stakeholder influence others. It encodes functional clusters (cost: Finance+Procurement, risk: Legal+Compliance, implementation: TechLead+Operations, authority: ExecSponsor) and authority hierarchy (ExecSponsor at level 5, Legal+Finance at 3, others at 2).

**How**:

**`CausalGraph` Dataclass**:
- `nodes`: List of stakeholder IDs.
- `edges`: Dict[Tuple[str, str], float] mapping directed edges to weights.
- `authority_weights`: Dict[str, float] normalized authority contributions.
- `scenario_type`: Which scenario params were used.
- `seed`: Graph seed for reproducibility.

**`sample_graph()` Function**:
- For each source-destination pair (no self-loops):
  - If source authority ≥ 4: always create edge (p_edge = 1.0).
  - If same functional cluster: p_edge = base_probability + intra_cluster_boost.
  - If different cluster: p_edge = base_probability - cross_cluster_penalty.
- Edge weights drawn from Normal(weight_mean, weight_std), clipped to [0.05, 0.95].
- Authority weights normalized by total authority sum.

**Scenario Parameters**:
| Scenario | base_prob | intra_boost | cross_penalty | auth_edge_prob | weight_mean | weight_std |
|---|---|---|---|---|---|---|
| aligned | 0.30 | 0.40 | 0.20 | 0.85 | 0.45 | 0.15 |
| conflicted | 0.45 | 0.50 | 0.15 | 0.80 | 0.50 | 0.18 |
| hostile_acquisition | 0.60 | 0.25 | 0.05 | 0.65 | 0.45 | 0.25 |

**`BeliefDistribution` Dataclass**:
- `distribution`: Dict[str, float] over 6 vendor types (competent, incompetent, trustworthy, deceptive, aligned, misaligned).
- `positive_mass()`: Sum of competent + trustworthy + aligned.
- `negative_mass()`: Sum of incompetent + deceptive + misaligned.
- `to_natural_language()`: Maps positive_mass ranges to human-readable strings.
- `confidence`: Derived from entropy, range [0, 1].

**`propagate_beliefs()` Function**:
- Takes beliefs_before_action and beliefs_after_action (both dicts of BeliefDistribution).
- Runs n_steps iterations (default 3).
- For each destination stakeholder, sums weighted deltas from all incoming edges.
- Applies damping factor: damping = 0.85^step (prevents runaway amplification).
- Uses `_apply_belief_delta()` which transfers probability mass between positive and negative types.

**`get_betweenness_centrality()`**:
- BFS-based shortest path computation.
- Returns fraction of node pairs for which this node lies on shortest path.
- Used by the causal reward dimension.

**`compute_behavioral_signature()`**:
- For a given targeted stakeholder and belief delta (0.30 default), compute how all other stakeholders' engagement levels change.
- Returns dict of engagement deltas per non-targeted stakeholder.
- Used for graph identifiability testing.

### 4. `deal_room/committee/belief_tracker.py` — Bayesian Belief Updates

**What**: Functions for updating stakeholder beliefs based on vendor actions.

**Why**: Belief updates are the primary mechanism by which the agent's actions affect stakeholder dispositions. The update uses likelihood ratios from action types and documents.

**How**:

**`ACTION_LIKELIHOODS` Dictionary**:
- Maps action signatures (e.g., "send_document(DPA)_proactive") to likelihood ratios per vendor type.
- Example: sending DPA proactively has likelihood 0.85 for competent, 0.20 for incompetent, 0.80 for trustworthy, 0.20 for deceptive.
- Uses exact substring matching on action_type, then document name matching.
- Falls back to uniform 0.5 likelihoods (default).

**`bayesian_update()` Function**:
- Gets likelihoods via `_get_likelihood()`.
- Damping factor: 1.0 for targeted stakeholders, 0.3 for non-targeted (targeted receives 3.3x stronger belief shift).
- For each vendor type: new_prob = prior_prob × (1 + damping × (likelihood - 1)).
- Normalizes to sum to 1.0.
- Confidence = 1 - (entropy / log2(6)), where log2(6) ≈ 2.585 is maximum entropy for 6 equally likely types.
- History tracks action type and damping per update.

**`compute_engagement_level()`**:
- Returns positive_mass - negative_mass (range approximately [-1, 1]).

### 5. `deal_room/committee/deliberation_engine.py` — Belief Propagation

**What**: Committee deliberation engine that wraps belief propagation with optional LLM-generated summaries.

**Why**: After the agent acts, the committee "discusses" the action, propagating beliefs through the causal graph. The optional LLM summary provides a human-readable deliberation dialogue for debugging and analysis.

**How**:

**`CommitteeDeliberationEngine` Class**:
- Takes the causal graph and n_deliberation_steps (default 3).
- `run()` calls `propagate_beliefs()` to get updated beliefs.
- If `render_summary=True` and there are target_ids, calls `_generate_summary()`.

**`_generate_summary()` Function**:
- Builds prompt with targeted stakeholder's belief shift and other stakeholder deltas.
- Calls `generate_deliberation_summary()` from `llm_client` (MiniMax API).
- Falls back to empty string on exception.
- Returns 2-4 sentence summary of committee discussion.

**`DeliberationResult` Dataclass**:
- `updated_beliefs`: Dict[str, BeliefDistribution] post-propagation.
- `summary_dialogue`: Optional[str] LLM-generated summary.
- `propagation_deltas`: Dict[str, float] of positive_mass delta per stakeholder.

### 6. `deal_room/environment/lookahead.py` — Lookahead Simulation

**What**: Simulator for previewing action outcomes before committing.

**Why**: Lookahead allows the agent to test hypotheses about what would happen if it takes an action. The cost (0.07 from goal) creates a trade-off preventing frivolous use.

**How**:

**`LookaheadSimulator.simulate()`**:
- Generates n_hypotheses (default 2) belief distributions for the target stakeholder: one optimistic (+0.15 competent, +0.10 trustworthy, +0.10 aligned) and one pessimistic (+0.15 incompetent, +0.10 deceptive, +0.10 misaligned).
- For each hypothesis, calls `_simulate_one_hypothesis()` which computes response scores based on action type and message content.
- Returns worst-case (minimum predicted_goal_delta) across hypotheses.

**`SimulationResult`**:
- `predicted_responses`: Dict of simulated stakeholder responses.
- `predicted_belief_deltas`: Dict of expected belief shifts.
- `cvar_impact`: Dict of expected CVaR changes (negative, since negative belief delta → higher CVaR risk).
- `graph_information_gain`: 0.1 if documents attached, 0.05 otherwise.
- `cost`: Always `LOOKAHEAD_COST` (0.07).

### 7. `deal_room/rewards/utterance_scorer.py` — Five-Dimensional Reward

**What**: World-state-based deterministic scoring with five dimensions.

**Why**: This is the core grading mechanism. It computes reward from world-state deltas (not message content), preventing gaming via empty messages or repetition. Each dimension captures a distinct aspect of deal progress.

**How**:

**Five Dimensions**:
1. **goal** (weight 0.30): Approval-weighted belief delta + blocker resolution score + veto headroom improvement.
2. **trust** (weight 0.25): For targeted stakeholders, positive_mass delta + trustworthy mass delta (60/40 weighted).
3. **info** (weight 0.20): Average entropy reduction across all stakeholders, normalized by max entropy.
4. **risk** (weight 0.15): CVaR improvement ratio for risk-averse stakeholders (lambda_risk > 0.30).
5. **causal** (weight 0.10): Betweenness centrality of targeted stakeholder normalized by max possible.

**`LOOKAHEAD_COST = 0.07`**: Subtracts exactly 0.07 from goal dimension when lookahead is used.

**`UtteranceScore.to_dict()`**: Returns {goal, trust, info, risk, causal} floats.

**`compute_prediction_accuracy()`**: Jaccard similarity between predicted and actual response token sets.

**`_score_goal()` Logic**:
- Approval delta weighted by authority: Σ(auth_i × Δpm_i) / Σ(auth_i).
- Blocker resolution: +0.15 per resolved blocker, -0.10 per newly created blocker.
- Veto headroom improvement: max(0, 1 - cvar_a/tau) - max(0, 1 - cvar_b/tau), averaged over risk-averse stakeholders.
- Final = clip(0.5 + 0.50×approval + 0.30×blocker + 0.20×veto, 0, 1).

### 8. `deal_room/rewards/pareto_efficiency.py` — Terminal Reward

**What**: Terminal reward computation when episode ends.

**Why**: Determines the outcome of the episode (deal_closed, veto, timeout, stage_regression, impasse) and assigns the appropriate terminal reward.

**How**:

**`compute_terminal_reward()`**:
- `deal_closed`: Returns `+1.0` (TERMINAL_REWARDS["deal_closed"]).
- `veto_triggered`: Returns `-1.0` (TERMINAL_REWARDS["veto"]), with terminal_outcome `f"veto_by_{veto_stakeholder}"`.
- `max_rounds_reached`: Checks Pareto optimality; if on frontier returns `0.0`, else returns `-0.8` (TERMINAL_REWARDS["max_rounds"]).
- `stage_regressions > 0`: Penalty = -0.3 × min(stage_regressions, 3).
- Default: Returns `-0.5` (TERMINAL_REWARDS["impasse"]).

**`check_pareto_optimality()`**:
- Checks no stakeholder's cvar exceeds their threshold.
- Checks no stakeholder is dominated (higher utility AND lower or equal cvar than another).

### 9. `deal_room/stakeholders/cvar_preferences.py` — Risk Modeling

**What**: CVaR computation and outcome distribution modeling.

**Why**: CVaR (Conditional Value at Risk) at quantile α captures the expected loss in the worst (1-α) quantile of outcomes. This is the core of the veto mechanism—EU can be positive while CVaR exceeds the threshold, triggering rejection.

**How**:

**`compute_outcome_distribution()`**:
- Determines `base_success_prob` based on uncertainty domain and deal terms:
  - compliance/data_protection: 0.80 base, 0.92 if has_dpa AND has_security_cert, 0.86 if has_dpa OR has_security_cert.
  - cost/payment: 0.70.
  - implementation/operational: 0.75.
- For each of n_samples (500):
  - If random() < base_success_prob: outcome = 1.0 - 0.1*random() + outcome_adjustment.
  - Else: outcome based on severity (0.6-0.7: mild, 0.3-0.5: moderate, 0.0-0.2: severe).
  - outcome_adjustment: -0.05 if liability_cap < 500000.
- Returns numpy array of outcomes in [0, 1].

**`compute_cvar(outcomes, alpha)`**:
- Sorts outcomes ascending.
- cutoff_index = int(len × (1 - alpha)).
- CVaR = average of (1 - outcomes[0:cutoff_index+1]) weighted uniformly.
- Coherent: CVaR ≤ max(outcomes).

**`evaluate_deal()`**: Convenience wrapping outcome distribution + CVaR + expected utility.

**`check_veto_trigger(cvar_loss, profile)`**: Returns cvar_loss > profile.tau.

### 10. `deal_room/stakeholders/archetypes.py` — Stakeholder Profiles

**What**: Six stakeholder archetypes with their risk parameters.

**Why**: Each stakeholder has different risk tolerance (tau), risk aversion (lambda_risk), and veto power. The ordering (Legal < Finance < ExecSponsor in tau) reflects real organizational dynamics where legal has lowest tolerance for risk.

**How**:

**Profiles**:
| Stakeholder | Alpha | Tau | Lambda | Veto | Primary Concerns |
|---|---|---|---|---|---|
| Legal | 0.95 | 0.10 | 0.70 | Yes | compliance_coverage, liability_limitation, data_protection |
| Finance | 0.90 | 0.15 | 0.50 | Yes | roi_clarity, payment_terms, cost_predictability |
| TechLead | 0.80 | 0.25 | 0.30 | No | implementation_feasibility, integration_quality, support_model |
| Procurement | 0.85 | 0.20 | 0.45 | No | contract_compliance, price_competitiveness, vendor_qualification |
| Operations | 0.80 | 0.30 | 0.35 | No | operational_continuity, timeline, change_management |
| ExecSponsor | 0.70 | 0.40 | 0.25 | Yes | strategic_alignment, organizational_consensus, reputational_risk |

**Alpha** (95% for Legal): Higher alpha means the CVaR considers a narrower (more extreme) tail—the worst 5% of outcomes. Legal has the highest alpha, meaning it cares most about catastrophic tail scenarios.

**Tau** (0.10 for Legal): The CVaR threshold. If CVaR exceeds tau, veto precursor fires. Legal's 0.10 means even small tail risk triggers warnings.

**Veto Power**: Legal, Finance, ExecSponsor have veto_power=True. Only these three can trigger episode termination via veto.

### 11. `deal_room/training/grpo_trainer.py` — GRPO Training Harness

**What**: Group Relative Policy Optimization training loop.

**Why**: Provides the training infrastructure for RL agents. Supports multiple policy adapters, episode trajectories, checkpointing, and metric tracking.

**How**:

**`TrainingMetrics` Dataclass**: Tracks goal/trust/info/risk/causal rewards, lookahead usage rate, prediction accuracy, total/weighted rewards, episodes completed, task mix, terminal outcomes, checkpoint path.

**`EpisodeTrajectory` Dataclass**: Collects observations, actions, reward vectors, scalar rewards, lookahead flags, prediction accuracies across an entire episode.

**`RandomPolicyAdapter`**: Baseline random agent that selects actions uniformly at random with 15% lookahead probability.

**`GRPOTrainer`**: Main training class. Key methods:
- `train()`: Main loop calling `_run_episode()` and `_compute_updates()`.
- `_run_episode()`: Runs environment to completion, collects trajectory.
- `_compute_updates()`: Processes batch of trajectories for policy gradient updates.
- `save_checkpoint()` / `load_checkpoint()`: State dict serialization.
- `generate_curriculum()`: Uses AdaptiveCurriculumGenerator for curriculum learning.

### 12. `deal_room/curriculum/adaptive_generator.py` — Curriculum Learning

**What**: Adaptive scenario selection based on agent failure analysis.

**Why**: Curriculum learning progressively increases difficulty as the agent improves. The generator analyzes failure modes to select appropriate scenarios.

**How**:

**`FailureAnalysis` Dataclass**: Tracks failure mode frequencies, worst graph/CVaR configs, agent capability estimate.

**`CurriculumConfig` Dataclass**: analysis_batch_size=10, easy_ratio=0.20, frontier_ratio=0.60, hard_ratio=0.20, max_graph_variation=0.3.

**Failure Mode Detection**:
- F1: CVaR veto despite positive EU (terminal_outcome = "veto").
- F2: Trust collapse (trust rewards drop > 0.20 in rounds 6-10).
- F3: Failed graph inference (causal rewards stuck in [0.15, 0.30] range).
- F4: Timeout without coalition.
- F5: Single-dimension reward hacking.
- F6: Authority shift blindness.

**`select_next_scenario()`**:
- If capability < 0.3: random selection.
- If capability < 0.5: prefer easy scenarios.
- If capability < 0.75: prefer frontier scenarios.
- Else: prefer hard scenarios.

### 13. `deal_room/environment/llm_client.py` — LLM Integration

**What**: Thin wrapper for MiniMax API calls (used in deliberation summaries and utterance generation).

**Why**: Provides LLM-generated deliberation summaries when the deliberation engine calls `_minimax_call()`. This is optional—the environment degrades gracefully without API keys.

**How**:
- `validate_api_keys()`: Checks MINIMAX_API_KEY presence.
- `generate_deliberation_summary(prompt, context, timeout)`: Calls MiniMax /api/v1/text/chatcompletion_pro.
- `generate_utterance(stakeholder_id, context, ...)`: Generates natural language stakeholder responses.

### 14. `server/app.py` — FastAPI HTTP Server

**What**: HTTP wrapper around the environment with session management.

**Why**: Provides network API for the environment, enabling containerized deployment and multi-session support. The server is thin—all business logic lives in the environment module.

**How**:

**Endpoints**:
- `GET /health`: Returns {status, service, tasks}.
- `GET /metadata`: Returns {name, version, tasks}.
- `POST /reset`: Creates new session, returns initial observation. Accepts {task_id, seed, episode_id}. Sets session cookie.
- `POST /step`: Advances environment with action. Returns {observation, reward, done, info}. Requires valid session cookie or explicit session_id in metadata.
- `GET /state`: Returns full DealRoomState for session.

**Session Management**:
- `DealRoomSessionPool` manages active sessions.
- Sessions persist until explicitly terminated or timeout.
- Cookie-based session tracking with fallback to explicit session_id in headers/query/metadata.

**Action Normalization**:
- `OutputValidator._normalize()` canonicalizes target IDs and action types.

### 15. `server/session_pool.py` — Session Pool

**What**: Manages multiple concurrent environment instances.

**Why**: Allows the server to handle multiple simultaneous sessions, each with its own DealRoomV3 instance and internal state.

**How**:
- `reset(seed, task_id, session_id)`: Creates new DealRoomV3, initializes via reset(), stores in sessions dict.
- `step(session_id, action)`: Retrieves session, calls env.step(), returns obs/reward/done/info/state.
- `state(session_id)`: Returns current DealRoomState.
- `has_session(session_id)`: Boolean check.

## File Structure Summary

```
deal_room/
├── __init__.py
├── committee/
│   ├── __init__.py
│   ├── belief_tracker.py        # Bayesian belief updates
│   ├── causal_graph.py          # Graph structure + belief propagation
│   └── deliberation_engine.py   # Committee deliberation + optional LLM summaries
├── curriculum/
│   ├── __init__.py
│   └── adaptive_generator.py     # Curriculum scenario selection
├── environment/
│   ├── __init__.py
│   ├── constants.py              # REWARD_WEIGHTS, TERMINAL_REWARDS, DEFAULT_MAX_ROUNDS, etc.
│   ├── dealroom_v3.py           # Main DealRoomV3 environment class
│   ├── llm_client.py            # MiniMax API wrapper
│   └── lookahead.py             # Lookahead simulation
├── rewards/
│   ├── __init__.py
│   ├── pareto_efficiency.py      # Terminal reward + Pareto optimality
│   └── utterance_scorer.py       # Five-dimensional grading
├── stakeholders/
│   ├── __init__.py
│   ├── archetypes.py             # Six stakeholder profiles
│   └── cvar_preferences.py       # CVaR + outcome distribution
└── training/
    ├── __init__.py
    ├── grpo_trainer.py           # GRPO training harness
    ├── grpo_colab.ipynb          # Colab notebook for training
    └── run_benchmark.py          # Benchmark runner

server/
├── __init__.py
├── app.py                        # FastAPI server
├── session_pool.py               # Session management
├── validator.py                  # Action normalization
├── gradio_custom.py              # Custom Gradio UI
├── gradio_standalone.py          # Standalone Gradio app
├── stakeholders.py               # Server-side stakeholder utilities
├── scenarios.py                  # Scenario definitions
├── semantics.py                  # Semantic utilities
├── claims.py                     # Research claims
├── grader.py                     # Legacy grader
└── deal_room_environment.py      # Legacy environment

models.py                         # Pydantic models (DealRoomAction, Observation, State)
inference.py                       # Inference utilities
client.py                         # Client example
```

## Key Implementation Details

### Engagement Noise

Engagement levels are noisy (σ = 0.03) and the noise is additive—meaning it cannot be cancelled by repeating the same action. This is verified in test_03_causal_inference.py test_3_4.

### Observation Fields (18 required)

`round_number`, `max_rounds`, `stakeholders`, `stakeholder_messages`, `engagement_level`, `engagement_level_delta`, `engagement_history`, `weak_signals`, `cross_stakeholder_echoes`, `veto_precursors`, `known_constraints`, `requested_artifacts`, `approval_path_progress`, `deal_momentum`, `deal_stage`, `active_blockers`, `days_to_deadline`, `done`

### Hidden Fields (must never appear in observation)

`G`, `causal_graph`, `graph`, `true_beliefs`, `belief_distributions`, `belief_state`, `B_i`, `V_i`, `tau`, `tau_i`, `risk_thresholds`, `cvar_thresholds`, `edge_weights`, `w_ij`, `deliberation_transcript`, `deliberation_log`, `internal_dialogue`, `u_i`, `u_ij`

### CVaR Veto Logic

1. Each step, `_evaluate_committee_risk()` computes cvar_loss per stakeholder.
2. `_compute_veto_precursors()` flags stakeholders where cvar_loss > tau×0.70 (warning) or > tau (critical).
3. `_update_veto_precursor_streaks()` increments streak for flagged stakeholders.
4. `_check_for_veto()` triggers veto if cvar_loss > tau AND streak ≥ 2.
5. Veto sets `done=True` and terminal_outcome = `veto_by_{stakeholder_id}`.

### Reward Weights

```python
REWARD_WEIGHTS = {
    "goal": 0.30,
    "trust": 0.25,
    "info": 0.20,
    "risk": 0.15,
    "causal": 0.10,
}
```