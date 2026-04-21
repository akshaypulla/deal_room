# DealRoom v3.2 Architecture Documentation

## Table of Contents
1. [Overview](#overview)
2. [Core Environment](#core-environment)
3. [Causal Graph System](#causal-graph-system)
4. [Belief Tracking & Bayesian Updates](#belief-tracking--bayesian-updates)
5. [Committee Deliberation Engine](#committee-deliberation-engine)
6. [CVaR-Based Risk Preferences](#cvar-based-risk-preferences)
7. [Reward System](#reward-system)
8. [Lookahead Simulator](#lookahead-simulator)
9. [Training Infrastructure](#training-infrastructure)
10. [Curriculum Generation](#curriculum-generation)
11. [Server Architecture](#server-architecture)

---

## Overview

**What:** DealRoom v3.2 is a multi-stakeholder negotiation environment designed for Reinforcement Learning (RL) training. It simulates a vendor deal room where an AI agent must navigate approvals from 6 stakeholders (Legal, Finance, TechLead, Procurement, Operations, ExecSponsor) across 3 scenario types (aligned, conflicted, hostile_acquisition).

**Why:** The environment was built to study:
- CVaR-based veto mechanisms (tail-risk aware decision making)
- Causal graph inference in multi-agent systems
- Partial observability with belief propagation
- Reward hacking resistance through deterministic world-state-based scoring

**How:** The system uses a FastAPI server wrapping a Python environment (`DealRoomV3`) that implements the OpenEnv interface. All business logic is in `deal_room/` package, with the server being a thin HTTP wrapper.

---

## Core Environment

### DealRoomV3 (`deal_room/environment/dealroom_v3.py`)

**What:** The main environment class that implements the RL interface with `reset()` and `step()` methods.

**Why:** Provides a standardized interface for RL training while encapsulating all deal room dynamics.

**How:**
- **Stakeholders:** 6 standard stakeholders with authority hierarchy (ExecSponsor=5, Legal/Finance=3, TechLead/Procurement/Operations=2)
- **Three scenarios:**
  - `aligned`: Cooperative scenario, low edge density (30% base probability), high authority edges (85%)
  - `conflicted`: Moderate conflict, medium edge density (45%), clustered stakeholders
  - `hostile_acquisition`: High conflict, dense edges (60%), weak authority
- **State management:** Uses `DealRoomState` Pydantic model for internal state, `DealRoomObservation` for agent-facing output
- **Key observation fields:** `engagement_level`, `weak_signals`, `cross_stakeholder_echoes`, `veto_precursors`, `active_blockers`

### Observation Configuration (`OBS_CONFIG`)

**What:** Noise and signal parameters for observation generation.

**Why:** Implements partial observability with calibrated noise to prevent reward hacking.

**How:**
- `engagement_noise_sigma: 0.03` - Controls engagement signal noise
- `echo_recall_probability: 0.70` - 70% chance cross-stakeholder echoes appear
- `weak_signal_hard_threshold: 0.12` - Threshold for weak signal generation
- `engagement_history_window: 5` - Number of historical engagement snapshots

---

## Causal Graph System

### CausalGraph (`deal_room/committee/causal_graph.py`)

**What:** Represents the committee's influence network as a directed weighted graph.

**Why:** The causal graph (G) is the core hidden state that agents must infer. It determines how actions on one stakeholder propagate to others.

**How:**
- **Nodes:** 6 stakeholders
- **Edges:** Directed weighted connections `(source, dest)` with weights `[0.05, 0.95]`
- **Authority weights:** Normalized authority hierarchy values
- **Graph sampling:** Probabilistic based on scenario type:
  - Authority nodes (level ≥ 4) always have outgoing edges
  - Same functional cluster boosts edge probability (+40% aligned, +50% conflicted)
  - Cross-cluster penalty (-20% aligned, -15% conflicted)
- **Functional clusters:**
  - cost: Finance, Procurement
  - risk: Legal, Compliance
  - implementation: TechLead, Operations
  - authority: ExecSponsor

### BeliefDistribution

**What:** Tracks a stakeholder's belief about the vendor across 6 vendor types.

**Why:** Provides a probabilistic model of stakeholder perceptions.

**How:**
- **Vendor types:** `competent`, `incompetent`, `trustworthy`, `deceptive`, `aligned`, `misaligned`
- **positive_mass():** Sum of competent + trustworthy + aligned
- **negative_mass():** Sum of incompetent + deceptive + misaligned
- **Initial beliefs:** Scenario-dependent distributions (e.g., `aligned` has higher positive mass, `hostile_acquisition` has higher negative mass)

### Belief Propagation

**What:** `propagate_beliefs()` function spreads belief changes across the graph.

**Why:** Models how one stakeholder's opinion affects others through the committee dynamics.

**How:**
- **Damping:** Each propagation step applies `0.85^step` damping to prevent runaway amplification
- **Direction:** Follows graph edge direction (A→B means A's belief change affects B)
- **Normalization:** All beliefs sum to 1.0 after each step
- **3 steps:** Default propagation depth for aligned/conflicted, 4 for hostile

### Betweenness Centrality

**What:** `get_betweenness_centrality()` measures how central a node is in the graph.

**Why:** Determines the "causal influence" score for reward calculation.

**How:**
- Counts shortest paths through the node
- Normalized by `(n-1)(n-2)` for nodes with >2 members
- Hub nodes (ExecSponsor) typically have highest centrality

### Graph Identifiability

**What:** `compute_behavioral_signature()` generates unique signatures per graph.

**Why:** Ensures every graph configuration produces distinguishable agent behavior.

**How:** Applies a positive delta to one node and measures the resulting engagement changes across all nodes.

---

## Belief Tracking & Bayesian Updates

### bayesian_update() (`deal_room/committee/belief_tracker.py`)

**What:** Updates stakeholder beliefs based on vendor actions.

**Why:** Provides a principled way to update beliefs given evidence (actions).

**How:**
- **Likelihood mapping:** Action-type to vendor-type likelihoods
  - `send_document(DPA)_proactive`: high on competent/trustworthy/aligned (0.80-0.85)
  - `send_document(security_cert)_proactive`: similar but slightly lower
  - `send_document(roi_model)_to_finance`: Finance-specific likelihoods
  - `direct_message_role_specific`: moderate likelihoods
  - `default`: 0.5 for all types
- **Damping:** Targeted actions get full update (damping=1.0), non-targeted get 0.3
- **Entropy-based confidence:** `confidence = 1.0 - (entropy / LOG2_6)`

### Action Likelihoods

**What:** Predefined likelihood mappings for different action types.

**Why:** Calibrates how different actions shift beliefs in different directions.

**How:** Each action type has likelihood ratios for each vendor type. Bayesian update multiplies prior by likelihood ratio.

---

## Committee Deliberation Engine

### CommitteeDeliberationEngine (`deal_room/committee/deliberation_engine.py`)

**What:** Orchestrates belief propagation after vendor actions.

**Why:** Models the committee discussion that happens after a vendor action.

**How:**
- Takes pre-action beliefs, post-action beliefs, and runs 3 steps of propagation
- Optionally generates natural language deliberation summary via MiniMax LLM
- Returns `DeliberationResult` with updated beliefs, summary, and propagation deltas

### Deliberation Summary Generation

**What:** `generate_deliberation_summary()` creates committee dialogue.

**Why:** Provides interpretable feedback about belief evolution.

**How:**
- Calls MiniMax API with deliberation context
- Falls back gracefully if API unavailable
- Summary includes: target stakeholder, belief shift magnitude, confidence change, other stakeholder deltas

---

## CVaR-Based Risk Preferences

### StakeholderRiskProfile (`deal_room/stakeholders/cvar_preferences.py`)

**What:** Risk profile for each stakeholder archetype.

**Why:** Different stakeholders have different risk tolerances and veto power.

**How:**
- **alpha:** CVaR percentile (0.70-0.95, higher = more risk-averse)
- **tau:** Veto threshold (0.10-0.40)
- **lambda_risk:** Risk aversion weight (0.25-0.70)
- **veto_power:** Boolean for whether stakeholder can veto
- **utility_weights:** Domain-specific priorities (compliance, ROI, feasibility, etc.)
- **uncertainty_domains:** Risk categories per stakeholder

### Stakeholder Archetypes (`deal_room/stakeholders/archetypes.py`)

**What:** Pre-configured risk profiles for 6 stakeholders.

**Why:** Realistic committee composition with diverse risk profiles.

**How:**
- **Legal:** alpha=0.95, tau=0.10, lambda=0.70, veto_power=True (most risk-averse)
- **Finance:** alpha=0.90, tau=0.15, lambda=0.50, veto_power=True
- **TechLead:** alpha=0.80, tau=0.25, lambda=0.30
- **Procurement:** alpha=0.85, tau=0.20, lambda=0.45
- **Operations:** alpha=0.80, tau=0.30, lambda=0.35
- **ExecSponsor:** alpha=0.70, tau=0.40, lambda=0.25, veto_power=True (least risk-averse)

### compute_outcome_distribution()

**What:** Monte Carlo simulation of deal outcomes.

**Why:** Generates the distribution of possible outcomes for CVaR calculation.

**How:**
- **Base success probability:** 0.75 default, 0.80 for compliance/risk domains, 0.70 for cost domains
- **DPA/security cert bonus:** +12-17% for Legal risk domain
- **Price/timeline/liability adjustments:** Modulates outcome based on deal terms
- **Severity-based outcomes:** 3 tiers (mild: 0.6-0.7, moderate: 0.3-0.5, severe: 0.0-0.2)
- **500 samples** default per evaluation

### compute_cvar()

**What:** Conditional Value at Risk calculation.

**Why:** Measures expected loss in the worst `alpha` percentile - captures tail risk.

**How:**
- Sorts outcomes and takes the `(1-alpha)` worst percentile
- Computes average loss in that tail
- **Key property:** Can veto deals with positive EU if tail risk is high

### Veto Mechanism

**What:** Two-stage veto detection in `DealRoomV3._check_for_veto()`.

**Why:** Implements the core research claim that CVaR veto fires even when EU > 0.

**How:**
- **Stage 1 - Precursor detection:** When `cvar_loss > tau * 0.70`, warning issued
- **Stage 2 - Veto trigger:** When `cvar_loss > tau` AND precursor streak ≥ 2 consecutive rounds
- **Veto selection:** Stakeholder with highest `(cvar_loss - tau)` fires veto
- **Terminal rewards:** `veto: -1.0`, `deal_closed: 1.0`, `max_rounds: 0.0`

---

## Reward System

### UtteranceScorer (`deal_room/rewards/utterance_scorer.py`)

**What:** Deterministic world-state-based reward computation with 5 dimensions.

**Why:** Prevents reward hacking by computing rewards from state deltas, not message content.

**How:**
- **goal (weight=0.25):** Authority-weighted belief change + blocker resolution + veto headroom
- **trust (weight=0.20):** Trustworthy mass delta for targeted stakeholder
- **info (weight=0.20):** Entropy reduction across all beliefs
- **risk (weight=0.20):** CVaR improvement for risk-averse stakeholders
- **causal (weight=0.15):** Betweenness centrality of targeted node
- **Lookahead cost:** Exactly 0.07 deducted from goal when lookahead used

### compute_terminal_reward() (`deal_room/rewards/pareto_efficiency.py`)

**What:** Determines terminal reward based on deal outcome.

**Why:** Provides appropriate terminal rewards for different episode endings.

**How:**
- **deal_closed:** +1.0
- **veto:** -1.0
- **max_rounds (Pareto optimal):** 0.0
- **max_rounds (no deal):** 0.0 (from TERMINAL_REWARDS but returns 0.0 in this case)
- **stage_regression:** -0.5 × min(stage_regressions, 3)
- **impasse:** -0.75

### Reward Weights

**What:** Fixed weights for combining 5 reward dimensions.

**Why:** Calibrated to balance multiple objectives.

**How:**
```python
REWARD_WEIGHTS = {
    "goal": 0.25,
    "trust": 0.20,
    "info": 0.20,
    "risk": 0.20,
    "causal": 0.15,
}
```

---

## Lookahead Simulator

### LookaheadSimulator (`deal_room/environment/lookahead.py`)

**What:** Mental state hypothesis generation for robust planning.

**Why:** Allows the agent to anticipate stakeholder responses and CVaR impact before committing to actions.

**How:**
- **Hypothesis generation:** Creates optimistic and pessimistic belief distributions
- **Simulation:** Evaluates action under each hypothesis
- **Worst-case selection:** Returns results from worst-case hypothesis
- **Output:** predicted_responses, predicted_belief_deltas, cvar_impact, graph_information_gain
- **Cost:** 0.07 (LOOKAHEAD_COST) deducted from goal reward

### SimulationResult

**What:** Output of lookahead simulation.

**Why:** Structured output for downstream reward calculation and decision-making.

**How:**
- `predicted_responses`: Dict[str, str] - predicted stakeholder responses
- `predicted_belief_deltas`: Dict[str, float] - predicted belief changes
- `cvar_impact`: Dict[str, float] - predicted CVaR changes
- `graph_information_gain`: float - expected information gain
- `cost`: float - lookahead cost (0.07)

---

## Training Infrastructure

### GRPOTrainer (`deal_room/training/grpo_trainer.py`)

**What:** Group Relative Policy Optimization training harness.

**Why:** Enables RL training with the DealRoom environment.

**How:**
- **Self-play episodes:** Runs episodes with policy adapter acting in environment
- **Batch training:** Accumulates trajectories, updates policy
- **Group relative advantage:** Computes advantages relative to batch peers
- **Checkpointing:** Saves training state and metrics
- **Policy adapters:**
  - `RandomPolicyAdapter`: Random baseline
  - `HeuristicPolicyAdapter`: Rule-based policy with veto awareness
  - `ModelPolicyAdapter`: Wraps arbitrary policy functions

### EpisodeTrajectory

**What:** Records all data from a single episode.

**Why:** Provides complete episode history for training and analysis.

**How:**
- task_id, terminal_outcome, observations, actions, rewards, scalar_rewards
- lookahead_used, prediction_accuracies, lookahead_diagnostics
- weighted_reward, seed

### TrainingMetrics

**What:** Aggregated training metrics per batch.

**Why:** Tracks training progress across dimensions.

**How:**
- Per-dimension rewards (goal, trust, info, risk, causal)
- lookahead_usage_rate, prediction_accuracy
- total_reward, weighted_reward
- episodes_completed, task_mix, terminal_outcomes

---

## Curriculum Generation

### AdaptiveCurriculumGenerator (`deal_room/curriculum/adaptive_generator.py`)

**What:** Scenario selection based on agent performance.

**Why:** Progressive difficulty increase for better learning.

**How:**
- **Difficulty levels:** easy (aligned), frontier (conflicted), hard (hostile_acquisition)
- **Failure analysis:** Detects 6 failure modes:
  - F1: CVaR veto despite positive EU
  - F2: Trust collapse mid-episode
  - F3: Failed graph inference
  - F4: Timeout without coalition
  - F5: Single-dimension reward hacking
  - F6: Authority shift blindness
- **Adaptive selection:** Chooses scenario based on agent_capability_estimate
- **Pool:** 15 scenarios (5 seeds × 3 base configs)

---

## Server Architecture

### FastAPI Server (`server/app.py`)

**What:** Thin HTTP wrapper for the environment.

**Why:** Provides network-accessible API for RL training and web interface.

**How:**
- **Endpoints:**
  - `GET /health` - Service health check
  - `GET /metadata` - Service metadata
  - `POST /reset` - Initialize new episode
  - `POST /step` - Execute action
  - `GET /state` - Get internal state
- **Session management:** Cookie-based session tracking
- **Gradio UI:** Mounted at `/__gradio_ui__`, accessed via `/web`

### DealRoomSessionPool (`server/session_pool.py`)

**What:** Per-client session environment management.

**Why:** Maintains independent environment state per client/browser session.

**How:**
- **Thread-safe:** Uses Lock for concurrent access
- **TTL:** 6-hour session expiry
- **Max sessions:** 128 concurrent sessions
- **Auto-pruning:** Removes expired sessions on reset

### OutputValidator (`server/validator.py`)

**What:** Action normalization and validation.

**Why:** Ensures incoming actions are well-formed.

**How:** Normalizes action fields (target resolution, message truncation, target_ids deduplication)

---

## Data Models

### DealRoomAction (`models.py`)

**What:** Agent action specification.

**Fields:**
- `action_type`: direct_message, send_document, group_proposal, backchannel, exec_escalation, concession
- `target_ids`: List of stakeholder IDs to address
- `message`: Text message (truncated to 1200 chars)
- `documents`: List of {name, content} dicts
- `proposed_terms`: Dict of deal term modifications
- `lookahead`: Optional LookaheadRequest for planning

### DealRoomObservation (`models.py`)

**What:** Agent-facing state observation (18 required fields).

**Required fields:**
- round_number, max_rounds, stakeholders, stakeholder_messages
- engagement_level, engagement_level_delta, engagement_history
- weak_signals, cross_stakeholder_echoes, veto_precursors
- known_constraints, requested_artifacts, approval_path_progress
- deal_momentum, deal_stage, active_blockers, days_to_deadline, done

**Hidden fields (never exposed):**
- G, causal_graph, graph, true_beliefs, belief_distributions
- tau, tau_i, risk_thresholds, cvar_thresholds, edge_weights
- deliberation_transcript, internal_dialogue, u_i, u_ij

### DealRoomState (`models.py`)

**What:** Internal environment state.

**Why:** Tracks all private state for simulation dynamics.

**Key fields:**
- episode_id, step_count, task_id, round_number, max_rounds
- stakeholder_private: per-stakeholder trust/approval/perceived_fit/private_resistance
- hidden_constraints, relationship_edges, commitment_ledger
- offer_state, feasibility_state, active_blockers
- deal_stage, deal_momentum, stage_regressions
- deal_closed, deal_failed, failure_reason, terminal_outcome, veto_stakeholder

---

## Key Implementation Details

### Engagement Noise

**What:** Stochasticity in observed engagement levels.

**Why:** Prevents perfect observability and reward hacking.

**How:**
- Sigma = 0.03 (from OBS_CONFIG)
- Applied via `np.clip(0.5 + normal(0, sigma), 0, 1)`
- Cannot be cancelled by repeating same action

### Cross-Stakeholder Echoes

**What:** Probability that non-targeted stakeholders notice the action.

**Why:** Models information propagation through the graph.

**How:**
- 70% probability per non-targeted stakeholder
- Only generated if action has target_ids
- Structure: `{"from": targeted, "to": sid, "content": "cross_reference"}`

### Weak Signals

**What:** Hints about stakeholder state derived from engagement history.

**Why:** Provides partial information about belief distribution without full transparency.

**How:**
- Generated from engagement level thresholds (>0.7 = high, <0.3 = low)
- Delta-based signals (improving/declining engagement)
- Confidence-based signals (high_uncertainty when confidence < 0.4)
- Format: Dict[str, List[str]] - stakeholder → list of signal tags

### Deal Stage Progression

**What:** Automatic deal stage based on round number.

**Why:** Models natural negotiation progression.

**How:**
- Rounds 0-2: "evaluation"
- Rounds 3-5: "negotiation"
- Rounds 6+: "final_review"

### Offer State Updates

**What:** Track and modify deal terms based on actions.

**Why:** Dynamic deal terms that evolve through negotiation.

**How:**
- Action type effects:
  - `concession`: price × 0.90, liability_cap × 1.10
  - `exec_escalation`: price × 1.10, liability_cap × 0.25
  - `walkaway_signal`: price × 1.05, liability_cap × 0.35
- Document effects:
  - "dpa" → has_dpa = True
  - "security"/"cert" → has_security_cert = True
  - "implementation"/"timeline" → timeline_weeks = min(current, 12)
  - "roi" → price = max(75000, price × 0.95)

---

## Dependencies

**Core:**
- openenv-core >= 0.2.2
- fastapi >= 0.115.0
- numpy >= 1.24.0
- openai >= 1.0.0
- pydantic >= 2.0.0
- scikit-learn >= 1.5.0
- uvicorn >= 0.24.0

**Optional:**
- gradio (for web UI)
- MiniMax API key (for LLM deliberation summaries)

---

## File Structure

```
deal_room/
├── __init__.py
├── environment/
│   ├── __init__.py
│   ├── constants.py          # REWARD_WEIGHTS, TERMINAL_REWARDS, etc.
│   ├── dealroom_v3.py        # Main environment class
│   ├── lookahead.py          # Lookahead simulation
│   └── llm_client.py         # MiniMax API client
├── committee/
│   ├── __init__.py
│   ├── belief_tracker.py     # Bayesian belief updates
│   ├── causal_graph.py       # Graph structure & propagation
│   └── deliberation_engine.py # Committee deliberation
├── stakeholders/
│   ├── __init__.py
│   ├── archetypes.py         # Stakeholder risk profiles
│   └── cvar_preferences.py   # CVaR computation
├── rewards/
│   ├── __init__.py
│   ├── utterance_scorer.py   # 5-dim reward computation
│   └── pareto_efficiency.py  # Terminal reward & Pareto check
├── curriculum/
│   ├── __init__.py
│   └── adaptive_generator.py # Curriculum generation
└── training/
    ├── __init__.py
    └── grpo_trainer.py       # GRPO training harness

server/
├── __init__.py
├── app.py                    # FastAPI server
├── session_pool.py           # Per-client session management
├── validator.py              # Action normalization
├── gradio_custom.py          # Custom Gradio components
├── stakeholders.py
├── scenarios.py
├── claims.py
├── semantics.py
└── grader.py

models.py                     # Pydantic data models
```
