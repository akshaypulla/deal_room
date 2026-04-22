# DealRoom v3.6 — Architecture Document

**Version:** 3.6  
**Generated:** 2026-04-21  
**Purpose:** Research-grade OpenEnv RL environment for enterprise B2B negotiation

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Core Environment (`deal_room/environment/`)](#3-core-environment-deal_roomenvironment)
4. [Committee Module (`deal_room/committee/`)](#4-committee-module-deal_roomcommittee)
5. [Rewards Module (`deal_room/rewards/`)](#5-rewards-module-deal_roomrewards)
6. [Stakeholders Module (`deal_room/stakeholders/`)](#6-stakeholders-module-deal_roomstakeholders)
7. [Curriculum Module (`deal_room/curriculum/`)](#7-curriculum-module-deal_roomcurriculum)
8. [Training Module (`deal_room/training/`)](#8-training-module-deal_roomtraining)
9. [Server Module (`server/`)](#9-server-module-server)
10. [Models (`models.py`)](#10-models-modelspy)
11. [Jupyter Notebooks](#11-jupyter-notebooks)
12. [Bug Fixes Applied in v3.6](#12-bug-fixes-applied-in-v36)
13. [Design Decisions and Rationale](#13-design-decisions-and-rationale)

---

## 1. Executive Summary

**DealRoom v3.6** is a multi-stakeholder enterprise B2B negotiation environment designed for reinforcement learning research. It models a vendor negotiating with a committee of 6 decision-makers (Legal, Finance, TechLead, Procurement, Operations, ExecSponsor) across three scenario types: `aligned`, `conflicted`, and `hostile_acquisition`.

### Key Properties
- **Partially Observable:** Stakeholders have hidden constraints and private state
- **Causal Graph Inference:** Actions targeting one stakeholder ripple through the committee via belief propagation
- **CVaR-based Risk Management:** Committee members evaluate deals using Conditional Value at Risk (CVaR) at their individual alpha levels
- **5-Dimensional Reward:** goal, trust, info, risk, causal — all computed from world state (no LLM required for scoring)
- **Lookahead Mechanism:** Optional planning action that simulates outcomes before committing
- **Veto Mechanism:** High-risk stakeholders can block the deal even when expected utility is positive

### What Makes This Architecture Unique
The environment combines three sophisticated mechanisms that are rarely found together:
1. **Bayesian belief tracking** with causal graph-based propagation
2. **CVaR terminal grading** that prevents reward hacking
3. **Adaptive curriculum generation** for curriculum-based RL training

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            DealRoom v3.6                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐   │
│  │  Agent/Policy  │────▶│   DealRoomV3     │◀────│  GRPOTrainer        │   │
│  │  (acts)        │     │  Environment     │     │  (trains policy)    │   │
│  └─────────────────┘     └──────────────────┘     └─────────────────────┘   │
│                                   │                                           │
│         ┌──────────────────────────┼──────────────────────────┐              │
│         ▼                          ▼                          ▼              │
│  ┌─────────────┐          ┌───────────────┐          ┌──────────────┐        │
│  │ Committee   │          │   Rewards     │          │  Curriculum │        │
│  │ - CausalGraph│          │ - Utterance   │          │  Adaptive   │        │
│  │ - BeliefTracker│       │   Scorer      │          │  Generator  │        │
│  │ - Deliberation│         │ - Pareto Eff. │          └──────────────┘        │
│  └─────────────┘          └───────────────┘                                    │
│                                   │                                           │
│                          ┌───────────────┐                                    │
│                          │ Stakeholders  │                                    │
│                          │ - CVaR Prefs  │                                    │
│                          │ - Archetypes   │                                    │
│                          └───────────────┘                                    │
└─────────────────────────────────────────────────────────────────────────────┘

                        Server (FastAPI + Gradio)
┌─────────────────────────────────────────────────────────────────────────────┐
│  ┌───────────┐  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐   │
│  │ app.py    │  │ DealRoomEnv  │  │ SessionPool  │  │  Gradio Custom UI │   │
│  │ (FastAPI) │  │ (V2.5 hybrid)│  │ (state mgmt) │  │  (deal_room_lab)  │   │
│  └───────────┘  └──────────────┘  └──────────────┘  └───────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Dual Environment Design

DealRoom v3.6 ships two distinct environment implementations:

| Aspect | `DealRoomV3` (v3) | `DealRoomEnvironment` (V2.5) |
|--------|------------------|------------------------------|
| **Location** | `deal_room/environment/dealroom_v3.py` | `server/deal_room_environment.py` |
| **Scoring** | 5D world-state scorer | Dense milestone rewards |
| **Committee** | Causal graph + belief propagation | Relationship propagation engine |
| **Risk Model** | CVaR per stakeholder | Private state tracks |
| **Terminal Grading** | CVaR veto + pareto efficiency | CCIGrader |
| **Observation** | Engagement noise + weak signals | Approval bands |
| **Use Case** | RL training + research | Web UI demo + API |

Both share the same Pydantic models (`models.py`) for actions and observations.

---

## 3. Core Environment (`deal_room/environment/`)

### 3.1 `dealroom_v3.py` — Main RL Environment

**Purpose:** OpenEnv-compatible RL environment for negotiation with causal committee dynamics.

**Key Classes:**
- `DealRoomV3` — Main environment class implementing `reset()` and `step()`
- `ScenarioConfig` — Configuration dataclass for episode setup
- `StateSnapshot` — Immutable snapshot of environment state for reward computation

**Environment Constants** (`constants.py`):
```python
REWARD_WEIGHTS = {"goal": 0.30, "trust": 0.18, "info": 0.18, "risk": 0.17, "causal": 0.17}
TERMINAL_REWARDS = {"deal_closed": 1.0, "veto": -1.0, "max_rounds": 0.0, "stage_regression": -0.5, "impasse": -0.75}
DEFAULT_MAX_ROUNDS = 10
SUMMARY_TIMEOUT_SECONDS = 5.0
VETO_WARNING_THRESHOLD_RATIO = 0.70
```

**Three Scenario Types:**
1. **`aligned`:** Cooperative negotiation, 2-3 stakeholders, high observability
2. **`conflicted`:** Mixed-motive negotiation, 3-4 stakeholders, medium observability
3. **`hostile_acquisition`:** Adversarial negotiation, 4 stakeholders, low observability, authority shift events

**Standard Stakeholders:**
```python
STANDARD_STAKEHOLDERS = ["Legal", "Finance", "TechLead", "Procurement", "Operations", "ExecSponsor"]
STANDARD_HIERARCHY = {"Legal": 3, "Finance": 3, "TechLead": 2, "Procurement": 2, "Operations": 2, "ExecSponsor": 5}
```

**Initial Belief Distributions** are scenario-dependent. For `conflicted` scenario, different stakeholder clusters receive different priors:
- `cost_cluster` (Finance, Procurement): Higher weight on cost concerns
- `risk_cluster` (Legal, Compliance): Higher weight on risk/deceptive concerns
- `impl_cluster` (TechLead, Operations): Higher weight on competence/aligned concerns

**Reset Flow:**
1. Initialize RNG with seed
2. Sample causal graph based on scenario type
3. Initialize belief distributions for each stakeholder
4. Set initial noisy engagement levels
5. Initialize state with scenario-specific offer terms
6. Build and return observation

**Step Flow:**
1. Normalize action (resolve target IDs)
2. Run lookahead simulation if requested
3. Capture state snapshot (pre-action)
4. Apply action to offer state
5. Update deal stage
6. Update beliefs via Bayesian update
7. Run committee deliberation (belief propagation)
8. Update noisy engagement levels
9. Generate stakeholder responses
10. Compute reward via UtteranceScorer
11. Evaluate committee risk (CVaR computation)
12. Check for veto trigger
13. Compute terminal reward if done
14. Build observation and return

**Observation Components:**
- `engagement_level`: Noisy measure of stakeholder receptiveness
- `weak_signals`: Environmental hints about stakeholder concerns
- `veto_precursors`: Warning signals from high-risk stakeholders
- `cross_stakeholder_echoes`: Information flow indicators
- `engagement_history`: Sliding window of recent engagement
- `active_blockers`: Currently blocking stakeholders

### 3.2 `lookahead.py` — Planning Simulator

**Purpose:** Simulates outcomes of proposed actions before committing, enabling minimax-style robustness.

**Key Class:** `LookaheadSimulator`

**How It Works:**
1. Takes a draft action and current beliefs
2. Generates optimistic and pessimistic hypotheses about target stakeholder
3. Simulates each hypothesis through `_simulate_one_hypothesis()`
4. Returns the **worst-case** outcome (minimax strategy)

**SimulationResult Contains:**
- `predicted_responses`: Expected stakeholder response text
- `predicted_belief_deltas`: Expected belief shifts
- `cvar_impact`: Expected risk impact
- `graph_information_gain`: Causal graph learning value
- `cost`: LOOKAHEAD_COST (0.07)

**Why Worst-Case?** Using worst-case instead of expected case makes the agent robust to adversarial stakeholder behavior.

### 3.3 `llm_client.py` — MiniMax API Client

**Purpose:** Interface to MiniMax M2.5 model for generating deliberation summaries and stakeholder responses.

**Key Components:**
- `llm_call_text()` — Main API call function with retry logic
- `RetryPolicy` — Configurable backoff with jitter
- `LLMErrorType` — Enum of error categories
- `LLMCallStats` — Statistics tracking
- `classify_error()` — Error type detection

**Error Handling:**
- Auto-recoverable errors (timeout, connection reset, 5xx): Automatic retry with exponential backoff
- Rate limit errors: Wait with countdown display
- Auth errors: Interactive pause for manual resolution
- Unknown errors: Skip with fallback

**Configuration via Environment Variables:**
- `MINIMAX_API_KEY` — API key (required for LLM features)
- `MINIMAX_BASE_URL` — API endpoint (default: `https://api.minimax.chat/v1`)
- `MINIMAX_MODEL` — Model name (default: `MiniMax-Text-01`)
- `DEALROOM_LLM_INTERACTIVE` — Enable interactive error handling

---

## 4. Committee Module (`deal_room/committee/`)

### 4.1 `causal_graph.py` — Committee Influence Network

**Purpose:** Models how actions and beliefs propagate through the stakeholder network.

**Key Class:** `CausalGraph`
- `nodes`: List of stakeholder IDs
- `edges`: Dict mapping (source, dest) → influence weight
- `authority_weights`: Normalized authority per stakeholder
- `scenario_type`: Which scenario configuration to use

**Scenario Parameters:**
```python
SCENARIO_PARAMS = {
    "aligned": {"base_edge_probability": 0.30, "intra_cluster_boost": 0.40, ...},
    "conflicted": {"base_edge_probability": 0.45, "intra_cluster_boost": 0.50, ...},
    "hostile_acquisition": {"base_edge_probability": 0.60, "intra_cluster_boost": 0.25, ...},
}
```

**Functional Clusters:**
- `cost`: Finance, Procurement
- `risk`: Legal, Compliance
- `implementation`: TechLead, Operations
- `authority`: ExecSponsor

**Key Functions:**
- `sample_graph()` — Generate random causal graph for scenario
- `BeliefDistribution` — Dataclass tracking belief state
- `propagate_beliefs()` — Multi-step belief propagation through graph
- `apply_positive_delta()` — Transfer probability mass between belief types
- `get_betweenness_centrality()` — Measure of stakeholder's network importance
- `compute_behavioral_signature()` — Expected response pattern for targeting
- `compute_engagement_level()` — positive_mass - negative_mass

**BeliefDistribution States:**
- `positive_mass()`: Sum of competent + trustworthy + aligned
- `negative_mass()`: Sum of incompetent + deceptive + misaligned
- `confidence`: 1.0 - (entropy / LOG2_6)

### 4.2 `belief_tracker.py` — Bayesian Belief Updates

**Purpose:** Updates stakeholder beliefs based on vendor actions using Bayesian inference.

**Likelihood Table (ACTION_LIKELIHOODS):**
Each action type maps to likelihood ratios for each belief dimension. For example:
- `send_document(DPA)_proactive` → high likelihood for competent/trustworthy/aligned
- `direct_message_role_specific` → moderate likelihood boost
- Default → 0.5/0.5 (no information gain)

**Bayesian Update Formula:**
```python
new_dist[vendor_type] = prior_prob * (1.0 + damping * (likelihood - 1.0))
```
Where `damping = 1.0` for targeted actions, `0.7` for non-targeted.

**Confidence Update:**
Entropy of belief distribution is computed, then:
```python
confidence = 1.0 - (entropy / LOG2_6)
```

### 4.3 `deliberation_engine.py` — Committee Deliberation

**Purpose:** Simulates committee discussion where initial beliefs ripple through the causal graph.

**Key Class:** `CommitteeDeliberationEngine`
- Takes vendor action and resulting beliefs
- Runs multi-step belief propagation (3 steps for aligned/conflicted, 4 for hostile)
- Optionally generates natural language summary via MiniMax

**DeliberationResult:**
- `updated_beliefs`: Propagated beliefs after deliberation
- `summary_dialogue`: LLM-generated summary of committee discussion
- `propagation_deltas`: Per-stakeholder belief shifts from propagation

**Why Belief Propagation Matters:**
When vendor action improves Legal's view of the vendor, Finance may notice (through the relationship graph) and update their beliefs as well. This captures the "ripple effect" in committee dynamics.

---

## 5. Rewards Module (`deal_room/rewards/`)

### 5.1 `utterance_scorer.py` — World-State Reward Computation

**Purpose:** Computes 5-dimensional reward signal entirely from environment state (no LLM needed).

**Key Class:** `UtteranceScorer`

**Why Deterministic World-State Scoring?**
- Reproducible results enable reliable policy comparison
- Immune to LLM API failures or inconsistency
- Prevents reward hacking via prompt engineering
- Aligns with academic rigor requirements for RL research

**The Five Dimensions:**

1. **goal (weight: 0.30)**
   - Measures: Approval improvement weighted by authority, blocker resolution, veto headroom improvement
   - Formula: Authority-weighted belief delta + blocker_score + veto_score

2. **trust (weight: 0.18)**
   - Measures: Targeted stakeholder's positive mass and trustworthy mass delta
   - Formula: `0.6 * pm_delta + 0.4 * tw_delta` for targeted stakeholders

3. **info (weight: 0.18)**
   - Measures: Entropy reduction in belief distributions
   - Formula: `(H_before - H_after) / LOG2_6` normalized to [0,1]

4. **risk (weight: 0.17)**
   - Measures: CVaR improvement for risk-averse stakeholders
   - Formula: Relative CVaR improvement for stakeholders with lambda_risk > 0.30

5. **causal (weight: 0.17)**
   - Measures: Causal graph information gain from targeting decision
   - Formula: Betweenness centrality of targeted node normalized

**Tanh Transformation:**
All dimensions are transformed via `0.5 + 0.5 * tanh(G * raw * SCALE)` where:
- `REWARD_GAIN = 3.0`
- `REWARD_SCALE = 6.0`

This ensures bounded [0, 1] output while preserving ranking.

**Lookahead Cost:**
When `lookahead_used=True`, goal score is reduced by `LOOKAHEAD_COST = 0.07` to discourage unnecessary planning.

### 5.2 `pareto_efficiency.py` — Terminal Reward Determination

**Purpose:** Determines terminal reward when episode ends.

**Key Functions:**
- `check_pareto_optimality()` — Check if deal is on Pareto frontier
- `compute_terminal_reward()` — Assign terminal reward based on outcome
- `get_pareto_frontier_stakeholders()` — List non-dominated stakeholders

**Terminal Outcomes:**
- `deal_closed`: +1.0 reward
- `veto_by_{stakeholder}`: -1.0 reward
- `max_rounds_pareto`: 0.0 reward (acceptable timeout)
- `max_rounds_no_deal`: 0.0 reward
- `stage_regression_{N}`: -0.5 * min(N, 3) penalty
- `impasse`: -0.75 reward

**Why Pareto Check?** A deal that exhausts rounds without veto is only rewarded if it's Pareto-optimal — otherwise the agent could "game" the system by making all stakeholders equally miserable.

---

## 6. Stakeholders Module (`deal_room/stakeholders/`)

### 6.1 `archetypes.py` — Stakeholder Risk Profiles

**Purpose:** Define the 6 standard committee member archetypes with their risk preferences.

**The Six Archetypes:**

| Stakeholder | Alpha | Tau | Lambda | Veto | Primary Concern |
|-------------|-------|-----|--------|------|------------------|
| Legal | 0.95 | 0.10 | 0.70 | Yes | Compliance coverage, liability limitation |
| Finance | 0.90 | 0.15 | 0.50 | Yes | ROI clarity, payment terms, cost predictability |
| TechLead | 0.80 | 0.25 | 0.30 | No | Implementation feasibility, integration quality |
| Procurement | 0.85 | 0.20 | 0.45 | No | Contract compliance, price competitiveness |
| Operations | 0.80 | 0.30 | 0.35 | No | Operational continuity, timeline |
| ExecSponsor | 0.70 | 0.40 | 0.25 | Yes | Strategic alignment, organizational consensus |

**Alpha (CVaR percentile):** Higher alpha means more risk-averse. Legal at 0.95 evaluates the worst 5% of outcomes.

**Tau (Veto threshold):** CVaR loss must exceed this before veto is possible. Legal's 0.10 tau means even small tail risk triggers warning.

**Lambda (Risk weight):** Used in expected utility calculation. Balances expected value vs CVaR.

### 6.2 `cvar_preferences.py` — CVaR Preference Model

**Purpose:** Implements the Conditional Value at Risk (CVaR) preference model for stakeholder decision-making.

**Key Functions:**
- `compute_outcome_distribution()` — Monte Carlo simulation of deal outcomes
- `compute_cvar()` — Calculate CVaR at stakeholder's alpha level
- `compute_expected_utility()` — Mean outcome
- `evaluate_deal()` — Combined expected utility and CVaR
- `check_veto_trigger()` — Compare CVaR loss against tau threshold
- `get_observable_signals()` — Environmental indicators of stakeholder concerns
- `compute_deal_quality_score()` — Weighted combination: `(1-lambda)*E[U] - lambda*CVaR`

**Outcome Distribution Simulation:**
- Base success probability depends on domain (compliance: 0.80-0.92, cost: 0.70, implementation: 0.75)
- Modifiers based on deal terms (liability cap, DPA presence, security cert)
- Outcome in [0, 1] representing deal quality

**CVaR Calculation:**
```python
sorted_outcomes = np.sort(outcomes)
cutoff_index = int(len(sorted_outcomes) * (1 - alpha))
cvar = mean of lowest (alpha * 100)% of outcomes
```

**Why CVaR over Expected Utility?**
CVaR captures tail risk — the worst-case scenarios that actually matter for high-stakes enterprise deals. A deal with high expected utility but 10% chance of complete failure is worse than a deal with slightly lower expected utility but no tail risk.

---

## 7. Curriculum Module (`deal_room/curriculum/`)

### 7.1 `adaptive_generator.py` — Curriculum-Based Training

**Purpose:** Generates adaptive curricula that target the agent's weaknesses.

**Key Class:** `AdaptiveCurriculumGenerator`

**Configuration:**
```python
@dataclass
class CurriculumConfig:
    analysis_batch_size: int = 10
    easy_ratio: float = 0.20
    frontier_ratio: float = 0.60
    hard_ratio: float = 0.20
    max_graph_variation: float = 0.3
```

**Failure Mode Taxonomy:**
- F1: CVaR veto despite positive expected outcome
- F2: Trust collapse mid-episode
- F3: Failed graph inference
- F4: Timeout without coalition formation
- F5: Single-dimension reward hacking
- F6: Authority shift blindness

**Adaptation Logic:**
1. Analyze batch trajectories for failure patterns
2. Estimate agent capability (weighted recent rewards)
3. Select next scenario difficulty based on capability:
   - < 0.3 capability → random (exploration)
   - 0.3-0.5 capability → easy scenarios
   - 0.5-0.75 capability → frontier (challenging but fair)
   - > 0.75 capability → hard scenarios
4. Apply failure-specific modifiers (e.g., F1 > 0.3 frequency → reduce CVaR tension)

---

## 8. Training Module (`deal_room/training/`)

### 8.1 `grpo_trainer.py` — GRPO Training Harness

**Purpose:** Group Relative Policy Optimization-style training for DealRoom agent policies.

**Key Classes:**
- `GRPOTrainer` — Main training coordinator
- `EpisodeTrajectory` — Episode data structure
- `TrainingMetrics` — Aggregated metrics per batch
- `PolicyAdapter` — Protocol for pluggable policies
- `RandomPolicyAdapter` — Random baseline
- `HeuristicPolicyAdapter` — Rule-based policy with flag toggling
- `ModelPolicyAdapter` — Wrapper for custom policy functions

**PolicyAdapter Protocol:**
```python
class PolicyAdapter(Protocol):
    name: str
    def act(self, observation: DealRoomObservation, rng: np.random.Generator) -> DealRoomAction: ...
    def update_from_batch(self, trajectories: Sequence[EpisodeTrajectory]) -> None: ...
    def state_dict(self) -> Dict[str, Any]: ...
    def load_state_dict(self, state: Dict[str, Any]) -> None: ...
```

**Training Flow:**
1. Generate scenario via curriculum generator
2. Run episode with current policy
3. Collect trajectory (observations, actions, rewards)
4. Compute batch metrics
5. Update policy via `update_from_batch()`
6. Save checkpoint
7. Analyze failures for curriculum update

**GRPO Advantage Computation:**
```python
advantage[i] = (reward[i] - mean(group_rewards)) / (std(group_rewards) + 1e-8)
```
Group-relative advantage prevents reward hacking by ensuring relative improvement.

**Key Metrics Tracked:**
- Per-dimension reward means
- Lookahead usage rate
- Prediction accuracy
- Terminal outcome distribution
- Task mix (which scenarios trained on)

---

## 9. Server Module (`server/`)

### 9.1 `app.py` — FastAPI Application

**Purpose:** HTTP API wrapper around DealRoom environment.

**Endpoints:**
- `GET /` → Redirect to `/web`
- `GET /web` → Main UI (Gradio iframe)
- `GET /health` → Health check with task list
- `GET /metadata` → Service metadata
- `POST /reset` → Start new episode, returns observation
- `POST /step` → Execute action, returns (observation, reward, done, info)
- `GET /state` → Current state query

**Session Management:**
Sessions are identified by `session_id` and tracked via cookie (`dealroom_session_id`) or header (`x-session-id`).

**Gradio UI Setup:**
The app tries to mount a Gradio interface at `/__gradio_ui__`:
1. First tries `server.gradio_standalone` (custom standalone)
2. Falls back to OpenEnv's built-in Gradio UI
3. If Gradio unavailable, serves unavailable message at `/ui`

### 9.2 `deal_room_environment.py` — V2.5 Hybrid Environment

**Purpose:** Alternative environment implementation with different dynamics.

**Key Differences from V3:**
- Uses `StakeholderEngine` for relationship propagation
- Semantic analyzer for intent/tone detection
- Commitment ledger for tracking claimed commitments
- Stage progression: evaluation → negotiation → legal_review → final_approval → closed
- Dense milestone rewards for constraint discovery and resolution
- CCIGrader for terminal scoring

**Constraint System:**
Four hidden constraints that must be discovered and resolved:
- `budget_ceiling`: Price ≤ $185,000 (requires ROI model)
- `delivery_window`: Timeline ≤ 16 weeks (requires implementation timeline)
- `compliance_addendum`: Must include "gdpr" (requires DPA)
- `supplier_process`: Must include "named_support_lead" (requires vendor packet)

**Stage Advancement Logic:**
- evaluation → negotiation: Contact all mandatory stakeholders OR discover constraint
- negotiation → legal_review: All mandatory contacted + all constraints known
- legal_review → final_approval: All mandatory workable + all artifacts requested fulfilled
- final_approval → closed: Deal feasible + no active blockers

### 9.3 `session_pool.py` — Session State Management

**Purpose:** Manages multiple concurrent environment instances per browser session.

**DealRoomSessionPool:**
- Thread-safe session management
- TTL-based session expiration (6 hours default)
- LRU-style eviction when max sessions reached
- Lazy environment creation

### 9.4 `stakeholders.py` — Stakeholder Relationship Engine

**Purpose:** Models how stakeholder relationships affect approval dynamics.

**Key Class:** `StakeholderEngine`

**Document Effects:**
```python
DOCUMENT_EFFECTS = {
    "roi_model": {"finance": 0.10, "executive_sponsor": 0.06, "procurement": 0.04},
    "implementation_timeline": {"technical": 0.09, "operations": 0.10, ...},
    "security_cert": {"technical": 0.06, "legal_compliance": 0.10, ...},
    "dpa": {"legal_compliance": 0.12, "procurement": 0.04},
    ...
}
```

**Relationship Propagation:**
- Alliance edges: Approvals spread between aligned stakeholders
- Conflict edges: Resistance increases when conflict partner worsens
- Sponsor edges: Executive support boosts approval

**Approval Bands:**
- `blocker`: resistance > 0.65 OR approval < 0.48
- `neutral`: 0.48 ≤ approval < 0.62
- `workable`: 0.62 ≤ approval < 0.72
- `supporter`: approval ≥ 0.72

### 9.5 `semantics.py` — Semantic Analysis Engine

**Purpose:** Analyzes messages for intent, tone, and semantic content.

**Three Backends (in priority order):**
1. **Embedding:** Sentence transformer model (`paraphrase-MiniLM-L3-v2`)
2. **TF-IDF:** Sklearn vectorizer fallback
3. **Lexical:** Jaccard similarity on token sets

**Intent Bank (12 categories):**
discover_budget, discover_timeline, discover_compliance, reassure, pressure, close_attempt, share_roi, share_implementation, share_security, share_dpa, share_vendor_packet, share_support_plan

**Tone Bank (6 categories):**
collaborative, credible, specific, pushy, evasive, adaptive

**Artifact Aliases:**
Maps document types to common synonyms (e.g., "roi_model" → ["roi", "business case", "payback"])

**Claim Extraction:**
Parses messages for numeric claims (price, timeline) and qualitative claims (security_posture, liability, support_level, implementation_commitment).

### 9.6 `validator.py` — Output Validation

**Purpose:** Normalizes and validates LLM-produced actions.

**Valid Action Types:**
direct_message, group_proposal, backchannel, send_document, concession, walkaway_signal, reframe_value_prop, exec_escalation

**Validation Modes:**
- `strict`: Reject unknown targets, filter invalid proposed_terms
- `permissive` (default): Accept any target, pass through terms

**JSON Extraction:**
Tries multiple regex patterns to extract JSON from LLM output:
```
```json\n...\n```
```\n...\n```
{...}
```

### 9.7 `grader.py` — Terminal Grader (V2.5)

**Purpose:** Computes terminal score for V2.5 environment.

**Score Components:**
- Approval completeness (35%): Fraction of mandatory stakeholders approving
- Constraint satisfaction (25%): Resolved constraints / total constraints
- Term feasibility (15%): Penalty for infeasible terms
- Relationship durability (15%): Average trust - permanent mark penalty
- Efficiency (10%): Time remaining relative to max rounds

**Score Range:** [0.01, 0.99] — Never exactly 0 or 1 to enable gradient-based learning

### 9.8 `claims.py` — Commitment Ledger

**Purpose:** Tracks semantic commitments made during negotiation and detects contradictions.

**Numeric Tolerances:**
- price: 8% tolerance
- timeline_weeks: 15% tolerance

**Contradiction Detection:**
- Numeric: |new_value - old_value| / old_value > tolerance
- Polarity: value or polarity changed for polarity slots

### 9.9 `scenarios.py` — Scenario Generation

**Purpose:** Generates seeded episode configurations with dynamic stakeholders.

**Scenario Templates:**
Each scenario defines:
- `max_rounds`, `days_to_deadline`
- `stakeholder_count` range, `constraint_count`
- `edge_count` range (relationship graph)
- `base_tracks`: Initial trust/approval/resistance ranges
- `roles`: Which archetypes participate
- `constraint_pool`: Which constraints are available
- `observability`: How much is visible initially (high/medium/low)
- `event_round`: When authority shift occurs (hostile only)

**Relationship Templates:**
Pre-defined alliance, conflict, and sponsor relationships between roles

### 9.10 `gradio_custom.py` — Custom Gradio UI

**Purpose:** Visual "DealRoom Lab" interface with stakeholder round table visualization.

**UI Components:**
- Round table with stakeholder seats (positioned around center)
- Seat color indicates status: aligned (green), blocking (red), uncertain (amber)
- Popup cards showing stakeholder messages and requests
- Action chips for quick responses
- Score panel with delta display
- Weak signals display
- "Why this score?" breakdown panel

**CSS Styling:**
Dark theme with orange (#FF6A00) accent color, custom seat styling, popup animations

### 9.11 `walkthrough_data.py` — Guided Walkthrough Data

**Purpose:** Curated walkthrough for medium-difficulty scenario demonstrating best practices.

**Steps:**
1. Map the buying committee (identify blockers)
2. Finance first (ROI model)
3. Legal unblocks stage progression (DPA)
4. Operations reveals hidden constraint (implementation timeline)
5. Procurement completes process path (vendor packet)
6. Close only when feasibility is green (group proposal)

---

## 10. Models (`models.py`)

**Purpose:** Pydantic model definitions shared across all modules.

**DealRoomAction:**
```python
action_type: str  # direct_message, send_document, etc.
target: str       # "all" or comma-separated or role name
target_ids: List[str]  # Resolved stakeholder IDs
message: str       # Content (truncated to 1200 chars)
documents: List[Dict]  # [{"type": "...", "name": "..."}]
proposed_terms: Optional[Dict]  # {price: 100000, ...}
lookahead: Optional[LookaheadRequest]  # Planning action
```

**DealRoomObservation:**
Contains full state: round_number, max_rounds, stakeholders, stakeholder_messages, engagement_level, weak_signals, veto_precursors, active_blockers, deal_stage, deal_momentum, etc.

**DealRoomState:**
Full episode state including private stakeholder state, hidden_constraints, commitment_ledger, offer_state, feasibility_state, etc.

**LookaheadRequest:**
```python
action_draft: DealRoomAction  # Proposed action to simulate
n_hypotheses: int = 2         # Number of hypothesis branches
depth: int = 2                 # Simulation depth
```

**SimulationResult (models.py):**
Alternative SimulationResult for API compatibility (differs from `lookahead.py`)

---

## 11. Jupyter Notebooks

### 11.1 `01_heuristic_training.ipynb` — Rule-Based Baseline

**Purpose:** Demonstrates that heuristic (flag-toggling) training has limited expressiveness.

**Key Finding:** HeuristicPolicyAdapter only has 3 learnable flags (enable_lookahead, prefer_concessions, trained) — cannot learn continuous behaviors.

### 11.2 `02_neural_training.ipynb` — Neural Network Policy

**Purpose:** Shows how a PyTorch MLP policy can learn via REINFORCE-style gradient updates.

**Architecture:**
```python
input_dim=128 → hidden_dim=256 → hidden_dim=256 → action_dim=32
```

**Observation Embedding:**
Flattens key observation features (round_number, veto_precursors, active_blockers, engagement_levels, etc.) to fixed-size vector.

### 11.3 `03_hybrid_training.ipynb` — Hybrid Policy

**Purpose:** Combines neural and heuristic policies with confidence-based selection.

**HybridPolicyAdapter:**
- Uses neural policy when confidence >= threshold
- Falls back to heuristic when:
  - Neural hasn't trained enough (< 50 steps)
  - Veto precursors present (high risk)
  - Early rounds (exploration phase)

### 11.4 `04_validation_dashboard.ipynb` — Environment Visualization

**Purpose:** Generates visualizations for causal graphs, learning curves, and reward progression.

**Visualizations:**
1. Causal graph structure (NetworkX graph)
2. Q-value convergence plots
3. Reward progression across training

### 11.5 `deal_room/training/grpo_colab.ipynb` — Colab Training Runbook

**Purpose:** Step-by-step guide for running GRPO training on Google Colab.

**Steps:**
1. Install dependencies
2. Configure MiniMax API
3. Initialize trainer
4. Run 50 episodes
5. Plot learning curves
6. Evaluate final policy

---

## 12. Bug Fixes Applied in v3.6

### Bug 1: LOG2_6 Constant Error (CRITICAL)

**File:** `deal_room/rewards/utterance_scorer.py` and `deal_room/committee/belief_tracker.py`

**Problem:** `math.log(2, 6)` computes log base 2 of 2 (which equals 1), not log base 6 of 2.

**Wrong:** `LOG2_6 = math.log(2, 6)` → 0.3868528072345416
**Correct:** `LOG2_6 = math.log(6) / math.log(2)` → 2.584962500721156

**Impact:** 
- Confidence calculations were incorrect (max confidence would be ~15% of true value)
- Information entropy reduction scoring was off by factor of ~6.7x
- This affected belief update confidence and UtteranceScorer info dimension

**Verification:**
```python
>>> import numpy as np
>>> np.log2(6)  # Ground truth
2.584962500721156
```

---

## 13. Design Decisions and Rationale

### 13.1 Why World-State Scoring?

**Decision:** Compute all rewards from environment state, not from LLM evaluation.

**Rationale:**
1. **Reproducibility:** Same seed + same action = same reward
2. **No API dependency:** Tests run without network access
3. **No reward hacking:** Agent cannot manipulate LLM prompts to inflate scores
4. **Academic rigor:** Standard in RL research papers

**Trade-off:** Less nuanced than LLM-based evaluation, but more rigorous.

### 13.2 Why CVaR over Expected Utility?

**Decision:** Use CVaR (Conditional Value at Risk) for stakeholder risk evaluation.

**Rationale:**
1. Enterprise deals have fat-tail risk distributions
2. Expected utility can be dominated by small probability catastrophic outcomes
3. CVaR directly models "what's the worst case and how bad is it?"
4. Aligns with real-world risk management practices (VaR/CVaR in finance)

**Trade-off:** CVaR is harder to optimize for, but produces more robust policies.

### 13.3 Why Duplicate Classes?

**Observation:** `BeliefDistribution` and `SimulationResult` appear in multiple modules.

**Rationale:**
1. **Encapsulation:** Each module is self-contained with its own data structures
2. **No cross-imports:** lookahead.py doesn't need committee imports and vice versa
3. **Testability:** Each module can be tested in isolation

**Trade-off:** Slight redundancy but maximum decoupling.

### 13.4 Why Dual Environment?

**Decision:** Maintain both `DealRoomV3` (v3) and `DealRoomEnvironment` (V2.5).

**Rationale:**
1. **V3:** Research-focused, causal graph inference, CVaR scoring
2. **V2.5:** Demo-focused, relationship propagation, dense milestone rewards
3. **Shared models:** Both use same Pydantic models for interoperability

**Trade-off:** Dual maintenance burden but serves both research and demo use cases.

### 13.5 Why Belief Propagation?

**Decision:** Model committee dynamics as belief propagation through causal graph.

**Rationale:**
1. **Realistic ripple effects:** Action toward one stakeholder affects others
2. **Coalition formation:** Groups of aligned stakeholders emerge naturally
3. **Authority dynamics:** High-authority stakeholders influence more

**Trade-off:** More complex than independent stakeholder models, but captures realistic committee behavior.

---

## Appendix A: File Structure

```
deal_room/
├── __init__.py              # Main package export
├── models.py                 # Pydantic models (shared)
├── deal_room/
│   ├── __init__.py
│   ├── environment/
│   │   ├── __init__.py
│   │   ├── constants.py      # Reward weights, terminal rewards
│   │   ├── dealroom_v3.py    # Main RL environment
│   │   ├── llm_client.py    # MiniMax API client
│   │   └── lookahead.py     # Planning simulator
│   ├── committee/
│   │   ├── __init__.py
│   │   ├── causal_graph.py   # Graph structure + belief propagation
│   │   ├── belief_tracker.py # Bayesian update logic
│   │   └── deliberation_engine.py # Committee deliberation
│   ├── rewards/
│   │   ├── __init__.py
│   │   ├── utterance_scorer.py # 5D world-state scorer
│   │   └── pareto_efficiency.py # Terminal reward logic
│   ├── stakeholders/
│   │   ├── __init__.py
│   │   ├── archetypes.py     # 6 stakeholder profiles
│   │   └── cvar_preferences.py # CVaR computation
│   ├── curriculum/
│   │   ├── __init__.py
│   │   └── adaptive_generator.py # Curriculum generation
│   └── training/
│       ├── __init__.py
│       └── grpo_trainer.py    # GRPO training harness
├── server/
│   ├── __init__.py           # V2.5 module exports
│   ├── app.py               # FastAPI server
│   ├── deal_room_environment.py # V2.5 environment
│   ├── session_pool.py      # Session state management
│   ├── stakeholders.py      # Relationship engine
│   ├── semantics.py        # Intent/tone analysis
│   ├── validator.py         # Action validation
│   ├── grader.py           # Terminal grader
│   ├── claims.py           # Commitment ledger
│   ├── scenarios.py        # Scenario generation
│   ├── gradio_custom.py    # Custom Gradio UI
│   └── walkthrough_data.py # Guided walkthrough
└── tests/
    ├── conftest.py          # Shared test fixtures
    ├── unit/               # Unit tests
    ├── integration/        # Integration tests
    ├── e2e/               # End-to-end tests
    ├── performance/        # Performance tests
    └── v3/                # V3-specific tests
```

---

## Appendix B: Import Dependencies

```
DealRoomV3
├── models.DealRoomAction
├── models.DealRoomObservation
├── models.DealRoomState
├── deal_room.committee.belief_tracker.bayesian_update
├── deal_room.committee.causal_graph (BeliefDistribution, sample_graph, compute_engagement_level)
├── deal_room.committee.deliberation_engine.CommitteeDeliberationEngine
├── deal_room.environment.constants (REWARD_WEIGHTS, TERMINAL_REWARDS, etc.)
├── deal_room.environment.lookahead.LookaheadSimulator
├── deal_room.rewards.pareto_efficiency.compute_terminal_reward
├── deal_room.rewards.utterance_scorer (UtteranceScorer, LOOKAHEAD_COST, compute_prediction_accuracy)
├── deal_room.stakeholders.archetypes (ARCHETYPE_PROFILES, get_archetype)
└── deal_room.stakeholders.cvar_preferences (compute_cvar, compute_outcome_distribution)
```

---

*Document Version: 3.6 — 2026-04-21*
