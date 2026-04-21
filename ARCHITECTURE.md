# DealRoom Architecture Documentation

## Overview

DealRoom is a multi-stakeholder enterprise negotiation environment that simulates a B2B sales cycle where a vendor negotiates with a buying committee to close a deal. The system is designed as a research testbed for studying committee dynamics, causal inference, CVaR-based preferences, and training strategic agents.

---

## Directory Structure

```
deal_room/
├── models.py                    # Core Pydantic data models
├── client.py                     # OpenEnv client wrapper
├── inference.py                  # Baseline inference script
├── calibrate.py                  # Calibration script (strategic vs random agent)
├── deal_room/                    # Main package
│   ├── environment/
│   │   ├── dealroom_v3.py       # V3 environment with causal graph inference
│   │   ├── llm_client.py        # Dual-API LLM client (MiniMax + OpenAI)
│   │   └── lookahead.py          # Lookahead simulator for minimax robustness
│   ├── committee/
│   │   ├── causal_graph.py      # Causal graph and belief propagation
│   │   ├── deliberation_engine.py # Two-layer deliberation engine
│   │   └── belief_tracker.py    # Bayesian belief updates
│   ├── rewards/
│   │   ├── utterance_scorer.py   # Five-dimensional deterministic reward scoring
│   │   └── pareto_efficiency.py # Terminal reward / Pareto efficiency checker
│   ├── stakeholders/
│   │   ├── archetypes.py         # Stakeholder risk profile definitions
│   │   └── cvar_preferences.py  # CVaR-based preference model
│   ├── curriculum/
│   │   └── adaptive_generator.py # Adaptive curriculum generator
│   └── training/
│       └── grpo_trainer.py      # GRPO trainer for multi-dimensional rewards
├── server/
│   ├── app.py                   # FastAPI HTTP server
│   ├── deal_room_environment.py # V2.5 deterministic environment
│   ├── grader.py               # Terminal grader (CCIGrader)
│   ├── scenarios.py             # Scenario generation with stakeholders/constraints
│   ├── stakeholders.py          # Stakeholder engine with response generation
│   ├── semantics.py             # Semantic analyzer (embedding/TF-IDF/lexical)
│   ├── claims.py                # Commitment ledger with contradiction detection
│   ├── validator.py            # Output validator (JSON-first parsing)
│   └── session_pool.py         # Session-scoped environment pool
```

---

## Core Data Models (`models.py`)

### DealRoomAction

The action space for the vendor agent. Fields:
- `action_type`: One of `direct_message`, `send_document`, `group_proposal`, `backchannel`, `exec_escalation`, `concession`, `walkaway_signal`, `reframe_value_prop`
- `target` / `target_ids`: Stakeholder targeting (e.g., "all", "Legal", "Finance")
- `message`: The communication content (truncated to 1200 chars)
- `documents`: List of document dictionaries with `type` and `specificity`
- `proposed_terms`: Optional dict with price, timeline_weeks, security_commitments, support_level, liability_cap
- `channel`: Communication channel (e.g., "formal", "backchannel")
- `mode`: Delivery mode (e.g., "async_email", "formal_meeting")
- `lookahead`: Optional `LookaheadRequest` for lookahead queries

Validators:
- `truncate_message`: Limits message to 1200 characters
- `normalize_target_ids`: Deduplicates and strips whitespace from target IDs
- `sync_targets`: If `target_ids` is set but `target == "all"`, updates `target` to comma-joined IDs

### DealRoomObservation

What the agent observes after each step. Key fields:
- `round_number` / `max_rounds`: Episode progress
- `stakeholders`: Dict mapping ID → {role} for each committee member
- `stakeholder_messages`: Responses from stakeholders
- `engagement_level`: Per-stakeholder engagement [0, 1]
- `engagement_level_delta`: Single float (change in first stakeholder's engagement)
- `engagement_history`: List of engagement snapshots
- `weak_signals`: Per-stakeholder signals (high_engagement, low_engagement, etc.)
- `cross_stakeholder_echoes`: List of {from, to, content} dicts for cross-stakeholder effects
- `veto_precursors`: Dict of stakeholder → warning message when CVaR risk is building
- `known_constraints`: List of constraint dicts that are in "known" status
- `requested_artifacts`: Per-stakeholder artifact requests
- `approval_path_progress`: Per-stakeholder {band, mandatory, authority}
- `deal_momentum`: "stalling" | "progressing" | "critical"
- `deal_stage`: "evaluation" → "negotiation" → "legal_review" → "final_approval" → "closed"
- `active_blockers`: List of stakeholder IDs currently blocking
- `days_to_deadline`: Countdown to deal deadline
- `done`: Episode termination flag

### DealRoomState

Full internal state (not exposed to agent). Contains:
- `episode_id`, `task_id`, `round_number`, `max_rounds`
- `stakeholders` + `stakeholder_private`: Public and private state per stakeholder
- `hidden_constraints`: Constraint dictionary with status (hidden/hinted/known/resolved)
- `relationship_edges`: Graph edges between stakeholders
- `commitment_ledger`: Rolling history of semantic commitments
- `deferred_effects`: Time-delayed effects (resistance spikes, blocker checks)
- `offer_state`: Proposed deal terms
- `feasibility_state`: Whether deal is feasible + list of violations
- `active_blockers`, `deal_stage`, `stage_regressions`
- `approval_caps`: Per-stakeholder maximum approval level
- `semantic_threshold_jitter`: Per-slot noise for semantic matching thresholds
- `milestone_flags`: Tracks which milestones have been awarded
- `validation_failures`, `malformed_actions`, `last_action_error`
- `deal_closed`, `deal_failed`, `failure_reason`, `final_terms`

Validator: `validate_tracks` ensures every `stakeholder_private` entry has required tracks: trust, approval, perceived_fit, private_resistance

### LookaheadRequest / SimulationResult

Lookahead query structure:
- `action_draft`: The action to simulate
- `n_hypotheses`: Number of mental state hypotheses (default 2)
- `depth`: Simulation depth (default 2)

Simulation result contains predicted responses, belief deltas, CVaR impact, graph information gain, and cost (0.07).

---

## V2.5 Environment (`server/deal_room_environment.py`)

The V2.5 deterministic environment processes actions and computes rewards without LLM calls.

### Stage Progression

Stages follow a fixed order: `evaluation → negotiation → legal_review → final_approval → closed`. The stage advances when:
- **evaluation → negotiation**: Mandatory stakeholders contacted OR constraints discovered
- **negotiation → legal_review**: All mandatory contacted AND all constraints known
- **legal_review → final_approval/closed**: All mandatory workable (approval ≥ 0.62, resistance ≤ 0.65), all requested artifacts cleared, and deal is feasible
- **final_approval → closed**: `_can_close()` returns True

Stage regression occurs when in legal_review/final_approval with active blockers present.

### Action Processing Pipeline

1. **Validation**: `OutputValidator` normalizes JSON input, extracts action type, resolves target IDs
2. **Semantic Analysis**: `SemanticAnalyzer` computes intent matches, tone scores, artifact matches, claim extraction, request matching
3. **Commitment Ledger**: Claims are ingested; contradictions are detected via tolerance comparison
4. **Stakeholder Update**: `StakeholderEngine.apply_action()` computes per-stakeholder deltas (trust, approval, perceived_fit, resistance) based on tone + artifacts + satisfied requests
5. **Relationship Propagation**: Alliance edges amplify approval/resistance changes; conflict edges introduce resistance; sponsor edges grant approval when source is strong
6. **Constraint Visibility**: Hidden constraints transition hidden → hinted → known based on intent/artifact overlap with thresholds
7. **Constraint Resolution**: Constraints check their specific conditions (e.g., price ≤ max_price)
8. **Term Application**: Proposed terms and extracted claims are stored in offer_state
9. **Deferred Effects**: Time-delayed effects (resistance spikes, blocker checks) are applied
10. **Feasibility Check**: Determines if deal is feasible based on unresolved constraints and term violations
11. **Dense Reward**: Milestone rewards for constraint hints (+0.03), constraint known (+0.04), artifact delivery (+0.03), band improvement (+0.02), blocker removal (+0.02), stage advance (+0.03)
12. **Terminal Check**: Checks for deal_closed, silent veto, or max rounds

### Approval Banding

Stakeholders are classified into bands based on approval and private_resistance:
- `blocker`: resistance > 0.65 OR approval < 0.48
- `supporter`: approval ≥ 0.72
- `workable`: approval ≥ 0.62
- `neutral`: otherwise

### Document Effects

Documents boost approval based on stakeholder role:
- `roi_model`: Finance +0.10, ExecSponsor +0.06, Procurement +0.04
- `dpa`: Legal +0.12, Procurement +0.04
- `security_cert`: Legal +0.10, TechLead +0.06, Procurement +0.04
- `implementation_timeline`: Operations +0.10, TechLead +0.09, ExecSponsor +0.03
- etc.

### Tone Impact

Message tone (from semantic analyzer) affects trust delta via role-specific weights:
- `collaborative`: +0.05
- `adaptive`: +0.04 × weight (role-specific)
- `credible`: +0.05 × weight (role-specific)
- `specific`: +0.05 × weight (role-specific)
- `pushy`: penalized
- `evasive`: penalized

### Hidden Constraints

Four constraint types with discovery intents and resolution checkers:
- `budget_ceiling`: Discovered via "discover_budget" intent, requires ROI model, resolved if price ≤ 185,000
- `delivery_window`: "discover_timeline", requires implementation timeline, resolved if timeline ≤ 16 weeks
- `compliance_addendum`: "discover_compliance", requires DPA, resolved if "gdpr" in commitments
- `supplier_process`: "share_vendor_packet", requires vendor_packet, resolved if "named_support_lead" in support_level

### Contradiction Detection

The `CommitmentLedger` detects contradictions when:
- Numeric slots (price, timeline_weeks) differ by more than 8% / 15% tolerance
- Polarity slots (security_posture, liability, support_level, implementation_commitment) change value or polarity

Contradictions cause: trust -0.10, perceived_fit -0.03, approval_cap → min(current, 0.58), resistance_spike deferred by 1 round.

---

## V3 Environment (`deal_room/environment/dealroom_v3.py`)

V3 introduces causal graph inference and a two-layer deliberation committee model. It uses LLM calls for deliberation summaries and stakeholder responses.

### Standard Stakeholders & Hierarchy

```
STANDARD_STAKEHOLDERS = ["Legal", "Finance", "TechLead", "Procurement", "Operations", "ExecSponsor"]
STANDARD_HIERARCHY = {
    "Legal": 3, "Finance": 3, "TechLead": 2, "Procurement": 2, "Operations": 2, "ExecSponsor": 5
}
```

### Causal Graph

The committee is modeled as a weighted directed graph where edges represent influence relationships. `sample_graph()` creates graphs with scenario-dependent parameters:

| Scenario | base_edge_prob | intra_cluster_boost | cross_cluster_penalty | authority_edge_prob |
|----------|---------------|--------------------|----------------------|---------------------|
| aligned | 0.30 | 0.40 | 0.20 | 0.85 |
| conflicted | 0.45 | 0.50 | 0.15 | 0.80 |
| hostile_acquisition | 0.60 | 0.25 | 0.05 | 0.65 |

Functional clusters: cost (Finance, Procurement), risk (Legal, Compliance), implementation (TechLead, Operations), authority (ExecSponsor).

### Belief Distribution

Each stakeholder maintains a `BeliefDistribution` over vendor types:
- **Positive**: competent, trustworthy, aligned
- **Negative**: incompetent, deceptive, misaligned

`positive_mass()` returns sum of positive types; `negative_mass()` returns sum of negative types; `engagement_level = positive_mass - negative_mass`.

Initial beliefs differ by scenario:
- **aligned**: uniform positive bias (competent/trustworthy/aligned = 0.25/0.25/0.20)
- **conflicted**: clusters have different biases (cost_cluster is skeptical, impl_cluster is aligned)
- **hostile_acquisition**: strong negative bias (incompetent/deceptive/misaligned = 0.22 each)

### Bayesian Belief Updates

When the vendor takes an action:
1. Likelihoods are assigned based on action type (e.g., `send_document(DPA)_proactive` → competent: 0.85, trustworthy: 0.80)
2. Targeted actions: damping = 1.0; non-targeted: damping = 0.3
3. Posterior = prior × dampened_likelihood for each vendor type
4. Beliefs are renormalized; confidence = 1 - (entropy / log2(6))

### Belief Propagation

After the vendor's action, the committee deliberates. `propagate_beliefs()` runs N steps where:
1. For each stakeholder, sum influence from all incoming edges: `total_delta = Σ(weight × delta_from_source)`
2. Apply dampened delta: `next_belief = _apply_belief_delta(current, total_delta, damping=0.85^step)`
3. Delta is positive mass transfer between positive/negative types, capped at 0.15 per type

This models how one stakeholder's changed opinion ripples through the influence graph.

### Committee Deliberation Engine

`CommitteeDeliberationEngine.run()`:
1. Runs `propagate_beliefs()` on beliefs_after_action to get updated_beliefs
2. Optionally generates a deliberation summary via MiniMax LLM call
3. Returns `DeliberationResult(updated_beliefs, summary_dialogue, propagation_deltas)`

### Engagement with Noise

True engagement = `compute_engagement_level(belief)`. Noisy engagement = true + Gaussian noise (σ=0.03). The agent only sees noisy engagement. Engagement history maintains a sliding window of 5 values.

### Veto Precursors

If a stakeholder's CVaR loss > tau × 0.70 (70% of veto threshold), their ID is added to `veto_precursors` with a warning message. This gives the agent an observable signal that a silent veto is building.

### Cross-Stakeholder Echoes

When a targeted stakeholder receives an action, other stakeholders have a 70% chance of "echoing" (awareness of the action) per the `echo_recall_probability` config.

### Reward Computation (V3)

V3 uses the `UtteranceScorer` to compute a 5-dimensional reward vector:
- `goal`: Weighted approval delta + blocker resolution bonus + CVaR veto headroom improvement
- `trust`: Positive mass delta + trustworthy mass delta for targeted stakeholders
- `information`: Entropy reduction in belief distributions (uncertainty reduction reward)
- `risk`: CVaR improvement for risk-averse stakeholders (lambda_risk > 0.30)
- `causal`: Betweenness centrality of targeted node in causal graph

Lookahead cost: -0.07 from goal dimension when lookahead is used.

---

## Causal Graph Module (`deal_room/committee/causal_graph.py`)

### BeliefDistribution Dataclass

```python
@dataclass
class BeliefDistribution:
    distribution: Dict[str, float]  # 6 vendor types
    stakeholder_role: str
    confidence: float = 1.0
    history: List[Tuple] = field(default_factory=list)
```

Methods:
- `positive_mass()`: Sum of competent, trustworthy, aligned probabilities
- `negative_mass()`: Sum of incompetent, deceptive, misaligned probabilities
- `to_natural_language()`: Generates readable belief description
- `copy()`: Deep copy with independent distribution dict

### Graph Construction (`sample_graph`)

For each (source, dest) pair:
- Authority nodes (authority ≥ 4): always create edge
- Same functional cluster: probability = base_edge_probability + intra_cluster_boost
- Cross-cluster: probability = base_edge_probability - cross_cluster_penalty
- Edge weight: clipped Normal(mean=0.45-0.50, std=0.15-0.25) → [0.05, 0.95]

Authority weights are normalized to sum to 1.0.

### Propagation Functions

- `propagate_beliefs(graph, beliefs_before, beliefs_after, n_steps)` → Dict[str, BeliefDistribution]
- `_apply_belief_delta(belief, delta, damping)` → BeliefDistribution: Transfers probability mass between positive/negative types
- `apply_positive_delta(belief, delta)` → BeliefDistribution: Helper for positive shift

### Centrality & Signatures

- `get_betweenness_centrality(graph, stakeholder)`: Fraction of shortest paths passing through the node
- `compute_behavioral_signature(graph, targeted_stakeholder, belief_delta, n_steps)`: Returns dict of engagement changes for all OTHER stakeholders when the target is influenced

### Engagement Level

`compute_engagement_level(belief)` = `positive_mass() - negative_mass()` ∈ [-1, 1]

---

## Utterance Scorer (`deal_room/rewards/utterance_scorer.py`)

Five deterministic scoring dimensions computed from world state deltas only:

### Goal Score

```
approval_delta = Σ(auth_weight × (pos_mass_after - pos_mass_before)) / Σ(auth_weights)
blocker_score = resolved_blockers × 0.15 - new_blockers × 0.10
veto_improvements = [max(0, 1 - cvar_a/tau) - max(0, 1 - cvar_b/tau)] for risk-averse stakeholders
veto_score = mean(veto_improvements)
goal = clip(0.5 + 0.50×approval_score + 0.30×blocker_score + 0.20×veto_score, 0, 1)
```

### Trust Score

For each targeted stakeholder:
```
delta = 0.6 × pos_mass_delta + 0.4 × trustworthy_mass_delta
trust = clip(0.5 + mean(deltas) × 3.0, 0, 1)
```

### Information Score

Measures entropy reduction (uncertainty removal):
```
h_before = entropy(before_distribution)
h_after = entropy(after_distribution)
info = clip(0.5 + mean((h_before - h_after) / LOG2_6) × 5.0, 0, 1)
```

### Risk Score

CVaR improvement for stakeholders with lambda_risk > 0.30:
```
improvement = (cvar_before - cvar_after) / cvar_before
risk = clip(0.5 + mean(improvements), 0, 1)
```

### Causal Score

Betweenness centrality of targeted node (graph-based influence power):
```
causal = centrality(target) / max_possible_centrality
```

### Lookahead Cost

When `lookahead_used=True`, goal score is reduced by exactly 0.07 (LOOKAHEAD_COST).

---

## CVaR Preferences (`deal_room/stakeholders/cvar_preferences.py`)

### StakeholderRiskProfile

Each archetype has:
- `alpha`: CVaR percentile (higher = more tail-sensitive, e.g., Legal = 0.95)
- `tau`: Veto threshold (lower = easier to veto, e.g., Legal = 0.10)
- `lambda_risk`: Risk weight in deal quality (higher = more CVaR-penalized)
- `utility_weights`: Domain-specific priorities
- `uncertainty_domains`: Risk categories relevant to this stakeholder

### CVaR Computation

`compute_cvar(outcomes, alpha)`:
1. Sort outcomes ascending
2. Cutoff = index at (1 - alpha) percentile
3. CVaR = mean of (1 - outcomes[:cutoff]) weighted uniformly

### Outcome Distribution

Monte Carlo sampling of deal outcomes based on:
- Base success probability by domain (compliance: 0.80-0.92, cost: 0.70, implementation: 0.75)
- Documentation effects (DPA + security cert boost compliance domain to 0.92)
- Price/liability adjustments

### Deal Quality Score

```
quality = (1 - lambda_risk) × expected_utility - lambda_risk × cvar_loss
```

### Veto Trigger

`check_veto_trigger(cvar_loss, profile)` returns `cvar_loss > profile.tau`

This is the "silent veto" mechanism: a stakeholder can kill the deal despite positive expected utility if the tail risk is too high.

---

## Stakeholder Archetypes (`deal_room/stakeholders/archetypes.py`)

Six archetypes with locked parameter values:

| Stakeholder | Alpha | Tau | Lambda | Top Utility Weight |
|-------------|-------|-----|--------|-------------------|
| Legal | 0.95 | 0.10 | 0.70 | compliance_coverage (0.40) |
| Finance | 0.90 | 0.15 | 0.50 | roi_clarity (0.35) |
| TechLead | 0.80 | 0.25 | 0.30 | implementation_feasibility (0.40) |
| Procurement | 0.85 | 0.20 | 0.45 | contract_compliance (0.35) |
| Operations | 0.80 | 0.30 | 0.35 | operational_continuity (0.40) |
| ExecSponsor | 0.70 | 0.40 | 0.25 | strategic_alignment (0.40) |

---

## LLM Client (`deal_room/environment/llm_client.py`)

Dual-API client with automatic fallback chain: **MiniMax M2.5 → OpenAI GPT-4o-mini**.

### Token Budgets

- `scorer_json`: 100 tokens (OpenAI json_object is compact)
- `stakeholder_response`: 1000 tokens (MiniMax thinking ~600-800 + text)
- `deliberation_summary`: 2000 tokens (MiniMax thinking ~1200-1500 + text)

### Error Classification

Errors are classified into `LLMErrorType` enum:
- **Auto-recoverable**: network timeout, DNS failure, connection reset, 5xx, server overloaded, empty response
- **Rate limit**: 429, quota exceeded
- **Auth error**: 401, 403 (DO NOT trigger fallback — bad key will fail on both APIs)

### Retry Policy

- Max auto-retries: 3 per API
- Exponential backoff with jitter (base=1s, factor=2, max=30s)
- Rate limit default wait: 60s, max wait: 300s
- JSON 3-strike limit: After 3 consecutive JSON parse failures on same context, prompts for manual intervention

### Fallback Logic

```
MiniMax (curl, Anthropic /v1/messages)
  ↓ on auto-recoverable error (retry 3×)
  ↓ on JSON parse failure (retry 3×)
  ↓ on non-recoverable error
OpenAI GPT-4o-mini (SDK, /chat/completions with json_object mode)
  ↓ on auto-recoverable error (retry 3×)
  ↓ on user intervention prompt
  → restart from MiniMax
```

### Public API

- `generate_stakeholder_response(prompt, context)` → text (allow_skip=True)
- `generate_deliberation_summary(prompt, context, timeout)` → text (allow_skip=True)
- `score_utterance_dimensions(scoring_prompt, context)` → dict with keys [goal, trust, info] (allow_skip=False)
- `validate_api_keys()` → prints warnings if keys missing

---

## Semantic Analyzer (`server/semantics.py`)

Three-tier semantic analysis (tries in order):

1. **Embedding** (preferred): SentenceTransformer `paraphrase-MiniLM-L3-v2`
2. **TF-IDF** (fallback): sklearn `TfidfVectorizer(ngram_range=(1,2))`
3. **Lexical** (last resort): Jaccard overlap on tokenized text

### Intent Bank

Predefined intent patterns: `discover_budget`, `discover_timeline`, `discover_compliance`, `reassure`, `pressure`, `close_attempt`, `share_roi`, `share_implementation`, `share_security`, `share_dpa`, `share_vendor_packet`

### Tone Bank

Tone patterns: `collaborative`, `credible`, `specific`, `pushy`, `evasive`, `adaptive`

### Artifact / Slot Aliases

Maps natural language aliases to structured types (e.g., "roi", "business case", "payback" → `roi_model`)

### Claim Extraction

Regex-based extraction of:
- Price: `\$?\s*(\d{2,3}(?:,\d{3})+|\d{5,6})`
- Timeline: `(\d{1,2})\s*(?:weeks?|wks?)`
- Security posture: "gdpr", "soc 2", "audit rights", "data residency"
- Liability: "liability cap", "unlimited liability", "indemnity"
- Support level: "named support lead", "24/7 support", "premium support"
- Implementation: "dedicated engineers", "implementation team", "named rollout lead"

### Request Matching

Matches artifacts in the message against `requested_artifacts` per stakeholder, using artifact aliases.

---

## Commitment Ledger (`server/claims.py`)

Rolling claim history (max 12 entries) with contradiction detection:

**Numeric tolerances**: price=8%, timeline_weeks=15%

**Polarity slots**: security_posture, liability, support_level, implementation_commitment

A contradiction occurs when:
- Numeric: `|previous - current| / previous > tolerance`
- Polarity: values or polarities differ

---

## Terminal Grader (`server/grader.py`)

`CCIGrader.compute(state)` returns a score in [0.01, 0.99] using weighted components:

| Component | Weight | Logic |
|----------|--------|-------|
| approval_completeness | 0.35 | Mean approval of mandatory stakeholders |
| constraint_satisfaction | 0.25 | Fraction of constraints resolved |
| term_feasibility | 0.15 | 1.0 - penalty for violations (max -0.20) |
| relationship_durability | 0.15 | Mean trust - 0.03 × permanent_marks |
| efficiency | 0.10 | 1.0 - (round_number/max_rounds)^1.25 × 0.45 |

Returns MIN_SCORE (0.01) if:
- Deal not closed or failed
- Deal infeasible
- Any constraint unresolved
- Mandatory stakeholder has approval < 0.62 or resistance > 0.65
- Any veto_power stakeholder has resistance > 0.65

---

## Pareto Efficiency (`deal_room/rewards/pareto_efficiency.py`)

Terminal reward determination:
- **deal_closed**: +1.0
- **veto**: -1.0
- **max_rounds + Pareto optimal**: 0.0
- **max_rounds + no deal**: 0.0
- **stage_regression**: -0.5 × min(stage_regressions, 3)
- **impasse**: -0.75

---

## Scenario Generation (`server/scenarios.py`)

### Scenario Templates

| Scenario | Max Rounds | Days to Deadline | Observability | Constraint Count | Event Round |
|----------|-----------|-----------------|---------------|-----------------|-------------|
| aligned | 8 | 45 | high | 1 | None |
| conflicted | 10 | 32 | medium | 1-2 | None |
| hostile_acquisition | 10 | 22 | low | 2 | 3-4 |

### Role Library

Six roles with: label, mandatory flag, authority weight, veto_power, style, requested_artifacts, utility_weights

### Constraint Library

Four constraints: budget_ceiling, delivery_window, compliance_addendum, supplier_process

Each has: slot, label, required_artifact, hint, weak_signal, checker

### Episode Generation

`generate_episode(task_id, seed)`:
1. Picks roles based on scenario template
2. Picks constraints from constraint pool
3. Picks relationship edges from template
4. Assigns names from ROLE_NAME_SETS
5. Samples track values (trust, approval, perceived_fit, private_resistance) from scenario ranges
6. Adds semantic threshold jitter

---

## FastAPI Server (`server/app.py`)

Endpoints:
- `GET /` → redirect to /web
- `GET /web` → HTML shell that loads Gradio in iframe
- `GET /health` → service status
- `GET /metadata` → name, version, tasks
- `POST /reset` → creates new session, returns initial observation
- `POST /step` → executes action, returns (observation, reward, done, info)
- `GET /state` → returns full DealRoomState

Session management via `DealRoomSessionPool` with 128 max sessions, 6-hour TTL.

---

## Session Pool (`server/session_pool.py`)

`DealRoomSessionPool` maintains one `DealRoomV3` instance per session. Sessions are pruned after 6 hours of inactivity or when exceeding 128 sessions (oldest is removed).

---

## Lookahead Simulator (`deal_room/environment/lookahead.py`)

`LookaheadSimulator.simulate()`:
1. Generates optimistic and pessimistic belief hypotheses for the target stakeholder
2. Simulates each hypothesis through `_simulate_one_hypothesis()`
3. Returns worst-case (minimum predicted_goal_delta) simulation result

Cost: 0.07 per lookahead query (deducted from goal reward).

---

## Adaptive Curriculum (`deal_room/curriculum/adaptive_generator.py`)

Failure analysis identifies 6 failure modes:
- F1: CVaR veto despite positive expected outcome
- F2: Trust collapse mid-episode
- F3: Failed graph inference
- F4: Timeout without coalition formation
- F5: Single-dimension reward hacking
- F6: Authority shift blindness

Curriculum config: 20% easy, 60% frontier, 20% hard. Scenario pool has 15 entries (5 seeds × 3 base configs). Selection is based on estimated agent capability.

---

## GRPO Trainer (`deal_room/training/grpo_trainer.py`)

Group Relative Policy Optimization for multi-dimensional rewards:

`compute_group_relative_advantage(episode_rewards, group_rewards)`:
1. Aggregate each step's 5D rewards using weights [0.25, 0.20, 0.20, 0.20, 0.15]
2. Compute mean and std of group rewards
3. Return (aggregated - mean) / std for each step

`run_self_play_episode()` runs an episode with the environment, collecting observations, actions, rewards, lookahead usage, and prediction accuracies.

`compute_training_metrics()` aggregates metrics across a batch of trajectories.

---

## Calibration Script (`calibrate.py`)

Calibration validates that a strategic agent outperforms random by ≥ 0.15 spread on all three tasks (aligned, conflicted, hostile_acquisition) over 50 episodes each. If calibration fails:
- Small spread on aligned → reduce initial_satisfaction by 0.05
- Small spread on hostile → increase round_3_hint detail or reduce veto_threshold to 0.40
