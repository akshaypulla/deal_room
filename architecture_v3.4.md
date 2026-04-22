# DealRoom v3 Architecture Documentation

## Overview

DealRoom v3 is an enterprise negotiation reinforcement learning environment designed to train autonomous agents to navigate complex multi-stakeholder negotiations. The system models realistic negotiation scenarios with a committee decision-making structure, causal belief propagation, and CVaR-based risk management.

## System Architecture

### High-Level Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        Server Layer                              │
│  ┌──────────┐  ┌───────────┐  ┌────────────┐  ┌──────────────┐  │
│  │ FastAPI  │  │  Gradio   │  │   Session  │  │  Semantic    │  │
│  │   App    │  │    UI     │  │    Pool    │  │   Analyzer  │  │
│  └────┬─────┘  └─────┬─────┘  └──────┬─────┘  └──────┬───────┘  │
│       │             │              │                │           │
│  ┌────┴─────────────┴──────────────┴────────────────┴───────┐  │
│  │              DealRoomEnvironment (Gateway)                  │  │
│  └────┬─────────────┬──────────────┬────────────────┬───────┘  │
└───────┼─────────────┼──────────────┼────────────────┼──────────┘
        │             │              │                │
        ▼             ▼              ▼                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Core Environment                             │
│  ┌────────────┐  ┌─────────────┐  ┌────────────┐  ┌───────────┐  │
│  │ DealRoomV3 │  │  Lookahead  │  │  LLM Client│  │ Constants │  │
│  │  Gym Env   │  │  Simulator  │  │  (MiniMax) │  │           │  │
│  └─────┬──────┘  └──────┬──────┘  └────────────┘  └───────────┘  │
│        │               │                                        │
│        ▼               ▼                                        │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    State Manager                             ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Stakeholder Layer                             │
│  ┌──────────────┐  ┌─────────────────┐  ┌──────────────────────┐ │
│  │  Archetypes  │  │ CVaR Preferences│  │   StakeholderEngine  │ │
│  │ (6 profiles) │  │   (Risk Model)  │  │   (Runtime Manager)  │ │
│  └──────────────┘  └─────────────────┘  └──────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Committee Layer                              │
│  ┌───────────────┐  ┌────────────────┐  ┌─────────────────────┐  │
│  │ Causal Graph  │  │Belief Tracker  │  │ Deliberation Engine │  │
│  │(DAG + Impact) │  │(Bayesian Updt) │  │  (Decision Making)  │  │
│  └───────────────┘  └────────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Reward Layer                                │
│  ┌──────────────────┐  ┌──────────────────────────────────────┐ │
│  │ Utterance Scorer │  │       Pareto Efficiency Checker      │ │
│  │  (5-dimension)   │  │      (Multi-objective Optimization)   │ │
│  └──────────────────┘  └──────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Training Layer                               │
│  ┌───────────────┐  ┌─────────────────┐  ┌───────────────────┐ │
│  │  GRPO Trainer │  │Policy Adapters  │  │Curriculum Generator│ │
│  │               │  │(Random/Heuristic│  │  (Adaptive Diff)   │ │
│  │               │  │ /Model-based)   │  │                   │ │
│  └───────────────┘  └─────────────────┘  └───────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Environment (`deal_room/environment/`)

#### DealRoomV3 Environment

**File:** `deal_room/environment/dealroom_v3.py`

The main RL environment implementing the Gym interface. Responsible for:
- Managing negotiation state transitions
- Processing agent actions and stakeholder responses
- Tracking conversation history and commitments
- Computing rewards based on multi-dimensional scoring

**Key State Variables:**
- `negotiation_outcome`: Final agreement or deadlock status
- `stakeholder_states`: Individual stakeholder belief/commitment states
- `committee_decision`: Aggregated committee decision outcome
- `conversation_history`: All utterances and claims made

**Key Methods:**
| Method | Description |
|--------|-------------|
| `reset()` | Initialize new negotiation scenario |
| `step(action)` | Execute agent action, return observation/reward |
| `get_observation()` | Return current state representation |
| `check_terminal()` | Determine if negotiation has concluded |
| `compute_reward()` | Calculate multi-dimensional reward |

#### Lookahead Simulator

**File:** `deal_room/environment/lookahead.py`

Provides M-step lookahead simulation for decision-making:
- Simulates future states given current trajectory
- Enables planning and tree search capabilities
- Supports branch exploration for strategic reasoning

#### LLM Client

**File:** `deal_room/environment/llm_client.py`

Interface to MiniMax LLM for:
- Generating stakeholder responses
- Producing committee deliberation text
- Creating scenario variations

#### Constants

**File:** `deal_room/environment/constants.py`

Centralized configuration for:
- Negotiation phase definitions
- Action space boundaries
- Reward scaling factors
- Default hyperparameters

---

### 2. Stakeholders (`deal_room/stakeholders/`)

#### Archetypes

**File:** `deal_room/stakeholders/archetypes.py`

Six distinct stakeholder profiles with unique negotiation styles:

| Archetype | Primary Concern | Risk Tolerance | Negotiation Style |
|-----------|-----------------|-----------------|-------------------|
| **Legal** | Compliance, liability | Low | Cautious, detail-oriented |
| **Finance** | Cost, ROI, budget | Medium | Metrics-driven |
| **TechLead** | Feasibility, integration | Medium | Technical, systematic |
| **Procurement** | Vendor relations, value | Medium | Relationship-focused |
| **Operations** | Process efficiency | Low | Practical, operational |
| **ExecSponsor** | Strategic alignment | High | Vision-driven |

Each archetype maintains:
- **Priority weighting**: Importance of different deal aspects
- **Constraint boundaries**: Acceptable value ranges
- **Communication preferences**: Tone and detail level

#### CVaR Preferences

**File:** `deal_room/stakeholders/cvar_preferences.py`

Conditional Value at Risk model for risk-sensitive decision-making:
- Computes worst-case scenario rewards at confidence level α
- Used for veto decisions when CVaR drops below threshold
- Integrates with committee deliberation for risk-aware outcomes

**Key Formulation:**
```
CVaR_α(reward) = E[reward | reward ≤ VaR_α]
```

---

### 3. Committee (`deal_room/committee/`)

#### Causal Graph

**File:** `deal_room/committee/causal_graph.py`

Directed Acyclic Graph (DAG) modeling negotiation causality:

**Node Types:**
- **Deal Terms**: Price, timeline, deliverables, warranties
- **Beliefs**: Stakeholder positions, trust levels
- **Outcomes**: Agreement, deadlock, partial deal

**Edge Types:**
- **Causal Influence**: Conditional probability P(child|parent)
- **Impact Strength**: Magnitude of influence (0.0 to 1.0)
- **Time Delay**: Steps before effect manifests

#### Belief Tracker

**File:** `deal_room/committee/belief_tracker.py`

Bayesian belief propagation system:
- Maintains probability distributions over unknown states
- Updates beliefs based on new evidence (utterances, actions)
- Supports uncertain information with probability intervals

**Update Mechanism:**
```
P(Belief | Evidence) ∝ P(Evidence | Belief) × P(Belief)
```

#### Deliberation Engine

**File:** `deal_room/committee/deliberation_engine.py`

Multi-party decision-making process:
1. **Information Gathering**: Collect stakeholder positions
2. **Argument Synthesis**: Build unified committee perspective
3. **Veto Check**: Evaluate CVaR thresholds
4. **Decision Aggregation**: Combine individual preferences into committee action

---

### 4. Rewards (`deal_room/rewards/`)

#### Utterance Scorer

**File:** `deal_room/rewards/utterance_scorer.py`

Five-dimensional reward signal:

| Dimension | Description | Weight Range |
|-----------|-------------|--------------|
| **Goal Alignment** | Progress toward acceptable deal | 0.0 - 1.0 |
| **Trust Building** | Actions that increase mutual trust | -0.5 - 1.0 |
| **Information Gain** | New information disclosure/receipt | 0.0 - 1.0 |
| **Risk Management** | Mitigation of potential risks | 0.0 - 1.0 |
| **Causal Coherence** | Consistency with causal graph | 0.0 - 1.0 |

**Scoring Formula:**
```
score = Σ(weight_i × dimension_i) / Σ(weight_i)
```

#### Pareto Efficiency Checker

**File:** `deal_room/rewards/pareto_efficiency.py`

Determines if current outcome is Pareto-optimal:
- Identifies if any party can be made better off without harming others
- Used to encourage efficient negotiation outcomes
- Returns efficiency ratio and improvement suggestions

---

### 5. Training (`deal_room/training/`)

#### GRPO Trainer

**File:** `deal_room/training/grpo_trainer.py`

Group Relative Policy Optimization implementation:
- Custom loss function with advantage estimation
- Supports multiple policy adapters
- Handles episodic training with checkpointing

#### Policy Adapters

Three adapter types for different training modes:

| Adapter | Use Case | Characteristics |
|---------|----------|-----------------|
| **RandomPolicyAdapter** | Baseline testing | Uniform random actions |
| **HeuristicPolicyAdapter** | Rule-based baseline | Domain knowledge rules |
| **ModelPolicyAdapter** | Trained model inference | Neural network policy |

#### Curriculum Generator

**File:** `deal_room/curriculum/adaptive_generator.py`

Adaptive difficulty progression:
- Generates scenarios with increasing complexity
- Balances difficulty based on agent performance
- Supports domain randomization for generalization

---

### 6. Server (`server/`)

#### FastAPI Application

**File:** `server/app.py`

REST API for external integration:
- `POST /negotiate`: Execute negotiation turn
- `GET /history/{session_id}`: Retrieve conversation
- `POST /reset`: Start new scenario
- `GET /health`: System status

#### Gradio UI

**File:** `server/gradio_custom.py`

Interactive web interface:
- Real-time negotiation visualization
- Stakeholder state display
- Action selection interface

#### Session Pool

**File:** `server/session_pool.py`

Multi-session management:
- Tracks active negotiation sessions
- Manages resource allocation
- Handles session timeout and cleanup

#### Semantic Analyzer

**File:** `server/semantics.py`

Natural language understanding:
- Extracts commitments from utterances
- Identifies claim relationships
- Detects contradiction and consistency

#### Stakeholders Engine

**File:** `server/stakeholders.py`

Runtime stakeholder management:
- Spawns stakeholder agents per scenario
- Manages stakeholder lifecycle
- Routes messages to appropriate stakeholders

#### Claims Ledger

**File:** `server/claims.py`

Commitment tracking:
- Records all made commitments
- Tracks fulfillment status
- Identifies broken promises

#### Scenario Generator

**File:** `server/scenarios.py`

Procedural scenario creation:
- Defines stakeholder composition
- Sets initial positions and constraints
- Configures success criteria

#### Action Validator

**File:** `server/validator.py`

Ensures action validity:
- Checks action against schema
- Validates timing constraints
- Enforces business rules

---

## Data Models

### Core Entities (`models.py`)

```python
class Stakeholder:
    id: str
    archetype: Archetype
    position: Dict[str, float]
    constraints: Dict[str, Boundary]
    belief_state: BeliefDistribution

class Claim:
    id: str
    speaker_id: str
    content: str
    commitments: List[Commitment]
    timestamp: float
    causal_parents: List[str]

class CommitteeDecision:
    action: str
    voter_positions: Dict[str, float]
    veto_triggered: bool
    cvar_score: float
    confidence: float
```

---

## API Reference

### Environment Interface

```python
class DealRoomV3(gym.Env):
    def reset(self, scenario_config: ScenarioConfig) -> Observation
    def step(self, action: AgentAction) -> Tuple[Observation, float, bool, Info]
    def get_observation(self) -> Observation
    def render(self, mode: str = 'human') -> Any
```

### Stakeholder Interface

```python
class StakeholderAgent:
    def observe(self, utterance: Utterance) -> None
    def deliberate(self, state: GameState) -> CommitteeContribution
    def respond(self, prompt: str) -> str
    def check_veto(self, outcome: Outcome) -> VetoDecision
```

### Committee Interface

```python
class Committee:
    def add_stakeholder(self, agent: StakeholderAgent) -> None
    def deliberation_cycle(self, state: GameState) -> CommitteeDecision
    def propagate_belief(self, evidence: Evidence) -> None
    def check_termination(self) -> TerminationDecision
```

---

## Configuration

### Default Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_utterance_length` | 500 | Maximum tokens per utterance |
| `turn_limit` | 50 | Maximum negotiation turns |
| `cvr_threshold` | 0.3 | CVaR veto threshold |
| `reward_weights` | [0.3, 0.2, 0.2, 0.15, 0.15] | Dimension weights |
| `lookahead_steps` | 3 | M-step lookahead depth |

---

## Extension Points

### Adding Custom Archetypes

1. Subclass `StakeholderArchetype` in `archetypes.py`
2. Define priority weights and constraint boundaries
3. Register in `ARCHETYPE_REGISTRY`

### Custom Reward Functions

1. Implement `RewardFunction` interface
2. Override `compute(state, action)` method
3. Register in environment's reward pipeline

### Custom Deliberation Rules

1. Extend `DeliberationEngine` class
2. Override `synthesize_position()` method
3. Configure in committee initialization

---

## Dependencies

```
gym>=0.26.0
numpy>=1.24.0
torch>=2.0.0
transformers>=4.30.0
fastapi>=0.100.0
gradio>=3.40.0
pydantic>=2.0.0
```

---

## Version History

- **v3.4** (Current): CVaR veto mechanism, belief tracking enhancement
- **v3.3**: Pareto efficiency checker, utterance scorer
- **v3.2**: Committee deliberation engine
- **v3.1**: Initial environment implementation
