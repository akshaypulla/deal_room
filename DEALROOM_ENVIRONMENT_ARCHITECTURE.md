# DealRoom Environment Architecture

**Version:** 2.5  
**Type:** Multi-Stakeholder Enterprise Negotiation Simulator  
**Framework:** OpenEnv-compatible Python RL Environment

---

## Overview

DealRoom is a deterministic enterprise negotiation environment that simulates the complex committee dynamics of software procurement decisions. It models real-world challenges including hidden feasibility constraints, approval chain management, stakeholder relationship dynamics, and irreversible trust damage.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DEALROOM V2.5                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐     ┌──────────────────┐     ┌─────────────────────┐       │
│  │   Scenarios │────▶│ DealRoomEnv     │◀────│  DealRoomAction     │       │
│  │   (scenarios│     │                  │     │  (user input)       │       │
│  │    .py)     │     │ ┌────────────┐  │     └─────────────────────┘       │
│  └─────────────┘     │ │   State    │  │                                   │
│                      │ │  Manager   │  │     ┌─────────────────────┐       │
│                      │ └────────────┘  │     │  DealRoomObservation│       │
│                      │       │         │◀────│  (user output)       │       │
│                      │       ▼         │     └─────────────────────┘       │
│                      │ ┌────────────┐  │                                   │
│                      │ │  Stakeholder│  │                                   │
│                      │ │  Engine     │  │                                   │
│                      │ └────────────┘  │                                   │
│                      │       │         │                                   │
│                      │       ▼         │                                   │
│                      │ ┌────────────┐  │     ┌─────────────────────┐       │
│                      │ │  Grader    │  │     │   Semantic Analyzer │       │
│                      │ │  (CCI)     │  │────▶│   (claims/artifact)  │       │
│                      │ └────────────┘  │     └─────────────────────┘       │
│                      │       │         │                                   │
│                      │       ▼         │     ┌─────────────────────┐       │
│                      │ ┌────────────┐  │     │   Output Validator  │       │
│                      │ │ Commitment │  │────▶│   (target/terms)    │       │
│                      │ │  Ledger    │  │     └─────────────────────┘       │
│                      └──────────────────┘                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Models (`models.py`)

The V2.5 Pydantic models define the interface between the environment and external agents.

#### DealRoomAction

```python
class DealRoomAction(BaseModel):
    action_type: str = "direct_message"      # Action type
    target: str = "all"                       # Target stakeholder(s)
    target_ids: List[str] = []               # Resolved target IDs
    message: str = ""                         # Communication content (max 1200 chars)
    documents: List[Dict[str, str]] = []     # Attached documents
    proposed_terms: Optional[Dict] = None    # Commercial terms
    channel: str = "formal"                  # Communication channel
    mode: str = "async_email"                # Communication mode
```

**Valid Action Types:**
- `direct_message` - Standard communication
- `send_document` - Share artifacts (ROI, DPA, etc.)
- `backchannel` - Informal check-in
- `group_proposal` - Proposal to all stakeholders
- `concession` - Offer terms
- `walkaway_signal` - Signal potential failure
- `reframe_value_prop` - Reframe value proposition
- `exec_escalation` - Escalate to leadership

#### DealRoomObservation

```python
class DealRoomObservation(BaseModel):
    round_number: int                        # Current round
    max_rounds: int                          # Maximum rounds allowed
    stakeholders: Dict[str, Dict]            # Public stakeholder info
    stakeholder_messages: Dict[str, str]    # Messages from stakeholders
    engagement_level: Dict[str, float]      # Engagement scores
    weak_signals: Dict[str, List[str]]      # Hints about constraints
    known_constraints: List[Dict]           # Discovered constraints
    requested_artifacts: Dict[str, List]   # Stakeholder requests
    approval_path_progress: Dict[str, Dict]  # Approval status per stakeholder
    deal_momentum: str                       # "stalling"|"progressing"|"critical"
    deal_stage: str                          # Current stage
    active_blockers: List[str]             # Blocking stakeholders
    days_to_deadline: int                   # Days remaining
    done: bool                               # Episode terminal state
```

#### DealRoomState

Full internal state including private stakeholder data and hidden constraints.

---

### 2. Scenarios (`server/scenarios.py`)

The scenario system generates deterministic episodes with controlled complexity.

#### Task Types

| Task | Rounds | Deadline | Stakeholders | Constraints | Description |
|------|--------|----------|--------------|-------------|-------------|
| **aligned** | 8 | 45 days | 2-3 | 1 | Low-friction deal with light politics |
| **conflicted** | 10 | 32 days | 3-4 | 1-2 | Conflicting incentives, approval drag |
| **hostile_acquisition** | 10 | 22 days | 4 | 2 | Authority shock, compressed timeline |

#### Stakeholder Roles

```python
ROLE_LIBRARY = {
    "finance": {
        "label": "Finance Lead",
        "mandatory": True,
        "authority": 0.92,
        "veto_power": True,
        "requested_artifacts": ["roi_model", "reference_case"],
    },
    "technical": {
        "label": "Technical Owner",
        "mandatory": True,
        "authority": 0.85,
        "veto_power": False,
        "requested_artifacts": ["implementation_timeline", "security_cert"],
    },
    "legal_compliance": {
        "label": "Legal & Compliance",
        "mandatory": True,
        "authority": 0.88,
        "veto_power": True,
        "requested_artifacts": ["dpa", "security_cert"],
    },
    "procurement": {
        "label": "Procurement",
        "mandatory": False,
        "authority": 0.74,
        "veto_power": False,
        "requested_artifacts": ["vendor_packet", "reference_case"],
    },
    "operations": {
        "label": "Operations Sponsor",
        "mandatory": False,
        "authority": 0.64,
        "veto_power": False,
        "requested_artifacts": ["implementation_timeline", "support_plan"],
    },
    "executive_sponsor": {
        "label": "Executive Sponsor",
        "mandatory": False,
        "authority": 0.95,
        "veto_power": True,
        "requested_artifacts": ["roi_model", "implementation_timeline"],
    },
}
```

#### Hidden Constraints

```python
CONSTRAINT_LIBRARY = {
    "budget_ceiling": {
        "slot": "price",
        "checker": {"max_price": 185000},
        "required_artifact": "roi_model",
    },
    "delivery_window": {
        "slot": "timeline_weeks",
        "checker": {"max_timeline_weeks": 16},
        "required_artifact": "implementation_timeline",
    },
    "compliance_addendum": {
        "slot": "security_posture",
        "checker": {"must_include": "gdpr"},
        "required_artifact": "dpa",
    },
    "supplier_process": {
        "slot": "support_level",
        "checker": {"must_include": "named_support_lead"},
        "required_artifact": "vendor_packet",
    },
}
```

---

### 3. Environment (`server/deal_room_environment.py`)

The main environment class implementing the OpenEnv interface.

#### Reset Flow

```
1. Generate episode from scenario (seeded random)
2. Initialize stakeholder private state (trust, approval, resistance)
3. Create hidden constraints based on task type
4. Initialize relationship edges
5. Generate opening messages from stakeholders
6. Return initial observation
```

#### Step Flow

```
1. Validate action (JSON parse or heuristic extraction)
2. Analyze semantics (claims, artifacts, intent)
3. Update commitment ledger (contradiction detection)
4. Apply action to stakeholder engine
5. Update constraint visibility and resolution
6. Compute feasibility state
7. Advance or regress deal stage
8. Compute dense reward
9. Check terminal conditions
10. Generate stakeholder responses
11. Return observation, reward, done, info
```

#### Deal Stages

```
evaluation ──▶ negotiation ──▶ legal_review ──▶ final_approval ──▶ closed
    │              │               │                │
    └──────────────┴───────────────┴────────────────┘
                    (can regress)
```

#### Stage Progression Rules

- **evaluation → negotiation:** All mandatory stakeholders contacted OR constraints discovered
- **negotiation → legal_review:** All mandatory contacted AND all constraints known
- **legal_review → final_approval:** Mandatory stakeholders workable AND no pending artifacts
- **final_approval → closed:** Feasible terms AND all stakeholders at approval ≥ 0.62

---

### 4. Stakeholder Engine (`server/stakeholders.py`)

Manages stakeholder state transitions and response generation.

#### Approval Bands

```python
def approval_band(approval: float, resistance: float) -> str:
    if approval >= 0.70 and resistance <= 0.35:
        return "supporter"
    elif approval >= 0.55 and resistance <= 0.50:
        return "workable"
    elif approval < 0.45 or resistance > 0.58:
        return "blocker"
    return "neutral"
```

#### Private State Tracks

Each stakeholder maintains four internal tracks:

| Track | Range | Description |
|-------|-------|-------------|
| `trust` | [0, 1] | Relationship quality, decreases with contradictions |
| `approval` | [0, 1] | Support level for the deal |
| `perceived_fit` | [0, 1] | How well solution meets stakeholder needs |
| `private_resistance` | [0, 1] | Hidden opposition, increases with pressure |

#### Permanent Marks

Irreversible trust damage markers:
- `semantic_contradiction` - Made claims that don't align
- `infeasible_promise` - Committed to infeasible terms
- `premature_close` - Tried to close without readiness
- `ignored_artifact_request` - Failed to provide requested materials

---

### 5. Semantic Analyzer (`server/semantics.py`)

Extracts meaning from natural language actions.

#### Intent Detection

```python
INTENT_BANK = {
    "discover_budget": ["help me understand the budget ceiling", ...],
    "discover_timeline": ["what timeline can your team actually support", ...],
    "share_roi": ["here is the roi analysis", ...],
    "close_attempt": ["let us move to final approval", ...],
}
```

#### Artifact Extraction

Detects mentions of:
- `roi_model` - Financial business case
- `implementation_timeline` - Delivery plan
- `dpa` - Data Processing Agreement
- `security_cert` - Security certifications
- `vendor_packet` - Supplier onboarding
- `reference_case` - Case studies
- `support_plan` - Support commitments

#### Claim Extraction

Extracts structured data:
- **Price:** "$180,000" → `{slot: "price", value: 180000}`
- **Timeline:** "14 weeks" → `{slot: "timeline_weeks", value: 14}`
- **Security:** "GDPR compliant" → `{slot: "security_posture", value: "gdpr"}`

---

### 6. Output Validator (`server/validator.py`)

Normalizes and validates agent actions.

#### Validation Modes

1. **JSON Parse (confidence 1.0):** Full structure validation
2. **Heuristic Extract (confidence 0.6):** Action type and target from text
3. **Fallback (confidence 0.0):** Default action with error flag

#### Target Resolution

Supports aliases:
```python
LEGACY_TARGET_ALIASES = {
    "cfo": ["finance"],
    "cto": ["technical"],
    "legal": ["legal_compliance"],
    "procurement": ["procurement"],
    "ops": ["operations"],
    "cto_cfo": ["technical", "finance"],
}
```

#### Proposed Terms Filtering

Only these keys are accepted:
- `price`
- `timeline_weeks`
- `security_commitments`
- `support_level`
- `liability_cap`

---

### 7. Commitment Ledger (`server/claims.py`)

Tracks claims made during negotiation for contradiction detection.

#### Ledger Entry

```python
{
    "stakeholder_id": "finance",
    "claim_type": "price",
    "slot": "price",
    "value": 180000,
    "polarity": "offer",
    "round": 2,
}
```

#### Contradiction Detection

Flags contradictions when:
- Same slot, different values, different polarity
- Example: "price cap at $150k" then "price at $200k"

---

### 8. Grader (`server/grader.py`)

Deterministic terminal grading for completed episodes.

#### CCI Score Formula

```
score = (
    approval_completeness × 0.35 +
    constraint_satisfaction × 0.25 +
    term_feasibility × 0.15 +
    relationship_durability × 0.15 +
    efficiency × 0.10
)
```

#### Terminal Conditions

**Score = 0.0 if:**
- Deal not closed (failed or timeout)
- Feasibility violations exist
- Mandatory stakeholder approval < 0.62
- Any veto holder resistance > 0.65

#### Component Scores

```python
# Approval Completeness
min(1.0, sum(mandatory_approvals) / len(mandatory_ids))

# Constraint Satisfaction
resolved_constraints / total_constraints

# Term Feasibility
max(0.0, 1.0 - (0.05 × violation_count))

# Relationship Durability
average_trust - (0.03 × permanent_marks)

# Efficiency
max(0.1, 1.0 - ((round_number / max_rounds) ^ 1.25) × 0.45)
```

---

## Workflow Examples

### Example 1: Aligned Scenario Walkthrough

```python
from server.deal_room_environment import DealRoomEnvironment
from models import DealRoomAction

env = DealRoomEnvironment()

# Reset the environment
obs = env.reset(seed=42, task_id="aligned")

print(f"Stage: {obs.deal_stage}")      # "evaluation"
print(f"Stakeholders: {list(obs.stakeholders.keys())}")  # ["finance", "technical"]

# Step 1: Send ROI to Finance
action = DealRoomAction(
    action_type="send_document",
    target="finance",
    documents=[{"type": "roi_model", "specificity": "high"}],
    message="Here is the ROI model with explicit payback assumptions."
)
obs, reward, done, info = env.step(action)

print(f"Reward: {reward:.4f}")          # ~0.03-0.05
print(f"Stage: {obs.deal_stage}")       # May advance to "negotiation"

# Step 2: Share implementation timeline
action = DealRoomAction(
    action_type="send_document",
    target="technical",
    documents=[{"type": "implementation_timeline", "specificity": "high"}],
    message="Here is the implementation timeline with milestones."
)
obs, reward, done, info = env.step(action)

# Step 3: Propose terms
action = DealRoomAction(
    action_type="group_proposal",
    target="all",
    message="I believe we have alignment to move to final approval.",
    proposed_terms={
        "price": 180000,
        "timeline_weeks": 14,
        "security_commitments": ["gdpr", "audit rights"],
        "support_level": "named_support_lead",
        "liability_cap": "mutual_cap",
    }
)
obs, reward, done, info = env.step(action)

if done:
    print(f"Final Score: {reward:.4f}")  # Terminal CCI score
```

### Example 2: Constraint Discovery

```python
# In conflicted scenario with hidden budget constraint
env = DealRoomEnvironment()
obs = env.reset(seed=64, task_id="conflicted")

# The budget_ceiling constraint is initially HIDDEN
print(obs.known_constraints)  # []

# Action: Probe Finance about budget
action = DealRoomAction(
    action_type="direct_message",
    target="finance",
    message="Help me understand the budget ceiling or board payback requirement."
)
obs, reward, done, info = env.step(action)

# Constraint transitions from HIDDEN → HINTED
print(obs.known_constraints)  # May still be empty (hinted only)
print(obs.weak_signals)       # {"finance": ["Finance keeps circling back..."]}

# Action: Send ROI document
action = DealRoomAction(
    action_type="send_document",
    target="finance",
    documents=[{"type": "roi_model", "specificity": "high"}],
    message="Here is the ROI model."
)
obs, reward, done, info = env.step(action)

# Constraint now KNOWN
print(obs.known_constraints)  # [{"id": "budget_ceiling", "status": "known", ...}]

# Constraint is RESOLVED when proposed price ≤ max_price (185000)
# This happens when terms are proposed
```

### Example 3: Premature Close Penalty

```python
# Attempt to close before readiness
action = DealRoomAction(
    action_type="group_proposal",
    target="all",
    message="Let's close the deal now."
)
obs, reward, done, info = env.step(action)

# Penalty applied: trust decreases for mandatory stakeholders
# But episode continues (not terminal)
print(f"Reward: {reward}")      # 0.0 (no milestone reward)
print(f"Done: {done}")          # False

# Check state for permanent mark
state = env.state
finance_trust = state.stakeholder_private["finance"]["trust"]
print(f"Finance trust: {finance_trust}")  # Decreased from initial
print(state.stakeholder_private["finance"]["permanent_marks"])  # ["premature_close"]
```

### Example 4: Using the API Server

```bash
# Start the server
python -m server.app

# In another terminal:

# Health check
curl http://localhost:7860/health

# Reset environment
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "aligned", "seed": 42}'

# Take a step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "action_type": "direct_message",
    "target": "finance",
    "message": "Hello Finance Lead."
  }'

# Get current state
curl http://localhost:7860/state
```

---

## State Transitions

### Hidden Constraint Lifecycle

```
[HIDDEN] ──(semantic probe+artifact)──▶ [HINTED] ──(aligned follow-up)──▶ [KNOWN]
                                                                                    │
                                                                                    ▼
                                                                              [RESOLVED]
                                                                                    │
                                                                                    ▼
                                                                            (checked per step)
```

### Stakeholder Lifecycle

```
[NEW] ──(first contact)──▶ [CONTACTED] ──(approval ≥ 0.62)──▶ [WORKABLE]
                                │                                    │
                                │                                    │
                                ▼                                    ▼
                           [BLOCKER] ◀──(resistance > 0.58 or approval < 0.45)
                                │
                                │
                                ▼
                    (veto_power AND resistance > 0.72)
                                │
                                ▼
                            [VETO]
                                │
                                ▼
                         [DEAL_FAILED]
```

---

## Reward System

### Dense Rewards (per step)

| Milestone | Reward |
|-----------|--------|
| Constraint hinted | +0.03 |
| Constraint known | +0.04 |
| Artifact satisfied | +0.03 |
| Band improved | +0.02 |
| Blocker removed | +0.02 |
| Stage advanced | +0.03 |
| **Maximum per step** | **0.15** |

### Terminal Rewards

| Outcome | Score |
|---------|-------|
| Feasible close | CCI score [0, 1] |
| Infeasible close | 0.0 |
| Silent veto | 0.0 |
| Timeout | 0.0 |

---

## Environment Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 7860 | Server port |
| `ENABLE_WEB_INTERFACE` | "true" | Enable Gradio UI |
| `HF_HOME` | `/opt/hf-home` | HuggingFace cache |

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Redirects to `/web` |
| GET | `/health` | Health check with task list |
| GET | `/metadata` | Environment metadata |
| POST | `/reset` | Reset environment |
| POST | `/step` | Execute action |
| GET | `/state` | Get full state |
| GET | `/web` | Gradio web interface |

---

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=term-missing

# Run specific test file
python -m pytest tests/unit/test_models.py -v

# Run tests matching pattern
python -m pytest tests/ -k "validator" -v
```

---

## Key Design Principles

1. **Deterministic Seeding:** Same seed always produces identical episode
2. **Partial Observability:** Hidden constraints must be discovered
3. **Irreversible Damage:** Trust marks cannot be undone
4. **Dense + Terminal Rewards:** Learning signal + true success criteria
5. **Stage-Gated Progression:** Cannot skip approval stages
6. **Multi-Stakeholder Utility:** Different roles with different priorities
