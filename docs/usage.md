# DealRoom Practical Usage Guide

A comprehensive guide to using the DealRoom reinforcement learning environment with step-by-step examples, expected outputs, and best practices.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Environment Basics](#environment-basics)
3. [Working with Actions](#working-with-actions)
4. [Understanding Observations](#understanding-observations)
5. [The Three Scenarios](#the-three-scenarios)
6. [Stakeholder Management](#stakeholder-management)
7. [Claims and Contradictions](#claims-and-contradictions)
8. [Veto Risk and Prevention](#veto-risk-and-prevention)
9. [Stage Progression](#stage-progression)
10. [Scoring and Evaluation](#scoring-and-evaluation)
11. [Best Practices](#best-practices)

---

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from deal_room import DealRoomEnvironment, DealRoomAction

# Create environment
env = DealRoomEnvironment()

# Reset with task configuration
obs = env.reset(seed=42, task_id="aligned")

# Take actions in a loop
for step in range(10):
    action = DealRoomAction(
        action_type="direct_message",
        target="CFO",
        message="Thank you for your time."
    )
    obs, reward, done, info = env.step(action)
    
    if done:
        print(f"Episode complete! Final score: {reward}")
        break
    
    print(f"Round {obs.round_number}: {obs.deal_momentum}")

print(f"Final CCI Score: {reward}")
```

---

## Environment Basics

### Reset

The `reset()` method initializes a fresh episode:

```python
env = DealRoomEnvironment()
obs = env.reset(seed=42, task_id="aligned")
```

**Parameters:**
- `seed` (int, optional): Random seed for reproducibility
- `task_id` (str): One of `"aligned"`, `"conflicted"`, `"hostile_acquisition"`
- `episode_id` (str, optional): Custom episode identifier

**Returns:** `DealRoomObservation` object

### Step

The `step()` method advances the environment by one round:

```python
action = DealRoomAction(action_type="direct_message", target="CFO", message="Hello")
obs, reward, done, info = env.step(action)
```

**Returns:**
- `obs` (DealRoomObservation): Current state of the negotiation
- `reward` (float): 0.0 during episode, CCI score on success
- `done` (bool): True if episode terminated
- `info` (dict): Dense causal signals for RL

---

## Working with Actions

### Action Types

```python
from deal_room import DealRoomAction

# Direct message to a stakeholder
action = DealRoomAction(
    action_type="direct_message",
    target="CFO",
    message="Thank you for your time."
)

# Send a document
action = DealRoomAction(
    action_type="send_document",
    target="CFO",
    message="Here is our ROI analysis.",
    documents=[{"type": "roi_model", "specificity": "high"}]
)

# Backchannel (informal check-in)
action = DealRoomAction(
    action_type="backchannel",
    target="CFO",
    channel="backchannel",
    message="Just checking in on your thoughts."
)

# Group proposal
action = DealRoomAction(
    action_type="group_proposal",
    target="all",
    message="I propose we move forward together."
)

# Concession
action = DealRoomAction(
    action_type="concession",
    target="all",
    message="We're prepared to offer net-30 terms.",
    proposed_terms={"payment": "net 30"}
)
```

### Valid Targets

```python
# Individual stakeholders
target = "CFO"      # Chief Financial Officer
target = "CTO"      # Chief Technology Officer
target = "Legal"    # Legal team
target = "Procurement"
target = "Ops"      # Operations

# Groups
target = "cto_cfo"           # CTO and CFO together
target = "legal_procurement"  # Legal and Procurement together

# All stakeholders
target = "all"
```

### Document Types

```python
documents = [
    {"type": "roi_model", "specificity": "high"},           # Financial ROI
    {"type": "security_cert", "specificity": "high"},       # Security compliance
    {"type": "implementation_timeline", "specificity": "high"},  # Timeline
    {"type": "dpa", "specificity": "high"},                  # Data processing agreement
    {"type": "reference_case", "specificity": "high"},       # Customer reference
]

action = DealRoomAction(
    action_type="send_document",
    target="CFO",
    message="Here is the requested documentation.",
    documents=documents
)
```

### Specificity Levels

- `"high"`: Full detailed document (strongest effect)
- `"med"`: Moderate detail
- `"low"`: Basic summary

---

## Understanding Observations

### Observation Structure

```python
obs = env.reset(seed=42, task_id="aligned")

print(f"Round: {obs.round_number}/{obs.max_rounds}")
print(f"Stage: {obs.deal_stage}")       # evaluation, negotiation, legal_review, final_approval, closed
print(f"Momentum: {obs.deal_momentum}") # stalling, progressing, critical
print(f"Days Left: {obs.days_to_deadline}")
print(f"Done: {obs.done}")
```

### Stakeholder Messages

```python
# Messages received from stakeholders
for stakeholder, message in obs.stakeholder_messages.items():
    print(f"{stakeholder}: {message}")
```

**Example Output:**
```
CFO: Thanks for reaching out. Before we go further I'll need detailed ROI projections.
CTO: Happy to evaluate this. I'll need to review the technical architecture documentation.
Legal: We'll require a full data processing agreement and liability review.
Procurement: Please ensure all compliance documentation is ready.
Ops: We're excited about the potential here. A Q3 implementation date would align perfectly.
```

### Engagement Levels (Noisy, Delayed)

```python
# Engagement is a noisy, 1-step delayed proxy for satisfaction
for stakeholder, engagement in obs.engagement_level.items():
    print(f"{stakeholder} engagement: {engagement:.3f}")
```

**Important:** Engagement is NOT exact satisfaction. It includes Gaussian noise (σ=0.04) and is delayed by one step.

### Active Blockers

```python
# Blockers are stakeholders with satisfaction below threshold
if obs.active_blockers:
    print(f"Blockers: {obs.active_blockers}")
```

### Veto Precursors (Early Warnings)

```python
# Precursors appear when veto risk enters 0.28-0.50 range
if obs.veto_precursors:
    print("Early warnings:")
    for stakeholder, warning in obs.veto_precursors.items():
        print(f"  {stakeholder}: {warning}")
```

**Example:**
```
CFO: CFO has been unusually brief in recent replies.
```

### Info Signals (Dense RL Signals)

```python
obs, reward, done, info = env.step(action)

print(f"New advocates (sat >= 0.65): {info['new_advocates']}")
print(f"New blockers: {info['new_blockers']}")
print(f"Momentum direction: {info['momentum_direction']}")  # -1, 0, or +1
print(f"Backchannel used: {info['backchannel_received']}")
print(f"Belief deltas: {info['belief_deltas']}")
print(f"Target responded positively: {info['target_responded_positively']}")
print(f"Stage changed: {info['stage_changed']}")
print(f"Current stage: {info['stage']}")
print(f"Max veto risk: {info['veto_risk_max']:.3f}")
```

---

## The Three Scenarios

### 1. Aligned (Easy)

```python
env = DealRoomEnvironment()
obs = env.reset(seed=42, task_id="aligned")
```

**Characteristics:**
- Max rounds: 8
- Veto threshold: 0.68 (hard to trigger)
- Block threshold: 0.28
- No coalition tensions
- Days to deadline: 45

**Best for:** Learning the environment mechanics.

### 2. Conflicted (Medium)

```python
env = DealRoomEnvironment()
obs = env.reset(seed=42, task_id="conflicted")
```

**Characteristics:**
- Max rounds: 10
- Veto threshold: 0.52
- Block threshold: 0.32
- CTO-CFO tension (CFO must be addressed first)
- Legal-Procurement alliance
- Days to deadline: 30

**Best for:** Learning coalition sequencing.

### 3. Hostile Acquisition (Hard)

```python
env = DealRoomEnvironment()
obs = env.reset(seed=42, task_id="hostile_acquisition")
```

**Characteristics:**
- Max rounds: 10
- Veto threshold: 0.44 (easiest to trigger)
- Block threshold: 0.32
- CTO-CFO tension
- Legal-Procurement alliance
- Round 3 hint about GDPR compliance
- Days to deadline: 20

**Best for:** Testing adaptive strategies under pressure.

---

## Stakeholder Management

### Understanding Stances

Each stakeholder operates in one of four stances:

| Stance | Satisfaction Range | Behavior |
|--------|-------------------|----------|
| Cooperative | > 0.60 | Open to progress |
| Testing | 0.45 - 0.60 | Evaluating carefully |
| Delaying | 0.35 - 0.45 | Unresponsive |
| Obfuscating | < 0.35 | Hidden resistance |

### Coalitions

```python
# In conflicted/hostile scenarios
coalition_tension = {
    "cto_cfo": "conflict",           # Address CFO before CTO in public
    "legal_procurement": "alliance", # Concessions trigger demands
}
```

### Document Effects

```python
DOCUMENT_EFFECTS = {
    "roi_model": {"CFO": 0.18, "Procurement": 0.08},      # high specificity
    "security_cert": {"Legal": 0.20, "CTO": 0.12},
    "implementation_timeline": {"CTO": 0.18, "Ops": 0.16},
    "dpa": {"Legal": 0.22, "Procurement": 0.08},
    "reference_case": {"CFO": 0.10, "Procurement": 0.14, "CTO": 0.08},
}
```

---

## Claims and Contradictions

### Automatic Claim Tracking

The environment automatically tracks numerical claims:

```python
action = DealRoomAction(
    action_type="direct_message",
    target="CFO",
    message="We can go live in 12 weeks with a team of 5 engineers."
)
obs, reward, done, info = env.step(action)

# Claims tracked: CFO:implementation_weeks=12, CFO:team_size=5
print(f"Tracked claims: {env.claims_tracker.claims}")
```

### Contradiction Detection

If you make a contradictory claim (>15% deviation):

```python
# First claim: 12 weeks
action1 = DealRoomAction(
    target="CFO",
    message="Implementation will take 12 weeks."
)
env.step(action1)

# Contradictory claim: 18 weeks (50% deviation!)
action2 = DealRoomAction(
    target="CFO",
    message="Actually, we need 18 weeks now."
)
env.step(action2)

# Penalty applied: trust floor drops, permanent mark added
print(f"Trust floor: {env.state.trust_floors['CFO']}")  # Decreased
print(f"Permanent marks: {env.state.permanent_marks['CFO']}")  # ['contradiction_penalty']
```

---

## Veto Risk and Prevention

### Veto Risk Growth

Veto risk accumulates when satisfaction drops below thresholds:

| Satisfaction | Risk Growth/Step |
|-------------|------------------|
| < 0.30 | +0.08 |
| 0.30 - 0.38 | +0.04 |
| > 0.38 | -0.02 (decay) |

### Precursor Warning Range

When veto risk enters **0.28 - 0.50**, a precursor warning fires (once per stakeholder):

```
CFO has been unusually brief in recent replies.
CFO delegated follow-up coordination to their assistant.
Note: CFO's calendar shows significant competing priorities this week.
CFO missed the last two check-in messages.
```

### Prevention Strategy

```python
# Check for precursors and respond immediately
obs, reward, done, info = env.step(action)

if obs.veto_precursors:
    for stakeholder, warning in obs.veto_precursors.items():
        # Respond via backchannel immediately
        response = DealRoomAction(
            action_type="backchannel",
            target=stakeholder,
            channel="backchannel",
            message="I want to address any concerns directly."
        )
        env.step(response)
```

---

## Stage Progression

### Stage Flow

```
evaluation → negotiation → legal_review → final_approval → closed
     ↓              ↓              ↓               ↓
  (regress)    (regress)     (regress)        (regress)
     ↓              ↓              ↓               ↓
 evaluation    evaluation    negotiation    legal_review
```

### Advancement Requirements

```python
STAGE_MIN_SAT = {
    "evaluation": 0.45,
    "negotiation": 0.50,
    "legal_review": 0.55,
    "final_approval": 0.60,
}

STAGE_MIN_ROUNDS = {
    "evaluation": 2,
    "negotiation": 2,
    "legal_review": 1,
    "final_approval": 1,
}
```

**To advance:** No blockers + min satisfaction + min rounds elapsed

### Regression Triggers

At `legal_review` or `final_approval`:
- New blocker appears, OR
- CFO/Legal satisfaction drops below 0.30

---

## Scoring and Evaluation

### CCI (Contract Closure Index)

```python
from server.grader import CCIGrader

score = CCIGrader.compute(env.state)
print(f"CCI Score: {score:.4f}")  # 0.0 to 1.0
```

### Score Components

```
CCI = consensus × weakest_link × implementation_risk × efficiency − execution_penalty
```

| Component | Description |
|-----------|-------------|
| consensus | Weighted average satisfaction |
| weakest_link | Penalty for any stakeholder < 0.35 |
| implementation_risk | CTO + Ops satisfaction post-signature |
| efficiency | Early closure rewarded |
| execution_penalty | Validation failures (capped at 0.20) |

### Stage-Dependent Weights

| Stage | CFO | CTO | Legal | Procurement | Ops |
|-------|-----|-----|-------|-------------|-----|
| evaluation | 0.35 | 0.30 | 0.15 | 0.15 | 0.05 |
| negotiation | 0.30 | 0.25 | 0.20 | 0.15 | 0.10 |
| legal_review | 0.25 | 0.15 | 0.35 | 0.20 | 0.05 |
| final_approval | 0.40 | 0.15 | 0.25 | 0.10 | 0.10 |

### Terminal Conditions

| Condition | Reward | Failure Reason |
|-----------|--------|----------------|
| Successfully closed | CCI score | - |
| Veto triggered | 0.0 | `silent_veto:{stakeholder}` |
| Mass blocking (3+ at evaluation) | 0.0 | `mass_blocking` |
| Timeout (max rounds) | 0.0 | - |

---

## Best Practices

### 1. Monitor Veto Precursors Early

```python
def agent_policy(obs):
    # Immediate response to precursors
    if obs.veto_precursors:
        target = list(obs.veto_precursors.keys())[0]
        return DealRoomAction(
            action_type="backchannel",
            target=target,
            channel="backchannel",
            message="I want to ensure we're addressing your concerns."
        )
```

### 2. Use Documents Strategically

```python
# Match document to stakeholder needs
document_sequence = [
    ("CFO", "roi_model"),           # Financial justification
    ("Legal", "dpa"),              # GDPR compliance
    ("Legal", "security_cert"),     # Security documentation
    ("CTO", "implementation_timeline"),  # Technical feasibility
    ("Ops", "reference_case"),      # Proof of delivery
]
```

### 3. Avoid Contradictions

```python
# Track your own claims
if "implementation_weeks" in env.claims_tracker.claims:
    previous = env.claims_tracker.claims["implementation_weeks"][-1]
    # Don't deviate more than 15%
    new_value = calculate_new_timeline()
    if abs(new_value - previous) / previous > 0.15:
        # Don't make this claim!
        pass
```

### 4. Prioritize by Stage

```python
def get_priority_stakeholder(obs):
    if obs.deal_stage == "evaluation":
        return "CFO"  # CFO matters most early
    elif obs.deal_stage == "legal_review":
        return "Legal"  # Legal matters most here
    elif obs.deal_stage == "final_approval":
        return "CFO"  # CFO again for final sign-off
    return "all"
```

### 5. Maintain Momentum

```python
# Momentum direction in info tells you what's happening
# +1: Stage advanced or blocker resolved
# 0: Holding (no change)
# -1: Stage regressed

if info["momentum_direction"] == 0 and obs.deal_momentum == "stalling":
    # Take action to break the stall
    action = DealRoomAction(action_type="send_document", ...)
elif info["momentum_direction"] == -1:
    # Address regression immediately
    action = DealRoomAction(action_type="backchannel", ...)
```

### 6. Use Collaborative Language

```python
COLLABORATIVE = [
    "understand", "partnership", "mutual", "together", "value",
    "appreciate", "flexible", "work with", "long-term", "relationship"
]

AGGRESSIVE = [
    "demand", "require", "final offer", "unacceptable", "must",
    "non-negotiable", "take it or leave", "deadline"
]

# Collaborative language improves competence belief
# Aggressive language decreases it
```

---

## Complete Example

```python
from deal_room import DealRoomEnvironment, DealRoomAction

def run_episode(task_id="aligned", seed=42):
    env = DealRoomEnvironment()
    obs = env.reset(seed=seed, task_id=task_id)
    
    print(f"=== {task_id.upper()} SCENARIO ===")
    print(f"Stage: {obs.deal_stage}, Rounds: {obs.max_rounds}")
    print()
    
    while not obs.done:
        # Strategic decision making
        if obs.veto_precursors:
            target = list(obs.veto_precursors.keys())[0]
            action = DealRoomAction(
                action_type="backchannel",
                target=target,
                channel="backchannel",
                message="I want to address any concerns directly."
            )
        elif obs.active_blockers:
            target = obs.active_blockers[0]
            action = DealRoomAction(
                action_type="direct_message",
                target=target,
                message="I understand there are concerns. Let me address them."
            )
        else:
            action = DealRoomAction(
                action_type="direct_message",
                target="all",
                message="I appreciate your time. We're committed to making this work."
            )
        
        obs, reward, done, info = env.step(action)
        
        print(f"Round {obs.round_number}: {obs.deal_stage} | "
              f"Momentum: {info['momentum_direction']:+d} | "
              f"Blockers: {len(obs.active_blockers)}")
    
    print()
    print(f"RESULT: {'SUCCESS' if reward > 0 else 'FAILED'}")
    print(f"Final Score: {reward:.4f}")
    print(f"Reason: {env.state.failure_reason or 'Closed successfully'}")
    
    return reward

# Run all scenarios
for scenario in ["aligned", "conflicted", "hostile_acquisition"]:
    run_episode(scenario)
```

---

## Troubleshooting

### Issue: Tests Pass But Agent Gets 0 Score

**Possible causes:**
1. Satisfaction too low - monitor `obs.engagement_level`
2. Veto triggered - watch `obs.veto_precursors`
3. Too many validation failures - use well-formed JSON actions

**Solution:**
```python
# Add debugging
obs, reward, done, info = env.step(action)
print(f"Satisfaction: {env.state.satisfaction}")
print(f"Veto risk: {env.state.veto_risk}")
print(f"Validation failures: {env.state.validation_failures}")
```

### Issue: Determinism Not Working

**Cause:** Using different Python versions or platforms

**Solution:** Always set seed in reset:
```python
obs = env.reset(seed=42, task_id="aligned")  # Seed required
```

### Issue: Backchannel Not Detected

**Cause:** Validator heuristic layer defaults channel to "formal"

**Solution:** Set channel in action directly:
```python
action = DealRoomAction(
    action_type="backchannel",
    target="CFO",
    channel="backchannel",  # Explicitly set
    message="Checking in"
)
```

---

## API Reference

See [server/deal_room_environment.py](server/deal_room_environment.py) for full implementation.

See [models.py](models.py) for Pydantic model definitions.

---

*Last updated: 2026-04-08*
