# DealRoom

An OpenEnv-compatible reinforcement learning environment for enterprise software negotiation. The agent acts as a vendor-side negotiator working through a realistic B2B deal with a buying committee that may include finance, legal/compliance, procurement, technical leadership, operations, and executive sponsors.

## Overview

DealRoom models real enterprise negotiation challenges:
- Discovering hidden blockers before a deal stalls
- Sequencing conversations across multiple stakeholders
- Sending the right evidence at the right time
- Avoiding premature escalation
- Turning partial support into durable approval

This is a real-world coordination problem, not a toy game. The task is partially observable and long-horizon — language and sequencing both matter, and successful behavior requires more than local pattern matching.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run local validation
pytest -q
openenv validate

# Start the server
uvicorn server.app:app --reload --port 7860

# Open the web UI
# http://127.0.0.1:7860/web

# Run baseline inference
python inference.py
```

## Three Benchmark Tasks

| Task | Difficulty | Description |
|------|------------|-------------|
| `aligned` | Easy | Low-friction enterprise deal. Correct document sequencing and engagement order. |
| `conflicted` | Medium | CTO-CFO tension, Legal-Procurement alliance. Coalition sequencing is primary skill. |
| `hostile_acquisition` | Hard | Post-acquisition authority shift, new compliance requirements. Adaptive mapping under pressure. |

## Baseline Scores

With `seed=42`:

| Task | Baseline Score |
|------|---------------|
| `aligned` | 0.86 |
| `conflicted` | 0.83 |
| `hostile_acquisition` | 0.80 |

## Project Structure

```
deal_room/
├── deal_room/              # Python package (root src)
│   ├── models.py           # Pydantic models: DealRoomAction, Observation, State
│   ├── scenarios.py        # Task configs: aligned, conflicted, hostile_acquisition
│   ├── grader.py          # CCIGrader
│   ├── validator.py       # OutputValidator
│   ├── claims.py          # ClaimsTracker
│   └── stakeholders.py    # StakeholderEngine + templates
├── server/                 # Server components
│   ├── app.py             # FastAPI application
│   ├── deal_room_environment.py  # Main RL environment
│   ├── semantics.py       # Semantic analyzer (TF-IDF/sentence-transformer fallback)
│   ├── session_pool.py    # Session pool for multi-user
│   └── gradio_custom.py   # Custom Gradio interface
├── docs/                  # Documentation
│   ├── architecture.md    # Environment architecture
│   ├── usage.md          # Practical usage guide
│   ├── testing.md        # Test suite documentation
│   └── openenv.yaml      # OpenEnv spec copy
├── tests/                 # Test suite
├── Dockerfile             # Multi-stage Docker build
├── openenv.yaml          # OpenEnv environment spec
├── calibrate.py          # Calibration script
└── inference.py          # Baseline inference script
```

## Architecture

```
Client/Agent
    │
    │ REST / OpenEnv API
    ▼
server/app.py (FastAPI wrapper)
    │
    ▼
deal_room_environment.py (Main Environment)
    ├── StakeholderEngine
    ├── ScenarioGenerator
    ├── CommitmentLedger
    ├── CCIGrader
    ├── SemanticAnalyzer
    └── OutputValidator
    │
    ▼
models.py / scenarios.py (Pydantic Models + Task Configurations)
```

Key design decisions:
- Zero LLM calls inside the `deal_room/` package — fully deterministic
- Hidden state (`stakeholder_private`, `hidden_constraints`) is only revealed through observation proxies
- Seeding guarantees reproducible episodes for RL training and evaluation
- ClaimsTracker uses regex-only contradiction detection

## API Reference

### Core Classes

**DealRoomEnvironment**
```python
from deal_room import DealRoomEnvironment
env = DealRoomEnvironment()
obs = env.reset(seed=42, task_id="aligned")
```

**DealRoomAction** — Action object the agent sends to the environment

| Field | Type | Description |
|-------|------|-------------|
| `action_type` | `str` | One of 8 action families |
| `target` | `str` | Target stakeholder or group |
| `target_ids` | `list[str]` | Explicit recipient IDs |
| `message` | `str` | Natural language negotiation move |
| `documents` | `list[dict]` | Supporting artifacts |
| `proposed_terms` | `dict\|null` | Structured commercial terms |
| `channel` | `str` | Communication mode |
| `mode` | `str` | Communication style |

**DealRoomObservation** — What the agent observes at each step

| Field | Type | Description |
|-------|------|-------------|
| `round_number` | `int` | Current round |
| `max_rounds` | `int` | Episode budget |
| `stakeholders` | `dict` | Active roster with role and authority |
| `stakeholder_messages` | `dict` | Visible stakeholder replies |
| `engagement_level` | `dict` | Noisy proxy for movement and support |
| `weak_signals` | `dict` | Indirect hints about hidden blockers |
| `known_constraints` | `list` | Constraints revealed to act on |
| `approval_path_progress` | `dict` | Public approval band and authority |
| `deal_momentum` | `str` | `progressing`, `stalling`, or `critical` |
| `deal_stage` | `str` | Stage in the approval pipeline |
| `active_blockers` | `list` | Stakeholders currently blocking |
| `days_to_deadline` | `int` | Remaining time pressure |
| `done` | `bool` | Whether episode has ended |

### REST API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/metadata` | GET | Environment metadata |
| `/reset` | POST | Reset environment with `task_id` and `seed` |
| `/step` | POST | Send action, receive observation |
| `/state` | GET | Get full internal state |
| `/web` | GET | Open Gradio web interface |

### Grading: Contract Closure Index (CCI)

The CCIGrader computes a score in `[0, 1]` across 5 dimensions:

| Component | Weight | Description |
|-----------|--------|-------------|
| `approval_completeness` | 40% | Weighted satisfaction with weakest-link penalty |
| `constraint_satisfaction` | 20% | Hidden feasibility constraints resolved |
| `term_feasibility` | 20% | Proposed terms pass feasibility checks |
| `relationship_durability` | 10% | Trust floors maintained |
| `efficiency` | 10% | Pacing relative to deadline |

## Action Types

| Action | Meaning |
| --- | --- |
| `direct_message` | Send a targeted message to a stakeholder |
| `backchannel` | Use a quieter coordination move to gather signal or reduce escalation risk |
| `send_document` | Share concrete evidence (ROI, DPA, security material, rollout plans) |
| `group_proposal` | Propose terms to multiple stakeholders or the whole committee |
| `concession` | Offer ground on terms or process |
| `walkaway_signal` | Signal risk of disengagement |
| `reframe_value_prop` | Reposition the value proposition for a role or coalition |
| `exec_escalation` | Push toward executive attention or formal approval pressure |

## License

BSD 3-Clause License. See LICENSE file for details.