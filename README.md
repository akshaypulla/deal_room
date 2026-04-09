---
title: Deal Room Environment Server
emoji: 🏢
colorFrom: yellow
colorTo: blue
sdk: docker
pinned: false
app_port: 7860
base_path: /web
tags:
  - openenv
---

# DealRoom

## Environment Description & Motivation
DealRoom is an OpenEnv-compatible reinforcement learning environment for enterprise software negotiation. The agent acts as a vendor-side negotiator working through a realistic B2B deal with a buying committee that may include finance, legal/compliance, procurement, technical leadership, operations, and executive sponsors.

This environment models a real task humans actually do:
- discovering hidden blockers before a deal stalls,
- sequencing conversations across multiple stakeholders,
- sending the right evidence at the right time,
- avoiding premature escalation,
- and turning partial support into durable approval.

Why this environment is useful:
- enterprise negotiation is a real-world coordination problem, not a toy game,
- the task is partially observable and long horizon,
- language and sequencing both matter,
- and successful behavior requires more than local pattern matching.

DealRoom is built for:
- RL training under partial observability,
- evaluation of long-horizon planning,
- benchmarking negotiation and coordination policies,
- and testing whether an agent can recover from mistakes without collapsing deal feasibility.

## Action Space
The environment uses a typed object action represented by `DealRoomAction`.

Action fields:
- `action_type` (`str`): one discrete action family
- `target` (`str`): compatibility alias such as `all` or a stakeholder id
- `target_ids` (`list[str]`): explicit recipient ids for the current episode roster
- `message` (`str`): natural-language negotiation move
- `documents` (`list[dict]`): optional supporting artifacts
- `proposed_terms` (`dict | null`): optional structured offer terms
- `channel` (`str`): communication metadata
- `mode` (`str`): communication mode metadata

Supported action families:

| Action | Meaning |
| --- | --- |
| `direct_message` | Send a targeted message to a stakeholder |
| `backchannel` | Use a quieter coordination move to gather signal or reduce escalation risk |
| `send_document` | Share concrete evidence such as ROI, DPA, security material, or rollout plans |
| `group_proposal` | Propose terms to multiple stakeholders or the whole committee |
| `concession` | Offer ground on terms or process |
| `walkaway_signal` | Signal risk of disengagement |
| `reframe_value_prop` | Reposition the value proposition for a role or coalition |
| `exec_escalation` | Push toward executive attention or formal approval pressure |

Common document types:
- `roi_model`
- `reference_case`
- `dpa`
- `security_cert`
- `vendor_packet`
- `implementation_timeline`
- `support_plan`

Common structured term fields:
- `price`
- `timeline_weeks`
- `security_commitments`
- `support_level`
- `liability_cap`

Action space type:
- discrete action family with structured typed parameters

## Observation Space
The observation space is a typed object represented by `DealRoomObservation`.

The agent sees:

| Field | Type | Meaning |
| --- | --- | --- |
| `round_number` | `int` | Current round |
| `max_rounds` | `int` | Episode budget |
| `stakeholders` | `dict[str, dict]` | Active roster with role and authority summary |
| `stakeholder_messages` | `dict[str, str]` | Visible stakeholder replies |
| `engagement_level` | `dict[str, float]` | Noisy public proxy for movement and support |
| `weak_signals` | `dict[str, list[str]]` | Indirect hints about hidden blockers |
| `known_constraints` | `list[dict]` | Constraints sufficiently revealed to act on |
| `requested_artifacts` | `dict[str, list[str]]` | Evidence still being requested |
| `approval_path_progress` | `dict[str, dict]` | Public approval band and authority info |
| `deal_momentum` | `str` | `progressing`, `stalling`, or `critical` |
| `deal_stage` | `str` | Stage in the approval pipeline |
| `competitor_events` | `list[str]` | External pressure events |
| `veto_precursors` | `dict[str, str]` | Early warning signs before a silent veto |
| `active_blockers` | `list[str]` | Stakeholders currently blocking movement |
| `days_to_deadline` | `int` | Remaining time pressure |
| `done` | `bool` | Whether the episode has ended |
| `info` | `dict` | Auxiliary signals for analysis/debugging |

Stakeholder-level visible attributes:
- `display_name`
- `role`
- `mandatory`
- `authority`

What is intentionally hidden:
- true utility weights,
- hidden feasibility constraints until inferred,
- private resistance beyond what weak signals imply,
- and internal relationship effects until they become externally visible.

Observation space type:
- structured object with dynamic roster and partial observability

## Task Description
The agent must close a feasible, durable enterprise deal before timeout.

Success criteria:
- the deal closes,
- all mandatory approvers are workable or supportive,
- no hard constraints remain unresolved,
- the final terms are feasible,
- and veto-level resistance is avoided.

Failure modes:
- timeout,
- silent veto,
- infeasible terms,
- unresolved hidden constraints,
- and cumulative trust damage that caps recovery.

## Difficulty Levels
DealRoom includes three benchmark levels that map naturally to a learning progression.

### Simple
Task id: `aligned`

What it models:
- a lower-friction buying committee,
- fewer active stakeholders,
- one main hidden blocker,
- and higher observability.

Why it is easiest:
- shorter approval path,
- fewer conflicting incentives,
- and easier diagnosis of which evidence matters.

### Medium
Task id: `conflicted`

What it adds:
- more stakeholders,
- conflicting incentives,
- approval drag,
- and relationship-sensitive sequencing.

Why it is harder than simple:
- the agent must manage stakeholder tension instead of a mostly aligned committee,
- new evidence requests matter more,
- and one action can help one role while slowing another.

### Hard
Task id: `hostile_acquisition`

What it models:
- post-acquisition pressure,
- authority shift events,
- multiple hidden constraints,
- and lower tolerance for inconsistency or premature pressure.

Why it is harder than medium:
- time pressure is tighter,
- recovery from mistakes is harder,
- feasibility can shift mid-episode,
- and multi-party coordination is less forgiving.

## Setup Instructions
### 1. Clone the repository
```bash
git clone <your-repo-url>
cd deal_room
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run local validation
```bash
pytest -q
openenv validate
```

### 4. Start the server
```bash
uvicorn server.app:app --reload --port 7860
```

### 5. Optional Docker setup
```bash
docker build -t deal-room-env:latest -f Dockerfile .
docker run --rm -p 7860:7860 deal-room-env:latest
```

## Usage Instructions
### Start the environment locally
```bash
uvicorn server.app:app --reload --port 7860
```

Primary endpoints:
- `GET /health`
- `GET /metadata`
- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /web`

### Open the web UI
```text
http://127.0.0.1:7860/web
```

The UI behavior:
- `Playground` stays as the direct interaction tab
- `Custom` is the teaching and explanation tab

### Run the baseline
```bash
python inference.py
```

Hackathon credential behavior:
- during evaluation, `inference.py` prefers `API_KEY` + `API_BASE_URL`,
- for local development, it can fall back to `OPENAI_API_KEY` or `HF_TOKEN`,
- and it uses the OpenAI client when the LiteLLM proxy is injected.

### Run the submission validator
```bash
bash scripts/validate-submission.sh
```

### Run the route smoke test
```bash
python scripts/container_route_smoke.py http://127.0.0.1:7860
```

## Baseline Scores
Current local benchmark with `seed=42`:

| Level | Task | Baseline score |
| --- | --- | --- |
| Simple | `aligned` | `0.86` |
| Medium | `conflicted` | `0.83` |
| Hard | `hostile_acquisition` | `0.80` |

Recent stress snapshot:

| Task | Mean score |
| --- | --- |
| `aligned` | `0.8761` |
| `conflicted` | `0.8118` |
| `hostile_acquisition` | `0.5905` |

Expected interpretation:
- `aligned` should be consistently solvable by the baseline,
- `conflicted` should remain strong but require more sequencing discipline,
- `hostile_acquisition` should be the least stable and closest to real-world pressure.

## Project Layout
- [models.py](/Users/akshaypulla/Documents/deal_room/models.py)
- [server/app.py](/Users/akshaypulla/Documents/deal_room/server/app.py)
- [server/deal_room_environment.py](/Users/akshaypulla/Documents/deal_room/server/deal_room_environment.py)
- [server/grader.py](/Users/akshaypulla/Documents/deal_room/server/grader.py)
- [server/gradio_custom.py](/Users/akshaypulla/Documents/deal_room/server/gradio_custom.py)
- [server/scenarios.py](/Users/akshaypulla/Documents/deal_room/server/scenarios.py)
- [server/semantics.py](/Users/akshaypulla/Documents/deal_room/server/semantics.py)
- [inference.py](/Users/akshaypulla/Documents/deal_room/inference.py)
