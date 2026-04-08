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

# Deal Room

Deal Room is an OpenEnv environment for enterprise software negotiation. The agent acts as a vendor-side negotiator trying to close a realistic B2B deal against a dynamic internal buying committee with hidden constraints, approval chains, and irreversible trust damage.

This V2.5 environment is built to be useful for RL and agent evaluation, not just scripted prompting. Each episode generates a seeded scenario with `2-4` stakeholders, `1-2` hidden hard constraints, up to `2` internal relationship edges, dense milestone rewards, and a deterministic terminal grader.

The web UI at `/web` keeps the native OpenEnv `Playground` untouched and adds a second `Custom` tab for judges. That custom tab includes a guided walkthrough, live sandbox, judge lens, counterfactual warnings, mistake tracking, and replay diff views.

## Why this is different

- Stakeholders are dynamic, not fixed. A deal may include finance, technical, legal/compliance, procurement, operations, or an executive sponsor.
- Constraints are partially observable. Budget, compliance, delivery, and process blockers appear through weak signals before they become fully known.
- Language matters. The environment uses a deterministic local semantic layer to score paraphrase-sensitive request matching, contradiction, and role-aware tone.
- Closing early is punished. Deals only close when terms are feasible and all mandatory approvers are workable.

## Tasks

- `aligned`: easier deal, `2-3` stakeholders, one hidden constraint, high observability.
- `conflicted`: medium difficulty, `3-4` stakeholders, one coalition edge, medium ambiguity.
- `hostile_acquisition`: hardest deal, `4` stakeholders, authority shift, two hidden constraints, low tolerance for inconsistency.

All tasks return rewards in `[0.0, 1.0]`. Intermediate steps use bounded dense reward and successful terminal close returns the deterministic grader score.

## Action Space

`DealRoomAction`

- `action_type`: `direct_message | group_proposal | backchannel | send_document | concession | walkaway_signal | reframe_value_prop | exec_escalation`
- `target`: compatibility alias such as `all`, `finance`, `technical`, `legal_compliance`, or legacy aliases like `CFO`
- `target_ids`: explicit dynamic stakeholder IDs for the active episode
- `message`: natural-language negotiation move
- `documents`: optional artifacts such as `roi_model`, `implementation_timeline`, `security_cert`, `dpa`, `vendor_packet`, `reference_case`, `support_plan`
- `proposed_terms`: optional structured offer with keys `price`, `timeline_weeks`, `security_commitments`, `support_level`, `liability_cap`
- `channel`, `mode`: communication metadata

## Observation Space

`DealRoomObservation`

- `stakeholders`: active roster with display name, role, authority, and mandatory status
- `stakeholder_messages`: current visible stakeholder replies
- `engagement_level`: noisy proxy for approval
- `weak_signals`: ambiguous hints about internal blockers
- `known_constraints`: constraints that have been fully uncovered
- `requested_artifacts`: still-missing artifacts by stakeholder
- `approval_path_progress`: approval band and authority for each stakeholder
- `deal_stage`: `evaluation -> negotiation -> legal_review -> final_approval -> closed`
- `active_blockers`, `veto_precursors`, `scenario_hint`, `competitor_events`, `days_to_deadline`

## Reward Model

- Dense milestone reward per step, capped at `0.15`
- Milestones include discovering hidden constraints, satisfying requested artifacts, moving mandatory approvers up a band, removing blockers, and legitimate stage advances
- Bad actions do not create large negative reward, but they apply permanent marks and feasibility damage that reduce future returns and the final score
- Terminal score uses approval completeness, constraint satisfaction, term feasibility, relationship durability, and efficiency

## Local Setup

```bash
pip install -r requirements.txt
pytest -q
openenv validate
uvicorn server.app:app --reload --port 7860
```

## Docker

```bash
docker build -t deal-room-env:latest -f Dockerfile .
docker run --rm -p 7860:7860 deal-room-env:latest
```

The Docker image is optimized for reliable startup on limited hardware. The semantic analyzer will use the lightweight embedding path when the dependency is available, and otherwise falls back to deterministic lexical similarity so the server still starts cleanly and reproducibly.

## Hugging Face Spaces

### Prerequisites

1. **Hugging Face Account**: Create an account at [huggingface.co](https://huggingface.co)
2. **Get an HF Token**: Generate a token at [hf.co/settings/tokens](https://hf.co/settings/tokens) with "write" permissions
3. **Set the token**:

```bash
export HF_TOKEN="your_huggingface_token_here"
```

Or use the `huggingface_hub` CLI:
```bash
huggingface-cli login
```

### Deploy to Hugging Face Spaces

```bash
# Option 1: Using openenv CLI (recommended)
openenv push

# Option 2: With custom repository ID
openenv push -r <your_username>/deal-room

# Option 3: Using the deploy script
./deploy.sh <your_username>/deal-room
```

### Expected Space

After deployment, your space will be available at:
```
https://huggingface.co/spaces/<your_username>/deal-room
```

### Space Features

- **Web UI**: Interactive Gradio interface at `/web`
- **Playground Tab**: Native OpenEnv playground for testing
- **Custom Tab**: Judge tools including walkthrough, sandbox, and diff views
- **API Endpoints**:
  - `/health` - Health check
  - `/metadata` - Environment metadata
  - `/reset` - Reset environment
  - `/step` - Execute action
  - `/state` - Get current state

Expected endpoints:

- `/health`
- `/metadata`
- `/reset`
- `/step`
- `/state`

## Baseline

`inference.py` runs a seeded protocol baseline across all three tasks and prints the required structured logs:

```bash
python inference.py
```

Credential resolution behavior:

- During hackathon evaluation, `inference.py` prefers the injected `API_BASE_URL` + `API_KEY` pair and will use the OpenAI client against that LiteLLM proxy by default.
- For local development, it can fall back to `OPENAI_API_KEY` or `HF_TOKEN` if `API_KEY` is not present.
- Set `DEALROOM_ENABLE_LLM_MESSAGES=0` only if you intentionally want fully local fallback behavior without proxy calls.

Current local heuristic smoke run with `seed=42`:

- `aligned`: `0.85`
- `conflicted`: `0.83`
- `hostile_acquisition`: `0.79`

Recent 4-seed stress snapshot:

- `aligned` mean: `0.8761`
- `conflicted` mean: `0.8118`
- `hostile_acquisition` mean: `0.5905`

The task ladder is deterministic and currently satisfies the expected ordering `aligned > conflicted > hostile_acquisition`.

## Pre-Submission Validation

Run the local validator before pushing:

```bash
bash scripts/validate-submission.sh
```

## Project Layout

- [models.py](/Users/akshaypulla/Documents/deal_room/models.py)
- [server/deal_room_environment.py](/Users/akshaypulla/Documents/deal_room/server/deal_room_environment.py)
- [server/scenarios.py](/Users/akshaypulla/Documents/deal_room/server/scenarios.py)
- [server/stakeholders.py](/Users/akshaypulla/Documents/deal_room/server/stakeholders.py)
- [server/semantics.py](/Users/akshaypulla/Documents/deal_room/server/semantics.py)
- [server/grader.py](/Users/akshaypulla/Documents/deal_room/server/grader.py)
- [inference.py](/Users/akshaypulla/Documents/deal_room/inference.py)
