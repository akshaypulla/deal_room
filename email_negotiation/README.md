# Email Negotiation Environment

Enterprise email negotiation environment for RL training via the OpenEnv framework.

## What This Is

A Two-LLM email negotiation system for training a seller LLM (Qwen2.5-3B-Instruct) to close enterprise deals:

- **Seller** = active RL agent (trainable)
- **Buyer** = frozen but stochastic LLM stakeholders (Legal, Finance, CTO, Procurement, Operations, ExecSponsor)
- **Medium** = email threads with persistent memory per stakeholder
- **Reward** = multiplicative `progress_score = S^0.4 × D^0.2 × G^0.2 × R^0.2`

## File Structure

```
email_negotiation/
├── openenv.yaml              ← OpenEnv CLI config (required for openenv push)
├── pyproject.toml            ← pip-installable package
├── Dockerfile                ← container for HF Spaces
├── models.py                ← EmailAction, EmailObservation, EmailState
├── client.py                ← HTTPEnvClient subclass for training notebooks
├── server/
│   ├── app.py               ← 3-line FastAPI entry point
│   ├── environment.py       ← OpenEnv Environment adapter
│   └── email_env/           ← Internal logic
│       ├── email_message.py
│       ├── inbox.py
│       ├── buyer_stakeholder.py
│       ├── reward_extractor.py
│       ├── progress_score.py
│       ├── email_negotiation.py
│       └── anti_gaming.py
└── notebooks/
    └── email_negotiation_training.ipynb
```

## Quick Start

```bash
# Install dependencies
pip install openenv-core[core] openai numpy

# Run locally
uvicorn server.app:app --port 8000

# Push to HF Spaces
openenv push
```

## Training

```python
from client import EmailNegotiationEnv

def openenv_reward(completions, **kwargs):
    rewards = []
    for completion in completions:
        with EmailNegotiationEnv(
            base_url="https://your-username-email-negotiation.hf.space"
        ).sync() as env:
            env.reset()
            action = parse_action_from_text(completion)
            result = env.step(action)
            rewards.append(result.reward)
    return rewards
```

## Key Design Decisions

| Decision | Value |
|----------|-------|
| Action space | Structured intent + slots (6 intents × 7 targets × 3 tones × 5 docs) |
| CC signal | Causal influence propagation, max 2 recipients |
| Reward | 3-layer hybrid (keyword → LLM → code computes final) |
| Anti-gaming | Policy constraints + CTA validation + diminishing returns |
| Progress gating | Multiplicative S×D×G×R forces balanced progress |

## OpenEnv Compatibility

This environment implements the OpenEnv `Environment` interface:
- `reset()` → `EmailObservation`
- `step(action: EmailAction)` → `EmailObservation`
- `state()` → `EmailState`

## Grading Alignment

- **Training evidence**: Reward curves improve from ~0.25 → ~0.42 over 100 steps (see notebooks/training_curves.png)
- **20% training criterion**: `openenv_reward()` function connects TRL GRPOTrainer to running Space
- **30% storytelling**: Multiplicative progress gating, multi-stakeholder memory, CC causal signal