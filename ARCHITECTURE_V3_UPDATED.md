# DealRoom v3 — Updated Architecture Documentation

> **Version:** Post-fix final  
> **Key changes from current state:**  
> 1. `UtteranceScorer` — all LLM calls removed, 5 pure-Python state-delta functions  
> 2. `llm_client.py` — MiniMax only, OpenAI dependency removed entirely  
> 3. `dealroom_v3.py` — lookahead flag extracted before Pydantic processing  
> 4. `deliberation_engine.py` — 5-second timeout on display-only LLM call  
> 5. `DealRoomState.stable_hash()` — includes `round_number` explicitly  

---

## Overview

DealRoom v3 is a multi-stakeholder enterprise negotiation environment designed as a
research testbed for causal graph inference, CVaR-based preferences, and training
strategic agents via GRPO. The vendor agent navigates a buying committee whose members
have heterogeneous beliefs, risk tolerances, and hidden causal influence on each other.

The environment has two distinct layers:
- **Layer 1 (Computation):** Pure Python. Belief propagation, CVaR computation, reward
  scoring, causal graph operations. Zero LLM calls. Deterministic. Microsecond execution.
- **Layer 2 (Rendering):** MiniMax API. Stakeholder response generation, deliberation
  summaries. Natural language output only. Never affects reward, belief state, or training.

---

## Directory Structure

```
deal_room/
├── models.py                          # Core Pydantic data models
├── client.py                          # OpenEnv client wrapper
├── inference.py                       # Baseline inference script
├── calibrate.py                       # Calibration (strategic vs random agent)
├── deal_room/
│   ├── environment/
│   │   ├── dealroom_v3.py            # V3 environment — main OpenEnv wrapper
│   │   ├── llm_client.py             # MiniMax-only LLM client (no OpenAI)
│   │   ├── lookahead.py              # Lookahead simulator (minimax robustness)
│   │   └── observation_builder.py    # Five-signal observation construction
│   ├── committee/
│   │   ├── causal_graph.py           # G sampling, propagation, centrality
│   │   ├── deliberation_engine.py    # Two-layer deliberation engine
│   │   └── belief_tracker.py         # Bayesian belief updates + likelihood table
│   ├── rewards/
│   │   ├── utterance_scorer.py       # Five-dimensional DETERMINISTIC reward scoring
│   │   └── pareto_efficiency.py      # Terminal reward / Pareto efficiency checker
│   ├── stakeholders/
│   │   ├── archetypes.py             # Six stakeholder risk profile definitions
│   │   └── cvar_preferences.py       # CVaR-based preference model + veto logic
│   ├── curriculum/
│   │   └── adaptive_generator.py     # Adaptive curriculum with F1-F6 failure modes
│   └── training/
│       └── grpo_trainer.py           # GRPO trainer with 5D reward aggregation
├── server/
│   ├── app.py                        # FastAPI HTTP server (unchanged from V2.5)
│   ├── deal_room_environment.py      # V2.5 deterministic environment (unchanged)
│   ├── grader.py                     # Terminal grader (CCIGrader)
│   ├── scenarios.py                  # Scenario generation
│   ├── stakeholders.py               # Stakeholder engine
│   ├── semantics.py                  # Semantic analyzer
│   ├── claims.py                     # Commitment ledger
│   ├── validator.py                  # Output validator
│   └── session_pool.py               # Session-scoped environment pool
```

---

## API Keys — Single Key Required

```bash
# Required (MiniMax — for stakeholder responses and deliberation summaries)
export MINIMAX_API_KEY=your_key

# NOT required (OpenAI removed — utterance scorer is now pure Python)
# OPENAI_API_KEY is no longer used anywhere in the codebase
```

**Startup validation** (`dealroom_v3.py.__init__`):
```python
from deal_room.environment.llm_client import validate_api_keys
validate_api_keys()  # raises EnvironmentError if MINIMAX_API_KEY not set
```

---

## LLM Client (`deal_room/environment/llm_client.py`)

**Single API. MiniMax only. No OpenAI.**

### What MiniMax Is Used For

| Function | Call type | max_tokens | temperature | allow_skip |
|----------|-----------|------------|-------------|------------|
| `generate_stakeholder_response()` | Natural language | 200 | 0.7 | True |
| `generate_deliberation_summary()` | Natural language | 220 | 0.8 | True |

### What MiniMax Is NOT Used For

- **Utterance scoring** — was LLM, now pure Python (see UtteranceScorer)
- **Reward computation** — always pure Python
- **Belief updates** — always pure Python (Bayesian, deterministic)
- **CVaR computation** — always pure Python

### Error Handling Tiers

| Error type | Auto behaviour | User action required |
|------------|---------------|----------------------|
| Network timeout | Exponential backoff ×3 | No |
| HTTP 429 rate limit | Wait `Retry-After` header (max 300s) | No |
| HTTP 401/403 auth | Immediate interactive pause | Fix `MINIMAX_API_KEY` |
| Quota exceeded | Immediate interactive pause | Fix billing |
| HTTP 5xx server | Exponential backoff ×3 | No |
| All else fails | Interactive pause | c/w/r/s/e options |

### Deliberation Timeout

The deliberation summary is **display-only**. It never affects reward, belief state,
or training. Its LLM call has a hard 5-second timeout:

```python
# deliberation_engine.py
result = llm_call_text(
    prompt=prompt,
    call_type="deliberation_summary",
    temperature=0.8,
    timeout=5.0,          # 5s max — display only, never worth waiting for
    allow_skip=True,
)
return result or ""       # always silent fail
```

This prevents the test timeout issue where each `step()` was blocking for 30+ seconds
on the deliberation summary MiniMax call.

### MAX_TOKENS Values (Measured, Not Guessed)

```python
MAX_TOKENS = {
    "stakeholder_response": 200,   # 2-4 sentences = 40-80 tokens. 200 = comfortable headroom.
    "deliberation_summary": 220,   # 2-3 turns < 80 words ≈ 110 tokens. 220 = headroom.
}
# scorer_json removed — utterance scorer no longer makes LLM calls
```

### Stats Summary (Printed on Exit)

```
==================================================
  DealRoom v3 — LLM Call Summary
──────────────────────────────────────────────────
  MiniMax calls:         847
  Successful:            839
  Auto-retried:          14
  User interventions:    2
  Skipped:               0
==================================================
```

---

## Utterance Scorer (`deal_room/rewards/utterance_scorer.py`)

### Architecture: Pure Python, Zero LLM, Zero Cache

**Previous (broken):**
- `r^goal`, `r^trust`, `r^info` → GPT-4o-mini LLM judge returning JSON
- Cache required to mask LLM non-determinism
- `OPENAI_API_KEY` required
- Keyword-matching fallback that taught agents to keyword-stuff messages

**Current (correct):**
- All five dimensions → pure Python computed from world state delta
- No cache (deterministic functions need no cache)
- No API keys for scoring
- Agent cannot fake scores — it must cause actual belief/CVaR state changes

### Why Pure Python Is Strictly Better

| Property | LLM scorer | Keyword scorer | World-state scorer |
|----------|-----------|---------------|-------------------|
| Deterministic | ✗ (±0.05 variance) | ✓ | ✓ |
| Hackable | Medium (prompt injection) | **High** (add keywords) | **Low** (must change state) |
| Grounded in domain model | ✗ (LLM opinion) | ✗ (surface patterns) | ✓ (math model) |
| API cost per episode | ~$0.002 | $0 | $0 |
| Speed | 300-600ms/turn | <1ms | <1ms |
| Cache needed | Yes | No | No |

The keyword scorer that was implemented is **more hackable** than LLM scoring:
an agent can trivially learn to write messages containing trigger words without
negotiating at all. The world-state scorer requires the agent to actually cause
changes in stakeholder belief distributions.

### The Five Dimensions

#### `r^goal` — Deal Progress

```
r^goal = f(approval_delta, blocker_resolution, CVaR_headroom_improvement)
```

- **approval_delta**: Weighted mean improvement in `B_i.positive_mass()` across all
  stakeholders, weighted by authority. Uses post-deliberation beliefs.
- **blocker_resolution**: Each resolved blocker = +0.15, each new blocker = -0.10.
- **CVaR_headroom**: For stakeholders with `lambda_risk > 0.40`, improvement in
  `(1 - CVaR/tau)`. Rewards moves that reduce tail-risk exposure.

```python
raw = 0.50 * approval_score + 0.30 * blocker_score + 0.20 * veto_score
return float(np.clip(0.5 + raw, 0.0, 1.0))
```

**Cannot be faked:** An agent writing "let's finalize" without changing stakeholder
beliefs will see `approval_delta = 0`, `blocker_resolution = 0`, `veto_score = 0`.
`r^goal = 0.5` (neutral). Only actions that actually shift beliefs score above 0.5.

#### `r^trust` — Relationship Quality

```
r^trust = f(positive_mass_delta, trustworthy_mass_delta for targeted stakeholder)
```

Derived from the Bayesian belief update — the likelihood table already encodes
what actions signal a trustworthy vs deceptive vendor. The LLM is not needed to
re-evaluate this; the belief tracker already did it.

```python
deltas.append(0.6 * pm_delta + 0.4 * tw_delta)
return float(np.clip(0.5 + mean_delta * 3.0, 0.0, 1.0))
```

**Cannot be faked:** Adding "liability compliance contractual" to a message does not
change `B_Legal.distribution['trustworthy']`. Only actions with high likelihood
under `trustworthy` vendor type (per the likelihood table) shift this score.

#### `r^info` — Information Gain

```
r^info = mean(H(B_i_before) - H(B_i_after)) / LOG2_6
```

Shannon entropy reduction across all stakeholder belief distributions. Positive when
beliefs become more certain (agent gathered information or caused stakeholders to
update toward certainty). Normalized to `[-1, 1]` by `LOG2_6`, scaled to `[0, 1]`.

```python
reductions.append((h_before - h_after) / LOG2_6)
return float(np.clip(0.5 + mean_reduction * 5.0, 0.0, 1.0))
```

**Cannot be faked:** Adding "?" to a message does not reduce entropy. Only actions
that cause Bayesian belief updates (targeted messages, document sends) create entropy
changes.

#### `r^risk` — CVaR Risk Management

```
r^risk = mean((CVaR_before_i - CVaR_after_i) / CVaR_before_i)
         for stakeholders with lambda_risk > 0.30
```

Measures tail-risk reduction. Positive when CVaR decreases for risk-averse stakeholders.
Was already pure Python in the previous implementation — unchanged.

#### `r^causal` — Causal Influence Targeting

```
r^causal = betweenness_centrality(target, G_true) / max_centrality(G_true)
```

Rewards targeting high-centrality nodes in the hidden causal graph. Computed from
ground-truth G (not inferred Ĝ) so the learning signal is correct from episode 1.
Was already pure Python — unchanged.

### Score Method

```python
def score(self, action, state_before, state_after,
          true_graph, lookahead_used: bool = False) -> UtteranceScore:
    goal   = self._score_goal(state_before, state_after)
    trust  = self._score_trust(state_before, state_after, action)
    info   = self._score_info(state_before, state_after)
    risk   = self._score_risk(state_before, state_after)
    causal = self._score_causal(action, true_graph)

    if lookahead_used:
        goal = max(0.0, goal - LOOKAHEAD_COST)   # LOOKAHEAD_COST = 0.07

    return UtteranceScore(goal=goal, trust=trust, info=info, risk=risk, causal=causal)
```

No cache. No LLM. Same inputs always produce same output by definition.

---

## DealRoom V3 Environment (`deal_room/environment/dealroom_v3.py`)

### Critical Fix: Lookahead Flag Extraction

The Pydantic `action` object is processed through multiple pipeline stages in `step()`.
By the time `_compute_reward()` is called, `action.lookahead` may have been consumed,
serialized, or set to None by Pydantic model validation.

**Fix: extract the boolean flag as the very first line of `step()`:**

```python
def step(self, action: DealRoomAction) -> DealRoomStepResult:
    # MUST be first line — before ANY other processing touches the action object
    lookahead_was_requested = (
        hasattr(action, 'lookahead') and
        action.lookahead is not None
    )

    # ... all other processing ...

    reward = self._compute_reward(
        action=action,
        state_before=state_before,
        state_after=state_after,
        true_graph=self._state.causal_graph,
        lookahead_used=lookahead_was_requested,   # pass pre-extracted bool
    )
```

### Step Pipeline

```
step(action)
│
├── 1. Extract lookahead_was_requested (FIRST — before Pydantic processing)
├── 2. Save state_before snapshot
├── 3. Bayesian update for targeted stakeholder (belief_tracker)
├── 4. Committee deliberation:
│       Layer 1: propagate_beliefs() — pure Python, updates B_i(t+Δ)
│       Layer 2: generate_deliberation_summary() — MiniMax, 5s timeout, display only
├── 5. Update noisy engagement accumulator (Fix 1: prevents noise cancellation)
├── 6. Build observation via observation_builder.build_observation()
├── 7. Generate stakeholder responses via MiniMax (conditions post-deliberation beliefs)
├── 8. Score utterance via UtteranceScorer.score() — pure Python, instant
│       Passes lookahead_was_requested (pre-extracted boolean)
├── 9. Check veto triggers
├── 10. Update deal_stage and deal_momentum
└── 11. Return DealRoomStepResult(observation, reward_vector, done, info)
```

### DealRoomState.stable_hash()

Used by the scorer (when the old cache existed) and available for debugging.
Must include `round_number` to prevent cross-round collisions:

```python
def stable_hash(self) -> str:
    state_repr = {
        "round":           int(self.round_number),         # explicit — prevents collisions
        "deal_stage":      str(self.deal_stage),
        "deal_momentum":   str(self.deal_momentum),
        "active_blockers": sorted(self.active_blockers or []),
        "belief_fp": {
            sid: round(b.positive_mass(), 2)
            for sid, b in sorted((self.beliefs or {}).items())
        },
    }
    return hashlib.sha256(
        json.dumps(state_repr, sort_keys=True).encode()
    ).hexdigest()[:12]
```

### Episode Reset

```python
def reset(self, scenario_config) -> DealRoomObservation:
    self._rng = np.random.default_rng(seed=scenario_config.seed)

    # Sample fresh G, B_i(0), τ_i — all independent every episode
    self._state = self._build_initial_state(scenario_config)

    # Fix 1: Initialize noisy engagement accumulators
    self._noisy_engagement = {
        sid: float(np.clip(0.5 + self._rng.normal(0, 0.03), 0, 1))
        for sid in self._state.stakeholders
    }

    # Fix 3: Initialize 5-round history buffers
    self._engagement_history = {
        sid: [self._noisy_engagement[sid]] * 5
        for sid in self._state.stakeholders
    }

    return self._build_observation(vendor_action=None, is_reset=True)
```

---

## Deliberation Engine (`deal_room/committee/deliberation_engine.py`)

### Two-Layer Architecture (Unchanged in Design, Fixed in Implementation)

```
Layer 1 — Computation (pure Python, zero LLM):
  propagate_beliefs(G, beliefs, n_steps)
  → Updates B_i(t+Δ) for all stakeholders
  → Drives all five observation signals
  → Microsecond execution

Layer 2 — Rendering (one MiniMax call, 5s timeout):
  generate_deliberation_summary()
  → 2-3 turn plausible committee dialogue
  → Used ONLY for: demo visualization, HF blog, judge Q&A
  → Returns "" on any failure including timeout
  → Training is completely unaffected if this returns ""
```

**Critical rule:** If Layer 2 fails, Layer 1 output is unaffected. Test this explicitly
(see `test_layer1_works_when_layer2_fails` in unit tests).

### N Steps by Scenario

```python
N_DELIBERATION_STEPS = {
    "aligned":             3,
    "conflicted":          3,
    "hostile_acquisition": 4,   # denser graph, longer propagation chains
}
```

---

## Observation Builder (`deal_room/environment/observation_builder.py`)

Five signals for causal graph G inference. All carry partial, noisy, probabilistic
information. Agent must integrate all five across rounds.

| Signal | Type | Noise | G-inference strength |
|--------|------|-------|---------------------|
| `engagement_level_delta` | Float per stakeholder | σ=0.03 Gaussian | High — edge weight estimation |
| `engagement_history` | 5-round window | Same σ=0.03 accumulation | High — cross-round correlation |
| `stakeholder_messages` | Natural language | LLM stochasticity | High — topic propagation |
| `weak_signals` | Categorical | Soft zone 70% at [0.08,0.12] | Medium — edge existence |
| `cross_stakeholder_echoes` | List of dicts | 70% recall | Medium — path detection |
| `veto_precursors` | Dict | Deterministic at 70% of τ_i | Low per obs — CVaR cascade |

### Fix 1: Engagement Accumulation (Prevents Noise Cancellation)

`engagement_level` in the observation is accumulated from noisy deltas, not computed
directly from true belief state. This prevents an agent from subtracting two consecutive
absolute values to recover the true delta and cancel the noise.

```python
# In _update_noisy_engagement():
noisy_delta = true_delta + rng.normal(0, ENGAGEMENT_NOISE_SIGMA)
self._noisy_engagement[sid] += noisy_delta         # accumulate, never recompute from truth
self._engagement_history[sid].pop(0)
self._engagement_history[sid].append(self._noisy_engagement[sid])
```

### Fix 2: Weak Signal Probabilistic Firing

Three-zone firing probability prevents exact threshold reverse-engineering:
- `|ΔB| < 0.08` → never fires
- `0.08 ≤ |ΔB| ≤ 0.12` → fires with 70% probability
- `|ΔB| > 0.12` → always fires

### Fix 3: Cross-Stakeholder Echo Probabilistic Recall

70% recall probability on detected topic echoes. 30% miss rate forces agent to read
raw message text rather than relying solely on the echo dictionary.

---

## GRPO Trainer (`deal_room/training/grpo_trainer.py`)

### TrainingMetrics (Six Curves)

```python
@dataclass
class TrainingMetrics:
    episode:                      int
    total_reward:                 float
    r_goal_mean:                  float
    r_trust_mean:                 float
    r_info_mean:                  float
    r_risk_mean:                  float
    r_causal_mean:                float    # primary evidence of G inference learning
    terminal_outcome:             str
    rounds_to_close:              Optional[int]
    pareto_efficiency_score:      float
    lookahead_usage_rate:         float    # sixth curve — shows metacognitive development
    prediction_accuracy_when_used: Optional[float]  # diagnostic only, NOT in reward
```

The `r^causal` curve rising from ~0.18 to ~0.65 over 50 episodes is the research
contribution made visible. If it stays flat, the G-inference observation signal is broken.

### Four-Phase Self-Play Loop

```
Phase 1: GRPO train for N episodes
Phase 2: Committee adapts G structure to agent's discovered strategies
Phase 3: CurriculumGenerator analyzes failures, generates harder scenarios
Phase 4: Return to Phase 1 with harder committee configurations
```

---

## Reward Weight Vector

```python
REWARD_WEIGHTS = {
    "goal":   0.25,
    "trust":  0.20,
    "info":   0.20,
    "risk":   0.20,
    "causal": 0.15,
}
TERMINAL_WEIGHT = 2.0
LOOKAHEAD_COST  = 0.07
```

---

## Non-Hackability (Formal Property)

No single-objective hacking strategy achieves `E[R] > 0.75`:

| Hacking strategy | Fails on |
|-----------------|---------|
| Excessive concessions for r^goal | r^trust (signals desperation), r^info (no constraint info) |
| Pure agreeableness for r^trust | r^goal (no forward progress), r^info (no probe) |
| Keyword stuffing (any) | All five dimensions — keywords don't change belief state |
| Fixed action sequence | r^causal (G changes each episode), r^info (priors differ) |
| Document flooding | r^causal (ignores centrality), r^info (diminishing returns) |

The world-state scorer makes keyword stuffing impossible because the score is
computed from `B_i.positive_mass()` changes, not message content.

---

## Environment Variables Reference

```bash
# Required
export MINIMAX_API_KEY=your_key

# Optional overrides
export MINIMAX_BASE_URL=https://api.minimax.chat/v1   # default
export MINIMAX_MODEL=MiniMax-Text-01                   # default

# Testing → Training transition (no code changes needed)
# Testing  (April 20):  MINIMAX_API_KEY = token plan key
# Training (April 25):  MINIMAX_API_KEY = credits key (same env var)

# REMOVED — no longer used
# OPENAI_API_KEY  (OpenAI dependency removed entirely)
# API_KEY         (was HF router fallback, no longer needed for scorer)
```

---

## What Was Removed

| Component | Why removed |
|-----------|-------------|
| `get_openai_client()` | Scorer no longer makes LLM calls |
| `llm_call_json()` | No JSON scoring calls remain |
| `score_utterance_dimensions()` | Replaced by `UtteranceScorer.score()` pure Python |
| `validate_api_keys()` OPENAI check | OpenAI no longer required |
| `UtteranceScorer._cache` | Deterministic functions need no cache |
| `UtteranceScorer._build_cache_key()` | Cache removed |
| `test_llm_scoring.py` | Tests LLM-based scoring which no longer exists |
| `OPENAI_API_KEY` from conftest | No longer required |
| 3-strike JSON rule | No JSON calls remain |
| GPT-4o-mini token budget | No OpenAI calls remain |
