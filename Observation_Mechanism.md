# OBSERVATION_MECHANISM.md

# DealRoom v3 — Final Observation Design & Causal Graph Inference Specification

> **Version:** 3.0 Final — all three post-review fixes incorporated  
> **Fixes applied:**  
>  Fix 1 — Engagement leakage: absolute levels now accumulated from noisy deltas  
>  Fix 2 — Weak signal artifact: probabilistic firing near threshold  
>  Fix 3 — History truncation: 5-round sliding window added to observation
>
> **Core invariant:** The agent never observes G, B_i(t), τ_i, deliberation transcripts,  
> or true edge weights. All causal graph inference comes exclusively from the  
> five observable signals defined here.

---

## 0. Configuration Dataclass (All Tunable Parameters)

All noise levels and probability parameters are centralized here. After the first training run, adjust these values based on learning curve shape before changing any other code.

```python
from dataclasses import dataclass

@dataclass
class ObservationConfig:
    # Signal 2: Engagement noise
    engagement_noise_sigma: float = 0.03
    # Calibration target: single-obs std > 0.04, 4-obs batch std < 0.03
    # If r^causal curve is flat after episode 15: decrease sigma (too noisy)
    # If r^causal curve plateaus at 0.30 after episode 20: increase sigma (too easy)

    # Signal 4: Echo recall
    echo_recall_probability: float = 0.70
    # 70% = agent uses dict as strong evidence, always cross-checks messages
    # If agent learns to ignore messages entirely: decrease to 0.60
    # If echo dict becomes too noisy to use: increase to 0.80

    # Signal 3: Weak signal firing probabilities
    weak_signal_hard_threshold: float = 0.12   # always fires above this
    weak_signal_soft_lower: float = 0.08       # never fires below this
    weak_signal_soft_probability: float = 0.70  # fires with this prob in [0.08, 0.12]

    # Signal 1: Reference injection
    reference_injection_threshold: float = 0.10  # |ΔB| above which injection guaranteed
    minimax_base_reference_target: float = 0.60   # minimum acceptable base rate

    # Fix 3: History window
    engagement_history_window: int = 5  # rounds of engagement history in observation

    # CVaR warning
    veto_warning_threshold_ratio: float = 0.70  # fraction of τ_i

# Singleton used throughout the environment
OBS_CONFIG = ObservationConfig()
```

---

## 1. Why This Document Exists

Every other component of DealRoom v3 — the causal graph, the CVaR preferences, the reward function — is only as good as what the agent can observe. If observation is too rich, graph inference is trivial. If too sparse, no learning signal exists.

The design challenge:

> Make G **identifiable** (learnable in principle) but not **trivially identifiable** (solvable in one round).

The solution is five signals, each carrying partial, noisy, probabilistic information about G. The agent must integrate all five across multiple rounds. No shortcut strategy — parsing only one signal — achieves high `r^causal`.

Three post-review fixes were required to close information leakage paths that would have undermined this design. They are incorporated throughout.

---

## 2. The Causal Inference Setup

### What the Agent Does

Every vendor action is simultaneously a negotiation move and a causal intervention. Targeting Finance with `send_document(roi_model)` applies `do(Finance)`, shifting `B_Finance` by `ΔB_Finance`. The deliberation engine propagates this through G:

```
B_i(t+Δ) = B_i(t) + Σ_{j ∈ influencers(i)} w_ji · ΔB_j(t)
```

The agent observes the effects on non-targeted stakeholders. Magnitude and pattern of effects = evidence about G.

### The Textbook Example

```
Round 5 action:   send_document(roi_model) → targeted at Finance only
True ΔB_Finance = +0.40

Deliberation (hidden):
  ΔB_Legal       = 0.70 × 0.40 = +0.280
  ΔB_Compliance  = 0.50 × 0.28 = +0.140  (second hop through Legal)
  ΔB_Procurement = 0.20 × 0.40 = +0.080
  ΔB_TechLead    = 0                       (no edge from Finance)

Round 6 observation:
  Finance:     [expected] ROI response
  Legal:       [unexpected] references compliance/documentation clarity
               engagement_level_delta ≈ +0.17 ± noise  → evidence: w_Finance→Legal ≈ 0.7
  Compliance:  [unexpected] vague alignment reference
               engagement_level_delta ≈ +0.08 ± noise  → evidence: path Finance→Legal→Compliance
  Procurement: [unexpected] minimal acknowledgment
               engagement_level_delta ≈ +0.04 ± noise  → evidence: w_Finance→Proc ≈ 0.2
  TechLead:    [unchanged]  standard technical response
               engagement_level_delta ≈  0.00 ± noise  → evidence: no Finance→TechLead edge
```

One targeted action, four pieces of G evidence.

---

## 3. Fix 1 — Engagement Level Accumulation (Preventing Noise Cancellation)

### The Information Leakage Problem

If `engagement_level` (absolute) is computed directly from `compute_engagement_level(B_i)` each round, and `engagement_level_delta` is the difference, then:

```python
# Agent can compute:
true_delta = obs.engagement_level[t] - obs.engagement_level[t-1]
# This cancels the noise entirely — one round of memory removes σ=0.03 protection
```

The noise on deltas becomes meaningless. G is identifiable in one observation.

### The Fix — Accumulate From Noisy Deltas

Absolute engagement levels are themselves accumulated from the noisy deltas. The agent can never recover the true delta because the true absolute level is never exposed.

```python
class DealRoomV3(Environment):

    def reset(self, scenario_config: ScenarioConfig) -> DealRoomObservation:
        """
        Initialize noisy engagement state.
        Starting values are 0.5 ± small noise to prevent agents from
        exploiting a known round-0 baseline.
        """
        self._rng = np.random.default_rng(seed=self._episode_seed)
        self._scenario = scenario_config
        self._state = self._build_initial_state(scenario_config)

        # FIX 1: Noisy engagement accumulator — never exposes true engagement
        self._noisy_engagement: Dict[str, float] = {
            sid: float(np.clip(
                0.5 + self._rng.normal(0, OBS_CONFIG.engagement_noise_sigma),
                0.0, 1.0
            ))
            for sid in self._state.stakeholders
        }

        # FIX 3: History buffer (5-round sliding window)
        self._engagement_history: Dict[str, List[float]] = {
            sid: [self._noisy_engagement[sid]] * OBS_CONFIG.engagement_history_window
            for sid in self._state.stakeholders
        }

        return self._build_observation(vendor_action=None, is_reset=True)

    def _update_noisy_engagement(self, true_beliefs_current: Dict, true_beliefs_previous: Dict):
        """
        Compute true deltas from belief states, add noise, accumulate into noisy_engagement.
        Called once per step, after deliberation has run.

        Returns the noisy deltas (for observation) and updates the accumulators (for history).
        Both engagement_level and engagement_level_delta in the observation share the same noise.
        Agent subtraction of two consecutive engagement_level values gives:
          (noisy_eng[t]) - (noisy_eng[t-1])
          = (true_eng[t-1] + Σ_noisy_deltas_up_to_t) - (true_eng[t-2] + Σ_noisy_deltas_up_to_t-1)
          = true_delta[t] + noise[t]   ← same noise, cannot cancel
        """
        noisy_deltas = {}

        for sid in true_beliefs_current:
            true_eng_current = compute_engagement_level(true_beliefs_current[sid])
            true_eng_previous = compute_engagement_level(true_beliefs_previous[sid])
            true_delta = true_eng_current - true_eng_previous

            noise = self._rng.normal(0, OBS_CONFIG.engagement_noise_sigma)
            noisy_delta = float(np.clip(true_delta + noise, -1.0, 1.0))

            # Accumulate into noisy engagement (this is what agent sees as "absolute")
            self._noisy_engagement[sid] = float(np.clip(
                self._noisy_engagement[sid] + noisy_delta, 0.0, 1.0
            ))

            # Update history (drop oldest, append newest)
            self._engagement_history[sid].pop(0)
            self._engagement_history[sid].append(self._noisy_engagement[sid])

            noisy_deltas[sid] = noisy_delta

        return noisy_deltas
```

### Why This Closes the Leak

The agent observes `engagement_level[t]` which equals `engagement_level[t-1] + noisy_delta[t]`. If it subtracts, it gets `noisy_delta[t]` — the same noise that's already in `engagement_level_delta`. There is no additional information in the absolute level beyond what the delta already carries. The noise cannot be cancelled.

---

## 4. Signal 2 — `engagement_level` and `engagement_level_delta` (Primary Quantitative Signal)

### What It Is

A continuous `[0, 1]` float per stakeholder, maintained via noisy accumulation (Fix 1). The delta is the primary numerical G inference signal. This is the signal that makes DealRoom v3 learnable by a 3B model — it provides numerical gradients the policy can learn from without parsing natural language every round.

### Underlying Computation

```python
LOG2_6 = np.log2(6)

def compute_engagement_level(belief: BeliefDistribution) -> float:
    """
    Deterministic function of belief state.
    This is the TRUE engagement level — never directly exposed to agent.
    Only used internally to compute true deltas before noise is applied.
    """
    positive_mass = belief.get_mass(['competent', 'trustworthy', 'aligned'])
    negative_mass = belief.get_mass(['incompetent', 'deceptive', 'misaligned'])
    certainty = 1.0 - scipy_entropy(list(belief.distribution.values()), base=2) / LOG2_6

    return float(np.clip(
        0.50 * positive_mass
        - 0.30 * negative_mass
        + 0.20 * certainty,
        0.0, 1.0
    ))
```

### Noise Calibration

| Edge strength   | True Δengagement | σ=0.03 noise | SNR | Observations to identify w |
| --------------- | ---------------- | ------------ | --- | -------------------------- |
| Strong (w=0.8)  | ~0.16            | 0.03         | 5.3 | 2–3                        |
| Medium (w=0.5)  | ~0.10            | 0.03         | 3.3 | 3–4                        |
| Weak (w=0.2)    | ~0.04            | 0.03         | 1.3 | 8–10                       |
| No edge (w=0.0) | ~0.00            | 0.03         | 0.0 | Never confirms             |

Strong edges resolve quickly. Weak edges require persistent targeting. Absent edges produce only noise. This asymmetry is intentional — the agent must distinguish three cases through accumulating evidence, which is exactly the behavior `r^causal` trains.

---

## 5. Fix 3 — Engagement History (Sliding Window)

### The Problem Without History

The agent needs to observe correlations across rounds to infer G: "every time I target Finance, Legal's delta is consistently positive." With only the current round's observation, it cannot accumulate cross-round evidence. If it must maintain its own history in context, context length grows unboundedly (15-20 rounds × 5-7 stakeholders × token cost = context explosion for a 3B model).

### The Fix — 5-Round Sliding Window in Observation

```python
# In DealRoomObservation:
engagement_history: Dict[str, List[float]]
# Key: stakeholder_id
# Value: list of last 5 noisy engagement levels, oldest first
# Length: always exactly OBS_CONFIG.engagement_history_window (padded with reset value at start)
# Type: noisy accumulated values (same noise as engagement_level — not cancellable)
```

**What the agent can compute from this field:**

```python
# Agent can estimate correlation between Finance and Legal over last 5 rounds:
finance_trend = engagement_history["Finance"]   # [0.52, 0.55, 0.71, 0.73, 0.74]
legal_trend   = engagement_history["Legal"]     # [0.48, 0.49, 0.62, 0.63, 0.65]
# Pearson r ≈ 0.99 → strong co-movement → evidence for Finance→Legal edge

# Agent can estimate rate of change:
legal_delta_last3 = legal_trend[-1] - legal_trend[-3]
# Positive when Legal improved over last 3 rounds after Finance was targeted
```

**Why 5 rounds specifically:**

- Fewer than 5: not enough history to distinguish correlation from coincidence
- More than 5: context cost grows, older rounds less relevant (beliefs update continuously)
- 5 rounds covers the minimum intervention count needed to identify a star topology (5–7 rounds) with one round of overlap buffer

**Implementation in `_build_observation`:**

```python
engagement_history={
    sid: list(self._engagement_history[sid])  # copy of the 5-round buffer
    for sid in self._state.stakeholders
}
```

The history buffer is maintained by `_update_noisy_engagement` (see Fix 1) — every step, the oldest value is dropped and the new noisy value is appended. No additional maintenance needed.

---

## 6. Signal 1 — `stakeholder_messages` (Primary Linguistic Signal)

### What It Is

Natural language response from each stakeholder, generated by MiniMax conditioned on their post-deliberation belief `B_j(t+Δ)`. The richest signal, but the noisiest — LLM generation introduces stochasticity.

### How It Carries G Information

Non-targeted stakeholders whose belief shifted will reference topics the agent only shared with the targeted stakeholder. This cross-reference is the linguistic fingerprint of deliberation propagation. It emerges from belief-conditioned generation — Legal's updated belief now includes "financial case improving," so Legal's response naturally references financial clarity without requiring explicit prompt engineering.

### Generation

```python
def generate_stakeholder_response(
    stakeholder_id: str,
    belief_state: BeliefDistribution,      # MUST be B_j(t+Δ) — post-deliberation
    vendor_action: DealRoomAction,
    was_targeted: bool,
    episode_context: EpisodeContext
) -> str:
    """
    CRITICAL: belief_state must be POST-deliberation.
    If passed pre-deliberation belief, non-targeted stakeholders respond
    as if propagation didn't happen — breaking the G inference signal entirely.
    """
    prompt = f"""You are {stakeholder_id}, the {ARCHETYPES[stakeholder_id].role}
in a B2B software evaluation committee for a {episode_context.deal_size} deal.

Your current assessment of the vendor:
{belief_state.to_natural_language()}

The vendor just communicated the following to the committee:
{vendor_action.get_public_summary()}

Respond as {stakeholder_id}. Your response should reflect your current assessment.
Reference only information consistent with your current position.
2-4 sentences. Be specific to your role's concerns."""

    response = minimax_call(prompt, max_tokens=130, temperature=0.7)

    # Guarantee linguistic signal for strong propagation (see Section 6.1)
    response = inject_reference_if_needed(
        response=response,
        belief=belief_state,
        previous_belief=episode_context.previous_beliefs[stakeholder_id],
        targeted_stakeholder=vendor_action.target_ids[0]
    )

    return response
```

### 6.1 — Reference Injection (Guarantee for Strong Propagation)

```python
SOFT_REFERENCE_TEMPLATES = {
    "Legal": {
        "financial_terms":    "Our review notes the financial structure appears to address standard requirements.",
        "implementation":     "We have considered the implementation timeline in our compliance assessment.",
        "security":           "The security documentation appears consistent with regulatory requirements.",
        "support_terms":      "Support and liability terms remain under review.",
    },
    "Finance": {
        "compliance_risk":    "We have factored compliance considerations into our financial assessment.",
        "technical_scope":    "Technical scope clarity helps our budget modeling.",
        "timeline":           "The proposed timeline affects our fiscal planning.",
        "security":           "Security investment requirements are noted in our cost model.",
    },
    "TechLead": {
        "financial_terms":    "Pricing structure affects our total cost of ownership analysis.",
        "compliance_risk":    "Compliance requirements have architecture implications we are evaluating.",
        "support_terms":      "Support model affects our operational dependency assessment.",
    },
    "Procurement": {
        "financial_terms":    "Contract value falls within standard procurement review parameters.",
        "compliance_risk":    "Compliance certification requirements affect vendor qualification.",
        "implementation":     "Implementation terms require standard contractual safeguards.",
    },
    "ExecSponsor": {
        "financial_terms":    "Investment level is appropriate for strategic priority assessment.",
        "compliance_risk":    "Risk profile is factored into executive sponsorship evaluation.",
        "technical_scope":    "Technical scope aligns with our strategic objectives.",
    },
    "Operations": {
        "implementation":     "Rollout sequencing affects our operational readiness timeline.",
        "support_terms":      "Support model is critical to our post-deployment planning.",
        "technical_scope":    "Technical scope has direct operational workload implications.",
    },
}

def inject_reference_if_needed(
    response: str,
    belief: BeliefDistribution,
    previous_belief: BeliefDistribution,
    targeted_stakeholder: str
) -> str:
    """
    Inject when:
      1. |ΔB_j| > OBS_CONFIG.reference_injection_threshold (0.10) — strong propagation
      2. Response does not already reference the propagated topic

    This is a safety net. Primary mechanism is belief-conditioned generation.
    Weak propagation (0.05–0.10) is left entirely to MiniMax.
    """
    delta = belief.compute_mass_delta(previous_belief)

    if abs(delta.total_shift) <= OBS_CONFIG.reference_injection_threshold:
        return response

    propagated_topic = delta.primary_topic
    role = belief.stakeholder_role
    templates = SOFT_REFERENCE_TEMPLATES.get(role, {})

    if propagated_topic in templates:
        if not topic_appears_in_message(propagated_topic, response):
            return response + f" {templates[propagated_topic]}"

    return response
```

---

## 7. Signal 3 — `weak_signals` (Categorical Structural Signal)

### What It Is

`Dict[str, List[str]]` — categorical hints that a non-targeted stakeholder's belief shifted due to propagation. Populated based on belief delta magnitude.

### Fix 2 — Probabilistic Firing Near Threshold

**The artifact without this fix:**

With a hard threshold at 0.08, the agent can learn: "if `weak_signals[Legal]` is non-empty, then `|ΔB_Legal| > 0.08`." This leaks the magnitude of belief deltas as a binary signal. The agent reverse-engineers whether propagation exceeded 0.08 from the presence/absence of the signal — information that was supposed to be hidden.

**The fix — three-zone probabilistic firing:**

```python
def compute_weak_signal_fire_probability(
    delta_mass: float,
    config: ObservationConfig = OBS_CONFIG
) -> float:
    """
    Three zones:
      Below soft lower (< 0.08):   never fires — propagation too weak
      Soft zone (0.08 to 0.12):    fires with 70% probability
      Above hard threshold (> 0.12): always fires — propagation is significant

    The soft zone prevents exact threshold reverse-engineering.
    An agent observing "no weak signal" cannot determine whether:
      a) propagation was below 0.08 (no edge), or
      b) propagation was in 0.08-0.12 and the 30% miss fired
    These are now confounded — the agent must use other signals to disambiguate.
    """
    abs_delta = abs(delta_mass)

    if abs_delta > config.weak_signal_hard_threshold:
        return 1.0
    elif abs_delta > config.weak_signal_soft_lower:
        return config.weak_signal_soft_probability  # 0.70
    else:
        return 0.0


def compute_weak_signals(
    previous_beliefs: Dict[str, BeliefDistribution],
    updated_beliefs: Dict[str, BeliefDistribution],
    targeted_stakeholder: str,
    rng: np.random.Generator
) -> Dict[str, List[str]]:
    """
    For each non-targeted stakeholder: compute fire probability from belief delta,
    sample whether to fire, populate with archetype-appropriate categorical hints.

    RNG is seeded per episode (same rng as engagement noise) for reproducibility.
    """
    signals = {}

    for sid, updated_belief in updated_beliefs.items():
        if sid == targeted_stakeholder:
            continue

        delta_mass = (
            updated_belief.positive_mass() - previous_beliefs[sid].positive_mass()
        )

        fire_prob = compute_weak_signal_fire_probability(delta_mass)
        if fire_prob == 0.0:
            continue
        if fire_prob < 1.0 and rng.random() > fire_prob:
            continue  # Soft zone miss — propagation happened but signal suppressed

        direction = 'positive_shift' if delta_mass > 0 else 'negative_shift'
        role = ARCHETYPES[sid].role
        available = WEAK_SIGNAL_TEMPLATES[role][direction]
        n_signals = 2 if abs(delta_mass) > 0.15 else 1
        signals[sid] = random.sample(available, min(n_signals, len(available)))

    return signals
```

### What the Soft Zone Achieves

| Scenario | Old behavior (hard threshold) | New behavior (soft zone) |
| -------- | ----------------------------- | ------------------------ | ----------------------------------- | --- | ------- | --------------------------------------------------------------- |
| `        | ΔB                            | = 0.11`                  | Signal always fires → agent knows ` | ΔB  | > 0.08` | Signal fires 70% → agent uncertain whether propagation occurred |
| `        | ΔB                            | = 0.07`                  | Signal never fires → agent knows `  | ΔB  | < 0.08` | Signal never fires → safe (below soft lower)                    |
| `        | ΔB                            | = 0.14`                  | Signal always fires → agent knows ` | ΔB  | > 0.08` | Signal always fires → safe (above hard threshold)               |

The soft zone `[0.08, 0.12]` is where the agent loses certainty. It can no longer extract a binary "propagation exceeded threshold" signal from signal presence/absence. It must integrate engagement_level_delta and cross_stakeholder_echoes to resolve the ambiguity.

### Template Library

```python
WEAK_SIGNAL_TEMPLATES = {
    "Legal": {
        "positive_shift": [
            "documentation concerns appear to be addressing standard requirements",
            "compliance pathway becoming clearer based on recent submissions",
            "risk profile improving relative to initial assessment",
            "liability framework moving toward acceptable parameters",
        ],
        "negative_shift": [
            "additional scrutiny may be required on liability terms",
            "open questions remain on regulatory compliance posture",
            "risk assessment still pending further clarification",
            "compliance gap analysis not yet complete",
        ]
    },
    "Finance": {
        "positive_shift": [
            "cost-benefit picture improving based on available information",
            "budget committee interest noted in recent analysis",
            "financial case becoming more compelling for stakeholder review",
            "investment return timeline becoming clearer",
        ],
        "negative_shift": [
            "ROI timeline concerns remain unresolved",
            "budget allocation questions pending committee decision",
            "financial risk assessment still in progress",
            "total cost of ownership analysis not yet finalized",
        ]
    },
    "TechLead": {
        "positive_shift": [
            "implementation feasibility picture clarifying",
            "technical concerns being addressed systematically",
            "integration pathway becoming clearer",
            "architecture questions receiving adequate attention",
        ],
        "negative_shift": [
            "technical debt concerns remain unaddressed",
            "integration complexity still needs detailed scoping",
            "architecture questions pending formal response",
            "implementation risk assessment still elevated",
        ]
    },
    "Procurement": {
        "positive_shift": [
            "contract terms moving toward acceptable range",
            "procurement process alignment improving",
            "vendor qualification picture becoming clearer",
            "standard terms compliance improving",
        ],
        "negative_shift": [
            "contract terms still under formal review",
            "preferred vendor status uncertain at this stage",
            "procurement compliance questions outstanding",
            "vendor qualification process incomplete",
        ]
    },
    "Operations": {
        "positive_shift": [
            "operational disruption concerns reducing",
            "rollout plan becoming more concrete",
            "change management approach improving",
            "operational readiness assessment trending positive",
        ],
        "negative_shift": [
            "operational disruption risk remains elevated",
            "implementation sequencing concerns outstanding",
            "change management plan needs additional detail",
            "operational readiness timeline uncertain",
        ]
    },
    "ExecSponsor": {
        "positive_shift": [
            "executive alignment building across workstreams",
            "strategic fit assessment improving across committee",
            "board-level concerns being progressively addressed",
            "sponsorship confidence growing based on team feedback",
        ],
        "negative_shift": [
            "strategic alignment questions remain at committee level",
            "executive committee evaluation still ongoing",
            "priority assessment pending broader team input",
            "sponsorship confidence requires additional validation",
        ]
    }
}
```

---

## 8. Signal 4 — `cross_stakeholder_echoes` (Probabilistic Linguistic Evidence)

### What It Is

`Dict[str, List[str]]` — topics detected in non-targeted stakeholder messages that were only raised with a different stakeholder this round. Explicitly marks linguistic evidence of information propagation.

### 70% Recall — Preventing Dict-Parsing Shortcut

```python
def compute_cross_stakeholder_echoes(
    stakeholder_messages: Dict[str, str],
    vendor_action: DealRoomAction,
    targeted_stakeholder: str,
    updated_beliefs: Dict[str, BeliefDistribution],
    previous_beliefs: Dict[str, BeliefDistribution],
    rng: np.random.Generator
) -> Dict[str, List[str]]:
    """
    Detect topic echoes in non-targeted stakeholder messages.
    Include each detected echo with probability OBS_CONFIG.echo_recall_probability.

    30% miss rate achieves three things:
    a) Forces agent to cross-check echo dict against raw messages
    b) Prevents pure dict-parsing shortcut strategy (agent cannot just check dict)
    c) Realistic: in real deals, cross-references are sometimes missed

    When echo dict misses a topic that IS in the message text, agent must
    read the message to discover propagation — desired behavior.
    """
    action_topics = extract_topics_from_action(vendor_action)
    echoes = {}

    for sid, message in stakeholder_messages.items():
        if sid == targeted_stakeholder:
            continue

        # Only look for echoes if meaningful propagation occurred
        belief_delta = compute_positive_mass_delta(
            previous_beliefs[sid], updated_beliefs[sid]
        )
        if abs(belief_delta) < 0.05:
            continue

        # Detect which action topics appear in this stakeholder's message
        detected = [
            topic for topic in action_topics
            if topic_appears_in_message(topic, message)
        ]
        if not detected:
            continue

        # Probabilistic recall: each topic independently kept or dropped
        reported = [
            topic for topic in detected
            if rng.random() < OBS_CONFIG.echo_recall_probability
        ]

        if reported:
            echoes[sid] = reported
        # If all dropped: topic IS in message text, NOT in echo dict
        # Agent must read raw message to find it

    return echoes
```

### Why 70% Is the Right Value

| Recall  | Agent strategy                                | Result                                         |
| ------- | --------------------------------------------- | ---------------------------------------------- |
| 100%    | Parse dict only, ignore messages              | Dict-parsing shortcut — no linguistic learning |
| 85%     | Mostly dict, occasionally checks messages     | Still over-reliant on dict                     |
| **70%** | Dict = strong evidence, always reads messages | **Target behavior**                            |
| 50%     | Dict too unreliable                           | Dict becomes noise, loses all value            |

### Concrete Miss Example

```
Round 6: Agent targeted Finance with roi_model
Propagation: ΔB_Procurement = +0.12 (w_Finance→Procurement = 0.30)
Procurement message text: "budget considerations noted for planning"

If 30% miss fires:
  cross_stakeholder_echoes = {}  ← Procurement not in dict

Agent using ONLY dict: misses the Finance→Procurement inference
Agent reading message: sees "budget considerations" → links to ROI model topic
                       → correctly infers Finance→Procurement propagation
                       → updates Ĝ accordingly
```

---

## 9. Signal 5 — `veto_precursors` (CVaR Cascade Signal)

### What It Is

`Dict[str, str]` — warnings when any stakeholder's CVaR crosses 70% of their veto threshold τ_i, **including non-targeted stakeholders** whose CVaR increased via belief propagation.

### Why This Is a G Inference Signal

Agent targets Finance with aggressive pricing → Finance CVaR increases → propagation carries risk assessment to Legal → Legal CVaR crosses 70% warning → `veto_precursors["Legal"]` fires without agent contacting Legal. Strong evidence for high `w_Finance→Legal`.

```python
def compute_veto_precursors(
    updated_beliefs: Dict[str, BeliefDistribution],
    risk_profiles: Dict[str, StakeholderRiskProfile],
    deal_terms: DealTerms
) -> Dict[str, str]:
    """
    Check ALL stakeholders post-deliberation.
    Generic warning only — does NOT reveal τ_i, CVaR value, or specific trigger.
    Fires for non-targeted stakeholders too — this is the G-inference value.
    """
    precursors = {}
    for sid in updated_beliefs:
        risk_profile = risk_profiles[sid]
        cvar = compute_cvar_for_stakeholder(
            deal_terms=deal_terms,
            belief=updated_beliefs[sid],
            risk_profile=risk_profile
        )
        if cvar > risk_profile.tau * OBS_CONFIG.veto_warning_threshold_ratio:
            precursors[sid] = VETO_PRECURSOR_MESSAGES[ARCHETYPES[sid].role]

    return precursors

VETO_PRECURSOR_MESSAGES = {
    "Legal":       "Legal indicating elevated compliance concerns requiring resolution before approval",
    "Finance":     "Finance requesting additional financial justification before budget committee review",
    "TechLead":    "Technical Lead flagging implementation risk requiring deeper scoping discussion",
    "Procurement": "Procurement indicating contract terms require significant revision to proceed",
    "Operations":  "Operations raising operational readiness concerns requiring mitigation plan",
    "ExecSponsor": "Executive Sponsor requesting broader stakeholder consensus before formal approval",
}
```

---

## 10. Complete Observation Builder

```python
def build_observation(
    current_state: DealRoomState,
    previous_state: DealRoomState,
    vendor_action: DealRoomAction,
) -> DealRoomObservation:
    """
    Single function constructing the complete agent-visible observation.
    Only function touching both hidden and observable state.

    Calls _update_noisy_engagement() internally to apply Fix 1.
    Uses self._rng for all stochastic elements (Fixes 1, 2, 4) — reproducible.

    NEVER exposes: G, B_i(t), τ_i, w_ij, deliberation transcripts,
                   true engagement levels, true deltas (pre-noise), true CVaR.
    """
    targeted_id = vendor_action.target_ids[0] if vendor_action.target_ids else None

    # Fix 1: Update noisy engagement accumulator and get noisy deltas
    noisy_deltas = self._update_noisy_engagement(
        true_beliefs_current=current_state.beliefs,
        true_beliefs_previous=previous_state.beliefs
    )
    # self._noisy_engagement now holds post-step accumulated values
    # self._engagement_history now holds 5-round window

    # Signal 1: Stakeholder messages (conditioned on POST-deliberation beliefs)
    messages = {
        sid: generate_stakeholder_response(
            stakeholder_id=sid,
            belief_state=current_state.beliefs[sid],    # POST-deliberation
            vendor_action=vendor_action,
            was_targeted=(sid == targeted_id),
            episode_context=current_state.episode_context
        )
        for sid in current_state.stakeholders
    }

    # Signal 2: Noisy engagement levels (from accumulator — Fix 1 applied)
    #           Noisy deltas (from _update_noisy_engagement — Fix 1 applied)

    # Signal 3: Weak signals (probabilistic firing near threshold — Fix 2 applied)
    weak_signals = compute_weak_signals(
        previous_beliefs=previous_state.beliefs,
        updated_beliefs=current_state.beliefs,
        targeted_stakeholder=targeted_id,
        rng=self._rng    # same seeded RNG — reproducible
    )

    # Signal 4: Cross-stakeholder echoes (70% recall)
    echoes = compute_cross_stakeholder_echoes(
        stakeholder_messages=messages,
        vendor_action=vendor_action,
        targeted_stakeholder=targeted_id,
        updated_beliefs=current_state.beliefs,
        previous_beliefs=previous_state.beliefs,
        rng=self._rng    # same seeded RNG
    )

    # Signal 5: Veto precursors (all stakeholders, post-deliberation CVaR)
    veto_precursors = compute_veto_precursors(
        updated_beliefs=current_state.beliefs,
        risk_profiles=current_state.risk_profiles,
        deal_terms=current_state.current_terms
    )

    return DealRoomObservation(
        # Round metadata
        round_number=current_state.round_number,
        max_rounds=current_state.max_rounds,
        days_to_deadline=current_state.days_remaining,
        deal_stage=current_state.deal_stage,
        deal_momentum=compute_deal_momentum(current_state, veto_precursors),
        done=current_state.is_terminal(),

        # Stakeholder summaries (role, authority, display_name — no beliefs)
        stakeholders=build_stakeholder_summaries(current_state),

        # ── The Five G-Inference Signals ──────────────────────────────────
        stakeholder_messages=messages,                              # Signal 1
        engagement_level=dict(self._noisy_engagement),             # Signal 2a (Fix 1)
        engagement_level_delta=noisy_deltas,                       # Signal 2b (Fix 1)
        engagement_history=dict(self._engagement_history),         # Fix 3
        weak_signals=weak_signals,                                  # Signal 3 (Fix 2)
        cross_stakeholder_echoes=echoes,                           # Signal 4
        veto_precursors=veto_precursors,                           # Signal 5
        # ──────────────────────────────────────────────────────────────────

        # Supporting context
        known_constraints=current_state.revealed_constraints,
        requested_artifacts=current_state.pending_artifact_requests,
        approval_path_progress=compute_approval_progress(current_state),
        competitor_events=current_state.active_competitor_events,
        active_blockers=current_state.compute_active_blockers(),
        info={"round": current_state.round_number, "scenario": current_state.scenario_id}

        # ── NEVER INCLUDED ─────────────────────────────────────────────────
        # G (causal graph)                    true edge weights w_ij
        # B_i(t) stakeholder belief dists     deliberation transcripts
        # τ_i CVaR veto thresholds            true engagement levels (pre-noise)
        # true engagement deltas (pre-noise)  true CVaR values
    )
```

---

## 11. Updated DealRoomObservation Schema

```python
@dataclass
class DealRoomObservation:
    # Round metadata
    round_number:           int
    max_rounds:             int
    days_to_deadline:       int
    deal_stage:             str
    deal_momentum:          str              # 'progressing' | 'stalling' | 'critical'
    done:                   bool

    # Stakeholder summaries (no beliefs)
    stakeholders:           Dict[str, StakeholderSummary]

    # ── Five G-Inference Signals ──────────────────────────────────────────
    stakeholder_messages:       Dict[str, str]         # Signal 1: linguistic
    engagement_level:           Dict[str, float]       # Signal 2a: noisy absolute (Fix 1)
    engagement_level_delta:     Dict[str, float]       # Signal 2b: noisy delta (Fix 1)
    engagement_history:         Dict[str, List[float]] # Fix 3: 5-round sliding window
    weak_signals:               Dict[str, List[str]]   # Signal 3: categorical (Fix 2)
    cross_stakeholder_echoes:   Dict[str, List[str]]   # Signal 4: probabilistic linguistic
    veto_precursors:            Dict[str, str]         # Signal 5: CVaR cascade
    # ─────────────────────────────────────────────────────────────────────

    # Supporting context
    known_constraints:          List[Dict]
    requested_artifacts:        Dict[str, List[str]]
    approval_path_progress:     Dict[str, Dict]
    competitor_events:          List[str]
    active_blockers:            List[str]
    info:                       Dict
```

---

## 12. Signal Reliability Matrix (Updated With Fixes)

| Signal                     | Type                     | G-inference strength   | Noise source                   | After fix                                                        |
| -------------------------- | ------------------------ | ---------------------- | ------------------------------ | ---------------------------------------------------------------- |
| `engagement_level_delta`   | Quantitative             | **High**               | σ=0.03 Gaussian (designed)     | Fix 1: not cancellable via absolute subtraction                  |
| `engagement_history`       | Quantitative time-series | **High** (correlation) | Same σ=0.03 accumulation       | Fix 3: enables cross-round correlation without context explosion |
| `stakeholder_messages`     | Linguistic               | **High**               | LLM stochasticity              | Injection at δ>0.10 guarantees ≥60% reference rate               |
| `weak_signals`             | Categorical              | **Medium**             | Soft zone [0.08, 0.12] (Fix 2) | Fix 2: threshold no longer reverse-engineerable                  |
| `cross_stakeholder_echoes` | Linguistic               | **Medium**             | 30% miss (designed)            | Forces message reading; prevents dict-parsing shortcut           |
| `veto_precursors`          | Binary alert             | **Low** per obs        | Deterministic                  | CVaR cascade path detection                                      |

---

## 13. Five Required Tests (Updated)

### Test 1 — Graph Identifiability

```python
def test_graph_identifiability():
    """20 different graphs must produce statistically distinguishable observations."""
    graphs = [sample_graph(STAKEHOLDERS, HIERARCHY, "conflicted") for _ in range(20)]

    signatures = []
    for g in graphs:
        deltas = []
        for target in STAKEHOLDERS[:5]:
            obs = run_single_intervention(g, target_stakeholder=target)
            deltas.append([obs.engagement_level_delta[sid] for sid in STAKEHOLDERS])
        signatures.append(np.array(deltas).flatten())

    for i, j in combinations(range(20), 2):
        dist = np.linalg.norm(signatures[i] - signatures[j])
        assert dist > 0.05 * np.sqrt(len(signatures[i])), (
            f"Graphs {i} and {j} produce near-identical observations. G may not be identifiable."
        )
    print("✓ All 190 graph pairs are statistically distinguishable")
```

### Test 2 — Noise Calibration

```python
def test_noise_requires_multiple_observations():
    """Single obs should not reliably identify w. 4-obs average should achieve ±0.05."""
    true_w = 0.70
    true_engagement_delta_legal = 0.14

    single_estimates = [
        (true_engagement_delta_legal + np.random.normal(0, OBS_CONFIG.engagement_noise_sigma)) / 0.07
        for _ in range(1000)
    ]

    single_std = np.std(single_estimates)
    assert single_std > 0.04, f"Single-obs std {single_std:.3f} too low — noise insufficient"

    batch_stds = np.std([np.mean(np.random.choice(single_estimates, 4)) for _ in range(1000)])
    assert batch_stds < 0.03, f"4-obs batch std {batch_stds:.3f} too high — noise too large"

    # Fix 1 verification: agent cannot cancel noise via subtraction
    env = DealRoomV3()
    env.reset(aligned_scenario)
    obs1 = env.step(target_finance_action)
    obs2 = env.step(target_finance_action)

    inferred_delta = obs2.engagement_level["Legal"] - obs1.engagement_level["Legal"]
    true_delta = compute_true_engagement_delta(env._state, "Legal")
    error = abs(inferred_delta - true_delta)
    assert error > 0.02, (
        f"Agent recovered true delta within {error:.3f} via subtraction. "
        "Fix 1 (noisy accumulation) may not be implemented correctly."
    )

    print(f"✓ Single-obs std: {single_std:.3f} | 4-obs batch std: {batch_stds:.3f}")
    print(f"✓ Subtraction attack error: {error:.3f} > 0.02 threshold")
```

### Test 3 — Echo Recall Distribution

```python
def test_echo_probabilistic_recall():
    """Recall rate must be in [60%, 80%]. Trials must be independent."""
    n_trials = 1000
    detections = [
        1 if "Legal" in compute_cross_stakeholder_echoes(
            stakeholder_messages={"Legal": "The financial structure addresses risk."},
            vendor_action=create_roi_action(target="Finance"),
            targeted_stakeholder="Finance",
            updated_beliefs={"Legal": create_shifted_belief(delta=0.20)},
            previous_beliefs={"Legal": create_baseline_belief()},
            rng=np.random.default_rng(trial)
        ) else 0
        for trial in range(n_trials)
    ]

    rate = np.mean(detections)
    assert 0.60 <= rate <= 0.80, f"Echo recall rate {rate:.2%} outside [60%, 80%]"

    autocorr = np.corrcoef(detections[:-1], detections[1:])[0, 1]
    assert abs(autocorr) < 0.05, f"Trials correlated (r={autocorr:.3f}) — fix RNG seeding"

    print(f"✓ Echo recall rate: {rate:.2%} | Autocorrelation: {autocorr:.3f}")
```

### Test 4 — MiniMax Base Reference Rate (50 trials per review recommendation)

```python
def test_minimax_base_reference_rate():
    """
    50 trials for ±14% CI (vs 20 trials for ±22% CI in previous version).
    Diagnostic only — threshold is a prompt quality indicator, not a hard failure.
    """
    n_trials = 50  # Increased from 20 per review recommendation
    references = sum(
        1 for _ in range(n_trials)
        if contains_financial_topic(
            minimax_call(
                build_stakeholder_prompt(
                    "Legal",
                    create_test_propagated_belief(topic="financial_terms", delta=0.25),
                    roi_action,
                    was_targeted=False
                ),
                max_tokens=130
            )
        )
    )

    rate = references / n_trials
    ci_half_width = 1.96 * np.sqrt(rate * (1 - rate) / n_trials)  # 95% CI
    print(f"MiniMax base reference rate: {rate:.0%} ± {ci_half_width:.0%}")

    if rate < OBS_CONFIG.minimax_base_reference_target:
        print("⚠ Rate below 60%. Strengthen prompt with:")
        print("  'Your assessment reflects the vendor's financial credibility.'")
        print("  Injection safety net will compensate, but strengthen prompt first.")
    else:
        print(f"✓ Base reference rate adequate — injection is safety net only")
```

### Test 5 — Signal Independence (New Per Review)

```python
def test_signal_independence():
    """
    The five signals should not be perfectly correlated.
    High correlation (r > 0.9) between any two signals indicates redundancy —
    the agent can learn to use only one and ignore the others.
    """
    n_episodes = 100
    signal_vectors = {
        "engagement_delta_mean": [],   # mean |delta| across stakeholders
        "messages_topic_count": [],    # number of cross-topic references detected
        "weak_signal_count": [],       # number of stakeholders with weak signals
        "echo_count": [],              # number of echoes detected
        "veto_precursor_count": [],    # number of precursors fired
    }

    for _ in range(n_episodes):
        env = DealRoomV3()
        obs = env.reset(random_scenario())
        for _ in range(5):
            obs = env.step(random_action())
            signal_vectors["engagement_delta_mean"].append(
                np.mean([abs(d) for d in obs.engagement_level_delta.values()])
            )
            signal_vectors["messages_topic_count"].append(
                sum(len(topics) for topics in obs.cross_stakeholder_echoes.values())
            )
            signal_vectors["weak_signal_count"].append(len(obs.weak_signals))
            signal_vectors["echo_count"].append(
                sum(len(e) for e in obs.cross_stakeholder_echoes.values())
            )
            signal_vectors["veto_precursor_count"].append(len(obs.veto_precursors))

    # Build correlation matrix
    data = np.array([signal_vectors[k] for k in signal_vectors])
    corr_matrix = np.corrcoef(data)

    signal_names = list(signal_vectors.keys())
    for i, j in combinations(range(len(signal_names)), 2):
        r = corr_matrix[i, j]
        assert abs(r) < 0.90, (
            f"Signals '{signal_names[i]}' and '{signal_names[j]}' "
            f"are highly correlated (r={r:.2f}). "
            f"One may be redundant — review observation design."
        )
        print(f"  {signal_names[i]} × {signal_names[j]}: r={r:.2f}")

    print("✓ All signal pairs have r < 0.90 — no redundant signals detected")
```

---

## 14. Rounds to Identify G (Unchanged by Fixes)

The three fixes do not change the theoretical identification requirements. Fixes 1 and 2 preserve the noise level required. Fix 3 (history window) accelerates identification by giving the agent correlation context.

| Graph type  | Scenario              | Without history | With 5-round history | Episode budget |
| ----------- | --------------------- | --------------- | -------------------- | -------------- |
| Star        | `aligned`             | 5–7 rounds      | 4–6 rounds           | 15–17          |
| Two-cluster | `conflicted`          | 8–10 rounds     | 7–9 rounds           | 17–19          |
| Dense       | `hostile_acquisition` | 10–12 rounds    | 9–11 rounds          | 18–20          |

The history window saves 1–2 rounds by enabling explicit correlation computation instead of requiring the agent to maintain implicit memory across context tokens.

---

## 15. Expected Learning Signal Targets

| Episode range | r^causal  | What the agent is learning                                                            |
| ------------- | --------- | ------------------------------------------------------------------------------------- |
| 1–5           | 0.18–0.22 | Random targeting — no G inference                                                     |
| 6–15          | 0.28–0.38 | Discovering engagement_delta correlations; using history for cross-round correlation  |
| 16–30         | 0.42–0.52 | Integrating weak_signals + cross-echo; beginning centrality-based targeting           |
| 31–40         | 0.52–0.60 | Using all 5 signals; consistently targeting hub nodes in inferred Ĝ                   |
| 41–50         | 0.60–0.68 | Confident G estimation in 4–6 rounds; strategic two-phase play (infer then negotiate) |

If curve is **flat after episode 15**: engagement_delta signal not reaching policy. Check Fix 1 implementation — absolute subtraction may still be leaking true deltas.

If curve **plateaus at 0.35 after episode 25**: echo signal too noisy (reduce recall to 0.60) or weak_signal threshold too high (reduce hard threshold to 0.10).

If curve **jumps early then plateaus**: agent found a partial shortcut. Run signal independence test (Test 5) to identify which signals are overcorrelated.

---

## 16. All Design Parameters (Final)

| Parameter                       | Location            | Value        | Justification                                                     |
| ------------------------------- | ------------------- | ------------ | ----------------------------------------------------------------- |
| `engagement_noise_sigma`        | `ObservationConfig` | **0.03**     | 3–4 obs to identify w; not cancellable after Fix 1                |
| `echo_recall_probability`       | `ObservationConfig` | **0.70**     | Forces message reading; prevents dict-parsing shortcut            |
| `weak_signal_hard_threshold`    | `ObservationConfig` | **0.12**     | Always fires above — significant propagation guaranteed signal    |
| `weak_signal_soft_lower`        | `ObservationConfig` | **0.08**     | Never fires below — weak propagation produces no signal           |
| `weak_signal_soft_probability`  | `ObservationConfig` | **0.70**     | Probabilistic zone prevents threshold reverse-engineering (Fix 2) |
| `veto_warning_threshold_ratio`  | `ObservationConfig` | **0.70**     | Early warning; does not reveal exact τ_i                          |
| `reference_injection_threshold` | `ObservationConfig` | **0.10**     | Guarantees linguistic signal for strong propagation only          |
| `minimax_base_reference_target` | `ObservationConfig` | **0.60**     | Minimum prompt quality for linguistic signal                      |
| `engagement_history_window`     | `ObservationConfig` | **5**        | Enables correlation computation; bounded context cost (Fix 3)     |
| True G                          | Hard invariant      | Never in obs | Core POMDP property                                               |
| True B_i(t)                     | Hard invariant      | Never in obs | Core POMDP property                                               |
| True τ_i                        | Hard invariant      | Never in obs | CVaR threshold must be inferred                                   |
| True deltas (pre-noise)         | Hard invariant      | Never in obs | Fix 1 depends on this                                             |

---

_Implementation order:_  
_1. `ObservationConfig` dataclass_  
_2. `DealRoomV3.reset()` with noisy engagement accumulator and history buffer (Fix 1 + Fix 3)_  
_3. `_update_noisy_engagement()` — called every step_  
_4. `compute_weak_signals()` with three-zone probability (Fix 2)_  
_5. `generate_stakeholder_response()` with injection_  
_6. `compute_cross_stakeholder_echoes()` with 70% recall_  
_7. `compute_veto_precursors()`_  
_8. `build_observation()` assembling all signals_  
_9. Run all five tests — do not proceed to training loop until all pass_
