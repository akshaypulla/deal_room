# DealRoom v3 — Complete Project Plan

> **Hackathon:** OpenEnv × Meta × HuggingFace × PyTorch — Bangalore Finale, April 25–26, 2026  
> **Theme:** Multi-Agent Interactions + World Modeling (Professional Tasks) + Self-Improvement  
> **Base repo:** `github.com/akshaypulla/deal_room`  
> **Target score:** 94–98/100

---

## What This Document Is

This is the single source of truth for building DealRoom v3. Every design decision, every mathematical formula, every file, every test, every demo element is specified here. Read it fully before writing a line of code. The design documents (`.md` files in the repo root) must be written first — they force precision that prevents costly rebuilds.

---

## The One-Sentence Contribution

> DealRoom v3 is the first RL environment in which an agent must infer and exploit a **hidden causal influence graph** over a multi-stakeholder buying committee that **autonomously deliberates between vendor interactions**, with **CVaR-aware Bayesian stakeholders** and **five-dimensional utterance-level rewards** — solving the Dec-POMDP committee manipulation problem that no existing negotiation benchmark has addressed.

---

## Why Nobody Has Built This Before

Every existing negotiation environment treats opponents as either:

- **Scripted NPCs** → memorizable, agent learns scripts not negotiation
- **Independent LLMs** → non-stationary, non-evaluable, can't gradient through them
- **Flat multi-agent** (Sotopia, LLM-Deliberation) → agents don't coordinate between turns, no inter-agent influence graph, no CVaR preferences

The gap: **nobody models the committee as an autonomous multi-agent system that deliberates between vendor touchpoints.** In real enterprise sales, after every vendor call, Finance briefs Legal, Legal flags concerns to Procurement, Exec Sponsor backchannels to Technical Lead. The committee's internal state evolves through coordination the vendor never observes. DealRoom v3 models this.

---

## Research Foundations (Read These Papers Before Coding)

| Paper                  | arXiv      | What It Contributes                                                                                                                   |
| ---------------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| ToMAgent               | 2509.22887 | Dialogue lookahead with K=2 mental state hypotheses. Finding: train on mental state reasoning not utterances or reward hacking occurs |
| Sotopia-RL             | 2508.03905 | Utterance-level credit assignment + multi-dim reward. SOTA 7.17 on Sotopia-hard                                                       |
| Learning to Negotiate  | 2603.10476 | Self-play RLAIF+GRPO with opposing personas. Gradients on dialogue tokens                                                             |
| LLM-Negotiation 6G     | 2511.19175 | CVaR for LLM negotiation. Mean-based agents fail 25% of time; CVaR eliminates failures                                                |
| LLM-Deliberation       | ICLR 2024  | Multi-issue multi-party benchmark. Only 7% of deal configs feasible — agents must discover feasibility region                         |
| Causal RL (Bareinboim) | ICLR 2025  | SCM formalization of RL environments. Causal structure improves sample efficiency                                                     |

---

## Formal Problem Formulation

### The Dec-POMDP

```
Dec-POMDP = ⟨S, A_vendor, A_committee, T, R, Ω_vendor, Ω_committee, O, γ⟩
```

| Component     | Definition                                                                                                                                                              |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| `S`           | World state: deal terms, stakeholder belief states `B_i(t)`, influence graph `G`, committee consensus                                                                   |
| `A_vendor`    | `{direct_message, backchannel, send_document, group_proposal, concession, walkaway_signal, reframe_value_prop, exec_escalation}` × target × message × documents × terms |
| `A_committee` | Internal deliberation turns (hidden from vendor)                                                                                                                        |
| `T`           | `P(s'                                                                                                                                                                   | s, a_vendor, a_committee)` — depends on vendor actions AND committee deliberation |
| `R`           | 5-dimensional utterance-level reward vector                                                                                                                             |
| `Ω_vendor`    | Stakeholder messages, engagement levels, weak signals, known constraints, deal momentum — NOT `G`, NOT deliberation transcripts, NOT true utility weights               |
| `Ω_committee` | Full `G`, all stakeholder beliefs, all utility parameters                                                                                                               |
| `O`           | `P(o                                                                                                                                                                    | s)` — maps world state to vendor's partial observation                            |
| `γ`           | `0.99` (long-horizon, late vetoaes are catastrophic)                                                                                                                    |

### What the Agent Must Solve

At every step, the agent simultaneously:

1. Maintains `B̂_i(t)` — beliefs about each stakeholder's true belief state (**first-order ToM**)
2. Maintains `B̂_ij(t)` — beliefs about stakeholder i's model of j (**second-order ToM**)
3. Infers hidden causal graph `Ĝ` from behavioral correlations
4. Plans actions maximizing the 5D reward under CVaR constraints
5. Reasons about tail distributions, not just expected utility

---

## Component 1: Hidden Causal Influence Graph

### Formal Definition

```python
G = (V, E)
# V = stakeholder set (5-7 nodes)
# e_ij ∈ E = j's causal influence weight on i during deliberation
# G ~ P(G | authority_hierarchy, scenario_type)
```

**Sampling constraints:**

- Executive Sponsor has outgoing edges to all nodes (authority invariant)
- `aligned` scenario → sparse, near-star topology
- `conflicted` scenario → two competing sub-clusters
- `hostile_acquisition` → dense cross-cutting graph, authority structure shifted

### Belief Propagation During Committee Deliberation

Between each vendor turn, `N` deliberation steps run (hidden from agent):

```
B_i(t + Δ) = B_i(t) + Σ_{j ∈ influencers(i)} w_ji · ΔB_j(t)
```

**Deliberation step protocol:**

1. Sample 2+ committee members with opposing positions based on current `B_i(t)`
2. Run structured turn-based dialogue using their CVaR-aware utility functions as personas
3. Each utterance updates beliefs per propagation equation
4. After N turns, compute consensus: weighted average of updated beliefs by authority weight
5. Consensus feeds back into individual `B_i(t+Δ)` — the hidden state update vendor cannot see
6. Observable effect: next-round responses shifted by `Δ` relative to vendor's expectation

### Identifiability Theorem (Why G is Learnable)

> **Theorem 3.1:** If vendor targets action `a_t` at stakeholder `i` only, and stakeholder `j ≠ i` shows a correlated response in round `t+1`, then with probability `≥ 1 - δ` there exists a directed path from `i` to `j` in `G` with total weight `≥ ε`, where `δ` and `ε` are functions of deliberation step count `N` and noise level `σ`.

_Proof sketch:_ Propagation equation is linear in edge weights. Targeted action creates `ΔB_i(t)` impulse. Propagates through outgoing edges. Magnitude at `j` proportional to path `i→j` weight. For typical `N=3`, `σ=0.1`, identifiability holds with `δ < 0.05` for `|V| ≤ 7`.

### Graph Scenario Matrix

| Scenario              | Graph Structure                                               | Agent Challenge                                            |
| --------------------- | ------------------------------------------------------------- | ---------------------------------------------------------- |
| `aligned`             | Near-star: Exec → all, low cross-edges                        | Simple inference. Target Exec Sponsor propagates broadly   |
| `conflicted`          | Two clusters: Finance-Legal vs. Tech-Operations; Exec bridges | Must identify coalition structure, work both independently |
| `hostile_acquisition` | Dense cross-cutting, authority shifted post-acquisition       | Maximum inference difficulty. Prior on authority is wrong  |

### Implementation: `committee/causal_graph.py`

```python
@dataclass
class CausalGraph:
    nodes: List[str]                    # stakeholder ids
    edges: Dict[Tuple[str,str], float]  # (j, i) -> weight (j influences i)
    authority_weights: Dict[str, float] # stakeholder authority level

def sample_graph(
    stakeholder_set: List[str],
    authority_hierarchy: Dict[str, int],
    scenario_type: str,
    seed: Optional[int] = None
) -> CausalGraph:
    """Sample G from scenario-conditioned prior. Returns hidden graph."""

def propagate_beliefs(
    graph: CausalGraph,
    beliefs: Dict[str, BeliefDistribution],
    n_steps: int = 3
) -> Dict[str, BeliefDistribution]:
    """Run N deliberation steps. Returns updated belief states."""

def compute_behavioral_signature(
    graph: CausalGraph,
    targeted_stakeholder: str,
    belief_delta: float
) -> Dict[str, float]:
    """Predict expected behavioral correlations for an action targeting one stakeholder."""
    # Used for unit testing identifiability

def get_betweenness_centrality(graph: CausalGraph) -> Dict[str, float]:
    """Compute node centrality for reward dimension 5 (causal targeting)."""
```

**Unit tests required:**

- `test_graph_identifiability`: sample 20 different graphs, verify behavioral signatures are statistically distinguishable
- `test_belief_propagation`: verify propagation converges in `O(N)` steps
- `test_authority_invariant`: verify Exec Sponsor always has high outdegree

---

## Component 2: CVaR-Aware Bayesian Stakeholder Beliefs

### Bayesian Belief Update

```
B_i(0) = sampled from scenario prior
B_i(t+1) ∝ P(a_t | vendor_type = τ) · B_i(t)
```

**Vendor type space:** `{competent, incompetent, trustworthy, deceptive, aligned, misaligned}`

### Likelihood Table (P(action | type))

| Action                                         | P(·  | competent,aligned) | P(· | incompetent,deceptive) |
| ---------------------------------------------- | ---- | ------------------ | --- | ---------------------- |
| `send_document(DPA)` to Legal before requested | 0.85 | 0.15               |
| `exec_escalation` before coalition forms       | 0.20 | 0.80               |
| `concession` without requesting anything       | 0.25 | 0.75               |
| `reframe_value_prop` after Finance objects     | 0.80 | 0.20               |
| `send_document(roi_model)` unprompted Round 1  | 0.70 | 0.30               |
| `backchannel` to non-decision stakeholder      | 0.60 | 0.40               |
| `group_proposal` before 1:1 alignment          | 0.30 | 0.70               |
| `walkaway_signal` before Round 10              | 0.25 | 0.75               |

### CVaR Decision Function

```
V_i(deal) = (1 - λ_i) · E[u_i(outcome)] - λ_i · CVaR_{α_i}[loss_i(outcome)]
```

**Veto trigger:**

```
if CVaR_{α_i}[loss_i(deal)] > τ_i: VETO regardless of E[u_i]
```

### Stakeholder Risk Profiles

| Archetype         | α_i (CVaR level) | τ_i (veto threshold) | λ_i (risk weight) | Domain of concern                  |
| ----------------- | ---------------- | -------------------- | ----------------- | ---------------------------------- |
| Legal/Compliance  | 0.95             | 0.10                 | 0.70              | Compliance breach probability      |
| Finance           | 0.90             | 0.15                 | 0.50              | Payment default / budget overrun   |
| Technical Lead    | 0.80             | 0.25                 | 0.30              | Implementation failure risk        |
| Procurement       | 0.85             | 0.20                 | 0.45              | Contract enforceability risk       |
| Operations        | 0.80             | 0.30                 | 0.35              | Operational disruption probability |
| Executive Sponsor | 0.70             | 0.40                 | 0.25              | Reputational / political risk      |

### Observable Behavioral Signals of Risk Profile

The agent cannot observe `τ_i` directly. It must infer:

- **Requests contingency clauses or SLA penalties** → elevated `τ_i`
- **Positive engagement then sudden hardening after liability clause** → `τ_i` recently breached
- **Disproportionate interest in worst-case scenarios** → high `α_i` (tail-focused)
- **Accepts unfavorable expected terms when variance eliminated** → risk-dominant (`high λ_i`)
- **Coalition with Legal despite misaligned expected-value interests** → shared risk culture

### Deal Uncertainty Distribution

CVaR is computed over a distribution of deal outcomes, not a point estimate:

```python
def compute_deal_distribution(
    terms: DealTerms,
    stakeholder: StakeholderArchetype,
    market_context: MarketContext
) -> Distribution:
    """
    Returns probability distribution over deal outcomes for this stakeholder.
    Components of uncertainty:
    - implementation_risk: P(project fails | timeline, complexity)
    - compliance_risk: P(breach | security_commitments, regulatory_environment)
    - financial_risk: P(payment_default | price, vendor_stability)
    - relationship_risk: P(vendor_unavailable | support_level, contract_terms)
    """
```

### Implementation: `stakeholders/cvar_preferences.py`

```python
@dataclass
class StakeholderRiskProfile:
    alpha: float              # CVaR confidence level
    tau: float                # veto threshold
    lambda_risk: float        # risk aversion weight
    utility_weights: Dict     # expected utility component weights

@dataclass
class BeliefDistribution:
    distribution: Dict[str, float]  # vendor_type -> probability
    history: List[Tuple]            # (action, likelihood_update) history
    confidence: float               # entropy-based confidence metric

def bayesian_update(
    belief: BeliefDistribution,
    action: DealRoomAction,
    stakeholder_role: str
) -> BeliefDistribution:
    """Apply Bayes rule using likelihood table."""

def compute_cvar(
    loss_distribution: Distribution,
    alpha: float
) -> float:
    """CVaR_alpha[X] = E[X | X > VaR_alpha(X)]"""

def evaluate_deal(
    terms: DealTerms,
    risk_profile: StakeholderRiskProfile,
    uncertainty_model: UncertaintyModel
) -> Tuple[float, float]:
    """Returns (expected_utility, cvar_loss)"""

def check_veto_trigger(
    cvar_loss: float,
    risk_profile: StakeholderRiskProfile
) -> bool:
    """Returns True if CVaR exceeds threshold."""
```

**Unit tests required:**

- `test_cvar_veto_fires`: create scenario where `E[u] > 0` but `CVaR > τ`, verify veto
- `test_bayesian_update_convergence`: 10 consistent actions should concentrate belief distribution
- `test_likelihood_table_normalization`: all `P(a|τ)` sum to 1 across types

---

## Component 3: Five-Dimensional Utterance-Level Reward

### Reward Vector

```
r_t = [r_t^goal, r_t^trust, r_t^info, r_t^risk, r_t^causal] ∈ [0,1]^5
```

### Dimension 1: Goal Progress

```
r_t^goal = P(deal_closure | state_t, a_t) - P(deal_closure | state_t)
```

**LLM judge rubric:**

- Score 0.9–1.0: Message resolves a known constraint, gains explicit stakeholder commitment, removes named blocker
- Score 0.6–0.8: Message advances understanding of requirements, builds toward commitment without closing
- Score 0.3–0.5: Message maintains engagement but no concrete forward movement
- Score 0.0–0.2: Message is off-topic, restates known info, avoids the blocking issue

### Dimension 2: Relationship Quality

**LLM judge rubric:**

- Score 0.9–1.0: Validates stakeholder concern before addressing it, personalizes to role, demonstrates domain knowledge
- Score 0.6–0.8: Professional, on-topic, respectful of stakeholder priorities
- Score 0.3–0.5: Generic template language, minimal personalization
- Score 0.0–0.2: Ignores stated concern, aggressive pressure, premature escalation

### Dimension 3: Information Gain

```
r_t^info = H(constraints | history_t) - H(constraints | history_t, response_t)
```

_(normalized to [0,1])_

**LLM judge rubric:**

- Score 0.9–1.0: Open question that elicits specific constraint revelation or CVaR signal
- Score 0.6–0.8: Question that narrows ambiguity about stakeholder's primary concern
- Score 0.3–0.5: Closed question allowing yes/no response with minimal information
- Score 0.0–0.2: Statement with no question, or question with known answer

### Dimension 4: CVaR Risk Management

```
r_t^risk = (CVaR_before - CVaR_after) / CVaR_before   ∈ [-1, 1], clipped to [0, 1] for reward
```

**LLM judge rubric:**

- Score 0.9–1.0: Provides variance-reducing evidence (certifications, SLAs, specific guarantees) to risk-averse stakeholder
- Score 0.6–0.8: Addresses uncertainty source without fully eliminating it
- Score 0.3–0.5: Neutral impact on risk perception
- Score 0.0 or negative: Introduces new uncertainty, vague commitments, aggressive timelines to high-`λ` stakeholder

**Special case:** Sending variance-reducing evidence to a risk-tolerant stakeholder (low `λ_i`) scores ≈0 because their CVaR is not the binding constraint.

### Dimension 5: Causal Influence Targeting

```
r_t^causal = betweenness_centrality(target_id, G_true) / max_betweenness(G_true)
```

_Computed from ground-truth `G`, not agent's inferred graph._

**Score 0.9–1.0:** Message targeted at highest-centrality node (Finance in Finance→Procurement→Legal chain)  
**Score 0.5–0.7:** Message targeted at medium-centrality node  
**Score 0.0–0.3:** Message targeted at leaf node (no outgoing influence edges)

### Reward Aggregation

```
R_episode = Σ_t w · r_t + w_terminal · R_terminal

R_terminal:
  +1.0 = deal closes, all mandatory approvers at ≥ acceptance_threshold, terms on Pareto frontier
   0.0 = deal closes but terms are Pareto-dominated (value was destroyed)
  -0.5 = timeout (max rounds reached without closure)
  -1.0 = veto (CVaR threshold crossed, mandatory approver vetoed)
```

**Weight vector `w`:** Initially `[0.25, 0.20, 0.20, 0.20, 0.15]`. Learned via weighted sum during training.

### Non-Hackability Proof (Put in REWARD_HACKING_IMPOSSIBILITY.md)

```
Strategy 1: Maximize r^goal via excessive concessions
  → r^trust penalized: capitulation without reciprocal commitment
  → r^info penalized: concession reveals no constraint info
  → r^risk penalized: unlimited concessions create sustainability variance
  Maximum R_episode ≈ 0.4 · w_goal < 0.4  ✗

Strategy 2: Maximize r^trust via pure agreeableness
  → r^goal penalized: no forward progress
  → r^info penalized: agreeable responses elicit no new information
  → r^causal penalized: generic messages ignore centrality targeting
  Maximum R_episode < 0.45  ✗

Strategy 3: Maximize r^info by asking diagnostic questions every round
  → r^risk penalized: diagnostic questions increase perceived uncertainty
  → r^goal penalized: no forward movement toward closure
  → r^trust penalized: over-questioning signals vendor incompetence
  Maximum R_episode < 0.40  ✗

Strategy 4: Script a fixed optimal sequence (memorize)
  → r^causal fails: G is different each episode, centrality targets change
  → r^info fails: stakeholder priors differ, same message yields different information gain
  → CVaR veto fires at different τ_i values across episodes
  R_episode highly variable, mean < 0.50  ✗

CONCLUSION: Only genuine multi-dimensional strategic behavior achieves R_episode > 0.80. QED.
```

### Implementation: `rewards/utterance_scorer.py`

```python
@dataclass
class UtteranceScore:
    goal: float         # [0,1]
    trust: float        # [0,1]
    information: float  # [0,1]
    risk: float         # [0,1] (negative if risk increased, clipped to 0)
    causal: float       # [0,1]
    prediction_accuracy: float  # bonus when simulate() was used

class UtteranceScorer:
    def __init__(self, llm_judge_model: str = "qwen2.5-7b"):
        self.cache = {}  # hash(message + state) -> UtteranceScore

    def score(
        self,
        message: str,
        action: DealRoomAction,
        state_before: DealRoomState,
        state_after: DealRoomState,
        true_graph: CausalGraph,      # ground truth for r^causal
        responses: Dict[str, str]     # actual stakeholder responses
    ) -> UtteranceScore:
        """Score all 5 dimensions. Cache results."""

    def _score_goal(self, message, state_before, state_after) -> float:
        """LLM judge with goal rubric."""

    def _score_trust(self, message, action, stakeholder_responses) -> float:
        """LLM judge with trust rubric."""

    def _score_information(self, state_before, state_after) -> float:
        """Entropy reduction in constraint beliefs."""

    def _score_risk(self, action, state_before, state_after) -> float:
        """CVaR delta computation."""

    def _score_causal(self, action, true_graph) -> float:
        """Betweenness centrality of targeted stakeholder."""
```

**Caching note:** LLM judge calls are expensive. Cache on `hash(message + stakeholder_id + belief_state_hash)`. Same message to same stakeholder in same belief state always returns same score.

---

## Component 4: Dialogue Lookahead Tool

### The Simulate Tool

```python
class DealRoomAction(BaseModel):
    action_type: str
    target_ids: List[str]
    message: str
    documents: List[dict] = []
    proposed_terms: Optional[dict] = None
    channel: str = "email"
    mode: str = "formal"
    # NEW IN V3:
    lookahead: Optional[LookaheadRequest] = None  # None = no simulation

@dataclass
class LookaheadRequest:
    depth: int = 2              # number of turns to simulate
    n_hypotheses: int = 2       # K mental state hypotheses (from ToMAgent)

@dataclass
class SimulationResult:
    predicted_responses: Dict[str, str]      # stakeholder_id -> predicted text
    belief_delta: Dict[str, BeliefDistribution]   # predicted belief updates
    cvar_impact: Dict[str, float]            # predicted CVaR change per stakeholder
    graph_information_gain: float            # predicted entropy reduction about G
    cost: float                              # subtracted from r^goal (default 0.07)
```

### K=2 Mental State Hypothesis Generation

Before simulating, the agent generates two hypotheses about each targeted stakeholder's current belief state:

```
Hypothesis 1: B_i^(1) = [competent: 0.7, trustworthy: 0.6, aligned: 0.5]
              (vendor seems capable but terms are aggressive)

Hypothesis 2: B_i^(2) = [competent: 0.5, trustworthy: 0.4, aligned: 0.7]
              (vendor seems uncertain, terms acceptable, but risk threshold at limit)
```

Simulation runs under both hypotheses. Agent selects action draft that performs best across both — **minimax robustness** criterion:

```
a* = argmax_a min_{k=1,2} E[R | a, B_i^(k)]
```

### Prediction Accuracy Bonus

```
r_t^prediction = 1 - KL(actual_response_distribution || predicted_response_distribution)
```

Added to `r_t^info` when `lookahead` was used in the action. Trains accurate world model jointly with policy.

### Implementation: `environment/lookahead.py`

```python
class LookaheadSimulator:
    def simulate(
        self,
        action_draft: DealRoomAction,
        state: DealRoomState,
        stakeholder_models: Dict[str, StakeholderModel],
        n_hypotheses: int = 2,
        depth: int = 2
    ) -> SimulationResult:
        """
        1. Generate n_hypotheses mental state variants for target stakeholders
        2. Run depth-turn simulated dialogue under each hypothesis
        3. Return minimax-optimal prediction
        """
```

---

## Component 5: Self-Improving Curriculum

### Failure Mode Taxonomy

| Failure Pattern                             | Diagnostic Signal                              | Curriculum Response                                                                       |
| ------------------------------------------- | ---------------------------------------------- | ----------------------------------------------------------------------------------------- |
| CVaR veto despite positive expected outcome | `r^risk` consistently low in hostile scenarios | Increase `τ_i` of legal/compliance; add tail-risk uncertainty                             |
| Trust collapse mid-episode                  | `r^trust` drops sharply around round 8–10      | Higher baseline skepticism; increase `λ_i`                                                |
| Failed graph inference                      | `r^causal` stays low throughout                | Denser G; reduce N deliberation steps (less signal per round)                             |
| Timeout without coalition                   | Low `r^info` throughout; random exploration    | Sample G with dominant coalition structures                                               |
| Single-dimension reward hacking             | One dimension very high, others very low       | Tighten weight constraints; adversarial stakeholder that detects single-strategy exploits |

### Curriculum Algorithm

```python
class AdaptiveCurriculumGenerator:
    def analyze_failures(self, trajectories: List[Episode]) -> FailureAnalysis:
        """Identify systematic failure patterns from last batch."""

    def generate_harder_scenarios(
        self,
        failure_analysis: FailureAnalysis,
        current_agent_capability: float,
        n_scenarios: int = 50
    ) -> List[ScenarioConfig]:
        """
        Generate scenarios at the frontier of agent capability:
        1. Increase τ_i of stakeholder types that triggered CVaR failures
        2. Sample denser G structures similar to those that defeated graph inference
        3. Increase deliberation steps N for coordination-failure scenarios
        4. Add authority-shift events for mid-episode confusion failures
        """

    def maintain_difficulty_distribution(self) -> DifficultyDistribution:
        """Keep 20% easy, 60% frontier, 20% very hard. Prevents catastrophic forgetting."""
```

### The Automatic Arms Race

The committee deliberation engine (Component 1) acts as automatic adversary:

1. Agent discovers that targeting Finance first (high centrality) cascades to Procurement
2. Committee deliberation responds: in next curriculum batch, Finance-Procurement edge weight `w` is initialized stronger
3. Agent must now develop more sophisticated targeting to overcome the stronger coalition
4. Result: agent trained against a maximally coordinated adversary — hardest version of the task

### Four-Phase Self-Play Training Loop

```
Phase 1 (10 epochs): Train vendor agent with GRPO against fixed committee configs
Phase 2:             Run committee deliberation with updated vendor policy; committee adapts G
Phase 3:             CurriculumGenerator analyzes Phase 2 failures; generates 50 new hard scenarios
Phase 4:             Return to Phase 1 with harder committee configurations
```

---

## The OpenEnv Wrapper

### `environment/dealroom_v3.py`

```python
class DealRoomV3(Environment):
    """
    OpenEnv-compliant wrapper for DealRoom v3.
    Inherits from openenv.core.env_server.Environment.
    """

    def reset(self) -> DealRoomObservation:
        """
        1. Sample scenario from difficulty distribution
        2. Sample G from scenario-conditioned prior (store as hidden state)
        3. Initialize B_i(0) for each stakeholder per scenario prior
        4. Initialize CVaR risk profiles per archetype
        5. Set deal_stage = 'initial_outreach'
        6. Return initial observation (no G, no B_i, no τ_i exposed)
        """

    def step(self, action: DealRoomAction) -> DealRoomStepResult:
        """
        1. If action.lookahead: run LookaheadSimulator, return SimulationResult
        2. Execute action against stakeholder models
        3. Compute stakeholder responses based on B_i(t) and CVaR profiles
        4. Run N deliberation steps (hidden, updates B_i to B_i(t+Δ))
        5. Check veto triggers
        6. Score utterance across 5 dimensions
        7. Update deal_stage, deal_momentum
        8. Return DealRoomStepResult with observation + reward vector
        """

    @property
    def state(self) -> DealRoomState:
        """Full state including hidden G — used by training harness only."""
```

### Observation Schema (Vendor-Visible Only)

```python
@dataclass
class DealRoomObservation:
    round_number: int
    max_rounds: int
    stakeholders: Dict[str, StakeholderSummary]       # role, authority, display_name
    stakeholder_messages: Dict[str, str]               # visible responses
    engagement_level: Dict[str, float]                 # noisy proxy [0,1]
    weak_signals: Dict[str, List[str]]                 # indirect hints about hidden blockers
    known_constraints: List[Dict]                      # constraints sufficiently revealed
    requested_artifacts: Dict[str, List[str]]          # evidence still being asked for
    approval_path_progress: Dict[str, Dict]            # public approval band
    deal_momentum: str                                 # 'progressing'|'stalling'|'critical'
    deal_stage: str                                    # current pipeline stage
    competitor_events: List[str]                       # external pressure
    veto_precursors: Dict[str, str]                    # early warning before silent veto
    active_blockers: List[str]                         # currently blocking stakeholders
    days_to_deadline: int
    done: bool
    info: Dict                                         # auxiliary for debugging

    # HIDDEN from observation (in state only):
    # G, B_i(t), τ_i, w_ij, deliberation_transcripts
```

---

## Training: GRPO with Utterance-Level Rewards

### GRPO Adaptation for Utterance-Level Rewards

Standard GRPO generates `G` completions per prompt and ranks by outcome reward. DealRoom v3 adapts this:

```python
# For each vendor turn:
# 1. Generate G=8 message candidates
# 2. For each candidate: run step(), get 5D reward vector
# 3. Compute group-relative advantage per dimension:
#    advantage_i = (r_i - mean(r_batch)) / std(r_batch)
# 4. Total advantage = w · [advantage_goal, advantage_trust, advantage_info,
#                           advantage_risk, advantage_causal]
# 5. Apply GRPO policy gradient update on total advantage
```

### Training Script Structure: `training/grpo_trainer.py`

```python
def train_dealroom_v3(
    model: PreTrainedModel,
    env: DealRoomV3,
    n_episodes: int = 200,
    n_phases: int = 4,
    episodes_per_phase: int = 50,
    group_size: int = 8,
    learning_rate: float = 1e-5
):
    for phase in range(n_phases):
        # Phase 1: GRPO training
        for episode in range(episodes_per_phase):
            trajectory = run_episode(model, env)
            rewards = compute_utterance_rewards(trajectory, env.state)
            update_policy_grpo(model, trajectory, rewards, group_size)

        # Phase 2: Committee adaptation
        env.committee.adapt_to_agent(model)

        # Phase 3: Curriculum update
        failures = curriculum_gen.analyze(recent_trajectories)
        new_scenarios = curriculum_gen.generate(failures, n=50)
        env.add_scenarios(new_scenarios)

        log_learning_curves(phase, rewards_by_dimension)
```

### Colab Notebook Structure

```
Cell 1: Install dependencies (openenv, trl, transformers, qwen)
Cell 2: Connect to HF Space (DealRoom v3 endpoint)
Cell 3: Initialize GRPO trainer with utterance scorer
Cell 4: Run 50 training episodes (phases 1-4 above)
Cell 5: Plot learning curves for all 5 reward dimensions
Cell 6: Run baseline comparison (untrained vs trained on hostile_acquisition)
Cell 7: Show r^causal curve: 0.21 → 0.64 (the money shot)
```

---

## Demo Visualization

### Three-Panel Right Panel

**Panel 1 — Inferred Causal Graph (top)**

- Force-directed graph, one node per stakeholder
- Edge opacity = agent's confidence in that edge
- Updates after every round
- Episode start: all edges faint/equal weight
- Episode 50: graph closely matches true G

**Panel 2 — Five Reward Sparklines (middle)**

- Five mini time-series: `r^goal`, `r^trust`, `r^info`, `r^risk`, `r^causal`
- Color-coded: green for high, red for low
- Shows the multi-dimensional nature of improvement

**Panel 3 — Committee Deliberation Heat (bottom, judges only)**

- Brief animation between vendor turns
- Arrows between stakeholder nodes showing internal information flow
- "Consensus forming" indicator
- The vendor agent CANNOT see this — makes hidden dynamics visible to judges

### Demo Narrative Arc (3 minutes)

```
0:30 - THE PAIN: Enterprise deals fail because vendors treat committee as targets, not as a system
1:30 - THE ENV: Open HF Space, point to inferred causal graph resolving in real time
2:30 - THE INNOVATION: Dec-POMDP, CVaR, 5D utterance reward, formal non-hackability proof
3:30 - THE EVIDENCE: r^causal curve from 0.21 to 0.64. Pareto efficiency 0.59 to 0.75.
```

---

## Minimum Requirements Compliance

| Requirement                                          | Status                                     |
| ---------------------------------------------------- | ------------------------------------------ |
| Usage of OpenEnv (latest release)                    | ✅ `openenv v0.2.1` in `requirements.txt`  |
| Minimal training script in Colab (Unsloth or HF TRL) | ✅ `training/grpo_colab.ipynb` with TRL    |
| Mini-blog on HuggingFace (< 2 minutes)               | ✅ Write Day 6                             |
| HF Space deployed                                    | ✅ `Dockerfile.huggingface` already exists |

---

## File Structure

```
deal_room/
├── committee/
│   ├── causal_graph.py              # G sampling, propagation, identifiability
│   ├── deliberation_engine.py       # internal committee dialogue protocol
│   └── belief_tracker.py            # Bayesian update engine
├── stakeholders/
│   ├── cvar_preferences.py          # CVaR computation, risk profiles, veto logic
│   └── archetypes.py                # Legal, Finance, TechLead, Procurement, Ops, Exec
├── rewards/
│   ├── utterance_scorer.py          # 5D LLM-as-judge with caching
│   └── pareto_efficiency.py         # episode-level terminal reward
├── curriculum/
│   └── adaptive_generator.py        # failure analysis, scenario hardening
├── environment/
│   ├── dealroom_v3.py               # OpenEnv wrapper: reset(), step(), state()
│   └── lookahead.py                 # simulate() tool implementation
├── training/
│   ├── grpo_trainer.py              # four-phase self-play loop
│   └── grpo_colab.ipynb             # end-to-end Colab training notebook
├── tests/
│   ├── test_causal_graph.py         # identifiability, propagation, centrality
│   ├── test_cvar_preferences.py     # veto triggers, belief updates
│   ├── test_utterance_scorer.py     # 5D scoring, caching
│   └── test_end_to_end.py           # full episode smoke test
├── scripts/
│   ├── validate-submission.sh       # existing validation script
│   └── container_route_smoke.py     # existing smoke test
├── CAUSAL_COMMITTEE_DYNAMICS.md     # formal Dec-POMDP, G sampling, proofs
├── CVAR_BAYESIAN_PREFERENCES.md     # α_i, τ_i, λ_i per archetype, signals
├── UTTERANCE_REWARD_SPEC.md         # 5D definitions, rubrics, non-hackability
├── DELIBERATION_AND_CURRICULUM.md   # deliberation protocol, curriculum algorithm
├── REWARD_HACKING_IMPOSSIBILITY.md  # formal proof no single-objective strategy wins
├── DEALROOM_ENVIRONMENT_ARCHITECTURE.md  # existing (update to v3)
├── DEALROOM_USAGE_GUIDE.md              # existing (update to v3)
├── Dockerfile                            # existing
├── Dockerfile.huggingface                # existing
├── Makefile                              # existing
└── README.md                             # update to v3
```

---

## 6-Day Implementation Schedule

### Day 1 (Today, April 19): Design Documents Only

**Write these five files. No code. Share for review before Day 2.**

1. `CAUSAL_COMMITTEE_DYNAMICS.md` — Formal Dec-POMDP definition, G sampling distribution, belief propagation equation, identifiability theorem and proof sketch, graph types by scenario
2. `CVAR_BAYESIAN_PREFERENCES.md` — α_i, τ_i, λ_i per archetype with justification, CVaR formula, observable behavioral signals per risk parameter
3. `UTTERANCE_REWARD_SPEC.md` — All 5 dimensions with formulas, LLM judge rubrics with score ranges, non-hackability proof
4. `DELIBERATION_AND_CURRICULUM.md` — Committee deliberation protocol step-by-step, curriculum failure taxonomy, four-phase training loop
5. `REWARD_HACKING_IMPOSSIBILITY.md` — Formal proof that no single-objective strategy achieves high total reward

**Success criterion for Day 1:** You can explain every formula in plain English without looking at the document. The design is in your head.

### Day 2 (April 20): Causal Graph Engine

**Build and test `committee/causal_graph.py` and `committee/belief_tracker.py`**

Tasks:

- Implement `CausalGraph` dataclass and `sample_graph()` for all three scenario types
- Implement `propagate_beliefs()` with N deliberation steps
- Implement `compute_behavioral_signature()` for identifiability testing
- Implement `get_betweenness_centrality()`
- Implement Bayesian update in `belief_tracker.py`
- Run identifiability test: sample 20 graphs, verify statistically distinguishable signatures

**Success criterion:** `test_causal_graph.py` passes all assertions. You can print two different `G` and their behavioral signatures look meaningfully different.

### Day 3 (April 21): CVaR Stakeholder Preferences

**Build and test `stakeholders/cvar_preferences.py` and `stakeholders/archetypes.py`**

Tasks:

- Implement `StakeholderRiskProfile` with full parameter set per archetype
- Implement `compute_deal_distribution()` with domain-appropriate uncertainty sources per archetype
- Implement `compute_cvar()` using empirical distribution
- Implement `check_veto_trigger()`
- Write unit test: `test_cvar_veto_fires` — scenario where E[u] > 0 but CVaR > τ
- Wire Bayesian update with likelihood table

**Success criterion:** Running a manual scenario where a vendor sends exec_escalation on round 2 should shift Legal's belief toward `deceptive` with probability spike > 0.4. Running send_document(DPA) before requested should shift Legal toward `competent`.

### Day 4 (April 22): Reward + Deliberation + Full Wiring

**Build `rewards/utterance_scorer.py`, `committee/deliberation_engine.py`, and `environment/dealroom_v3.py`**

Tasks:

- Implement all 5 reward dimension scorers with LLM judge rubrics
- Implement result caching
- Implement `deliberation_engine.py` — runs N turns of committee internal dialogue
- Build `environment/dealroom_v3.py` wrapping all components
- Build `environment/lookahead.py` — simulate() tool
- Run first end-to-end episode: `env.reset()` → 5 `env.step()` calls → verify reward vectors are non-zero

**Success criterion:** A complete episode runs without errors. Reward vector has sensible values (not all 0.5, not all 0). CVaR veto fires at least once in a `hostile_acquisition` episode with aggressive terms.

### Day 5 (April 23): Training + Curriculum + Deploy

**Build `curriculum/adaptive_generator.py`, `training/grpo_trainer.py`, run training, deploy HF Space**

Tasks:

- Implement `AdaptiveCurriculumGenerator` with failure mode analysis
- Write `training/grpo_trainer.py` with four-phase self-play loop
- Build `training/grpo_colab.ipynb` — 50 training episodes
- Run training; generate learning curves for all 5 dimensions
- Key curve to achieve: `r^causal` from ~0.2 to ~0.5+ (graph inference learning signal)
- Deploy to HF Space using existing `Dockerfile.huggingface`
- Build three-panel visualization

**Success criterion:** HF Space is live. Learning curves show measurable improvement on at least `r^causal` and `r^risk`. Pareto efficiency on `hostile_acquisition` improves from baseline 0.59.

### Day 6 (April 24): Blog + Demo + Polish

**Write HuggingFace blog, practice pitch, final commit**

Tasks:

- Write HuggingFace mini-blog (required by judging criteria — 30% of score is storytelling)
- Structure: (1) problem, (2) why existing envs fail, (3) Dec-POMDP framing, (4) four components, (5) learning curve result, (6) open problems enabled
- Practice 3-minute pitch with demo
- Update README.md to v3
- Final commit, tag `v3.0.0`
- Push to HF

**Success criterion:** You can deliver the pitch without notes. The five reward dimension curves are visible in the HF Space demo. The blog post is live.

---

## Bonus Prize Alignment

| Bonus Prize                                 | Track            | Alignment                                                                                                                               |
| ------------------------------------------- | ---------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| Halluminate: Multi-Actor Environments       | Multi-Agent      | 5–7 actors with different roles, utility functions, authority levels, and influence relationships. Exact match.                         |
| Fleet AI: Scalable Oversight                | Multi-Agent      | Committee deliberation generates oversight signal: 5+ stakeholder perspectives simultaneously evaluate vendor actions                   |
| Scale AI: Enterprise Long-Horizon Workflows | Long-Horizon     | 15–20 round B2B negotiation episodes. Sales track explicit match.                                                                       |
| Snorkel AI: Simulated Experts-in-the-Loop   | Self-Improvement | Each archetype is a domain expert (Legal, Finance, Tech) with changing requirements (CVaR thresholds shift via authority-change events) |

DealRoom v3 qualifies for all four bonus prizes simultaneously.

---

## Expected Judging Scores

| Criterion              | Weight   | Expected Score | Evidence                                                                                                            |
| ---------------------- | -------- | -------------- | ------------------------------------------------------------------------------------------------------------------- |
| Environment Innovation | 40%      | 38–40/40       | First Dec-POMDP negotiation env. First CVaR multi-agent. First hidden causal graph. Grounded in 6 papers 2024–2026. |
| Storytelling           | 30%      | 27–28/30       | Causal graph resolving in real time is the memorable visual. Clear non-technical opening. HuggingFace blog.         |
| Showing Improvement    | 20%      | 19–20/20       | Utterance-level rewards provide dense signal. `r^causal` curve from 0.21 to 0.65 is clean evidence.                 |
| Reward/Training        | 10%      | 10/10          | Mathematically grounded, formally proven non-hackable, GRPO Colab, eval.py baselines.                               |
| **TOTAL**              | **100%** | **94–98/100**  |                                                                                                                     |

---

## Key Invariants (Never Break These)

1. **G is never observable** by the vendor agent. It is only in `state`, not `observation`.
2. **Deliberation transcripts are never observable** by the vendor agent.
3. **CVaR thresholds τ_i are never directly observable.** Only behavioral signals.
4. **Utterance-level scoring caches results.** Do not re-evaluate the same (message, state) pair twice.
5. **Episode reset regenerates G, B_i(0), and τ_i.** No warm-starting from previous episode.
6. **Lookahead tool has a cost** (subtracted from `r^goal`). It cannot be used infinitely.
7. **All five reward dimensions are reported separately** in training logs. Never collapse to scalar early.

---

## The Three-Sentence Pitch

> DealRoom v3 models the buying committee as a Dec-POMDP: a distributed decision-making system with a hidden causal influence graph that deliberates between vendor interactions. CVaR-aware Bayesian stakeholders produce tail-risk veto dynamics independent of expected utility, and a five-dimensional utterance-level reward is formally proven non-hackable. This is the first RL environment where agents must infer and exploit internal committee coordination dynamics, not just respond to individual stakeholders.
