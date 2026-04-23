# Architecture v3.7 — DealRoom Training Environment

## Overview

**Purpose**: DealRoom v3.7 is a reinforcement learning environment designed to train a Large Language Model (LLM) to negotiate B2B software deals with a multi-stakeholder buying committee. The environment simulates real business meetings where the LLM must navigate complex committee dynamics, address stakeholder concerns, and close deals that satisfy risk-averse stakeholders.

**Key Principle**: The environment is designed to train an LLM—not a neural network agent. All interactions are natural, meeting-like conversations. The LLM observes the deal state, takes actions (send documents, messages, concessions), and receives rewards based on world-state changes rather than human-judgment.

---

## 1. Environment Core (`dealroom_v3.py`)

### 1.1 DealRoomV3 Class

**What**: The main OpenEnv-compatible RL environment that orchestrates the entire negotiation simulation.

**Why**: Provides a standardized interface for RL training with reset/step semantics, manages all state (beliefs, causal graph, offer terms), and computes rewards based on world-state transitions.

**How**:

```python
class DealRoomV3:
    def __init__(self, use_llm_stakeholders: bool = False):
        self._rng: Optional[np.random.Generator] = None
        self._scenario: Optional[ScenarioConfig] = None
        self._state: Optional[DealRoomState] = None
        self._graph = None
        self._beliefs: Dict[str, BeliefDistribution] = {}
        self._utterance_scorer = UtteranceScorer()
        self._lookahead_simulator: Optional[LookaheadSimulator] = None
```

**Reset Flow**:
```python
def reset(self, seed: Optional[int] = None, task_id: str = "aligned", **kwargs) -> DealRoomObservation:
    self._rng = np.random.default_rng(seed)
    self._scenario = ScenarioConfig(task_id=task_id, seed=seed)
    self._episode_id = str(uuid.uuid4())[:8]
    # Samples a new causal graph per episode
    self._graph = sample_graph(
        stakeholder_set=STANDARD_STAKEHOLDERS,
        authority_hierarchy=STANDARD_HIERARCHY,
        scenario_type=task_id,
        rng=self._rng,
    )
    # Initializes beliefs for all stakeholders
    self._beliefs = {
        sid: BeliefDistribution(
            distribution=_get_initial_beliefs(task_id, sid),
            stakeholder_role=sid,
            confidence=1.0,
            history=[],
        )
        for sid in STANDARD_STAKEHOLDERS
    }
```

**Step Flow**:
```python
def step(self, action: DealRoomAction) -> Tuple[DealRoomObservation, float, bool, Dict[str, Any]]:
    # 1. Normalize action and run lookahead if requested
    action = self._normalize_action(action)
    lookahead_result = self._run_lookahead(action) if action.lookahead else None

    # 2. Apply action to offer state (modify deal terms)
    self._apply_action_to_offer_state(action)

    # 3. Update beliefs via Bayesian update for all stakeholders
    for sid in STANDARD_STAKEHOLDERS:
        is_targeted = sid in action.target_ids
        self._beliefs[sid] = bayesian_update(
            belief=self._beliefs[sid],
            action_type=action.action_type,
            documents=action.documents,
            stakeholder_role=sid,
            is_targeted=is_targeted,
        )

    # 4. Run committee deliberation (belief propagation through causal graph)
    deliberation_engine = CommitteeDeliberationEngine(graph=self._graph, n_deliberation_steps=3)
    deliberation_result = deliberation_engine.run(...)

    # 5. Compute reward via UtteranceScorer
    reward, reward_components = self._compute_reward(...)

    # 6. Evaluate CVaR and check for veto
    risk_snapshot = self._evaluate_committee_risk(self._state.offer_state)
    veto_triggered, veto_stakeholder = self._check_for_veto(risk_snapshot)
    done = veto_triggered or self._round_number >= self._state.max_rounds

    # 7. Build observation
    obs = self._build_observation(...)
    return obs, reward, done, info
```

---

### 1.2 Standard Stakeholders & Hierarchy

**What**: Six committee members with defined roles and authority levels.

**Why**: Different authority levels create realistic committee dynamics. ExecSponsor (authority=5) has veto power and influences all other members. Finance/Legal (authority=3) have moderate influence. TechLead/Procurement/Operations (authority=2) are leaves.

**How**:

```python
STANDARD_STAKEHOLDERS = ["Legal", "Finance", "TechLead", "Procurement", "Operations", "ExecSponsor"]
STANDARD_HIERARCHY = {
    "Legal": 3, "Finance": 3, "TechLead": 2,
    "Procurement": 2, "Operations": 2, "ExecSponsor": 5,
}
```

---

### 1.3 Initial Beliefs

**What**: Prior probability distributions over vendor types for each stakeholder.

**Why**: Different scenarios have different trust baselines. "Aligned" scenario starts with positive priors; "hostile_acquisition" starts with skepticism and deception priors.

**How**:

```python
INITIAL_BELIEFS = {
    "aligned": {
        "default": {
            "competent": 0.25, "trustworthy": 0.25, "aligned": 0.20,
            "incompetent": 0.10, "deceptive": 0.10, "misaligned": 0.10,
        }
    },
    "hostile_acquisition": {
        "default": {
            "competent": 0.12, "trustworthy": 0.12, "aligned": 0.10,
            "incompetent": 0.22, "deceptive": 0.22, "misaligned": 0.22,
        }
    },
    # "conflicted" has cluster-specific beliefs (cost_cluster, risk_cluster, impl_cluster)
}
```

---

### 1.4 Offer State Management

**What**: Tracks the current deal terms (price, timeline, security commitments, liability cap, DPA status).

**Why**: CVaR computations depend on actual deal terms. Documents sent change the offer state.

**How**:

```python
def _apply_action_to_offer_state(self, action: DealRoomAction) -> None:
    offer_state["days_to_deadline"] = max(0, offer_state.get("days_to_deadline", 30) - 1)
    for key, value in (action.proposed_terms or {}).items():
        offer_state[key] = value

    # Documents update specific deal attributes
    if any("dpa" in name for name in document_names):
        offer_state["has_dpa"] = True
    if any("security" in name for name in document_names):
        offer_state["has_security_cert"] = True
    if any("implementation" in name for name in document_names):
        offer_state["timeline_weeks"] = min(offer_state.get("timeline_weeks", 12), 12)
    if any("roi" in name for name in document_names):
        offer_state["price"] = max(75000, int(offer_state.get("price", 100000) * 0.95))

    # Action types modify terms
    if action.action_type == "concession":
        offer_state["price"] = max(70000, int(offer_state.get("price", 100000) * 0.90))
        offer_state["liability_cap"] = max(offer_state.get("liability_cap", 1000000), 1500000)
```

---

### 1.5 Veto Mechanism

**What**: CVaR-based veto that can terminate the episode even when expected utility is positive.

**Why**: The core research claim—CVaR captures tail risk that expected value ignores. A deal can have good average outcomes but still be vetoed due to dangerous tail scenarios.

**How**:

```python
def _check_for_veto(self, risk_snapshot: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    candidates: List[Tuple[float, str]] = []
    for sid in STANDARD_STAKEHOLDERS:
        profile = get_archetype(sid)
        if not profile or not profile.veto_power:
            continue
        cvar_loss = risk_snapshot["cvar_losses"].get(sid, 0.0)
        # Veto fires when CVaR exceeds tau AND precursor streak >= 2
        if cvar_loss > profile.tau and self._veto_precursor_streaks.get(sid, 0) >= 2:
            candidates.append((cvar_loss - profile.tau, sid))
    return (True, candidates[0][1]) if candidates else (False, None)
```

---

## 2. Causal Graph (`causal_graph.py`)

### 2.1 CausalGraph Structure

**What**: A stochastic directed graph representing committee influence relationships.

**Why**: Simulates how one stakeholder's opinion influences others. Actions targeted at one stakeholder propagate through the graph to affect others' beliefs.

**How**:

```python
@dataclass
class CausalGraph:
    nodes: List[str]                    # Stakeholder IDs
    edges: Dict[Tuple[str, str], float]  # (source, dest) -> influence weight
    authority_weights: Dict[str, float]  # Normalized authority per stakeholder
    scenario_type: str
    seed: int

def sample_graph(stakeholder_set, authority_hierarchy, scenario_type, rng):
    params = SCENARIO_PARAMS[scenario_type]
    edges = {}
    for source in stakeholder_set:
        for dest in stakeholder_set:
            if source == dest:
                continue
            source_authority = authority_hierarchy.get(source, 1)
            same_cluster = _same_functional_cluster(source, dest)

            # Authority nodes always have outgoing edges
            if source_authority >= 4:
                p_edge = 1.0
            elif same_cluster:
                p_edge = params["base_edge_probability"] + params["intra_cluster_boost"]
            else:
                p_edge = max(0.0, params["base_edge_probability"] - params["cross_cluster_penalty"])

            if rng.random() < p_edge:
                w = float(np.clip(rng.normal(params["weight_mean"], params["weight_std"]), 0.05, 0.95))
                edges[(source, dest)] = w
```

---

### 2.2 Belief Distribution

**What**: Represents a stakeholder's belief about the vendor across 6 types.

**Why**: The vendor can be competent/incompetent, trustworthy/deceptive, aligned/misaligned—independent dimensions tracked separately.

**How**:

```python
@dataclass
class BeliefDistribution:
    distribution: Dict[str, float]  # e.g., {"competent": 0.25, "incompetent": 0.10, ...}
    stakeholder_role: str
    confidence: float = 1.0
    history: List[Tuple] = field(default_factory=list)

    def positive_mass(self) -> float:
        return sum(self.distribution.get(t, 0) for t in ["competent", "trustworthy", "aligned"])

    def negative_mass(self) -> float:
        return sum(self.distribution.get(t, 0) for t in ["incompetent", "deceptive", "misaligned"])
```

---

### 2.3 Belief Propagation

**What**: Signal flows through graph edges, updating non-targeted stakeholders' beliefs.

**Why**: In real committees, information spreads. Targeting Finance also affects Legal if they communicate.

**How**:

```python
def propagate_beliefs(graph, beliefs_before_action, beliefs_after_action, n_steps=3):
    current_beliefs = {sid: b.copy() for sid, b in beliefs_after_action.items()}

    for step in range(n_steps):
        next_beliefs = {sid: b.copy() for sid, b in current_beliefs.items()}

        for dest_id in graph.nodes:
            influencers = graph.get_influencers(dest_id)
            if not influencers:
                continue

            total_delta = 0.0
            for source_id, weight in influencers.items():
                delta_source = (
                    current_beliefs[source_id].positive_mass()
                    - beliefs_before_action[source_id].positive_mass()
                )
                total_delta += weight * delta_source

            # K-gate non Linearity with epsilon floor
            k_gate = 10.0
            epsilon_floor = 0.003
            scaled_delta = total_delta * (abs(total_delta) / (abs(total_delta) + 1.0 / k_gate))
            if abs(total_delta) > 0 and abs(scaled_delta) < epsilon_floor:
                scaled_delta = np.sign(total_delta) * epsilon_floor

            if abs(scaled_delta) > 1e-10:
                next_beliefs[dest_id] = _apply_belief_delta(
                    current_beliefs[dest_id], scaled_delta, damping=0.95**step
                )

        current_beliefs = next_beliefs

    return current_beliefs
```

---

### 2.4 Scenario Parameters

**What**: Different graph densities and edge probabilities per scenario.

**Why**: "Hostile acquisition" is harder because the graph is denser (more ways for negative opinions to spread) and authority is lower.

**How**:

```python
SCENARIO_PARAMS = {
    "aligned": {
        "base_edge_probability": 0.30,
        "intra_cluster_boost": 0.40,
        "cross_cluster_penalty": 0.20,
        "authority_edge_prob": 0.85,
        "weight_mean": 0.45, "weight_std": 0.15,
    },
    "hostile_acquisition": {
        "base_edge_probability": 0.60,
        "intra_cluster_boost": 0.25,
        "cross_cluster_penalty": 0.05,
        "authority_edge_prob": 0.65,
        "weight_mean": 0.45, "weight_std": 0.25,
    },
}
```

---

## 3. Belief Tracker (`belief_tracker.py`)

### 3.1 Bayesian Update

**What**: Updates belief distribution based on vendor action using likelihood ratios.

**Why**: Actions have different implications. Sending a DPA proactively suggests competence and trustworthiness (0.85 likelihood). Direct message is less informative (0.70).

**How**:

```python
ACTION_LIKELIHOODS = {
    "send_document(DPA)_proactive": {
        "competent": 0.85, "incompetent": 0.15,
        "trustworthy": 0.80, "deceptive": 0.20,
        "aligned": 0.80, "misaligned": 0.20,
    },
    "send_document(security_cert)_proactive": {...},
    "direct_message_role_specific": {...},
    "default": {t: 0.50 for t in VENDOR_TYPES},
}

def bayesian_update(belief, action_type, documents, stakeholder_role, is_targeted):
    likelihoods = _get_likelihood(action_type, documents, stakeholder_role)
    damping = 1.0 if is_targeted else 0.7  # Targeted actions have full effect

    new_dist = {}
    for vendor_type, prior_prob in belief.distribution.items():
        likelihood = likelihoods.get(vendor_type, 0.5)
        dampened_likelihood = 1.0 + damping * (likelihood - 1.0)
        new_dist[vendor_type] = prior_prob * dampened_likelihood

    total = sum(new_dist.values())
    new_dist = {k: max(0.01, v / total) for k, v in new_dist.items()}

    # Confidence = 1 - (entropy / LOG2_6)
    probs = [new_dist.get(t, 0.01) for t in VENDOR_TYPES]
    entropy = -sum(p * math.log(p, 2) for p in probs if p > 0)
    confidence = 1.0 - (entropy / LOG2_6)

    return BeliefDistribution(distribution=new_dist, stakeholder_role=belief.stakeholder_role,
                              confidence=confidence, history=belief.history + [(action_type, damping)])
```

---

## 4. Committee Deliberation (`deliberation_engine.py`)

### 4.1 DeliberationEngine

**What**: Orchestrates belief propagation after a vendor action, optionally generating a summary via LLM.

**Why**: Models how the committee discusses and updates collective understanding. The propagation step models internal deliberation.

**How**:

```python
class CommitteeDeliberationEngine:
    def run(self, vendor_action, beliefs_before_action, beliefs_after_vendor_action, render_summary=True):
        # Propagate beliefs through graph
        updated_beliefs = propagate_beliefs(
            graph=self.graph,
            beliefs_before_action=beliefs_before_action,
            beliefs_after_action=beliefs_after_vendor_action,
            n_steps=self.n_steps,
        )

        # Optionally generate natural language summary
        summary = None
        if render_summary and vendor_action.target_ids:
            summary = self._generate_summary(
                beliefs_before=beliefs_before_action,
                beliefs_after=updated_beliefs,
                targeted_stakeholder=vendor_action.target_ids[0],
            )

        return DeliberationResult(
            updated_beliefs=updated_beliefs,
            summary_dialogue=summary,
            propagation_deltas={sid: updated_beliefs[sid].positive_mass() - beliefs_before_action[sid].positive_mass()
                               for sid in updated_beliefs},
        )
```

---

## 5. Stakeholder Profiles (`archetypes.py`, `cvar_preferences.py`)

### 5.1 StakeholderRiskProfile

**What**: Risk tolerance parameters for each committee member.

**Why**: Different stakeholders have different risk profiles. Legal is most risk-averse (tau=0.10, alpha=0.95). ExecSponsor is least risk-averse (tau=0.40, alpha=0.70).

**How**:

```python
@dataclass
class StakeholderRiskProfile:
    stakeholder_id: str
    role: str
    alpha: float       # CVaR percentile (0.95 = worst 5% of outcomes)
    tau: float          # Veto threshold
    lambda_risk: float  # Risk aversion weight
    veto_power: bool = False
    utility_weights: Dict[str, float] = field(default_factory=dict)
    uncertainty_domains: List[str] = field(default_factory=list)

# Example: Legal is most risk-averse
"Legal": StakeholderRiskProfile(
    alpha=0.95, tau=0.10, lambda_risk=0.70, veto_power=True,
    utility_weights={"compliance_coverage": 0.40, "liability_limitation": 0.30, ...},
    uncertainty_domains=["compliance_breach", "data_protection_failure", ...],
)
```

---

### 5.2 CVaR Computation

**What**: Conditional Value at Risk—expected loss in the worst alpha percentile of outcomes.

**Why**: CVaR captures tail risk. A deal with 90% chance of great outcome but 10% chance of catastrophic failure has high CVaR.

**How**:

```python
def compute_cvar(outcomes: np.ndarray, alpha: float) -> float:
    sorted_outcomes = np.sort(outcomes)
    cutoff_index = int(len(sorted_outcomes) * (1 - alpha))
    if cutoff_index >= len(sorted_outcomes):
        cutoff_index = len(sorted_outcomes) - 1
    if cutoff_index < 0:
        return 1.0

    tail_losses = 1.0 - sorted_outcomes[: cutoff_index + 1]
    tail_probs = np.ones(len(tail_losses)) / len(sorted_outcomes)
    total_tail_prob = sum(tail_probs)
    if total_tail_prob < 1e-8:
        return 0.0

    cvar = sum(l * p for l, p in zip(tail_losses, tail_probs)) / total_tail_prob
    return float(cvar)
```

---

### 5.3 Outcome Distribution

**What**: Monte Carlo simulation of deal outcomes based on terms and stakeholder profile.

**Why**: CVaR is computed over outcome samples. Deal terms affect the probability distribution.

**How**:

```python
def compute_outcome_distribution(deal_terms, stakeholder_profile, rng, n_samples=1000):
    domain = stakeholder_profile.uncertainty_domains[0] if stakeholder_profile.uncertainty_domains else "generic"

    # Base success probability depends on domain and documents
    base_success_prob = 0.75
    if "compliance" in domain or "data_protection" in domain:
        base_success_prob = 0.80
        if deal_terms.get("has_dpa") and deal_terms.get("has_security_cert"):
            base_success_prob = 0.92
        elif deal_terms.get("has_dpa") or deal_terms.get("has_security_cert"):
            base_success_prob = 0.86

    outcomes = []
    for _ in range(n_samples):
        if rng.random() < base_success_prob:
            outcome = 1.0 - (0.1 * rng.random()) + outcome_adjustment
        else:
            severity = rng.random()
            if severity < 0.3:
                outcome = 0.6 + 0.1 * rng.random() + outcome_adjustment
            elif severity < 0.7:
                outcome = 0.3 + 0.2 * rng.random() + outcome_adjustment
            else:
                outcome = 0.0 + 0.2 * rng.random() + outcome_adjustment
        outcomes.append(max(0.0, min(1.0, outcome)))

    return np.array(outcomes)
```

---

## 6. Utterance Scorer (`utterance_scorer.py`)

### 6.1 Five-Dimensional Reward

**What**: World-state-based deterministic scoring across 5 dimensions.

**Why**: Separating reward into dimensions provides richer learning signal and prevents reward hacking (single-dimensional optimization).

**How**:

```python
@dataclass
class UtteranceScore:
    goal: float      # Deal progress: approval delta + blocker resolution + veto headroom
    trust: float     # Trustworthy mass delta for targeted stakeholder
    information: float  # Entropy reduction across all beliefs
    risk: float      # CVaR improvement for risk-averse stakeholders
    causal: float     # Betweenness centrality of targeted node in graph
    lookahead_used: bool = False

def weighted_sum(self, weights: Dict[str, float]) -> float:
    return (weights["goal"] * self.goal + weights["trust"] * self.trust +
            weights["info"] * self.information + weights["risk"] * self.risk +
            weights["causal"] * self.causal)
```

---

### 6.2 Goal Score

**What**: Composite of approval delta (weighted by authority), blocker resolution, and veto headroom.

**Why**: Progress toward deal closure requires resolving blockers and improving CVaR headroom for risk-averse stakeholders.

**How**:

```python
def _score_goal(self, beliefs_before, beliefs_after, blockers_before, blockers_after,
               deal_stage, risk_profiles, deal_terms, authority_weights):
    # Authority-weighted approval delta
    for sid, b_after in beliefs_after.items():
        b_before = beliefs_before.get(sid)
        auth = authority_weights.get(sid, 1.0)
        approval_delta += (b_after.positive_mass() - b_before.positive_mass()) * auth
        total_auth += auth
    approval_score = (approval_delta / total_auth) if total_auth > 0 else 0.0

    # Blocker resolution
    resolved = len(blockers_before_set - blockers_after_set)
    new_created = len(blockers_after_set - blockers_before_set)
    blocker_score = resolved * 0.15 - new_created * 0.10

    # Veto headroom improvement (for lambda_risk > 0.40 stakeholders)
    for sid, profile in risk_profiles.items():
        if profile.lambda_risk > 0.40:
            cvar_b = self._compute_cvar(sid, beliefs_before, profile, deal_terms)
            cvar_a = self._compute_cvar(sid, beliefs_after, profile, deal_terms)
            headroom_delta = max(0.0, 1.0 - cvar_a/tau) - max(0.0, 1.0 - cvar_b/tau)
            veto_improvements.append(headroom_delta)

    raw = 0.50 * approval_score + 0.30 * blocker_score + 0.20 * veto_score
    return float(0.5 + 0.5 * np.tanh((REWARD_GAIN * raw) * REWARD_SCALE))
```

---

### 6.3 Lookahead Cost

**What**: Using lookahead reduces the goal dimension score by exactly 0.07.

**Why**: Lookahead is powerful but costly—prevents abuse while preserving the capability for strategic decisions.

**How**:

```python
LOOKAHEAD_COST = 0.07

# In UtteranceScorer.score():
if lookahead_used:
    goal = max(0.0, goal - LOOKAHEAD_COST)
```

---

## 7. Lookahead Simulator (`lookahead.py`)

### 7.1 LookaheadSimulator

**What**: Generates hypothetical outcomes by simulating optimistic and pessimistic belief states.

**Why**: Allows the LLM to "think ahead" before committing to an action, improving decision quality.

**How**:

```python
class LookaheadSimulator:
    def simulate(self, action_draft, current_beliefs, n_hypotheses=2, depth=2):
        # Generate optimistic and pessimistic hypotheses
        hypotheses = self._generate_hypotheses(target_id, current_beliefs)
        # For each hypothesis, simulate the response
        worst_case = min(simulation_results, key=lambda x: x.predicted_goal_delta)
        return SimulationResult(
            predicted_responses=worst_case.responses,
            predicted_belief_deltas=worst_case.belief_deltas,
            cvar_impact=worst_case.cvar_impact,
            cost=LOOKAHEAD_COST,
        )

    def _generate_hypotheses(self, target_stakeholder, current_beliefs):
        # Optimistic: shift probability mass toward positive types
        optimistic_dist["competent"] = min(0.95, optimistic_dist.get("competent", 0) + 0.15)
        optimistic_dist["trustworthy"] = min(0.95, optimistic_dist.get("trustworthy", 0) + 0.10)

        # Pessimistic: shift toward negative types
        pessimistic_dist["incompetent"] = min(0.95, pessimistic_dist.get("incompetent", 0) + 0.15)
        pessimistic_dist["deceptive"] = min(0.95, pessimistic_dist.get("deceptive", 0) + 0.10)
```

---

## 8. Text Environment Wrapper (`text_env.py`)

### 8.1 DealRoomTextEnv

**What**: TRL-compatible text-in/text-out wrapper for DealRoomV3.

**Why**: Allows training LLMs via TRL's GRPOTrainer with natural language prompts and completions.

**How**:

```python
class DealRoomTextEnv:
    def reset(self) -> str:
        self.env = DealRoomV3(use_llm_stakeholders=self.use_llm_stakeholders)
        self.current_obs = self.env.reset(seed=self.seed, task_id=self.task_id)
        return build_situation_prompt(self.current_obs)

    def step(self, action_text: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        action = parse_action_text(action_text)
        self.current_obs, reward, done, info = self.env.step(action)
        if done:
            next_prompt = _build_terminal_prompt(self.current_obs, self._episode_reward, info)
        else:
            next_prompt = build_situation_prompt(self.current_obs)
        return next_prompt, float(reward), bool(done), info
```

---

### 8.2 Situation Prompt Builder

**What**: Converts DealRoomObservation to natural language prompt for LLM.

**Why**: The LLM needs readable context about the current deal state.

**How**:

```python
def build_situation_prompt(obs: DealRoomObservation) -> str:
    parts = []
    parts.append("You are an AI vendor negotiating a B2B software deal with a buying committee.")
    parts.append("")
    parts.append(f"=== CURRENT SITUATION ===")
    parts.append(f"- Round: {obs.round_number}/{obs.max_rounds}")
    parts.append(f"- Deal Stage: {obs.deal_stage}")
    parts.append(f"- Deal Momentum: {obs.deal_momentum}")

    for sid in STAKEHOLDER_NAMES:
        role_info = obs.stakeholders.get(sid, {})
        eng = engagement.get(sid, 0.5)
        last_msg = stakeholder_msgs.get(sid, "")
        parts.append(f"  - {sid} ({role}): engagement={eng_pct}%, approval={band}. Last said: \"{last_msg[:80]}...\"")

    parts.append("")
    parts.append("=== AVAILABLE ACTIONS ===")
    parts.append("  send_document <target> <doc_type> [message]")
    parts.append("  direct_message <target> <message> ###")
    parts.append("  concession <target> <term>=<value> ###")
    parts.append("  group_proposal <message> ###")
    return "\n".join(parts)
```

---

## 9. Action Parsing (`prompts.py`)

### 9.1 Action Text Parser

**What**: Parses LLM text output into DealRoomAction objects.

**Why**: LLM generates text actions; environment needs structured actions.

**How**:

```python
ACTION_PATTERNS = [
    (re.compile(r"^\s*send_document\s+(\w+)\s+(\w+)(?:\s+([^#]+))?(?:\s*###\s*)?$", re.I), "send_document"),
    (re.compile(r"^\s*direct_message\s+(\w+)\s+(.+?)\s*###\s*$", re.I), "direct_message"),
    (re.compile(r"^\s*concession\s+(\w+)\s+(\w+)=([\d.]+)\s*###\s*$", re.I), "concession"),
    (re.compile(r"^\s*group_proposal\s+(.+?)\s*###\s*$", re.I), "group_proposal"),
    (re.compile(r"^\s*exec_escalation\s+(.+?)\s*###\s*$", re.I), "exec_escalation"),
]

def parse_action_text(text: str) -> DealRoomAction:
    for pattern, action_type in ACTION_PATTERNS:
        m = pattern.match(text)
        if m:
            groups = m.groups()
            if action_type == "send_document":
                return DealRoomAction(
                    action_type="send_document", target=target, target_ids=[target],
                    message=message, documents=[{"type": doc_type, "name": doc_type.upper()}],
                )
            # ... handle all action types
    return _fallback_action(text)
```

---

## 10. Training Infrastructure (`grpo_trainer.py`)

### 10.1 GRPOTrainer

**What**: GRPO-style training harness with self-play episodes and curriculum learning.

**Why**: Trains the LLM policy through episode experiences with adaptive difficulty.

**How**:

```python
class GRPOTrainer:
    def run_self_play_episode(self, env, policy_adapter, max_steps=10, task_id=None, seed=None):
        scenario = self.curriculum_generator.generate_adaptive_scenario(self._latest_failure_analysis)
        observation = env.reset(seed=episode_seed, task_id=scenario["task_id"])

        for _ in range(max_steps):
            action = policy_adapter.act(observation, self.rng)
            observation, reward, done, info = env.step(action)
            # Collect trajectory data
            if done:
                break

        return trajectory

    def run_training_loop(self, n_episodes=50, episodes_per_batch=4, max_steps=10):
        for batch_index in range(n_episodes):
            batch_trajectories = []
            for _ in range(episodes_per_batch):
                trajectory = self.run_self_play_episode(...)
                batch_trajectories.append(trajectory)

            # Analyze failures and update curriculum
            self._latest_failure_analysis = self.curriculum_generator.analyze_failures(batch_trajectories)
            self.policy_adapter.update_from_batch(batch_trajectories)
```

---

### 10.2 Policy Adapters

**What**: Pluggable policies that generate actions from observations.

**Why**: Allows comparison of different strategies (random, heuristic, trained).

**How**:

```python
class RandomPolicyAdapter:
    def act(self, observation: DealRoomObservation, rng) -> DealRoomAction:
        target = stakeholders[int(rng.integers(0, len(stakeholders)))]
        action_type = rng.choice(["direct_message", "send_document", "backchannel", "concession"])
        # ... build action

class HeuristicPolicyAdapter:
    def act(self, observation, rng) -> DealRoomAction:
        # Priority 1: Address veto precursors
        if observation.veto_precursors:
            target = next(iter(observation.veto_precursors))
            if target.lower() == "legal":
                return DealRoomAction(action_type="send_document", target=target,
                                      documents=[{"type": "dpa", "name": "DPA"}],
                                      proposed_terms={"liability_cap": 1500000})
        # Priority 2: Pick low-engagement stakeholder
        # Priority 3: Default to Finance with ROI model
```

---

## 11. Curriculum Generator (`adaptive_generator.py`)

### 11.1 Adaptive Curriculum

**What**: Dynamically selects scenarios based on agent performance.

**Why**: Training should start easy (aligned) and progress to hard (hostile_acquisition) as the agent improves.

**How**:

```python
class AdaptiveCurriculumGenerator:
    def analyze_failures(self, trajectories) -> FailureAnalysis:
        failure_counts = {k: 0 for k in FAILURE_MODE_DESCRIPTIONS}
        for traj in trajectories:
            detected = self._detect_failures(traj)
            for failure_id in detected:
                failure_counts[failure_id] += 1

        return FailureAnalysis(
            failure_modes={k: v/len(trajectories) for k, v in failure_counts.items()},
            agent_capability_estimate=mean(recent_rewards),
        )

    def generate_adaptive_scenario(self, failure_analysis=None) -> Dict:
        scenario = self.select_next_scenario(failure_analysis)
        # Adjust difficulty based on failure modes
        if failure_analysis and "F1" in failure_analysis.failure_modes:
            scenario["reduce_cvar_tension"] = True
        return scenario
```

---

## 12. Data Models (`models.py`)

### 12.1 DealRoomAction

**What**: Structured representation of vendor action.

**Why**: Normalized action format for environment processing.

**How**:

```python
class DealRoomAction(BaseModel):
    action_type: str = "direct_message"
    target: str = "all"
    target_ids: List[str] = []
    message: str = ""
    documents: List[Dict[str, str]] = []
    proposed_terms: Optional[Dict[str, Any]] = None
    lookahead: Optional["LookaheadRequest"] = None
    # Validators ensure message length limits and target normalization
```

---

### 12.2 DealRoomObservation

**What**: Everything the LLM sees about the current deal state.

**Why**: Partial observability—causal graph, true beliefs, CVaR parameters are hidden.

**How**:

```python
class DealRoomObservation(BaseModel):
    reward: Optional[float] = None
    round_number: int = 0
    max_rounds: int = 10
    stakeholders: Dict[str, Dict[str, Any]] = {}
    stakeholder_messages: Dict[str, str] = {}
    engagement_level: Dict[str, float] = {}     # Noisy proxy for trust
    weak_signals: Dict[str, List[str]] = {}     # Hints about hidden concerns
    veto_precursors: Dict[str, str] = {}        # Warning before veto
    cross_stakeholder_echoes: List[Dict[str, str]] = []  # Who noticed what from whom
    engagement_history: List[Dict[str, float]] = []  # Window of past engagement
    # Hidden (never exposed): causal_graph, belief_distributions, tau, edge_weights
```

---

## 13. Environment Constants (`constants.py`)

### 13.1 Reward Weights

**What**: Default weights for the 5 reward dimensions.

**Why**: Provides balanced signal across goal (0.30), trust (0.18), info (0.18), risk (0.17), causal (0.17).

**How**:

```python
REWARD_WEIGHTS = {
    "goal": 0.30, "trust": 0.18, "info": 0.18, "risk": 0.17, "causal": 0.17,
}

TERMINAL_REWARDS = {
    "deal_closed": 1.0, "veto": -1.0, "max_rounds": 0.0,
    "stage_regression": -0.5, "impasse": -0.75,
}

DEFAULT_MAX_ROUNDS = 10
VETO_WARNING_THRESHOLD_RATIO = 0.70  # Precursor fires at 70% of tau
```

---

## 14. LLM Client (`llm_client.py`)

### 14.1 GPT-4o-mini Integration

**What**: Optional LLM calls for stakeholder response generation and deliberation summaries.

**Why**: Provides more natural stakeholder responses when API key is available; falls back to templates otherwise.

**How**:

```python
def llm_call_text(prompt, call_type, temperature, context="", allow_skip=True, policy=DEFAULT_POLICY, timeout=30.0):
    # Retry with exponential backoff for auto-recoverable errors
    # Interactive pause for auth/rate_limit errors requiring user intervention
    # Statistics tracking via LLMCallStats
    client = OpenAI(api_key=key, timeout=timeout, max_retries=0)
    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens_val, temperature=temperature,
    )
    return response.choices[0].message.content

def generate_stakeholder_response(prompt, context="") -> Optional[str]:
    return llm_call_text(prompt=prompt, call_type="stakeholder_response", temperature=0.7, ...)
```

---

## 15. Observation Configuration (`dealroom_v3.py`)

### 15.1 ObservationConfig

**What**: Noise parameters for the observation mechanism.

**Why**: Creates realistic partial observability. Engagement is noisy (sigma=0.03), echoes are probabilistic (70% recall).

**How**:

```python
@dataclass
class ObservationConfig:
    engagement_noise_sigma: float = 0.03
    echo_recall_probability: float = 0.70
    weak_signal_hard_threshold: float = 0.12
    weak_signal_soft_lower: float = 0.08
    weak_signal_soft_probability: float = 0.70
    reference_injection_threshold: float = 0.10
    minimax_base_reference_target: float = 0.60
    engagement_history_window: int = 5
```

---

## 16. Pareto Efficiency (`pareto_efficiency.py`)

### 16.1 Terminal Reward Computation

**What**: Determines final reward based on deal outcome.

**Why**: Different terminal states have different values. Veto is worst (-1.0), deal closed is best (+1.0).

**How**:

```python
def compute_terminal_reward(deal_closed, veto_triggered, veto_stakeholder,
                           max_rounds_reached, stage_regressions,
                           all_utilities, cvar_losses, thresholds):
    if deal_closed:
        return TERMINAL_REWARDS["deal_closed"], "deal_closed"
    if veto_triggered:
        return TERMINAL_REWARDS["veto"], f"veto_by_{veto_stakeholder}"
    if max_rounds_reached:
        is_pareto = check_pareto_optimality(all_utilities, cvar_losses, thresholds)
        if is_pareto:
            return 0.0, "max_rounds_pareto"
        return TERMINAL_REWARDS["max_rounds"], "max_rounds_no_deal"
    if stage_regressions > 0:
        return TERMINAL_REWARDS["stage_regression"] * min(stage_regressions, 3), f"stage_regression_{stage_regressions}"
    return TERMINAL_REWARDS["impasse"], "impasse"
```

---

## Architecture Summary

The environment is designed around a **causal graph hidden from the agent**. The agent observes only:

- **Engagement levels** (noisy proxy for trust)
- **Stakeholder messages** (responses to actions)
- **Weak signals** (hints about concerns)
- **Cross-stakeholder echoes** (who noticed what from whom)
- **Veto precursors** (warning signs before veto)

The agent cannot see:
- The actual causal graph structure
- Belief distributions
- CVaR thresholds
- Edge weights

This creates a **Partially Observable Markov Decision Process (POMDP)** where the agent must infer the hidden state through observations and take actions that improve the deal while managing tail risk.
