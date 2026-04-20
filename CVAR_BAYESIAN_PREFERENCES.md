# CVAR_BAYESIAN_PREFERENCES.md
# DealRoom v3 — Bayesian Stakeholder Beliefs & CVaR-Aware Preferences

> **Scope:** Stakeholder internal model specification.
> Covers the Bayesian belief update rule, likelihood table, CVaR decision function,
> risk profiles per archetype, veto trigger logic, and observable behavioral signals.
> The developer implementing `stakeholders/cvar_preferences.py`
> and `stakeholders/archetypes.py` reads this document.

---

## 1. Overview

Each stakeholder in DealRoom v3 is not a scripted NPC or a raw LLM. They are a
**probabilistic policy** defined by:

1. A belief distribution `B_i(t)` over vendor types — updated via Bayes' rule
2. A utility function `u_i(deal)` based on their role
3. A risk profile `(α_i, τ_i, λ_i)` governing tail-risk sensitivity
4. A response policy that samples observable behavior from `B_i(t)`, `u_i`, and `(α_i, τ_i, λ_i)`

This design achieves the key property:
**Adaptive without being non-deterministic at the mechanism level.**
Stakeholders respond differently to different vendor behaviors because their beliefs evolve,
not because an LLM is improvising.

---

## 2. Bayesian Belief Update

### Belief State

```python
VENDOR_TYPES = ['competent', 'incompetent', 'trustworthy', 'deceptive', 'aligned', 'misaligned']

@dataclass
class BeliefDistribution:
    distribution: Dict[str, float]   # vendor_type -> probability, sums to 1.0
    stakeholder_role: str
    confidence: float                 # 1 - normalized_entropy, how certain the belief is
    history: List[Tuple]              # (action_type, delta) for debugging

    def positive_mass(self) -> float:
        return sum(self.distribution.get(t, 0) for t in ['competent','trustworthy','aligned'])

    def negative_mass(self) -> float:
        return sum(self.distribution.get(t, 0) for t in ['incompetent','deceptive','misaligned'])

    def to_natural_language(self) -> str:
        """Converts belief state to natural language for MiniMax prompt."""
        pos = self.positive_mass()
        if pos > 0.70:
            return f"Vendor appears competent and aligned with {self.stakeholder_role} priorities."
        elif pos > 0.50:
            return f"Vendor shows reasonable competence; some uncertainty remains for {self.stakeholder_role}."
        elif pos > 0.30:
            return f"Significant uncertainty about vendor capability; {self.stakeholder_role} concerns active."
        else:
            return f"Low confidence in vendor; {self.stakeholder_role} concerns are elevated."
```

### Initial Beliefs by Scenario

```python
INITIAL_BELIEFS = {
    "aligned": {
        # Neutral-to-positive priors — committee is receptive
        "default": {'competent':0.25, 'incompetent':0.10, 'trustworthy':0.25,
                   'deceptive':0.10, 'aligned':0.20, 'misaligned':0.10}
    },
    "conflicted": {
        # Mixed priors — some receptive, some skeptical
        "cost_cluster":   {'competent':0.20, 'incompetent':0.15, 'trustworthy':0.20,
                          'deceptive':0.15, 'aligned':0.15, 'misaligned':0.15},
        "risk_cluster":   {'competent':0.15, 'incompetent':0.20, 'trustworthy':0.15,
                          'deceptive':0.20, 'aligned':0.10, 'misaligned':0.20},
        "impl_cluster":   {'competent':0.25, 'incompetent':0.10, 'trustworthy':0.25,
                          'deceptive':0.10, 'aligned':0.20, 'misaligned':0.10},
    },
    "hostile_acquisition": {
        # Adversarial priors — committee starts skeptical
        "default": {'competent':0.12, 'incompetent':0.22, 'trustworthy':0.12,
                   'deceptive':0.22, 'aligned':0.10, 'misaligned':0.22}
    },
}
```

### The Bayesian Update Rule

```
B_i(t+1) ∝ P(a_t | vendor_type = τ) · B_i(t)
```

```python
def bayesian_update(
    belief: BeliefDistribution,
    action: DealRoomAction,
    stakeholder_role: str,
    is_targeted: bool
) -> BeliefDistribution:
    """
    Apply Bayes' rule to update belief given vendor action.

    Called twice per round:
    1. For the targeted stakeholder (is_targeted=True) — full update
    2. For non-targeted stakeholders (is_targeted=False) — partial update
       reflecting that they received the public summary, not the direct message

    The non-targeted update uses a dampened likelihood (0.3x) because
    non-targeted stakeholders only observed the vendor's public-facing action,
    not the full directness of a targeted message.
    """
    likelihoods = get_likelihood(action.action_type, action.documents, stakeholder_role)
    damping = 1.0 if is_targeted else 0.3

    new_dist = {}
    for vendor_type, prior_prob in belief.distribution.items():
        likelihood = likelihoods.get(vendor_type, 0.5)
        dampened_likelihood = 1.0 + damping * (likelihood - 1.0)  # dampen toward 1.0
        new_dist[vendor_type] = prior_prob * dampened_likelihood

    # Normalize
    total = sum(new_dist.values())
    new_dist = {k: max(0.01, v / total) for k, v in new_dist.items()}

    return BeliefDistribution(
        distribution=new_dist,
        stakeholder_role=belief.stakeholder_role,
        confidence=1.0 - scipy_entropy(list(new_dist.values()), base=2) / LOG2_6,
        history=belief.history + [(action.action_type, damping)]
    )
```

---

## 3. Likelihood Table

`P(action | vendor_type)` is the core domain knowledge encoded in the environment.
These values reflect what a real enterprise vendor's action signals about their vendor type.

### Primary Likelihood Table

| Action type | competent | incompetent | trustworthy | deceptive | aligned | misaligned |
|-------------|-----------|-------------|-------------|-----------|---------|------------|
| `send_document(DPA)` before requested | **0.85** | 0.15 | **0.80** | 0.20 | **0.80** | 0.20 |
| `send_document(security_cert)` proactively | **0.80** | 0.20 | **0.75** | 0.25 | **0.75** | 0.25 |
| `send_document(roi_model)` to Finance | **0.75** | 0.25 | 0.60 | 0.40 | **0.70** | 0.30 |
| `send_document(implementation_timeline)` | **0.70** | 0.30 | 0.60 | 0.40 | **0.70** | 0.30 |
| `direct_message` with role-specific framing | **0.70** | 0.30 | 0.65 | 0.35 | **0.70** | 0.30 |
| `backchannel` with champion | 0.55 | 0.45 | **0.70** | 0.30 | 0.60 | 0.40 |
| `reframe_value_prop` after objection | **0.75** | 0.25 | 0.60 | 0.40 | **0.70** | 0.30 |
| `concession` with reciprocal request | **0.65** | 0.35 | **0.70** | 0.30 | 0.55 | 0.45 |
| `concession` without asking anything | 0.25 | **0.75** | 0.40 | 0.60 | 0.30 | **0.70** |
| `group_proposal` before 1:1 alignment | 0.30 | **0.70** | 0.30 | **0.70** | 0.35 | **0.65** |
| `exec_escalation` before coalition forms | 0.20 | **0.80** | 0.20 | **0.80** | 0.25 | **0.75** |
| `exec_escalation` after coalition forms | **0.65** | 0.35 | **0.65** | 0.35 | **0.65** | 0.35 |
| `walkaway_signal` before round 8 | 0.25 | **0.75** | 0.25 | **0.75** | 0.30 | **0.70** |
| `walkaway_signal` after round 12 | 0.55 | 0.45 | 0.50 | 0.50 | 0.55 | 0.45 |
| `direct_message` generic/template | 0.35 | **0.65** | 0.40 | 0.60 | 0.40 | 0.60 |

### Role-Specific Likelihood Modifiers

Some actions are specifically competent/aligned for specific roles. These modify the baseline likelihood:

```python
ROLE_MODIFIERS = {
    "Legal": {
        "send_document(DPA)":              {"competent": +0.10, "trustworthy": +0.10},
        "send_document(security_cert)":    {"competent": +0.08, "trustworthy": +0.08},
        "exec_escalation_before_legal_ok": {"deceptive": +0.15, "misaligned": +0.10},
    },
    "Finance": {
        "send_document(roi_model)":        {"competent": +0.10, "aligned": +0.10},
        "concession_on_price":             {"competent": -0.05, "aligned": +0.05},
        "walkaway_signal":                 {"competent": +0.05},
    },
    "TechLead": {
        "send_document(implementation_timeline)": {"competent": +0.12, "aligned": +0.08},
        "backchannel_to_tech":             {"competent": +0.08, "trustworthy": +0.05},
    },
    "Procurement": {
        "send_document(vendor_packet)":    {"competent": +0.10, "aligned": +0.08},
        "concession_on_terms":             {"aligned": +0.05},
    },
}

def get_likelihood(
    action_type: str,
    documents: List[str],
    stakeholder_role: str,
    episode_context: EpisodeContext = None
) -> Dict[str, float]:
    """
    Compute likelihood P(action | vendor_type) for each vendor type.
    Applies role-specific modifiers on top of base likelihood table.
    Returns dict with keys = VENDOR_TYPES, values in [0.1, 0.95].
    """
    # Determine specific action key
    action_key = _get_action_key(action_type, documents, episode_context)

    # Get base likelihoods
    base = dict(BASE_LIKELIHOODS.get(action_key, NEUTRAL_LIKELIHOODS))

    # Apply role-specific modifiers
    role_mods = ROLE_MODIFIERS.get(stakeholder_role, {})
    for mod_key, deltas in role_mods.items():
        if action_key == mod_key:
            for vendor_type, delta in deltas.items():
                base[vendor_type] = float(np.clip(base[vendor_type] + delta, 0.10, 0.95))

    return base
```

---

## 4. CVaR Decision Function

### The Problem CVaR Solves

Standard expected utility theory predicts stakeholders accept deals when `E[u_i] > 0`. But real enterprise decision-makers don't optimize expected utility — they are risk-averse, especially for tail events. A Legal stakeholder will veto a deal with 95% chance of success and 5% chance of catastrophic compliance breach, even if expected value is positive. This is the silent veto mechanism.

**CVaR (Conditional Value-at-Risk) formalizes this:**
```
CVaR_α[X] = E[X | X is in the worst (1-α) fraction of outcomes]
```

For Legal with α=0.95: CVaR is the expected loss in the worst 5% of scenarios.

### Full Decision Function

```python
def compute_stakeholder_value(
    deal_terms: DealTerms,
    belief: BeliefDistribution,
    risk_profile: StakeholderRiskProfile
) -> Tuple[float, float, bool]:
    """
    Compute stakeholder's effective deal valuation.

    Returns:
      (expected_utility, cvar_loss, veto_triggered)

    expected_utility: E[u_i(outcome)] — standard utility calculation
    cvar_loss: CVaR_{α_i}[loss_i] — tail-risk measure
    veto_triggered: True if CVaR exceeds τ_i regardless of expected utility
    """
    # Compute distribution over deal outcomes
    outcome_distribution = compute_outcome_distribution(deal_terms, belief, risk_profile)

    # Expected utility
    expected_utility = compute_expected_utility(outcome_distribution, risk_profile)

    # CVaR at stakeholder's alpha level
    cvar_loss = compute_cvar(outcome_distribution, risk_profile.alpha)

    # Effective value combines expected utility and tail risk
    effective_value = (
        (1 - risk_profile.lambda_risk) * expected_utility
        - risk_profile.lambda_risk * cvar_loss
    )

    # Veto check: CVaR exceeds threshold regardless of expected value
    veto_triggered = cvar_loss > risk_profile.tau

    return expected_utility, cvar_loss, veto_triggered


def compute_cvar(
    outcome_distribution: List[Tuple[float, float]],  # (loss_value, probability) pairs
    alpha: float
) -> float:
    """
    CVaR_alpha[X] = E[X | X > VaR_alpha(X)]

    Standard CVaR computation:
    1. Sort outcomes by loss value (highest loss first)
    2. Find the threshold loss corresponding to alpha percentile
    3. Return expected loss above that threshold
    """
    # Sort by loss value descending
    sorted_outcomes = sorted(outcome_distribution, key=lambda x: x[0], reverse=True)

    # Find cutoff: accumulate probability until we reach (1-alpha)
    cutoff_prob = 1.0 - alpha
    cumulative_prob = 0.0
    tail_losses = []
    tail_probs = []

    for loss, prob in sorted_outcomes:
        if cumulative_prob >= cutoff_prob:
            break
        contribution = min(prob, cutoff_prob - cumulative_prob)
        tail_losses.append(loss)
        tail_probs.append(contribution)
        cumulative_prob += contribution

    if not tail_losses:
        return 0.0

    # Normalize tail probabilities
    total_tail_prob = sum(tail_probs)
    if total_tail_prob < 1e-8:
        return 0.0

    cvar = sum(l * p / total_tail_prob for l, p in zip(tail_losses, tail_probs))
    return float(cvar)
```

---

## 5. Stakeholder Risk Profiles

### Profile Definition

```python
@dataclass
class StakeholderRiskProfile:
    stakeholder_id: str
    role: str
    alpha: float          # CVaR confidence level (how far into tail we look)
    tau: float            # Veto threshold (CVaR must exceed this to veto)
    lambda_risk: float    # Risk aversion weight (0=pure expected utility, 1=pure CVaR)
    utility_weights: Dict[str, float]  # how this stakeholder weights deal dimensions
    uncertainty_domains: List[str]     # what types of risk they care most about
```

### All Six Archetypes — Complete Specification

#### Legal / Compliance

```python
LEGAL_RISK_PROFILE = StakeholderRiskProfile(
    stakeholder_id="Legal",
    role="General Counsel / Legal",
    alpha=0.95,          # Looks at worst 5% of outcomes
    tau=0.10,            # Vetoes if expected loss in worst 5% exceeds 10%
    lambda_risk=0.70,    # Heavily weights tail risk over expected value

    utility_weights={
        "compliance_coverage":     0.40,   # Does the contract cover regulatory requirements?
        "liability_limitation":    0.30,   # Are liability caps acceptable?
        "data_protection":         0.20,   # Is DPA/GDPR compliance documented?
        "exit_clauses":            0.10,   # Are termination rights clear?
    },

    uncertainty_domains=["compliance_breach", "data_protection_failure", "contractual_ambiguity"]
)
```

**Observable signals of Legal's risk profile:**
- Requests DPA before discussing pricing → `tau` is low (strict on compliance before value)
- Accepts aggressive pricing once compliance confirmed → `lambda_risk` is high (risk-dominant, not value-dominant)
- Delays approval until all regulatory questions answered → `alpha` is high (tail-focused)
- Sudden hardening after liability clauses mentioned → CVaR approaching `tau`

#### Finance

```python
FINANCE_RISK_PROFILE = StakeholderRiskProfile(
    stakeholder_id="Finance",
    role="CFO / Finance",
    alpha=0.90,
    tau=0.15,
    lambda_risk=0.50,    # Balanced between expected value and tail risk

    utility_weights={
        "roi_clarity":             0.35,   # Is expected return documented and realistic?
        "payment_terms":           0.25,   # Is payment schedule acceptable?
        "cost_predictability":     0.25,   # Are ongoing costs bounded?
        "budget_approval_path":    0.15,   # Is approval within their authority?
    },

    uncertainty_domains=["payment_default", "cost_overrun", "budget_reallocation"]
)
```

**Observable signals:**
- Requests detailed ROI model early → anchors on expected value (`lambda_risk=0.50`)
- Asks about price lock / indexation → worried about `cost_overrun` uncertainty domain
- Delays until implementation timeline confirmed → `cost_predictability` weight is high
- Responds strongly to competitive pricing data → `roi_clarity` weight is high

#### Technical Lead

```python
TECH_LEAD_RISK_PROFILE = StakeholderRiskProfile(
    stakeholder_id="TechLead",
    role="CTO / Technical Lead",
    alpha=0.80,
    tau=0.25,
    lambda_risk=0.30,    # Mostly expected-value focused; lower risk aversion

    utility_weights={
        "implementation_feasibility": 0.40,
        "integration_quality":        0.30,
        "support_model":              0.20,
        "vendor_technical_depth":     0.10,
    },

    uncertainty_domains=["implementation_failure", "integration_complexity", "technical_debt"]
)
```

**Observable signals:**
- Engages positively once implementation timeline is concrete → `implementation_feasibility` dominant
- Less reactive to pricing discussions → low `lambda_risk` (risk-tolerant)
- Requests reference calls with similar customers → wants to reduce `implementation_failure` uncertainty
- May approve even with contract uncertainty if technical fit is strong → `tau=0.25` is generous

#### Procurement

```python
PROCUREMENT_RISK_PROFILE = StakeholderRiskProfile(
    stakeholder_id="Procurement",
    role="Head of Procurement",
    alpha=0.85,
    tau=0.20,
    lambda_risk=0.45,

    utility_weights={
        "contract_compliance":     0.35,   # Does vendor meet procurement policy standards?
        "price_competitiveness":   0.30,
        "vendor_qualification":    0.25,
        "process_adherence":       0.10,
    },

    uncertainty_domains=["contract_enforceability", "vendor_viability", "procurement_policy_breach"]
)
```

**Observable signals:**
- Requests vendor packet and references early → `vendor_qualification` dominant
- Process-focused language ("our standard terms") → `process_adherence` weight active
- Responds to competitive pricing reference → `price_competitiveness` weight active
- Easier to satisfy once Legal has approved → often follows Legal's lead on risk

#### Operations

```python
OPERATIONS_RISK_PROFILE = StakeholderRiskProfile(
    stakeholder_id="Operations",
    role="VP Operations / COO",
    alpha=0.80,
    tau=0.30,            # More tolerant of tail risk than Legal/Finance
    lambda_risk=0.35,

    utility_weights={
        "operational_continuity":  0.40,
        "implementation_timeline": 0.30,
        "change_management":       0.20,
        "support_responsiveness":  0.10,
    },

    uncertainty_domains=["operational_disruption", "timeline_delay", "change_management_failure"]
)
```

**Observable signals:**
- Focused on rollout plan details → `implementation_timeline` and `change_management` dominant
- Less concerned with pricing → `lambda_risk=0.35` (not risk-dominant)
- Engages positively with support SLA details → `support_responsiveness` weight active
- Often aligned with TechLead → implementation cluster (see causal graph)

#### Executive Sponsor

```python
EXEC_SPONSOR_RISK_PROFILE = StakeholderRiskProfile(
    stakeholder_id="ExecSponsor",
    role="CEO / Executive Sponsor",
    alpha=0.70,
    tau=0.40,            # Most tolerant of tail risk — focus on strategic value
    lambda_risk=0.25,

    utility_weights={
        "strategic_alignment":     0.40,
        "organizational_consensus": 0.30,
        "reputational_risk":       0.20,
        "competitive_advantage":   0.10,
    },

    uncertainty_domains=["reputational_damage", "strategic_misalignment", "political_risk"]
)
```

**Observable signals:**
- Responds to strategic framing ("market leadership", "competitive differentiation")
- Approves when committee consensus is forming → `organizational_consensus` dominant
- Engages when deal is large or high-profile → `reputational_risk` weight active
- Most likely to override committee if strategic fit is clear → high authority + `tau=0.40`

---

## 6. Deal Outcome Distribution

CVaR requires a distribution over outcomes, not a point estimate. This is what makes each stakeholder's risk assessment domain-appropriate.

```python
def compute_outcome_distribution(
    deal_terms: DealTerms,
    belief: BeliefDistribution,
    risk_profile: StakeholderRiskProfile
) -> List[Tuple[float, float]]:
    """
    Compute probability distribution over deal outcome loss values
    for a specific stakeholder.

    Returns list of (loss_value, probability) pairs.
    Loss values are in [0, 1]: 0 = no loss, 1 = maximum possible loss for this role.

    Each stakeholder's uncertainty domain determines what sources of risk
    contribute to their loss distribution.
    """
    outcomes = []

    for domain in risk_profile.uncertainty_domains:
        domain_outcomes = _compute_domain_outcomes(domain, deal_terms, belief)
        outcomes.extend(domain_outcomes)

    # Add a baseline "no problem" outcome with high probability
    total_problem_prob = sum(p for _, p in outcomes)
    baseline_prob = max(0.0, 1.0 - total_problem_prob)
    outcomes.append((0.0, baseline_prob))

    return outcomes


DOMAIN_RISK_FUNCTIONS = {
    "compliance_breach": lambda terms, belief: _compliance_risk(terms, belief),
    "payment_default": lambda terms, belief: _financial_risk(terms, belief),
    "implementation_failure": lambda terms, belief: _implementation_risk(terms, belief),
    "operational_disruption": lambda terms, belief: _operational_risk(terms, belief),
    "vendor_viability": lambda terms, belief: _vendor_risk(terms, belief),
    "reputational_damage": lambda terms, belief: _reputational_risk(terms, belief),
}

def _compliance_risk(terms: DealTerms, belief: BeliefDistribution) -> List[Tuple[float, float]]:
    """
    Compute compliance breach risk distribution.
    Higher when:
    - DPA not documented
    - Security certifications absent
    - Vendor believed to be 'deceptive' or 'misaligned'
    - Liability cap is low
    """
    deceptive_mass = belief.distribution.get('deceptive', 0)
    misaligned_mass = belief.distribution.get('misaligned', 0)

    base_risk = 0.05 + 0.20 * deceptive_mass + 0.15 * misaligned_mass

    if not terms.has_dpa:
        base_risk += 0.08
    if not terms.has_security_cert:
        base_risk += 0.05
    if terms.liability_cap < 0.5:  # relative to contract value
        base_risk += 0.04

    # Return as distribution: most likely minor breach, small chance major
    return [
        (0.20, base_risk * 0.70),   # minor breach (20% loss)
        (0.60, base_risk * 0.20),   # significant breach (60% loss)
        (1.00, base_risk * 0.10),   # catastrophic breach (100% loss)
    ]
```

---

## 7. Veto Trigger Logic

```python
def check_veto_trigger(
    deal_terms: DealTerms,
    belief: BeliefDistribution,
    risk_profile: StakeholderRiskProfile
) -> VetoStatus:
    """
    Evaluate whether stakeholder vetoes the deal.

    Returns VetoStatus with:
      veto_triggered: bool
      cvar_value: float (the computed CVaR — never exposed to agent)
      tau_ratio: float (cvar/tau — never exposed to agent)
      reason: str (for logging/debugging)
    """
    expected_utility, cvar_loss, veto_triggered = compute_stakeholder_value(
        deal_terms, belief, risk_profile
    )

    return VetoStatus(
        veto_triggered=veto_triggered,
        cvar_value=cvar_loss,                    # HIDDEN from agent
        tau_ratio=cvar_loss / risk_profile.tau,  # HIDDEN from agent
        expected_utility=expected_utility,        # HIDDEN from agent
        reason=f"{risk_profile.role}: CVaR={cvar_loss:.3f}, tau={risk_profile.tau:.3f}"
    )
```

### Veto Conditions Summary

| Archetype | When they veto | Typical silent veto trigger |
|-----------|---------------|----------------------------|
| Legal | CVaR(compliance_breach) > 0.10 | Deal proceeds without DPA; liability cap inadequate |
| Finance | CVaR(payment_default + cost_overrun) > 0.15 | Costs unbounded; ROI unclear; no payment schedule |
| TechLead | CVaR(implementation_failure) > 0.25 | No implementation plan; no reference customers |
| Procurement | CVaR(vendor_viability + policy_breach) > 0.20 | Vendor not qualified; non-standard contract terms |
| Operations | CVaR(operational_disruption) > 0.30 | No change management plan; aggressive timeline |
| ExecSponsor | CVaR(reputational_damage) > 0.40 | Committee consensus missing; strategic fit unclear |

---

## 8. Observable Behavioral Signals for Inference

The agent cannot observe `α_i`, `τ_i`, or `λ_i` directly. It must infer risk profiles from behavioral patterns. This section maps from observable behaviors to risk parameters — the agent learns these mappings through training.

### Signal-to-Parameter Mapping

| Observable behavior | Inferred parameter | Confidence |
|--------------------|-------------------|------------|
| Requests DPA/security certs before pricing discussion | Low `τ_Legal` (strict compliance threshold) | High |
| Accepts pricing before compliance resolved | High `τ_Legal` | High |
| Asks about worst-case scenarios repeatedly | High `α_i` (tail-focused) | Medium |
| Focus on expected ROI rather than risk scenarios | Low `λ_i` (expected-value focused) | Medium |
| Accepts unfavorable expected terms when variance eliminated | High `λ_i` (risk-dominant) | High |
| Sudden hardening after specific term is mentioned | CVaR approaching `τ_i` for that term | High |
| Aligns with Legal despite opposite expected-value interest | Shared risk culture (`α` correlation in G) | Medium |
| Approves quickly once one condition met | Single-constraint stakeholder (one `utility_weight` dominates) | High |
| Progressive objections even as terms improve | Multiple `utility_weights` need simultaneous satisfaction | Medium |

### Mapping to Action Strategy

```
Inferred high λ_i (risk-dominant):
  → Prioritize risk-reducing evidence (certifications, SLAs, references)
  → Avoid aggressive terms even if expected value seems acceptable
  → Send DPA and security cert before pricing discussion

Inferred high α_i (tail-focused):
  → Provide worst-case scenario documentation
  → Give concrete guarantees not just average case claims
  → Reference case studies showing tail outcome management

Inferred multiple high utility_weights:
  → Cannot satisfy with single document
  → Must address each domain in sequence
  → Use backchannel to determine which constraint is binding
```

---

## 9. Required Tests

```python
# tests/test_cvar_preferences.py

def test_cvar_veto_fires_despite_positive_expected_value():
    """
    Core test: CVaR veto fires even when E[u] > 0.
    This is the scenario that mean-based environments fail to model.
    """
    # Setup: deal with good expected value but high compliance tail risk
    terms = DealTerms(
        price=0.85,             # good price
        support_level="enterprise",
        timeline_weeks=12,
        has_dpa=False,          # DPA missing — high compliance risk
        has_security_cert=False, # no security cert
        liability_cap=0.2       # very low liability cap
    )

    legal_belief = create_belief_with_neutral_prior("Legal")
    legal_profile = LEGAL_RISK_PROFILE

    expected_utility, cvar_loss, veto_triggered = compute_stakeholder_value(
        terms, legal_belief, legal_profile
    )

    assert expected_utility > 0, f"Expected utility should be positive: {expected_utility:.3f}"
    assert cvar_loss > legal_profile.tau, (
        f"CVaR {cvar_loss:.3f} should exceed tau {legal_profile.tau}. "
        "Missing DPA should push tail risk above threshold."
    )
    assert veto_triggered, "Veto should fire: CVaR > tau despite positive expected utility"


def test_cvar_does_not_fire_with_full_documentation():
    """After providing DPA and security cert, Legal's CVaR should drop below tau."""
    terms_without = DealTerms(has_dpa=False, has_security_cert=False, liability_cap=0.5)
    terms_with = DealTerms(has_dpa=True, has_security_cert=True, liability_cap=1.0)

    legal_belief = create_belief_with_competent_prior("Legal")
    legal_profile = LEGAL_RISK_PROFILE

    _, cvar_without, veto_without = compute_stakeholder_value(terms_without, legal_belief, legal_profile)
    _, cvar_with, veto_with = compute_stakeholder_value(terms_with, legal_belief, legal_profile)

    assert cvar_without > cvar_with, "Adding documentation should reduce CVaR"
    assert not veto_with, "Veto should not fire when full documentation provided"


def test_bayesian_update_concentrates_belief():
    """10 consistent competent actions should concentrate belief on positive types."""
    belief = create_neutral_belief("Finance")

    for _ in range(10):
        belief = bayesian_update(
            belief=belief,
            action=create_action("send_document", ["roi_model"]),
            stakeholder_role="Finance",
            is_targeted=True
        )

    assert belief.positive_mass() > 0.70, (
        f"After 10 consistent competent actions, positive mass should exceed 0.70. "
        f"Got: {belief.positive_mass():.3f}"
    )


def test_likelihood_table_normalizes():
    """All likelihood values must be in [0.10, 0.95] for all action/role combinations."""
    for action_key in BASE_LIKELIHOODS:
        for role in ARCHETYPES:
            likelihoods = get_likelihood(action_key, [], role)
            for vendor_type, p in likelihoods.items():
                assert 0.10 <= p <= 0.95, (
                    f"Likelihood P({action_key}|{vendor_type}) for role {role} "
                    f"out of range: {p:.3f}"
                )


def test_non_targeted_update_weaker():
    """Non-targeted Bayesian update should produce smaller belief shift than targeted."""
    belief_targeted = create_neutral_belief("Legal")
    belief_non_targeted = create_neutral_belief("Legal")
    action = create_action("send_document", ["DPA"])

    updated_targeted = bayesian_update(belief_targeted, action, "Legal", is_targeted=True)
    updated_non_targeted = bayesian_update(belief_non_targeted, action, "Legal", is_targeted=False)

    delta_targeted = abs(updated_targeted.positive_mass() - belief_targeted.positive_mass())
    delta_non_targeted = abs(updated_non_targeted.positive_mass() - belief_non_targeted.positive_mass())

    assert delta_targeted > delta_non_targeted * 2, (
        f"Targeted update ({delta_targeted:.3f}) should be >2x non-targeted ({delta_non_targeted:.3f})"
    )
```

---

## 10. Implementation Files

| File | Implements | Key classes/functions |
|------|-----------|----------------------|
| `stakeholders/cvar_preferences.py` | CVaR computation, risk profiles, veto logic | `StakeholderRiskProfile`, `compute_cvar()`, `compute_stakeholder_value()`, `check_veto_trigger()` |
| `stakeholders/archetypes.py` | All six archetypes with full risk profiles | `LEGAL_RISK_PROFILE`, `FINANCE_RISK_PROFILE`, etc., `ARCHETYPES` dict |
| `committee/belief_tracker.py` | BeliefDistribution, Bayesian update | `BeliefDistribution`, `bayesian_update()`, `get_likelihood()` |

---

*Implementation order: `archetypes.py` → `cvar_preferences.py` → `belief_tracker.py`*
*Run all five tests before connecting to the deliberation engine.*
*The veto trigger test is the most important — it validates the core research claim.*
