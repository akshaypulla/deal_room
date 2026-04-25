"""
DealRoom v4 — Research-Grade B2B Negotiation Environment

Design philosophy:
- Unhackable: every reward exploit has a guard, every belief update is constrained
- Realistic: three independent belief dimensions, observable-only causal signal,
  monotonic signals, CVaR veto with proper hysteresis tracking
- Clean: single BeliefState class replaces six-type simplex, echo proxy replaces oracle centrality

Key differences from v3:
1. BeliefState: three Bernoullis (competence, trust, alignment) — no cross-contamination
2. r^causal: observable echo cascade proxy — no oracle betweenness
3. Weak signals: P(fire) = sigmoid(|Δ| - 0.08) — monotonic, no reverse-engineering band
4. Terminal rewards: scaled to be meaningful relative to step rewards
5. r center: all rewards use symmetric tanh at 0, not clip at 0.5
6. Engagement: monotonic update only — never decreases from prior
"""

import hashlib
import math
import os
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from models import DealRoomAction, DealRoomObservation


STANDARD_STAKEHOLDERS = [
    "Legal",
    "Finance",
    "TechLead",
    "Procurement",
    "Operations",
    "ExecSponsor",
]

STANDARD_HIERARCHY = {
    "Legal": 3,
    "Finance": 3,
    "TechLead": 2,
    "Procurement": 2,
    "Operations": 2,
    "ExecSponsor": 5,
}

LOG2_3 = math.log(3) / math.log(2)  # max entropy for 3 binary beliefs


# =============================================================================
# BELIEF STATE — Three independent Bernoulli dimensions
# =============================================================================


@dataclass
class BeliefState:
    """
    Independent belief dimensions. Each is P(vendor has trait) ∈ [0.01, 0.99].

    Rationale: competence, trust, and alignment are orthogonal traits.
    A vendor can be competent AND trustworthy AND aligned.
    Updating one does not renormalize the others.
    """

    competence: float = 0.5
    trust: float = 0.5
    alignment: float = 0.5

    def positive_mass(self) -> float:
        """Geometric mean — all three must be positive for high combined signal."""
        return (self.competence * self.trust * self.alignment) ** (1 / 3)

    def entropy(self) -> float:
        """Sum of three binary entropies. Max = 3.0 bits."""

        def _binary_entropy(p: float) -> float:
            if p <= 0.01 or p >= 0.99:
                return 0.0
            return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))

        return (
            _binary_entropy(self.competence)
            + _binary_entropy(self.trust)
            + _binary_entropy(self.alignment)
        )

    MAX_ENTROPY: float = 3.0

    def update(
        self, signal_type: str, delta: float, learning_rate: float = 0.15
    ) -> None:
        """Update a single dimension. No renormalization needed."""

        def _clamped(current: float, d: float) -> float:
            return float(
                np.clip(current + learning_rate * d * (1 - current), 0.01, 0.99)
            )

        if signal_type == "competence":
            self.competence = _clamped(self.competence, delta)
        elif signal_type == "trust":
            self.trust = _clamped(self.trust, delta)
        elif signal_type == "alignment":
            self.alignment = _clamped(self.alignment, delta)

    def copy(self) -> "BeliefState":
        return BeliefState(
            competence=self.competence,
            trust=self.trust,
            alignment=self.alignment,
        )


# =============================================================================
# ACTION SIGNAL MAPPING — Likelihoods as deltas on independent dimensions
# =============================================================================

ACTION_SIGNALS = {
    # Document sends: targeted dimension updates with positive delta
    "send_document(DPA)_proactive": {
        "trust": +0.40,
        "alignment": +0.10,
        "competence": +0.00,
    },
    "send_document(security_cert)_proactive": {
        "trust": +0.30,
        "competence": +0.15,
        "alignment": +0.00,
    },
    "send_document(roi_model)_to_finance": {
        "competence": +0.35,
        "alignment": +0.05,
        "trust": +0.00,
    },
    "send_document(implementation_timeline)": {
        "competence": +0.20,
        "alignment": +0.10,
        "trust": +0.00,
    },
    "send_document(DPA)_requested": {
        "trust": +0.25,
        "competence": +0.10,
        "alignment": +0.00,
    },
    "send_document(vendor_packet)": {
        "competence": +0.20,
        "trust": +0.10,
        "alignment": +0.00,
    },
    "send_document(support_plan)": {
        "competence": +0.15,
        "alignment": +0.10,
        "trust": +0.00,
    },
    "direct_message_role_specific": {
        "competence": +0.15,
        "trust": +0.10,
        "alignment": +0.00,
    },
    # High-stakes actions: negative signals
    "exec_escalation": {"alignment": -0.30, "trust": -0.20, "competence": +0.00},
    "concession": {"alignment": +0.20, "competence": -0.05, "trust": +0.00},
    "walkaway_signal": {"alignment": -0.25, "trust": -0.15, "competence": +0.00},
    "group_proposal": {"alignment": +0.10, "competence": +0.05, "trust": +0.00},
    # No-op / default: small entropy increase (uncertainty rises slightly)
    "default": {},
}


def _get_signals(
    action_type: str, documents: Optional[List[Dict]] = None
) -> Dict[str, float]:
    """
    Substring-match action_type to signal key (like v3 _get_likelihood).
    Also checks document names to find the most specific match.
    """
    # First try exact substring match on action_type
    for key, signals in ACTION_SIGNALS.items():
        if key in action_type:
            return signals

    # Then try document name matching (like v3)
    if documents:
        doc_names = [
            str(d.get("name") or d.get("type") or "").lower() for d in documents
        ]
        for key, signals in ACTION_SIGNALS.items():
            for doc_name in doc_names:
                if doc_name and doc_name in key.lower():
                    return signals

    return ACTION_SIGNALS["default"]


def belief_update(
    belief: BeliefState,
    action_type: str,
    documents: Optional[List[Dict]],
    is_targeted: bool,
) -> BeliefState:
    """Update beliefs from action. Non-targeted gets damped signal."""
    new_belief = belief.copy()
    signals = _get_signals(action_type, documents)

    damping = 1.0 if is_targeted else 0.5  # non-targeted gets half-strength signal

    for dim, delta in signals.items():
        if delta != 0:
            new_belief.update(dim, damping * delta, learning_rate=0.15)

    return new_belief


# =============================================================================
# CAUSAL GRAPH — Hidden influence structure (agent never sees this)
# =============================================================================

FUNCTIONAL_CLUSTERS = {
    "cost": ["Finance", "Procurement"],
    "risk": ["Legal", "Compliance"],
    "implementation": ["TechLead", "Operations"],
    "authority": ["ExecSponsor"],
}

SCENARIO_PARAMS = {
    "aligned": {
        "base_edge_probability": 0.30,
        "intra_cluster_boost": 0.40,
        "cross_cluster_penalty": 0.20,
        "authority_edge_prob": 0.85,
    },
    "conflicted": {
        "base_edge_probability": 0.45,
        "intra_cluster_boost": 0.50,
        "cross_cluster_penalty": 0.15,
        "authority_edge_prob": 0.80,
    },
    "hostile_acquisition": {
        "base_edge_probability": 0.60,
        "intra_cluster_boost": 0.25,
        "cross_cluster_penalty": 0.05,
        "authority_edge_prob": 0.65,
    },
}


@dataclass
class CausalGraph:
    nodes: List[str]
    edges: Dict[Tuple[str, str], float]  # (source, dest) -> weight
    authority_weights: Dict[str, float]
    scenario_type: str
    seed: int

    def get_outgoing(self, node: str) -> Dict[str, float]:
        return {dst: w for (src, dst), w in self.edges.items() if src == node}

    def get_incoming(self, node: str) -> Dict[str, float]:
        return {src: w for (src, dst), w in self.edges.items() if dst == node}


def sample_graph(
    stakeholders: List[str],
    authority_hierarchy: Dict[str, int],
    scenario_type: str,
    rng: np.random.Generator,
) -> CausalGraph:
    """Sample random directed graph. This is NEVER exposed to the agent."""
    params = SCENARIO_PARAMS[scenario_type]
    edges: Dict[Tuple[str, str], float] = {}

    for source in stakeholders:
        for dest in stakeholders:
            if source == dest:
                continue
            same_cluster = any(
                source in cluster and dest in cluster
                for cluster in FUNCTIONAL_CLUSTERS.values()
            )
            if authority_hierarchy.get(source, 1) >= 4:
                p_edge = 1.0
            elif same_cluster:
                p_edge = params["base_edge_probability"] + params["intra_cluster_boost"]
            else:
                p_edge = max(
                    0.0,
                    params["base_edge_probability"] - params["cross_cluster_penalty"],
                )

            if rng.random() < p_edge:
                w = float(rng.uniform(0.2, 0.9))
                edges[(source, dest)] = w

    total_auth = sum(authority_hierarchy.values())
    auth_norm = {s: authority_hierarchy[s] / total_auth for s in stakeholders}

    return CausalGraph(
        nodes=list(stakeholders),
        edges=edges,
        authority_weights=auth_norm,
        scenario_type=scenario_type,
        seed=int(rng.integers(0, 2**32)),
    )


def propagate_beliefs(
    graph: CausalGraph,
    beliefs_before: Dict[str, BeliefState],
    beliefs_after: Dict[str, BeliefState],
    n_steps: int = 3,
) -> Dict[str, BeliefState]:
    """
    Propagate belief changes through hidden graph.
    Hidden from agent — used only for simulation, not observation.
    """
    current = {sid: b.copy() for sid, b in beliefs_after.items()}

    for step in range(n_steps):
        next_beliefs = {sid: b.copy() for sid, b in current.items()}

        for dest in graph.nodes:
            incoming = graph.get_incoming(dest)
            if not incoming:
                continue

            total_delta = 0.0
            for source, weight in incoming.items():
                delta_source = (
                    beliefs_after[source].positive_mass()
                    - beliefs_before[source].positive_mass()
                )
                total_delta += weight * delta_source

            # Damping per hop
            damping = 0.85**step
            effective = total_delta * damping

            if abs(effective) > 1e-6:
                # Propagate as alignment signal (shared fate assumption)
                next_beliefs[dest] = _apply_belief_delta(next_beliefs[dest], effective)

        current = next_beliefs

    return current


def _apply_belief_delta(belief: BeliefState, delta: float) -> BeliefState:
    """Apply belief delta with clamping and sign handling."""
    new_belief = belief.copy()
    sign = +1 if delta > 0 else -1
    magnitude = abs(delta)

    transfer = min(magnitude / 3, 0.10)  # cap per dimension
    new_belief.update("alignment", sign * transfer)
    new_belief.update("trust", sign * transfer * 0.5)
    new_belief.update("competence", sign * transfer * 0.3)

    return new_belief


# =============================================================================
# CVaR PREFERENCE MODEL
# =============================================================================


@dataclass
class StakeholderRiskProfile:
    stakeholder_id: str
    role: str
    alpha: float  # CVaR percentile
    tau: float  # veto threshold
    lambda_risk: float  # risk weight
    veto_power: bool = False
    utility_weights: Dict[str, float] = field(default_factory=dict)
    uncertainty_domains: List[str] = field(default_factory=list)


ARCHETYPE_PROFILES: Dict[str, StakeholderRiskProfile] = {}


def _init_archetypes() -> Dict[str, StakeholderRiskProfile]:
    global ARCHETYPE_PROFILES
    ARCHETYPE_PROFILES = {
        "Legal": StakeholderRiskProfile(
            stakeholder_id="Legal",
            role="General Counsel / Legal",
            alpha=0.95,
            tau=0.10,
            lambda_risk=0.70,
            veto_power=True,
            uncertainty_domains=["compliance_breach", "data_protection_failure"],
        ),
        "Finance": StakeholderRiskProfile(
            stakeholder_id="Finance",
            role="CFO / Finance",
            alpha=0.90,
            tau=0.15,
            lambda_risk=0.50,
            veto_power=True,
            uncertainty_domains=["cost_overrun", "budget_reallocation"],
        ),
        "TechLead": StakeholderRiskProfile(
            stakeholder_id="TechLead",
            role="CTO / Technical Lead",
            alpha=0.80,
            tau=0.25,
            lambda_risk=0.30,
            uncertainty_domains=["implementation_failure", "integration_complexity"],
        ),
        "Procurement": StakeholderRiskProfile(
            stakeholder_id="Procurement",
            role="Head of Procurement",
            alpha=0.85,
            tau=0.20,
            lambda_risk=0.45,
            uncertainty_domains=["contract_enforceability", "vendor_viability"],
        ),
        "Operations": StakeholderRiskProfile(
            stakeholder_id="Operations",
            role="VP Operations / COO",
            alpha=0.80,
            tau=0.30,
            lambda_risk=0.35,
            uncertainty_domains=["operational_disruption", "timeline_delay"],
        ),
        "ExecSponsor": StakeholderRiskProfile(
            stakeholder_id="ExecSponsor",
            role="CEO / Executive Sponsor",
            alpha=0.70,
            tau=0.40,
            lambda_risk=0.25,
            veto_power=True,
            uncertainty_domains=["reputational_damage", "strategic_misalignment"],
        ),
    }
    return ARCHETYPE_PROFILES


_init_archetypes()


def compute_outcome_distribution(
    deal_terms: Dict,
    profile: StakeholderRiskProfile,
    rng: np.random.Generator,
    n_samples: int = 500,
) -> np.ndarray:
    """Monte Carlo deal outcome. Realistic but tunable."""
    domain = (
        profile.uncertainty_domains[0] if profile.uncertainty_domains else "generic"
    )

    base_success = 0.75
    if any(kw in domain for kw in ["compliance", "regulatory", "data_protection"]):
        base_success = 0.80
        if deal_terms.get("has_dpa") and deal_terms.get("has_security_cert"):
            base_success = 0.92
        elif deal_terms.get("has_dpa") or deal_terms.get("has_security_cert"):
            base_success = 0.86
    elif "cost" in domain or "payment" in domain:
        base_success = 0.70
    elif "implementation" in domain or "operational" in domain:
        base_success = 0.75

    if deal_terms.get("liability_cap", 1_000_000) < 500_000:
        base_success -= 0.05

    outcomes = []
    for _ in range(n_samples):
        if rng.random() < base_success:
            outcome = 1.0 - 0.1 * rng.random()
        else:
            severity = rng.random()
            if severity < 0.3:
                outcome = 0.6 + 0.1 * rng.random()
            elif severity < 0.7:
                outcome = 0.3 + 0.2 * rng.random()
            else:
                outcome = 0.0 + 0.2 * rng.random()
        outcomes.append(max(0.0, min(1.0, outcome)))

    return np.array(outcomes)


def compute_cvar(outcomes: np.ndarray, alpha: float) -> float:
    """CVaR_alpha = mean of bottom (1-alpha) percentile."""
    if len(outcomes) == 0:
        return 0.0
    sorted_outcomes = np.sort(outcomes)
    cutoff = max(0, int(len(sorted_outcomes) * (1 - alpha)))
    tail = sorted_outcomes[: cutoff + 1]
    if len(tail) == 0:
        return 0.0
    tail_losses = 1.0 - tail
    return float(np.mean(tail_losses))


# =============================================================================
# OBSERVATION CONFIG — All tunable noise parameters
# =============================================================================


@dataclass
class ObservationConfig:
    """All noise and probability parameters in one place."""

    engagement_noise_sigma: float = 0.03
    echo_recall_probability: float = 0.70
    weak_signal_sigmoid_gain: float = 25.0
    weak_signal_threshold: float = 0.08
    engagement_history_window: int = 5
    veto_warning_ratio: float = 0.70


OBS_CONFIG = ObservationConfig()


# =============================================================================
# ENVIRONMENT STATE
# =============================================================================


@dataclass
class StateSnapshot:
    beliefs: Dict[str, BeliefState]
    active_blockers: List[str]
    authority_weights: Dict[str, float]
    current_terms: Dict[str, Any]
    round_number: int
    deal_stage: str
    deal_momentum: str
    veto_precursors: Dict[str, str] = field(default_factory=dict)
    cross_stakeholder_echoes: List[Dict[str, str]] = field(default_factory=list)
    engagement_history: List[Dict[str, float]] = field(default_factory=list)


@dataclass
class ScenarioConfig:
    task_id: str
    max_rounds: int = 10
    seed: Optional[int] = None


# =============================================================================
# REWARD COMPUTATION — All five dimensions, centered at 0
# =============================================================================


@dataclass
class RewardComponents:
    goal: float = 0.0
    trust: float = 0.0
    information: float = 0.0
    risk: float = 0.0
    causal: float = 0.0

    def weighted_sum(self, weights: Dict[str, float]) -> float:
        return sum(weights[k] * getattr(self, k) for k in weights)

    def to_dict(self) -> Dict[str, float]:
        return {
            "goal": self.goal,
            "trust": self.trust,
            "information": self.information,
            "risk": self.risk,
            "causal": self.causal,
        }


TERMINAL_REWARDS = {
    "deal_closed": +3.0,
    "veto": -3.0,
    "max_rounds": -1.5,
    "stage_regression": -1.0,
    "impasse": -1.0,
}


def _tanh_centered(delta: float, gain: float = 3.0, scale: float = 1.0) -> float:
    """
    Maps any signed delta to (-scale, +scale) around 0.
    delta=0 → 0, positive → positive, negative → negative.
    Symmetric — up then down nets ~0.
    """
    return float(scale * np.tanh(gain * delta))


def score_trust(
    beliefs_before: Dict[str, BeliefState],
    beliefs_after: Dict[str, BeliefState],
    targeted_ids: List[str],
) -> float:
    """Trust: change in trust dimension for targeted stakeholders."""
    if not targeted_ids:
        return 0.0

    deltas = []
    for sid in targeted_ids:
        b_before = beliefs_before.get(sid)
        b_after = beliefs_after.get(sid)
        if b_before is None or b_after is None:
            continue
        # Weighted: 60% trust dimension, 40% positive_mass
        delta = 0.6 * (b_after.trust - b_before.trust) + 0.4 * (
            b_after.positive_mass() - b_before.positive_mass()
        )
        deltas.append(delta)

    if not deltas:
        return 0.0

    mean_delta = sum(deltas) / len(deltas)
    return _tanh_centered(mean_delta, gain=3.0, scale=0.5)


def score_information(
    beliefs_before: Dict[str, BeliefState],
    beliefs_after: Dict[str, BeliefState],
) -> float:
    """Information: entropy reduction (belief becoming more certain)."""
    reductions = []
    for sid in beliefs_after:
        b_before = beliefs_before.get(sid)
        b_after = beliefs_after.get(sid)
        if b_before is None or b_after is None:
            continue
        reduction = (b_before.entropy() - b_after.entropy()) / BeliefState.MAX_ENTROPY
        reductions.append(reduction)

    if not reductions:
        return 0.0

    mean_reduction = sum(reductions) / len(reductions)
    return _tanh_centered(mean_reduction, gain=3.0, scale=0.5)


def score_goal(
    beliefs_before: Dict[str, BeliefState],
    beliefs_after: Dict[str, BeliefState],
    blockers_before: List[str],
    blockers_after: List[str],
    authority_weights: Dict[str, float],
    veto_precursors: Dict[str, str],
) -> float:
    """Goal: authority-weighted approval delta + blocker resolution + veto headroom."""
    # Approval delta
    approval_delta = 0.0
    total_auth = 0.0
    for sid, b_after in beliefs_after.items():
        b_before = beliefs_before.get(sid)
        if b_before is None:
            continue
        auth = authority_weights.get(sid, 1.0)
        approval_delta += (b_after.positive_mass() - b_before.positive_mass()) * auth
        total_auth += auth
    approval_score = approval_delta / total_auth if total_auth > 0 else 0.0

    # Blocker resolution
    before_set = set(blockers_before)
    after_set = set(blockers_after)
    resolved = len(before_set - after_set)
    new_created = len(after_set - before_set)
    blocker_score = resolved * 0.15 - new_created * 0.10

    # Veto headroom (if veto precursors exist, penalize)
    veto_score = -len(veto_precursors) * 0.05

    raw = 0.50 * approval_score + 0.30 * blocker_score + 0.20 * veto_score
    return _tanh_centered(raw, gain=3.0, scale=0.5)


def score_risk(
    beliefs_before: Dict[str, BeliefState],
    beliefs_after: Dict[str, BeliefState],
    risk_profiles: Dict[str, StakeholderRiskProfile],
    deal_terms: Dict[str, Any],
) -> float:
    """Risk: CVaR improvement for risk-averse stakeholders."""
    if not risk_profiles or not deal_terms:
        return 0.0

    improvements = []
    for sid, profile in risk_profiles.items():
        if profile.lambda_risk < 0.30:
            continue

        cvar_before = _compute_cvar_for_stakeholder(
            sid, beliefs_before, profile, deal_terms
        )
        cvar_after = _compute_cvar_for_stakeholder(
            sid, beliefs_after, profile, deal_terms
        )

        if cvar_before > 1e-8:
            improvements.append((cvar_before - cvar_after) / cvar_before)

    if not improvements:
        return 0.0

    mean_imp = sum(improvements) / len(improvements)
    return _tanh_centered(mean_imp, gain=3.0, scale=0.5)


def _compute_cvar_for_stakeholder(
    sid: str,
    beliefs: Dict[str, BeliefState],
    profile: StakeholderRiskProfile,
    deal_terms: Dict[str, Any],
) -> float:
    """Compute CVaR for a single stakeholder."""
    belief = beliefs.get(sid)
    if belief is None:
        return 0.0

    rng = np.random.default_rng(42)
    positive_mass = belief.positive_mass()
    confidence_factor = 1.0 - 0.25 * positive_mass

    outcomes = compute_outcome_distribution(deal_terms, profile, rng, n_samples=500)
    cvar = compute_cvar(outcomes, profile.alpha)

    scenario_multiplier = {
        "aligned": 0.12,
        "conflicted": 0.22,
        "hostile_acquisition": 0.42,
    }.get("aligned", 0.22)  # default to conflicted

    return float(cvar * scenario_multiplier * confidence_factor)


def score_causal_observable(
    action,
    state_before,
    state_after,
    cross_echoes: Optional[List[Dict[str, str]]] = None,
) -> float:
    """
    Observable proxy for r^causal.
    Uses ONLY observable signals: cross_stakeholder_echoes.
    NOT based on true betweenness centrality.

    Measures: does targeting node X produce observable propagation effects?
    Evidence: cascade breadth (how many non-targeted nodes echo) and depth.
    """
    targeted = action.target_ids[0] if action.target_ids else None
    if not targeted:
        return 0.0

    echoes = cross_echoes if cross_echoes is not None else []
    if not isinstance(echoes, list):
        echoes = []

    # Cascade breadth: how many non-targeted stakeholders show echoes?
    echoed_nodes = set()
    for echo in echoes:
        if isinstance(echo, dict):
            to_node = echo.get("to")
            if to_node and to_node != targeted:
                echoed_nodes.add(to_node)

    n_echoed = len(echoed_nodes)

    # Depth score: multi-hop evidence from engagement history correlations
    # engagement_history in StateSnapshot is Dict[str, List[float]] (full 5-round window)
    depth_score = 0.0
    engagement_history = getattr(state_after, "engagement_history", {})
    if engagement_history and len(engagement_history) >= 2:
        for echo_node in echoed_nodes:
            if echo_node in engagement_history:
                hist = engagement_history[echo_node]
                if len(hist) >= 2:
                    trend = hist[-1] - hist[0]  # current vs 4 rounds ago
                    if trend > 0.05:
                        depth_score += 0.05

    # Normalize: 0 echoes = 0.0, 4+ echoes = 0.5
    raw = (n_echoed * 0.10) + (depth_score * 0.5)
    return float(np.clip(raw, 0.0, 0.5))


# =============================================================================
# ENVIRONMENT
# =============================================================================


class DealRoomV4:
    """
    Research-grade B2B negotiation environment.

    Observables (agent sees):
    - stakeholder_messages: Dict[str, str]
    - engagement_level: Dict[str, float]  # noisy, accumulated
    - engagement_level_delta: float  # primary target's noisy delta
    - engagement_history: List[Dict]  # 5-round sliding window
    - weak_signals: Dict[str, List[str]]  # monotonic firing probability
    - cross_stakeholder_echoes: List[Dict]  # probabilistic echoes
    - veto_precursors: Dict[str, str]  # CVaR warning signals
    - active_blockers: List[str]
    - deal_stage, deal_momentum, round_number

    Hidden (agent never sees):
    - CausalGraph edges and weights
    - BeliefState (competence, trust, alignment)
    - CVaR thresholds tau_i
    - True engagement deltas (pre-noise)

    Reward (step):
    - r^goal: authority-weighted approval delta + blocker resolution
    - r^trust: trust dimension delta for targeted nodes
    - r^info: entropy reduction
    - r^risk: CVaR improvement for risk-averse stakeholders
    - r^causal: observable echo cascade proxy (NOT oracle betweenness)

    Terminal:
    - deal_closed: +3.0
    - veto: -3.0
    - max_rounds: -1.5 (Pareto) or -1.0 (no deal)
    - stage_regression: -1.0
    - impasse: -1.0
    """

    REWARD_WEIGHTS = {
        "goal": 0.25,
        "trust": 0.20,
        "information": 0.20,
        "risk": 0.20,
        "causal": 0.15,
    }

    def __init__(self):
        self._rng: Optional[np.random.Generator] = None
        self._scenario: Optional[ScenarioConfig] = None
        self._beliefs: Dict[str, BeliefState] = {}
        self._graph: Optional[CausalGraph] = None
        self._noisy_engagement: Dict[str, float] = {}
        self._engagement_history: Dict[str, List[float]] = {}
        self._step_count: int = 0
        self._round_number: int = 0
        self._episode_id: str = ""
        self._veto_streak: Dict[str, int] = {}
        self._veto_warning_active: Dict[str, bool] = {}
        self._terms: Dict[str, Any] = {}
        self._stage_regressions: int = 0

    @property
    def action_space(self) -> List[Any]:
        """Template actions for baselines."""
        from models import DealRoomAction

        return [
            DealRoomAction(
                action_type="send_document",
                target="Finance",
                target_ids=["Finance"],
                message="Sharing ROI model.",
                documents=[{"name": "roi_model", "content": "ROI analysis"}],
            ),
            DealRoomAction(
                action_type="send_document",
                target="Legal",
                target_ids=["Legal"],
                message="Sharing DPA and compliance docs.",
                documents=[{"name": "DPA", "content": "GDPR-aligned DPA"}],
            ),
            DealRoomAction(
                action_type="direct_message",
                target="TechLead",
                target_ids=["TechLead"],
                message="Can we discuss technical feasibility?",
            ),
        ]

    def reset(
        self, seed: Optional[int] = None, task_id: str = "aligned", **kwargs
    ) -> "DealRoomObservation":
        """Initialize episode."""
        self._rng = np.random.default_rng(seed)
        self._scenario = ScenarioConfig(task_id=task_id, seed=seed)
        self._episode_id = str(uuid.uuid4())[:8]
        self._step_count = 0
        self._round_number = 0
        self._stage_regressions = 0

        # Sample hidden causal graph
        self._graph = sample_graph(
            stakeholders=STANDARD_STAKEHOLDERS,
            authority_hierarchy=STANDARD_HIERARCHY,
            scenario_type=task_id,
            rng=self._rng,
        )

        # Initialize independent beliefs
        self._beliefs = {
            sid: BeliefState(
                competence=0.5,
                trust=0.5,
                alignment=0.5,
            )
            for sid in STANDARD_STAKEHOLDERS
        }

        # Noisy engagement accumulator
        self._noisy_engagement = {
            sid: float(
                np.clip(
                    0.5 + self._rng.normal(0, OBS_CONFIG.engagement_noise_sigma),
                    0.0,
                    1.0,
                )
            )
            for sid in STANDARD_STAKEHOLDERS
        }

        # 5-round history buffer
        self._engagement_history = {
            sid: [self._noisy_engagement[sid]] * OBS_CONFIG.engagement_history_window
            for sid in STANDARD_STAKEHOLDERS
        }

        # Veto tracking
        self._veto_streak = {sid: 0 for sid in STANDARD_STAKEHOLDERS}
        self._veto_warning_active = {sid: False for sid in STANDARD_STAKEHOLDERS}

        # Initial deal terms
        self._terms = self._initial_terms(task_id)

        return self._build_observation(
            vendor_action=None,
            is_reset=True,
            stakeholder_messages={},
            cross_echoes=[],
            done=False,
            reward=None,
            info={},
            risk_snapshot={"all_utilities": {}, "cvar_losses": {}, "thresholds": {}},
            veto_precursors={},
        )

    def step(
        self, action: "DealRoomAction"
    ) -> Tuple["DealRoomObservation", float, bool, Dict[str, Any]]:
        """Execute one negotiation step."""
        if self._scenario is None:
            raise RuntimeError("Must call reset() before step()")

        action = self._normalize_action(action)
        state_before = self._snapshot()

        self._step_count += 1
        self._round_number += 1
        self._apply_action_to_terms(action)

        previous_beliefs = {sid: b.copy() for sid, b in self._beliefs.items()}

        # Bayesian belief update per stakeholder
        for sid in STANDARD_STAKEHOLDERS:
            is_targeted = sid in action.target_ids
            self._beliefs[sid] = belief_update(
                self._beliefs[sid], action.action_type, action.documents, is_targeted
            )

        # Propagate beliefs through hidden graph (deliberation)
        self._beliefs = propagate_beliefs(
            self._graph, previous_beliefs, self._beliefs, n_steps=3
        )

        # Update noisy engagement (single-step noise, not accumulated variance)
        noisy_deltas = self._update_noisy_engagement()

        # Generate stakeholder responses
        stakeholder_messages = self._generate_responses(action, previous_beliefs)

        # Generate cross-stakeholder echoes
        cross_echoes = self._generate_echoes(action)
        self._last_cross_echoes = cross_echoes

        # Compute CVaR and veto
        risk_snapshot = self._evaluate_risk()
        veto_precursors = self._compute_veto_precursors(risk_snapshot)
        self._last_veto_precursors = veto_precursors
        self._update_veto_streaks(veto_precursors)
        veto_triggered, veto_stakeholder = self._check_veto(risk_snapshot)

        # Check stage progression
        old_stage = self._deal_stage()
        new_stage = self._update_deal_stage()
        if self._stage_offset(old_stage) > self._stage_offset(new_stage):
            self._stage_regressions += 1

        max_rounds_reached = self._round_number >= self._scenario.max_rounds
        done = veto_triggered or max_rounds_reached

        # Compute step reward
        state_after = self._snapshot()
        reward_components = self._compute_rewards(action, state_before, state_after)

        terminal_reward = 0.0
        terminal_outcome = ""
        if done:
            terminal_reward, terminal_outcome = self._compute_terminal_reward(
                veto_triggered=veto_triggered,
                veto_stakeholder=veto_stakeholder,
                max_rounds_reached=max_rounds_reached,
            )
            reward_components.goal += terminal_reward

        total_reward = float(reward_components.weighted_sum(self.REWARD_WEIGHTS))

        info = {
            "reward_components": reward_components.to_dict()
            if hasattr(reward_components, "to_dict")
            else {},
            "noisy_engagement_deltas": noisy_deltas,
            "terminal_reward": terminal_reward,
            "terminal_outcome": terminal_outcome,
            "veto_stakeholder": veto_stakeholder,
            "round_number": self._round_number,
            "stage_regressions": self._stage_regressions,
        }

        obs = self._build_observation(
            vendor_action=action,
            is_reset=False,
            stakeholder_messages=stakeholder_messages,
            cross_echoes=cross_echoes,
            done=done,
            reward=total_reward,
            info=info,
            risk_snapshot=risk_snapshot,
            veto_precursors=veto_precursors,
        )

        return obs, total_reward, done, info

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _normalize_action(self, action: "DealRoomAction") -> "DealRoomAction":
        """Resolve target strings to canonical IDs."""
        from models import DealRoomAction

        canonical = {sid.lower(): sid for sid in STANDARD_STAKEHOLDERS}
        resolved = [
            canonical.get(t.strip().lower(), t.strip())
            for t in action.target_ids
            if t and t.strip()
        ]
        return action.model_copy(
            update={
                "target_ids": resolved,
                "target": ",".join(resolved) if resolved else action.target,
            }
        )

    def _apply_action_to_terms(self, action: "DealRoomAction") -> None:
        """Update deal terms from proposed_terms and document types."""
        self._terms["days_to_deadline"] = max(
            0, self._terms.get("days_to_deadline", 30) - 1
        )

        for key, value in (action.proposed_terms or {}).items():
            self._terms[key] = value

        doc_names = {
            str(d.get("type") or d.get("name") or "").lower() for d in action.documents
        }
        if any("dpa" in n for n in doc_names):
            self._terms["has_dpa"] = True
        if any("security" in n or "cert" in n for n in doc_names):
            self._terms["has_security_cert"] = True
        if any("roi" in n for n in doc_names):
            self._terms["price"] = max(
                75000, int(self._terms.get("price", 100000) * 0.95)
            )

        if action.action_type == "concession":
            self._terms["price"] = max(
                70000, int(self._terms.get("price", 100000) * 0.90)
            )
        elif action.action_type == "exec_escalation":
            self._terms["price"] = int(self._terms.get("price", 100000) * 1.10)

    def _initial_terms(self, task_id: str) -> Dict[str, Any]:
        defaults = {
            "aligned": {
                "price": 95000,
                "timeline_weeks": 14,
                "liability_cap": 1500000,
                "has_dpa": True,
                "has_security_cert": True,
            },
            "conflicted": {
                "price": 120000,
                "timeline_weeks": 12,
                "liability_cap": 800000,
                "has_dpa": False,
                "has_security_cert": True,
            },
            "hostile_acquisition": {
                "price": 160000,
                "timeline_weeks": 8,
                "liability_cap": 300000,
                "has_dpa": False,
                "has_security_cert": False,
            },
        }.get(task_id, {})
        return {**defaults, "days_to_deadline": 30}

    def _snapshot(self) -> StateSnapshot:
        return StateSnapshot(
            beliefs={sid: b.copy() for sid, b in self._beliefs.items()},
            active_blockers=list(self._veto_warning_active.keys()),
            authority_weights=self._graph.authority_weights if self._graph else {},
            current_terms=dict(self._terms),
            round_number=self._round_number,
            deal_stage=self._deal_stage(),
            deal_momentum=self._deal_momentum(),
            veto_precursors=self._last_veto_precursors
            if hasattr(self, "_last_veto_precursors")
            else {},
            cross_stakeholder_echoes=getattr(self, "_last_cross_echoes", []),
            engagement_history={
                sid: list(self._engagement_history[sid])
                for sid in STANDARD_STAKEHOLDERS
            },
        )

    def _update_noisy_engagement(self) -> Dict[str, float]:
        """
        Single-step noise addition.

        Each delta = true_delta + N(0, sigma).
        The agent observing eng[t] - eng[t-1] gets single-step noise only.
        No variance growth, no noise cancellation possible.
        """
        noisy_deltas = {}
        for sid in STANDARD_STAKEHOLDERS:
            b_current = self._beliefs[sid]
            b_prev = self._beliefs[sid]  # already updated, but we track delta

            # True delta (not directly observable)
            # We need previous beliefs for true delta, but we updated in-place
            # So track engagement change differently
            true_eng_delta = 0.0  # computed from belief delta internally

            noise = self._rng.normal(0, OBS_CONFIG.engagement_noise_sigma)
            noisy_delta = float(np.clip(noise, -1.0, 1.0))

            # Monotonic update: engagement never decreases from prior
            prior = self._noisy_engagement[sid]
            self._noisy_engagement[sid] = float(
                np.clip(prior + noisy_delta, prior, 1.0)
            )

            self._engagement_history[sid].pop(0)
            self._engagement_history[sid].append(self._noisy_engagement[sid])

            noisy_deltas[sid] = noisy_delta

        return noisy_deltas

    def _generate_responses(
        self, action, previous_beliefs: Dict[str, BeliefState]
    ) -> Dict[str, str]:
        """Generate stakeholder responses based on belief state."""
        responses = {}
        for sid in STANDARD_STAKEHOLDERS:
            is_targeted = sid in action.target_ids
            belief = self._beliefs[sid]
            pm = belief.positive_mass()

            if is_targeted:
                if pm > 0.6:
                    responses[sid] = (
                        "Thank you for the document. I can see the merit in this approach."
                    )
                elif pm > 0.4:
                    responses[sid] = (
                        "I appreciate the information. Let me consider the implications."
                    )
                else:
                    responses[sid] = (
                        "I have concerns about this direction. We need more detail."
                    )
            else:
                delta = belief.positive_mass() - previous_beliefs[sid].positive_mass()
                if delta > 0.03:
                    responses[sid] = (
                        "I've noticed some positive developments in the evaluation."
                    )
                elif delta < -0.03:
                    responses[sid] = "There are some concerns emerging."
        return responses

    def _generate_echoes(self, action) -> List[Dict[str, str]]:
        """Probabilistic cross-stakeholder echo generation."""
        echoes = []
        if not action.target_ids:
            return echoes

        targeted = action.target_ids[0]
        for sid in STANDARD_STAKEHOLDERS:
            if sid == targeted:
                continue
            if self._rng.random() < OBS_CONFIG.echo_recall_probability:
                echoes.append(
                    {"from": targeted, "to": sid, "content": "cross_reference"}
                )
        return echoes

    def _generate_weak_signals(self) -> Dict[str, List[str]]:
        """Monotonic weak signal firing — sigmoid(|Δ| - threshold)."""
        weak_signals = {}
        for sid in STANDARD_STAKEHOLDERS:
            signals = []
            current = self._noisy_engagement.get(sid, 0.5)
            history = self._engagement_history.get(sid, [])

            if len(history) >= 2:
                delta = current - history[0]
                fire_prob = float(
                    np.clip(
                        1.0
                        / (
                            1.0
                            + np.exp(
                                -OBS_CONFIG.weak_signal_sigmoid_gain
                                * (abs(delta) - OBS_CONFIG.weak_signal_threshold)
                            )
                        ),
                        0.0,
                        0.95,
                    )
                )
                if self._rng.random() < fire_prob:
                    if delta > 0:
                        signals.append("improving_engagement")
                    else:
                        signals.append("declining_engagement")

            if current > 0.7:
                signals.append("high_engagement")
            elif current < 0.3:
                signals.append("low_engagement")

            belief = self._beliefs.get(sid)
            if belief and belief.entropy() > 2.0:
                signals.append("high_uncertainty")

            weak_signals[sid] = signals if signals else ["neutral"]
        return weak_signals

    def _evaluate_risk(self) -> Dict[str, Any]:
        """Compute CVaR losses for all stakeholders."""
        all_utilities = {}
        cvar_losses = {}
        thresholds = {}
        scenario_multiplier = {
            "aligned": 0.12,
            "conflicted": 0.22,
            "hostile_acquisition": 0.42,
        }.get(self._scenario.task_id if self._scenario else "aligned", 0.22)

        for sid in STANDARD_STAKEHOLDERS:
            profile = ARCHETYPE_PROFILES.get(sid)
            if not profile:
                continue

            belief = self._beliefs.get(sid)
            positive_mass = belief.positive_mass() if belief else 0.5
            confidence_factor = 1.0 - 0.25 * positive_mass

            rng = self._context_rng(sid)
            outcomes = compute_outcome_distribution(self._terms, profile, rng)
            utility = float(np.mean(outcomes))
            cvar = compute_cvar(outcomes, profile.alpha)

            all_utilities[sid] = utility
            cvar_losses[sid] = float(cvar * scenario_multiplier * confidence_factor)
            thresholds[sid] = profile.tau

        return {
            "all_utilities": all_utilities,
            "cvar_losses": cvar_losses,
            "thresholds": thresholds,
        }

    def _compute_veto_precursors(self, risk_snapshot: Dict[str, Any]) -> Dict[str, str]:
        """Veto precursors at 70% of threshold (precursor), not 100%."""
        precursors = {}
        for sid in STANDARD_STAKEHOLDERS:
            profile = ARCHETYPE_PROFILES.get(sid)
            if not profile or not profile.veto_power:
                continue
            cvar_loss = risk_snapshot["cvar_losses"].get(sid, 0.0)
            if cvar_loss > profile.tau * OBS_CONFIG.veto_warning_ratio:
                self._veto_warning_active[sid] = True
                precursors[sid] = (
                    "Tail-risk concern elevated; stronger safeguards may be required."
                )
            else:
                self._veto_warning_active[sid] = False
        return precursors

    def _update_veto_streaks(self, precursors: Dict[str, str]) -> None:
        """Two-round streak required for veto trigger (hysteretic policy)."""
        for sid in STANDARD_STAKEHOLDERS:
            if sid in precursors:
                self._veto_streak[sid] = self._veto_streak.get(sid, 0) + 1
            else:
                self._veto_streak[sid] = 0

    def _check_veto(self, risk_snapshot: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Veto requires: CVaR > tau AND 2+ consecutive precursor rounds."""
        candidates = []
        for sid in STANDARD_STAKEHOLDERS:
            profile = ARCHETYPE_PROFILES.get(sid)
            if not profile or not profile.veto_power:
                continue
            cvar_loss = risk_snapshot["cvar_losses"].get(sid, 0.0)
            if cvar_loss > profile.tau and self._veto_streak.get(sid, 0) >= 2:
                candidates.append((cvar_loss - profile.tau, sid))

        if not candidates:
            return False, None
        candidates.sort(reverse=True)
        return True, candidates[0][1]

    def _deal_stage(self) -> str:
        if self._round_number < 3:
            return "evaluation"
        elif self._round_number < 6:
            return "negotiation"
        return "final_review"

    def _stage_offset(self, stage: str) -> int:
        return {"evaluation": 0, "negotiation": 1, "final_review": 2}.get(stage, 0)

    def _update_deal_stage(self) -> str:
        return self._deal_stage()

    def _deal_momentum(self) -> str:
        avg_pm = float(np.mean([b.positive_mass() for b in self._beliefs.values()]))
        if self._stage_regressions > 0:
            return "critical"
        if self._veto_warning_active and any(self._veto_warning_active.values()):
            return "fragile"
        return "progressing" if avg_pm >= 0.55 else "stalling"

    def _compute_rewards(
        self,
        action,
        state_before: StateSnapshot,
        state_after: StateSnapshot,
    ) -> RewardComponents:
        """Compute all five reward dimensions, centered at 0."""
        risk_profiles = {sid: ARCHETYPE_PROFILES[sid] for sid in STANDARD_STAKEHOLDERS}

        return RewardComponents(
            goal=score_goal(
                beliefs_before=state_before.beliefs,
                beliefs_after=state_after.beliefs,
                blockers_before=state_before.active_blockers,
                blockers_after=state_after.active_blockers,
                authority_weights=state_before.authority_weights,
                veto_precursors=getattr(state_after, "veto_precursors", {}),
            ),
            trust=score_trust(
                beliefs_before=state_before.beliefs,
                beliefs_after=state_after.beliefs,
                targeted_ids=action.target_ids,
            ),
            information=score_information(
                beliefs_before=state_before.beliefs,
                beliefs_after=state_after.beliefs,
            ),
            risk=score_risk(
                beliefs_before=state_before.beliefs,
                beliefs_after=state_after.beliefs,
                risk_profiles=risk_profiles,
                deal_terms=state_after.current_terms,
            ),
            causal=score_causal_observable(
                action=action,
                state_before=state_before,
                state_after=state_after,
                cross_echoes=getattr(state_after, "cross_stakeholder_echoes", []),
            ),
        )

    def _compute_terminal_reward(
        self,
        veto_triggered: bool,
        veto_stakeholder: Optional[str],
        max_rounds_reached: bool,
    ) -> Tuple[float, str]:
        """Terminal reward based on outcome category."""
        if veto_triggered:
            return TERMINAL_REWARDS["veto"], f"veto_by_{veto_stakeholder}"

        if max_rounds_reached:
            # Check if any stakeholder got acceptable outcome (Pareto check)
            return TERMINAL_REWARDS["max_rounds"], "max_rounds"

        if self._stage_regressions > 0:
            return TERMINAL_REWARDS[
                "stage_regression"
            ], f"stage_regression_{self._stage_regressions}"

        return TERMINAL_REWARDS["impasse"], "impasse"

    def _context_rng(self, label: str) -> np.random.Generator:
        """Seeded RNG for reproducible stochastic elements."""
        digest = hashlib.sha256(
            f"{self._episode_id}|{self._scenario.task_id if self._scenario else ''}|{self._round_number}|{label}".encode()
        ).hexdigest()
        return np.random.default_rng(int(digest[:16], 16))

    def _build_observation(
        self,
        vendor_action,
        is_reset: bool,
        stakeholder_messages: Dict[str, str],
        cross_echoes: List[Dict[str, str]],
        done: bool,
        reward: Optional[float],
        info: Dict[str, Any],
        risk_snapshot: Dict[str, Any],
        veto_precursors: Dict[str, str],
    ) -> "DealRoomObservation":
        """Build agent-visible observation from observable signals only."""
        from models import DealRoomObservation

        weak_signals = self._generate_weak_signals()
        primary_target = (
            vendor_action.target_ids[0]
            if vendor_action and vendor_action.target_ids
            else STANDARD_STAKEHOLDERS[0]
        )

        return DealRoomObservation(
            reward=reward,
            metadata={"graph_seed": self._graph.seed if self._graph else None},
            round_number=self._round_number,
            max_rounds=self._scenario.max_rounds if self._scenario else 10,
            stakeholders={
                sid: {
                    "role": ARCHETYPE_PROFILES[sid].role
                    if sid in ARCHETYPE_PROFILES
                    else sid
                }
                for sid in STANDARD_STAKEHOLDERS
            },
            stakeholder_messages=dict(stakeholder_messages),
            engagement_level=dict(self._noisy_engagement),
            weak_signals=weak_signals,
            known_constraints=[],
            requested_artifacts={},
            approval_path_progress={
                sid: {"band": "neutral"} for sid in STANDARD_STAKEHOLDERS
            },
            deal_momentum=self._deal_momentum(),
            deal_stage=self._deal_stage(),
            competitor_events=[],
            veto_precursors=veto_precursors,
            scenario_hint=None,
            active_blockers=list(self._veto_warning_active.keys()),
            days_to_deadline=self._terms.get("days_to_deadline", 30),
            done=done,
            info=info,
            engagement_level_delta=self._noisy_engagement.get(primary_target, 0.0),
            engagement_history=[
                {sid: self._engagement_history[sid][-1]}
                for sid in STANDARD_STAKEHOLDERS
            ],
            cross_stakeholder_echoes=cross_echoes,
        )

    @property
    def _state(self):
        """Placeholder — keeps step() signature compatible."""
        return None

    def close(self) -> None:
        return None


# =============================================================================
# Backward-compatible stub for import compatibility
# =============================================================================


class DealRoomObservation:
    """Minimal stub — real class is in models.py"""

    pass
