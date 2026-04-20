# CAUSAL_COMMITTEE_DYNAMICS.md
# DealRoom v3 — Causal Committee Structure & Deliberation Engine

> **Scope:** Environment-internal specification.
> Covers how G is defined, sampled, and operated.
> Covers how committee deliberation updates B_i between vendor turns.
> The developer implementing `committee/causal_graph.py` and
> `committee/deliberation_engine.py` reads this document.
>
> **Companion doc:** `OBSERVATION_MECHANISM.md` specifies how the
> effects of G and deliberation are made visible to the agent.
> These two documents together define the full Dec-POMDP dynamics.

---

## 1. Formal Dec-POMDP Definition (Environment Side)

The buying committee and vendor agent interaction is a Dec-POMDP:

```
Dec-POMDP = ⟨S, A_vendor, A_committee, T, R, Ω_vendor, O, γ⟩
```

**This document specifies T — the transition function.**

```
T = P(s' | s, a_vendor, a_committee)
```

The transition has two independent components:

```
Step 1 (vendor action):    s → s_intermediate
  B_targeted(t+1) ∝ P(a_vendor | type) · B_targeted(t)
  All other beliefs unchanged.

Step 2 (committee deliberation):  s_intermediate → s'
  For each non-targeted stakeholder i:
    B_i(t+Δ) = B_i(t) + Σ_{j ∈ influencers(i)} w_ji · ΔB_j(t)
  Deliberation runs N steps (N = scenario parameter, hidden from agent).
  Outputs updated beliefs B_i(t+Δ) for all stakeholders.
  Agent observes only effects in next round, not deliberation itself.
```

The hidden state `s` contains: `G`, `{B_i(t)}`, `{τ_i}`, `{w_ij}`.
The observable state `o ∈ Ω_vendor` contains only what `OBSERVATION_MECHANISM.md` specifies.

---

## 2. The Causal Influence Graph G

### Formal Definition

```
G = (V, E)

V = set of stakeholder IDs in the episode roster
    |V| ∈ {5, 6, 7} depending on scenario

E: V × V → [0, 1]
   e_ij = w_ji = causal influence weight of stakeholder j on stakeholder i
          during internal committee deliberation
   e_ij = 0 means no edge (j does not influence i)
   e_ij = 1 means full influence (j's belief delta fully propagates to i)
```

**Key distinction:** `w_ji` is read as "j influences i" — j is the source, i is the destination. When j's belief updates (`ΔB_j`), this propagates to i by `w_ji · ΔB_j`.

### G is Sampled at Episode Reset

```python
G ~ P(G | authority_hierarchy, scenario_type, seed)
```

G is sampled once per episode at `reset()`. It is stored as hidden state. It never changes during the episode. It is never exposed to the agent.

### Sampling Distribution

```python
def sample_graph(
    stakeholder_set: List[str],
    authority_hierarchy: Dict[str, int],   # stakeholder_id -> authority level (1-5)
    scenario_type: str,                     # 'aligned' | 'conflicted' | 'hostile_acquisition'
    rng: np.random.Generator
) -> CausalGraph:
    """
    Sample G from scenario-conditioned prior.

    Two invariants hold across all scenarios:
    1. Authority invariant: high-authority stakeholders (ExecSponsor, authority=5)
       have outgoing edges to all stakeholders with probability >= 0.85.
       This reflects real committee dynamics: executives shape everyone's view.
    2. Role-cluster invariant: stakeholders in the same functional cluster
       (Finance+Procurement = cost cluster; Legal+Compliance = risk cluster;
       TechLead+Operations = implementation cluster) have higher intra-cluster
       edge probabilities than inter-cluster.

    Scenario-specific parameters:
      aligned:             sparse_factor=0.3, cluster_boost=0.4, cross_cut=0.1
      conflicted:          sparse_factor=0.5, cluster_boost=0.5, cross_cut=0.2
      hostile_acquisition: sparse_factor=0.7, cluster_boost=0.3, cross_cut=0.5
    """
```

### Scenario-Specific Graph Properties

#### `aligned` — Near-Star Topology

```
Structure:     ExecSponsor → all other nodes (high w, 0.6-0.9)
               Finance ↔ Procurement (moderate w, 0.3-0.5)
               Legal → Compliance (moderate w, 0.4-0.6)
               Cross-cluster edges: sparse (w < 0.2)

Mean density:  0.35 (35% of possible edges present above w=0.1)
Hub node:      ExecSponsor (betweenness centrality 0.7-0.9)

Agent challenge:
  Identify ExecSponsor as hub (straightforward from engagement correlations)
  Target ExecSponsor → cascades to all stakeholders efficiently
  Identification rounds: 5-7
```

#### `conflicted` — Two-Cluster Topology

```
Structure:     Cluster A: Finance ↔ Legal ↔ Compliance (dense, w=0.5-0.8)
               Cluster B: TechLead ↔ Operations ↔ Procurement (dense, w=0.5-0.8)
               Bridge:    ExecSponsor → both clusters (w=0.4-0.6)
               Cross-cluster edges: sparse (w=0.1-0.2)

Mean density:  0.50
Hub nodes:     ExecSponsor (inter-cluster), Finance (intra-cluster A)

Agent challenge:
  Identify two-cluster structure (requires targeting both clusters)
  Recognize ExecSponsor as bridge
  Must work both clusters — targeting only one yields partial progress
  Identification rounds: 8-10
```

#### `hostile_acquisition` — Dense Cross-Cutting Topology

```
Structure:     Many lateral edges across functional boundaries
               Authority structure shifted: new post-acquisition hierarchy
               Prior on authority may be wrong (ExecSponsor may be less central)
               Compliance elevated (regulatory scrutiny post-acquisition)
               Edge weights more variable (0.1-0.9, less predictable)

Mean density:  0.65
Hub nodes:     Unknown at episode start — must be inferred
               Sometimes Legal, sometimes Compliance, sometimes Finance
               depending on specific G sample

Agent challenge:
  Prior on committee structure is wrong — must update quickly
  Authority hierarchy different from aligned/conflicted
  Multiple high-centrality nodes (not a clear single hub)
  Identification rounds: 10-12
```

### Sampling Implementation

```python
@dataclass
class CausalGraph:
    nodes: List[str]                         # stakeholder IDs
    edges: Dict[Tuple[str, str], float]      # (source_j, dest_i) -> weight w_ji
    authority_weights: Dict[str, float]      # stakeholder -> authority level (normalized)
    scenario_type: str
    seed: int

    def get_weight(self, source: str, dest: str) -> float:
        """w_ji where j=source, i=dest. Returns 0.0 if no edge."""
        return self.edges.get((source, dest), 0.0)

    def get_influencers(self, stakeholder: str) -> Dict[str, float]:
        """Returns {j: w_ji} for all j that influence stakeholder i."""
        return {
            src: w for (src, dst), w in self.edges.items()
            if dst == stakeholder
        }

    def get_outgoing(self, stakeholder: str) -> Dict[str, float]:
        """Returns {i: w_ji} for all i that stakeholder j influences."""
        return {
            dst: w for (src, dst), w in self.edges.items()
            if src == stakeholder
        }


SCENARIO_PARAMS = {
    "aligned": {
        "base_edge_probability": 0.30,
        "intra_cluster_boost":   0.40,
        "cross_cluster_penalty": 0.20,
        "authority_edge_prob":   0.85,
        "weight_mean":           0.45,
        "weight_std":            0.15,
    },
    "conflicted": {
        "base_edge_probability": 0.45,
        "intra_cluster_boost":   0.50,
        "cross_cluster_penalty": 0.15,
        "authority_edge_prob":   0.80,
        "weight_mean":           0.50,
        "weight_std":            0.18,
    },
    "hostile_acquisition": {
        "base_edge_probability": 0.60,
        "intra_cluster_boost":   0.25,
        "cross_cluster_penalty": 0.05,
        "authority_edge_prob":   0.65,    # reduced — authority structure shifted
        "weight_mean":           0.45,
        "weight_std":            0.25,    # higher variance — less predictable
    },
}

FUNCTIONAL_CLUSTERS = {
    "cost":           ["Finance", "Procurement"],
    "risk":           ["Legal", "Compliance"],
    "implementation": ["TechLead", "Operations"],
    "authority":      ["ExecSponsor"],
}

def sample_graph(
    stakeholder_set: List[str],
    authority_hierarchy: Dict[str, int],
    scenario_type: str,
    rng: np.random.Generator
) -> CausalGraph:
    params = SCENARIO_PARAMS[scenario_type]
    edges = {}

    for source in stakeholder_set:
        for dest in stakeholder_set:
            if source == dest:
                continue

            # Compute edge probability
            source_authority = authority_hierarchy.get(source, 1)
            same_cluster = _same_functional_cluster(source, dest)

            if source_authority >= 4:  # ExecSponsor or equivalent
                p_edge = params["authority_edge_prob"]
            elif same_cluster:
                p_edge = params["base_edge_probability"] + params["intra_cluster_boost"]
            else:
                p_edge = max(0.0, params["base_edge_probability"] - params["cross_cluster_penalty"])

            # Sample edge existence
            if rng.random() < p_edge:
                # Sample edge weight
                w = float(np.clip(
                    rng.normal(params["weight_mean"], params["weight_std"]),
                    0.05, 0.95   # clip to meaningful range
                ))
                edges[(source, dest)] = w

    # Normalize authority weights
    total_authority = sum(authority_hierarchy.values())
    authority_normalized = {
        sid: authority_hierarchy[sid] / total_authority
        for sid in stakeholder_set
    }

    return CausalGraph(
        nodes=list(stakeholder_set),
        edges=edges,
        authority_weights=authority_normalized,
        scenario_type=scenario_type,
        seed=int(rng.integers(0, 2**32))
    )
```

---

## 3. Belief Propagation Equation

This is the core of committee dynamics. Pure Python — no LLM involved.

```
B_i(t+Δ) = B_i(t) + Σ_{j ∈ influencers(i)} w_ji · ΔB_j(t)
```

Where:
- `B_i(t)` = belief distribution over vendor types before deliberation
- `ΔB_j(t)` = change in j's belief from the vendor action this round
- `w_ji` = edge weight (j influences i)
- `B_i(t+Δ)` = belief distribution after N deliberation steps

### Implementation

```python
def propagate_beliefs(
    graph: CausalGraph,
    beliefs_before_action: Dict[str, BeliefDistribution],
    beliefs_after_action: Dict[str, BeliefDistribution],  # only targeted stakeholder updated
    n_steps: int = 3
) -> Dict[str, BeliefDistribution]:
    """
    Run N deliberation steps of belief propagation through G.

    beliefs_after_action has ONE stakeholder updated (the targeted one).
    All other beliefs are identical to beliefs_before_action.
    Propagation spreads the targeted stakeholder's update through the graph.

    n_steps=3 is standard. Higher n_steps allows longer propagation paths.
    hostile_acquisition uses n_steps=4 (denser graph needs more steps).
    """
    current_beliefs = {sid: b.copy() for sid, b in beliefs_after_action.items()}

    for step in range(n_steps):
        next_beliefs = {sid: b.copy() for sid, b in current_beliefs.items()}

        for dest_id in graph.nodes:
            influencers = graph.get_influencers(dest_id)
            if not influencers:
                continue

            # Compute delta from each influencer's change since before_action
            total_delta = 0.0
            for source_id, weight in influencers.items():
                delta_source = (
                    current_beliefs[source_id].positive_mass() -
                    beliefs_before_action[source_id].positive_mass()
                )
                total_delta += weight * delta_source

            # Apply propagation to dest's belief distribution
            if abs(total_delta) > 0.005:  # ignore negligible deltas
                next_beliefs[dest_id] = _apply_belief_delta(
                    belief=current_beliefs[dest_id],
                    delta=total_delta,
                    damping=0.85 ** step  # damping prevents runaway accumulation
                )

        current_beliefs = next_beliefs

    return current_beliefs


def _apply_belief_delta(
    belief: BeliefDistribution,
    delta: float,
    damping: float = 1.0
) -> BeliefDistribution:
    """
    Apply a signed delta to a belief distribution.
    Positive delta: shift mass toward positive vendor types.
    Negative delta: shift mass toward negative vendor types.
    Preserves distribution sum = 1.
    """
    effective_delta = delta * damping

    new_dist = dict(belief.distribution)
    positive_types = ['competent', 'trustworthy', 'aligned']
    negative_types = ['incompetent', 'deceptive', 'misaligned']

    if effective_delta > 0:
        # Shift mass from negative to positive types
        transfer = min(abs(effective_delta) / 3, 0.15)  # cap per-type transfer
        for t in negative_types:
            new_dist[t] = max(0.01, new_dist.get(t, 0) - transfer)
        for t in positive_types:
            new_dist[t] = min(0.95, new_dist.get(t, 0) + transfer)
    else:
        # Shift mass from positive to negative types
        transfer = min(abs(effective_delta) / 3, 0.15)
        for t in positive_types:
            new_dist[t] = max(0.01, new_dist.get(t, 0) - transfer)
        for t in negative_types:
            new_dist[t] = min(0.95, new_dist.get(t, 0) + transfer)

    # Renormalize
    total = sum(new_dist.values())
    new_dist = {k: v / total for k, v in new_dist.items()}

    return BeliefDistribution(
        distribution=new_dist,
        stakeholder_role=belief.stakeholder_role,
        history=belief.history + [('propagation', effective_delta)]
    )
```

### Why Damping

Without damping (`damping=1.0`), beliefs can run away to extremes in dense graphs — every propagation step amplifies the previous. Damping `0.85^step` means:
- Step 1: 100% of delta propagates
- Step 2: 85% of delta propagates
- Step 3: 72% of delta propagates

This reflects real committee dynamics: information degrades as it passes through intermediaries. The executive hears a summary of Finance's concern, not Finance's exact belief distribution.

---

## 4. Committee Deliberation Engine

### Architecture: Two Layers

```
Layer 1 — Computation (pure Python, no LLM):
  propagate_beliefs(G, beliefs, n_steps)
  → Updated belief states B_i(t+Δ) for all stakeholders
  → This drives EVERYTHING: observation signals, CVaR checks, reward computation
  → Microseconds to execute

Layer 2 — Rendering (one MiniMax call per vendor turn):
  generate_deliberation_summary(beliefs_before, beliefs_after, G_summary)
  → Natural language summary of what "happened" in the deliberation
  → Used ONLY for: demo visualization, HF blog examples, judge Q&A
  → Does NOT affect belief state, reward, or agent observation
  → ~150ms, ~$0.003 per call
```

**Critical:** Layer 2 is display-only. If it fails, comment it out and training still works. If Layer 1 has a bug, training breaks entirely. Test Layer 1 exhaustively before wiring Layer 2.

### Deliberation Step Protocol

```python
class CommitteeDeliberationEngine:

    def __init__(
        self,
        graph: CausalGraph,
        n_deliberation_steps: int = 3
    ):
        self.graph = graph
        self.n_steps = n_deliberation_steps

    def run(
        self,
        vendor_action: DealRoomAction,
        beliefs_before_action: Dict[str, BeliefDistribution],
        beliefs_after_vendor_action: Dict[str, BeliefDistribution],
        render_summary: bool = True
    ) -> DeliberationResult:
        """
        Full deliberation cycle for one vendor turn.

        1. Run belief propagation (Layer 1)
        2. Generate summary dialogue (Layer 2, optional)
        3. Return updated beliefs + summary

        Parameters:
          beliefs_before_action:     B_i(t) for all stakeholders (pre-action)
          beliefs_after_vendor_action: B_i(t) with ONLY targeted stakeholder updated
                                       (result of Bayesian update from vendor action)
        """
        # Layer 1: Deterministic belief propagation
        updated_beliefs = propagate_beliefs(
            graph=self.graph,
            beliefs_before_action=beliefs_before_action,
            beliefs_after_action=beliefs_after_vendor_action,
            n_steps=self.n_steps
        )

        # Layer 2: Rendering (display only)
        summary = None
        if render_summary:
            summary = self._generate_summary(
                beliefs_before=beliefs_before_action,
                beliefs_after=updated_beliefs,
                targeted_stakeholder=vendor_action.target_ids[0]
            )

        return DeliberationResult(
            updated_beliefs=updated_beliefs,
            summary_dialogue=summary,   # None if render_summary=False
            propagation_deltas={
                sid: (
                    updated_beliefs[sid].positive_mass() -
                    beliefs_before_action[sid].positive_mass()
                )
                for sid in updated_beliefs
            }
        )

    def _generate_summary(
        self,
        beliefs_before: Dict[str, BeliefDistribution],
        beliefs_after: Dict[str, BeliefDistribution],
        targeted_stakeholder: str
    ) -> str:
        """
        Generate 2-3 turn plausible dialogue for demo/logging purposes.
        Does NOT drive belief state. Called AFTER propagation completes.
        If MiniMax call fails, return empty string — training unaffected.
        """
        # Identify the two stakeholders with largest belief delta
        deltas = {
            sid: abs(beliefs_after[sid].positive_mass() - beliefs_before[sid].positive_mass())
            for sid in beliefs_after
            if sid != targeted_stakeholder
        }
        if not deltas:
            return ""

        top_two = sorted(deltas.keys(), key=lambda x: deltas[x], reverse=True)[:2]
        if len(top_two) < 2:
            return ""

        s1, s2 = top_two[0], top_two[1]
        d1 = beliefs_after[s1].positive_mass() - beliefs_before[s1].positive_mass()
        d2 = beliefs_after[s2].positive_mass() - beliefs_before[s2].positive_mass()

        try:
            prompt = f"""Two committee members briefly discuss the vendor's latest communication.
{s1} ({ARCHETYPES[s1].role}): current position shifted {'positively' if d1 > 0 else 'negatively'} 
{s2} ({ARCHETYPES[s2].role}): current position shifted {'positively' if d2 > 0 else 'negatively'}

Write a 2-3 turn realistic internal dialogue (not to the vendor) reflecting their positions.
Keep it under 80 words total. No vendor is present."""

            return minimax_call(prompt, max_tokens=100, temperature=0.8)
        except Exception:
            return ""  # Fail silently — rendering is non-critical
```

### N Steps by Scenario

```python
DELIBERATION_STEPS = {
    "aligned":             3,   # sparse graph, propagation terminates quickly
    "conflicted":          3,   # two-cluster, limited cross-cluster propagation
    "hostile_acquisition": 4,   # dense graph, longer propagation chains
}
```

---

## 5. Graph Centrality Computation

Used by the reward system (`r^causal`) and by the demo visualization.

```python
def get_betweenness_centrality(graph: CausalGraph) -> Dict[str, float]:
    """
    Compute betweenness centrality for each node in G.
    Used for r^causal reward: agent rewarded for targeting high-centrality nodes.

    Betweenness centrality = fraction of shortest paths passing through node.
    High centrality = information broker = high-leverage target for agent.

    Uses edge weights as path costs (higher weight = shorter effective path,
    because higher weight = faster propagation = more influence).
    """
    import networkx as nx

    # Build weighted directed graph
    G_nx = nx.DiGraph()
    G_nx.add_nodes_from(graph.nodes)
    for (src, dst), weight in graph.edges.items():
        # Convert weight to distance: higher weight = shorter path
        G_nx.add_edge(src, dst, weight=1.0 / (weight + 0.01))

    centrality = nx.betweenness_centrality(G_nx, weight='weight', normalized=True)
    return centrality


def get_max_centrality(graph: CausalGraph) -> float:
    """Returns the maximum betweenness centrality in graph. Used for normalization."""
    c = get_betweenness_centrality(graph)
    return max(c.values()) if c else 1.0
```

---

## 6. Identifiability Verification

This test must pass before the environment is considered valid. If G is not identifiable from observations, the entire learning signal for `r^causal` breaks.

```python
def verify_graph_identifiability(
    n_graphs: int = 20,
    n_interventions: int = 5,
    scenario_type: str = "conflicted",
    significance_threshold: float = 0.05
) -> bool:
    """
    Sample n_graphs different graphs.
    For each, run n_interventions targeted actions.
    Verify that behavioral signatures are pairwise distinguishable.

    If any pair of graphs is indistinguishable:
    - Increase n_deliberation_steps (more propagation = more signal)
    - Decrease engagement noise sigma (more precise signal)
    - Review propagation damping factor

    Returns True if all pairs distinguishable, raises AssertionError otherwise.
    """
    rng = np.random.default_rng(42)
    graphs = [
        sample_graph(STANDARD_STAKEHOLDERS, STANDARD_HIERARCHY, scenario_type, rng)
        for _ in range(n_graphs)
    ]

    signatures = []
    for graph in graphs:
        sig = []
        for target in STANDARD_STAKEHOLDERS[:n_interventions]:
            # Simulate a targeted action with ΔB = 0.3
            beliefs_before = create_neutral_beliefs(STANDARD_STAKEHOLDERS)
            beliefs_after_action = dict(beliefs_before)
            beliefs_after_action[target] = apply_positive_delta(beliefs_before[target], 0.3)

            # Run propagation
            updated = propagate_beliefs(graph, beliefs_before, beliefs_after_action, n_steps=3)

            # Record engagement deltas for all non-targeted stakeholders
            for sid in STANDARD_STAKEHOLDERS:
                if sid != target:
                    delta = (
                        updated[sid].positive_mass() -
                        beliefs_before[sid].positive_mass()
                    )
                    sig.append(delta)

        signatures.append(np.array(sig))

    # Check pairwise distinguishability
    min_distance = float('inf')
    for i, j in combinations(range(n_graphs), 2):
        dist = np.linalg.norm(signatures[i] - signatures[j])
        min_distance = min(min_distance, dist)
        dimension = len(signatures[i])
        threshold = significance_threshold * np.sqrt(dimension)
        assert dist > threshold, (
            f"Graphs {i} and {j} indistinguishable: dist={dist:.4f} < {threshold:.4f}. "
            f"G is not identifiable. Check propagation parameters."
        )

    print(f"✓ All {n_graphs*(n_graphs-1)//2} graph pairs distinguishable")
    print(f"  Minimum pairwise distance: {min_distance:.4f}")
    return True
```

---

## 7. Authority Shift Events (hostile_acquisition Only)

In the `hostile_acquisition` scenario, authority can shift mid-episode. This reflects post-acquisition reorganization: a new executive arrives, a department head is replaced, a compliance function is elevated.

```python
@dataclass
class AuthorityShiftEvent:
    round: int                 # round when event occurs (hidden from agent)
    stakeholder_id: str        # whose authority changes
    old_authority: int
    new_authority: int
    reason: str                # for demo/logging only

def apply_authority_shift(
    graph: CausalGraph,
    event: AuthorityShiftEvent,
    rng: np.random.Generator
) -> CausalGraph:
    """
    Modify G's edge weights when authority shifts.

    Authority increase: boost outgoing edge weights by 0.15-0.25
    Authority decrease: reduce outgoing edge weights by 0.15-0.25

    This makes the graph non-stationary in hostile_acquisition.
    The agent's inferred Ĝ from early rounds may be wrong after the shift.
    It must re-calibrate its graph estimate when engagement patterns change unexpectedly.
    """
    new_edges = dict(graph.edges)
    delta = 0.20 * (event.new_authority - event.old_authority) / 5

    for (src, dst), w in graph.edges.items():
        if src == event.stakeholder_id:
            new_edges[(src, dst)] = float(np.clip(w + delta, 0.05, 0.95))

    return CausalGraph(
        nodes=graph.nodes,
        edges=new_edges,
        authority_weights=graph.authority_weights,  # authority_weights not updated here
        scenario_type=graph.scenario_type,
        seed=graph.seed
    )
```

**Why this matters for training:** Authority shifts are unobservable. The agent only sees that its previously reliable graph inference breaks — engagement correlations that were stable suddenly change direction. It must learn to detect graph non-stationarity and rerun inference. This is the hardest variant. Only present in `hostile_acquisition`.

---

## 8. Behavioral Signature Computation

Used for identifiability testing. Also available to be used internally to debug the environment during development.

```python
def compute_behavioral_signature(
    graph: CausalGraph,
    targeted_stakeholder: str,
    belief_delta: float = 0.30,
    n_steps: int = 3
) -> Dict[str, float]:
    """
    Predict engagement deltas for all non-targeted stakeholders
    given a targeted action with specified belief_delta.

    Used for:
    1. Identifiability testing (verify graphs produce distinct signatures)
    2. Debug: visualize what the agent "should" learn from each intervention

    Returns: {stakeholder_id: predicted_true_engagement_delta}
    Note: this is the TRUE delta — in practice the agent sees noisy version.
    """
    beliefs_before = create_neutral_beliefs(graph.nodes)
    beliefs_after_action = dict(beliefs_before)
    beliefs_after_action[targeted_stakeholder] = apply_positive_delta(
        beliefs_before[targeted_stakeholder], belief_delta
    )

    updated = propagate_beliefs(graph, beliefs_before, beliefs_after_action, n_steps)

    signature = {}
    for sid in graph.nodes:
        if sid == targeted_stakeholder:
            continue
        true_before = compute_engagement_level(beliefs_before[sid])
        true_after = compute_engagement_level(updated[sid])
        signature[sid] = true_after - true_before

    return signature
```

---

## 9. Implementation Files

| File | Implements | Key functions |
|------|-----------|---------------|
| `committee/causal_graph.py` | CausalGraph, sampling, propagation, centrality | `sample_graph()`, `propagate_beliefs()`, `get_betweenness_centrality()`, `compute_behavioral_signature()` |
| `committee/deliberation_engine.py` | CommitteeDeliberationEngine, two-layer architecture | `CommitteeDeliberationEngine.run()`, `_generate_summary()` |
| `committee/belief_tracker.py` | BeliefDistribution, Bayesian update | `BeliefDistribution`, `bayesian_update()`, `apply_positive_delta()` |

---

## 10. Required Tests

```python
# test_causal_graph.py

def test_propagation_direction():
    """Positive belief delta for targeted stakeholder propagates positively to neighbors."""
    graph = sample_graph(['A', 'B', 'C'], {'A':3,'B':2,'C':1}, "aligned", rng)
    # Add edge A→B with weight 0.7, no edge A→C
    graph.edges[('A','B')] = 0.7

    before = create_neutral_beliefs(['A','B','C'])
    after_action = dict(before)
    after_action['A'] = apply_positive_delta(before['A'], 0.4)

    updated = propagate_beliefs(graph, before, after_action, n_steps=3)

    # B should improve (edge A→B exists)
    assert updated['B'].positive_mass() > before['B'].positive_mass() + 0.05
    # C should not improve (no edge A→C)
    assert abs(updated['C'].positive_mass() - before['C'].positive_mass()) < 0.03


def test_damping_prevents_runaway():
    """In dense graphs, damping should prevent belief distributions from collapsing to extremes."""
    graph = create_fully_connected_graph(['A','B','C','D','E'], weight=0.8)
    before = create_neutral_beliefs(['A','B','C','D','E'])
    after_action = dict(before)
    after_action['A'] = apply_positive_delta(before['A'], 0.5)

    updated = propagate_beliefs(graph, before, after_action, n_steps=5)

    for sid in ['B','C','D','E']:
        assert 0.0 < updated[sid].positive_mass() < 1.0, (
            f"Belief for {sid} collapsed to extreme — check damping"
        )


def test_graph_identifiability():
    """20 graphs must produce pairwise distinguishable behavioral signatures."""
    verify_graph_identifiability(n_graphs=20, n_interventions=5)


def test_authority_invariant():
    """ExecSponsor should have high outdegree in all scenario types."""
    for scenario in ['aligned', 'conflicted', 'hostile_acquisition']:
        for _ in range(10):
            g = sample_graph(STANDARD_STAKEHOLDERS, STANDARD_HIERARCHY, scenario, rng)
            exec_edges = g.get_outgoing('ExecSponsor')
            exec_outdegree = len([w for w in exec_edges.values() if w > 0.1])
            assert exec_outdegree >= 2, (
                f"ExecSponsor has only {exec_outdegree} outgoing edges in {scenario}. "
                "Authority invariant violated."
            )


def test_centrality_monotone():
    """Hub nodes should have higher centrality than leaf nodes."""
    g = create_star_graph(hub='ExecSponsor', leaves=['Finance','Legal','TechLead','Procurement'])
    centrality = get_betweenness_centrality(g)
    for leaf in ['Finance', 'Legal', 'TechLead', 'Procurement']:
        assert centrality['ExecSponsor'] > centrality[leaf], (
            f"Hub ExecSponsor (c={centrality['ExecSponsor']:.2f}) should have "
            f"higher centrality than {leaf} (c={centrality[leaf]:.2f})"
        )
```

---

*This document specifies the environment-internal causal dynamics.*
*Companion: `OBSERVATION_MECHANISM.md` specifies how effects are made visible to the agent.*
*Implementation order: `belief_tracker.py` → `causal_graph.py` → `deliberation_engine.py`*
*Run all five tests before wiring into `environment/dealroom_v3.py`.*
