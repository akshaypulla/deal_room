"""
CausalGraph for DealRoom v3 - committee dynamics and belief propagation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

import numpy as np


VENDOR_TYPES = [
    "competent",
    "incompetent",
    "trustworthy",
    "deceptive",
    "aligned",
    "misaligned",
]

SCENARIO_PARAMS = {
    "aligned": {
        "base_edge_probability": 0.30,
        "intra_cluster_boost": 0.40,
        "cross_cluster_penalty": 0.20,
        "authority_edge_prob": 0.85,
        "weight_mean": 0.45,
        "weight_std": 0.15,
    },
    "conflicted": {
        "base_edge_probability": 0.45,
        "intra_cluster_boost": 0.50,
        "cross_cluster_penalty": 0.15,
        "authority_edge_prob": 0.80,
        "weight_mean": 0.50,
        "weight_std": 0.18,
    },
    "hostile_acquisition": {
        "base_edge_probability": 0.60,
        "intra_cluster_boost": 0.25,
        "cross_cluster_penalty": 0.05,
        "authority_edge_prob": 0.65,
        "weight_mean": 0.45,
        "weight_std": 0.25,
    },
}

FUNCTIONAL_CLUSTERS = {
    "cost": ["Finance", "Procurement"],
    "risk": ["Legal", "Compliance"],
    "implementation": ["TechLead", "Operations"],
    "authority": ["ExecSponsor"],
}


def _same_functional_cluster(source: str, dest: str) -> bool:
    for cluster_stakeholders in FUNCTIONAL_CLUSTERS.values():
        if source in cluster_stakeholders and dest in cluster_stakeholders:
            return True
    return False


@dataclass
class CausalGraph:
    nodes: List[str]
    edges: Dict[Tuple[str, str], float]
    authority_weights: Dict[str, float]
    scenario_type: str
    seed: int

    def get_weight(self, source: str, dest: str) -> float:
        return self.edges.get((source, dest), 0.0)

    def get_influencers(self, stakeholder: str) -> Dict[str, float]:
        return {src: w for (src, dst), w in self.edges.items() if dst == stakeholder}

    def get_outgoing(self, stakeholder: str) -> Dict[str, float]:
        return {dst: w for (src, dst), w in self.edges.items() if src == stakeholder}


def sample_graph(
    stakeholder_set: List[str],
    authority_hierarchy: Dict[str, int],
    scenario_type: str,
    rng: np.random.Generator,
) -> CausalGraph:
    params = SCENARIO_PARAMS[scenario_type]
    edges = {}

    for source in stakeholder_set:
        for dest in stakeholder_set:
            if source == dest:
                continue

            source_authority = authority_hierarchy.get(source, 1)
            same_cluster = _same_functional_cluster(source, dest)

            if source_authority >= 4:
                p_edge = 1.0  # Always create edges from authority nodes
            elif same_cluster:
                p_edge = params["base_edge_probability"] + params["intra_cluster_boost"]
            else:
                p_edge = max(
                    0.0,
                    params["base_edge_probability"] - params["cross_cluster_penalty"],
                )

            if rng.random() < p_edge:
                w = float(
                    np.clip(
                        rng.normal(params["weight_mean"], params["weight_std"]),
                        0.05,
                        0.95,
                    )
                )
                edges[(source, dest)] = w

    total_authority = sum(authority_hierarchy.values())
    authority_normalized = {
        sid: authority_hierarchy[sid] / total_authority for sid in stakeholder_set
    }

    return CausalGraph(
        nodes=list(stakeholder_set),
        edges=edges,
        authority_weights=authority_normalized,
        scenario_type=scenario_type,
        seed=int(rng.integers(0, 2**32)),
    )


@dataclass
class BeliefDistribution:
    distribution: Dict[str, float]
    stakeholder_role: str
    confidence: float = 1.0
    history: List[Tuple] = field(default_factory=list)

    def positive_mass(self) -> float:
        return sum(
            self.distribution.get(t, 0) for t in ["competent", "trustworthy", "aligned"]
        )

    def negative_mass(self) -> float:
        return sum(
            self.distribution.get(t, 0)
            for t in ["incompetent", "deceptive", "misaligned"]
        )

    def to_natural_language(self) -> str:
        pos = self.positive_mass()
        if pos > 0.70:
            return f"Vendor appears competent and aligned with {self.stakeholder_role} priorities."
        elif pos > 0.50:
            return f"Vendor shows reasonable competence; some uncertainty remains for {self.stakeholder_role}."
        elif pos > 0.30:
            return f"Significant uncertainty about vendor capability; {self.stakeholder_role} concerns active."
        else:
            return f"Low confidence in vendor; {self.stakeholder_role} concerns are elevated."

    def copy(self) -> "BeliefDistribution":
        return BeliefDistribution(
            distribution=dict(self.distribution),
            stakeholder_role=self.stakeholder_role,
            confidence=self.confidence,
            history=list(self.history),
        )


def create_neutral_beliefs(stakeholder_ids: List[str]) -> Dict[str, BeliefDistribution]:
    neutral = {t: 1.0 / len(VENDOR_TYPES) for t in VENDOR_TYPES}
    return {
        sid: BeliefDistribution(
            distribution=dict(neutral), stakeholder_role=sid, confidence=1.0, history=[]
        )
        for sid in stakeholder_ids
    }


def apply_positive_delta(
    belief: BeliefDistribution, delta: float
) -> BeliefDistribution:
    new_dist = dict(belief.distribution)
    positive_types = ["competent", "trustworthy", "aligned"]
    negative_types = ["incompetent", "deceptive", "misaligned"]

    transfer = min(abs(delta) / 3, 0.15)
    for t in negative_types:
        new_dist[t] = max(0.01, new_dist.get(t, 0) - transfer)
    for t in positive_types:
        new_dist[t] = min(0.95, new_dist.get(t, 0) + transfer)

    total = sum(new_dist.values())
    new_dist = {k: v / total for k, v in new_dist.items()}

    return BeliefDistribution(
        distribution=new_dist,
        stakeholder_role=belief.stakeholder_role,
        history=belief.history + [("positive_delta", delta)],
    )


def propagate_beliefs(
    graph: CausalGraph,
    beliefs_before_action: Dict[str, BeliefDistribution],
    beliefs_after_action: Dict[str, BeliefDistribution],
    n_steps: int = 3,
) -> Dict[str, BeliefDistribution]:
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

            k_gate = 10.0
            epsilon_floor = 0.003
            scaled_delta = total_delta * (
                abs(total_delta) / (abs(total_delta) + 1.0 / k_gate)
            )
            if abs(total_delta) > 0 and abs(scaled_delta) < epsilon_floor:
                scaled_delta = np.sign(total_delta) * epsilon_floor
            if abs(scaled_delta) > 1e-10:
                next_beliefs[dest_id] = _apply_belief_delta(
                    current_beliefs[dest_id], scaled_delta, damping=0.95**step
                )

        current_beliefs = next_beliefs

    return current_beliefs


def _apply_belief_delta(
    belief: BeliefDistribution, delta: float, damping: float = 1.0
) -> BeliefDistribution:
    effective_delta = delta * damping
    new_dist = dict(belief.distribution)
    positive_types = ["competent", "trustworthy", "aligned"]
    negative_types = ["incompetent", "deceptive", "misaligned"]

    if effective_delta > 0:
        transfer = min(abs(effective_delta) / 3, 0.15)
        for t in negative_types:
            new_dist[t] = max(0.01, new_dist.get(t, 0) - transfer)
        for t in positive_types:
            new_dist[t] = min(0.95, new_dist.get(t, 0) + transfer)
    else:
        transfer = min(abs(effective_delta) / 3, 0.15)
        for t in positive_types:
            new_dist[t] = max(0.01, new_dist.get(t, 0) - transfer)
        for t in negative_types:
            new_dist[t] = min(0.95, new_dist.get(t, 0) + transfer)

    total = sum(new_dist.values())
    new_dist = {k: v / total for k, v in new_dist.items()}

    return BeliefDistribution(
        distribution=new_dist,
        stakeholder_role=belief.stakeholder_role,
        history=belief.history + [("propagation", effective_delta)],
    )


def get_betweenness_centrality(graph: CausalGraph, stakeholder: str) -> float:
    if len(graph.nodes) <= 2:
        return 1.0

    n = len(graph.nodes)
    betweenness = 0.0

    for source in graph.nodes:
        for dest in graph.nodes:
            if source == dest or source == stakeholder or dest == stakeholder:
                continue

            if _shortest_path_exists(
                graph, source, stakeholder
            ) and _shortest_path_exists(graph, stakeholder, dest):
                betweenness += 1.0

    return betweenness / ((n - 1) * (n - 2)) if n > 2 else 0.0


def _shortest_path_exists(graph: CausalGraph, source: str, dest: str) -> bool:
    visited = set()
    queue = [source]
    visited.add(source)

    while queue:
        current = queue.pop(0)
        if current == dest:
            return True

        for neighbor in graph.get_outgoing(current).keys():
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return False


def compute_behavioral_signature(
    graph: CausalGraph,
    targeted_stakeholder: str,
    belief_delta: float = 0.30,
    n_steps: int = 3,
) -> Dict[str, float]:
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


def compute_engagement_level(belief: BeliefDistribution) -> float:
    return belief.positive_mass() - belief.negative_mass()
