"""
Scenario generation for DealRoom V2.5.

Public task IDs stay stable, but each task now generates a seeded episode with
dynamic stakeholders, hidden constraints, and a small relationship graph.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, List

import numpy as np

ROLE_LIBRARY: Dict[str, Dict[str, object]] = {
    "finance": {
        "label": "Finance Lead",
        "mandatory": True,
        "authority": 0.92,
        "veto_power": True,
        "style": "analytical",
        "requested_artifacts": ["roi_model", "reference_case"],
        "utility_weights": {
            "cost": 0.34,
            "risk": 0.22,
            "timeline": 0.10,
            "compliance": 0.12,
            "control": 0.22,
        },
    },
    "technical": {
        "label": "Technical Owner",
        "mandatory": True,
        "authority": 0.85,
        "veto_power": False,
        "style": "direct",
        "requested_artifacts": ["implementation_timeline", "security_cert"],
        "utility_weights": {
            "cost": 0.10,
            "risk": 0.24,
            "timeline": 0.24,
            "compliance": 0.12,
            "control": 0.30,
        },
    },
    "legal_compliance": {
        "label": "Legal & Compliance",
        "mandatory": True,
        "authority": 0.88,
        "veto_power": True,
        "style": "cautious",
        "requested_artifacts": ["dpa", "security_cert"],
        "utility_weights": {
            "cost": 0.06,
            "risk": 0.28,
            "timeline": 0.06,
            "compliance": 0.44,
            "control": 0.16,
        },
    },
    "procurement": {
        "label": "Procurement",
        "mandatory": False,
        "authority": 0.74,
        "veto_power": False,
        "style": "process-heavy",
        "requested_artifacts": ["vendor_packet", "reference_case"],
        "utility_weights": {
            "cost": 0.28,
            "risk": 0.18,
            "timeline": 0.08,
            "compliance": 0.20,
            "control": 0.26,
        },
    },
    "operations": {
        "label": "Operations Sponsor",
        "mandatory": False,
        "authority": 0.64,
        "veto_power": False,
        "style": "pragmatic",
        "requested_artifacts": ["implementation_timeline", "support_plan"],
        "utility_weights": {
            "cost": 0.08,
            "risk": 0.20,
            "timeline": 0.34,
            "compliance": 0.08,
            "control": 0.30,
        },
    },
    "executive_sponsor": {
        "label": "Executive Sponsor",
        "mandatory": False,
        "authority": 0.95,
        "veto_power": True,
        "style": "political",
        "requested_artifacts": ["roi_model", "implementation_timeline"],
        "utility_weights": {
            "cost": 0.18,
            "risk": 0.22,
            "timeline": 0.22,
            "compliance": 0.10,
            "control": 0.28,
        },
    },
}

CONSTRAINT_LIBRARY: Dict[str, Dict[str, object]] = {
    "budget_ceiling": {
        "slot": "price",
        "label": "Budget Ceiling",
        "required_artifact": "roi_model",
        "hint": "Board scrutiny on payback has intensified.",
        "weak_signal": "Finance keeps circling back to timing and board optics.",
        "checker": {"max_price": 185000},
    },
    "delivery_window": {
        "slot": "timeline_weeks",
        "label": "Delivery Window",
        "required_artifact": "implementation_timeline",
        "hint": "Implementation window is tighter than the public timeline suggests.",
        "weak_signal": "Operations mentions internal commitments they cannot miss.",
        "checker": {"max_timeline_weeks": 16},
    },
    "compliance_addendum": {
        "slot": "security_posture",
        "label": "Compliance Addendum",
        "required_artifact": "dpa",
        "hint": "A new compliance review has started in parallel.",
        "weak_signal": "Legal references recent audit findings without sharing details.",
        "checker": {"must_include": "gdpr"},
    },
    "supplier_process": {
        "slot": "support_level",
        "label": "Supplier Process",
        "required_artifact": "vendor_packet",
        "hint": "Process exceptions will be hard to obtain this quarter.",
        "weak_signal": "Procurement insists on formal process language in replies.",
        "checker": {"must_include": "named_support_lead"},
    },
}

SCENARIOS: Dict[str, Dict[str, object]] = {
    "aligned": {
        "max_rounds": 8,
        "days_to_deadline": 45,
        "stakeholder_count": (2, 3),
        "constraint_count": 1,
        "edge_count": (0, 1),
        "base_tracks": {
            "trust": (0.58, 0.70),
            "approval": (0.48, 0.62),
            "perceived_fit": (0.50, 0.65),
            "private_resistance": (0.25, 0.40),
        },
        "roles": ["finance", "technical", "operations", "executive_sponsor"],
        "constraint_pool": ["budget_ceiling", "delivery_window"],
        "event_round": None,
        "observability": "high",
        "description": (
            "A relatively aligned enterprise deal with one hidden hard constraint and "
            "light internal politics."
        ),
    },
    "conflicted": {
        "max_rounds": 10,
        "days_to_deadline": 32,
        "stakeholder_count": (3, 4),
        "constraint_count": (1, 2),
        "edge_count": (1, 1),
        "base_tracks": {
            "trust": (0.40, 0.58),
            "approval": (0.35, 0.52),
            "perceived_fit": (0.38, 0.55),
            "private_resistance": (0.38, 0.55),
        },
        "roles": ["finance", "technical", "procurement", "operations", "legal_compliance"],
        "constraint_pool": ["budget_ceiling", "delivery_window", "supplier_process"],
        "event_round": None,
        "observability": "medium",
        "description": (
            "A conflict-heavy deal where sequencing, coalition management, and hidden "
            "process blockers matter."
        ),
    },
    "hostile_acquisition": {
        "max_rounds": 10,
        "days_to_deadline": 22,
        "stakeholder_count": (4, 4),
        "constraint_count": 2,
        "edge_count": (1, 2),
        "base_tracks": {
            "trust": (0.35, 0.52),
            "approval": (0.30, 0.48),
            "perceived_fit": (0.35, 0.50),
            "private_resistance": (0.45, 0.60),
        },
        "roles": ["finance", "technical", "legal_compliance", "executive_sponsor", "procurement"],
        "constraint_pool": ["budget_ceiling", "delivery_window", "compliance_addendum"],
        "event_round": (3, 4),
        "observability": "low",
        "description": (
            "A compressed, political deal with an authority shift, two hidden hard "
            "constraints, and very low tolerance for inconsistency."
        ),
    },
}

LEGACY_TARGET_ALIASES = {
    "cfo": ["finance"],
    "cto": ["technical"],
    "legal": ["legal_compliance"],
    "procurement": ["procurement"],
    "ops": ["operations"],
    "cto_cfo": ["technical", "finance"],
    "legal_procurement": ["legal_compliance", "procurement"],
}

ROLE_NAME_SETS = {
    "finance": ["Mira", "Jon", "Alma"],
    "technical": ["Ravi", "Noah", "Elena"],
    "legal_compliance": ["Priya", "Sara", "Nikhil"],
    "procurement": ["Anya", "Lee", "Jordan"],
    "operations": ["Tess", "Mason", "Ivy"],
    "executive_sponsor": ["Owen", "Lina", "Sofia"],
}

RELATIONSHIP_TEMPLATES = {
    "alliance": [
        ("finance", "procurement"),
        ("technical", "operations"),
        ("executive_sponsor", "finance"),
    ],
    "conflict": [
        ("finance", "technical"),
        ("legal_compliance", "operations"),
        ("procurement", "technical"),
    ],
    "sponsor": [
        ("executive_sponsor", "technical"),
        ("operations", "finance"),
        ("technical", "operations"),
    ],
}


def _sample_track(rng: np.random.Generator, low: float, high: float) -> float:
    return round(float(rng.uniform(low, high)), 3)


def _pick_roles(template: Dict[str, object], rng: np.random.Generator) -> List[str]:
    min_count, max_count = template["stakeholder_count"]
    count = int(rng.integers(min_count, max_count + 1))
    roles = list(template["roles"])
    if "finance" not in roles:
        roles.append("finance")
    selected = ["finance"]
    optional = [role for role in roles if role != "finance"]
    while len(selected) < count and optional:
        choice = str(rng.choice(optional))
        optional.remove(choice)
        selected.append(choice)
    if template is SCENARIOS["hostile_acquisition"]:
        for required in ["technical", "legal_compliance"]:
            if required not in selected:
                selected[-1] = required
    return list(dict.fromkeys(selected))


def _pick_constraints(template: Dict[str, object], rng: np.random.Generator) -> List[str]:
    count_setting = template["constraint_count"]
    if isinstance(count_setting, tuple):
        count = int(rng.integers(count_setting[0], count_setting[1] + 1))
    else:
        count = int(count_setting)
    pool = list(template["constraint_pool"])
    rng.shuffle(pool)
    return pool[:count]


def _pick_edges(template: Dict[str, object], roles: List[str], rng: np.random.Generator) -> List[Dict[str, str]]:
    min_edges, max_edges = template["edge_count"]
    if max_edges == 0:
        return []
    edge_count = int(rng.integers(min_edges, max_edges + 1))
    candidates: List[Dict[str, str]] = []
    for edge_type, pairs in RELATIONSHIP_TEMPLATES.items():
        for source, target in pairs:
            if source in roles and target in roles and source != target:
                candidates.append({"type": edge_type, "source": source, "target": target})
    rng.shuffle(candidates)
    return candidates[:edge_count]


def expand_targets(target: str, available_ids: List[str]) -> List[str]:
    if not target:
        return []
    lowered = target.lower().strip()
    if lowered == "all":
        return list(available_ids)
    if lowered in LEGACY_TARGET_ALIASES:
        return [item for item in LEGACY_TARGET_ALIASES[lowered] if item in available_ids]
    raw_parts = [part.strip() for part in target.split(",") if part.strip()]
    normalized = []
    for part in raw_parts:
        part_lower = part.lower()
        if part_lower in LEGACY_TARGET_ALIASES:
            normalized.extend(
                [item for item in LEGACY_TARGET_ALIASES[part_lower] if item in available_ids]
            )
        elif part_lower in available_ids:
            normalized.append(part_lower)
    return list(dict.fromkeys(normalized))


def generate_episode(task_id: str, seed: int | None = None) -> Dict[str, object]:
    if task_id not in SCENARIOS:
        raise ValueError(f"Unknown task_id '{task_id}'. Valid: {list(SCENARIOS)}")

    template = SCENARIOS[task_id]
    rng = np.random.default_rng(seed)
    roles = _pick_roles(template, rng)
    constraints = _pick_constraints(template, rng)
    edges = _pick_edges(template, roles, rng)

    stakeholders: Dict[str, Dict[str, object]] = {}
    stakeholder_private: Dict[str, Dict[str, object]] = {}
    requested_artifacts: Dict[str, List[str]] = {}
    approval_caps: Dict[str, float] = {}

    for role_id in roles:
        role_template = deepcopy(ROLE_LIBRARY[role_id])
        name = str(rng.choice(ROLE_NAME_SETS[role_id]))
        public_label = f"{name} ({role_template['label']})"
        stakeholders[role_id] = {
            "id": role_id,
            "name": name,
            "display_name": public_label,
            "role": role_id,
            "authority": role_template["authority"],
            "mandatory": role_template["mandatory"],
            "veto_power": role_template["veto_power"],
            "style": role_template["style"],
        }
        requested_artifacts[role_id] = list(role_template["requested_artifacts"])
        approval_caps[role_id] = 1.0
        stakeholder_private[role_id] = {
            "role": role_id,
            "trust": _sample_track(rng, *template["base_tracks"]["trust"]),
            "approval": _sample_track(rng, *template["base_tracks"]["approval"]),
            "perceived_fit": _sample_track(rng, *template["base_tracks"]["perceived_fit"]),
            "private_resistance": _sample_track(
                rng, *template["base_tracks"]["private_resistance"]
            ),
            "utility_weights": deepcopy(role_template["utility_weights"]),
            "mandatory": role_template["mandatory"],
            "veto_power": role_template["veto_power"],
            "authority": role_template["authority"],
            "style": role_template["style"],
            "requested_artifacts": list(role_template["requested_artifacts"]),
            "permanent_marks": [],
            "discovered_constraints": [],
            "band_history": [],
        }

    hidden_constraints = {}
    for constraint_id in constraints:
        payload = deepcopy(CONSTRAINT_LIBRARY[constraint_id])
        payload["id"] = constraint_id
        payload["status"] = "hidden"
        payload["revealed_by"] = []
        hidden_constraints[constraint_id] = payload

    event_round = template["event_round"]
    if isinstance(event_round, tuple):
        event_round = int(rng.integers(event_round[0], event_round[1] + 1))

    semantic_threshold_jitter = {
        slot: round(float(rng.uniform(-0.03, 0.03)), 4)
        for slot in [
            "price",
            "timeline_weeks",
            "security_posture",
            "liability",
            "support_level",
            "implementation_commitment",
        ]
    }

    return {
        "task_id": task_id,
        "template": deepcopy(template),
        "stakeholders": stakeholders,
        "stakeholder_private": stakeholder_private,
        "hidden_constraints": hidden_constraints,
        "relationship_edges": edges,
        "requested_artifacts": requested_artifacts,
        "approval_caps": approval_caps,
        "days_to_deadline": template["days_to_deadline"],
        "event_round": event_round,
        "semantic_threshold_jitter": semantic_threshold_jitter,
        "observability": template["observability"],
        "description": template["description"],
    }
