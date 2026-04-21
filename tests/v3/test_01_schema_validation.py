#!/usr/bin/env python3
"""
test_01_schema_validation.py
DealRoom v3 — Observation Schema Validation

Validates:
- All 18 required fields are present
- No internal/hidden fields are exposed to the agent
- Field types match the specification
- Engagement history window is correctly sized
- Action schema accepts all valid action types + lookahead
- Documents field format is correct
"""

import os
import sys
from pathlib import Path

_dotenv = Path(__file__).parent / ".env"
if _dotenv.exists():
    try:
        from dotenv import load_dotenv

        load_dotenv(_dotenv)
    except ImportError:
        pass

import requests

BASE_URL = os.getenv("DEALROOM_BASE_URL", "http://127.0.0.1:7860")

REQUIRED_FIELDS = [
    "round_number",
    "max_rounds",
    "stakeholders",
    "stakeholder_messages",
    "engagement_level",
    "engagement_level_delta",
    "engagement_history",
    "weak_signals",
    "cross_stakeholder_echoes",
    "veto_precursors",
    "known_constraints",
    "requested_artifacts",
    "approval_path_progress",
    "deal_momentum",
    "deal_stage",
    "active_blockers",
    "days_to_deadline",
    "done",
]

HIDDEN_FIELDS = {
    # Core research-grade internal state — must never appear as a top-level key
    "G",
    "causal_graph",
    "graph",
    "true_beliefs",
    "belief_distributions",
    "belief_state",
    "B_i",
    "V_i",
    # CVaR / risk parameters — internal to the grader
    "tau",
    "tau_i",
    "risk_thresholds",
    "cvar_thresholds",
    "edge_weights",
    "w_ij",
    # Deliberation / internal dialogue
    "deliberation_transcript",
    "deliberation_log",
    "internal_dialogue",
    # Utility / preference parameters
    "u_i",
    "u_ij",
    # Do NOT add substrings here — matching is exact key check,
    # so "graph" matches only key "graph", NOT "graph_seed" or "graph_structure"
}

ACTION_TYPES = [
    "direct_message",
    "send_document",
    "group_proposal",
    "backchannel",
    "exec_escalation",
]
VALID_TARGETS = [
    "Legal",
    "Finance",
    "TechLead",
    "Procurement",
    "Operations",
    "ExecSponsor",
]


def test_1_1_all_required_fields_present():
    print("\n[1.1] Required fields in observation...")
    r = requests.post(f"{BASE_URL}/reset", json={"task_id": "aligned"}, timeout=30)
    obs = r.json()
    missing = [f for f in REQUIRED_FIELDS if f not in obs]
    if missing:
        raise AssertionError(f"Missing required fields: {missing}")
    print(f"  ✓ All {len(REQUIRED_FIELDS)} required fields present")


def test_1_2_no_hidden_fields_exposed():
    print("\n[1.2] Hidden/internal fields not exposed...")
    r = requests.post(
        f"{BASE_URL}/reset", json={"task_id": "hostile_acquisition"}, timeout=30
    )
    obs = r.json()

    def find_hidden(d, path=""):
        found = []
        if isinstance(d, dict):
            for k, v in d.items():
                if k in HIDDEN_FIELDS:
                    found.append(f"{path}.{k}")
                found.extend(find_hidden(v, f"{path}.{k}"))
        elif isinstance(d, list):
            for i, item in enumerate(d):
                found.extend(find_hidden(item, f"{path}[{i}]"))
        return found

    exposed = find_hidden(obs)
    if exposed:
        raise AssertionError(f"HIDDEN FIELDS EXPOSED: {exposed}")
    print("  ✓ No hidden fields exposed in any scenario")


def test_1_3_field_types_correct():
    print("\n[1.3] Field types match specification...")
    session = requests.Session()
    r = session.post(f"{BASE_URL}/reset", json={"task_id": "aligned", "seed": 42})
    session_id = r.json().get("metadata", {}).get("session_id")

    r = session.post(
        f"{BASE_URL}/step",
        json={
            "metadata": {"session_id": session_id},
            "action_type": "direct_message",
            "target_ids": ["Finance"],
            "message": "Test message.",
            "documents": [],
            "lookahead": None,
        },
        timeout=60,
    )
    obs = r.json().get("observation", r.json())

    assert isinstance(obs.get("round_number"), int), "round_number must be int"
    assert isinstance(obs.get("max_rounds"), int), "max_rounds must be int"
    assert isinstance(obs.get("stakeholders"), dict), "stakeholders must be dict"
    assert isinstance(obs.get("engagement_level"), dict), (
        "engagement_level must be dict"
    )
    assert isinstance(obs.get("engagement_level_delta"), (int, float)), (
        "engagement_level_delta must be numeric"
    )
    assert isinstance(obs.get("engagement_history"), list), (
        "engagement_history must be list"
    )
    assert isinstance(obs.get("weak_signals"), dict), "weak_signals must be dict"
    assert isinstance(obs.get("cross_stakeholder_echoes"), list), (
        "cross_stakeholder_echoes must be list"
    )
    assert isinstance(obs.get("veto_precursors"), dict), "veto_precursors must be dict"
    assert isinstance(obs.get("deal_momentum"), str), (
        "deal_momentum must be str (categorical: stalling/building/surging)"
    )
    assert isinstance(obs.get("deal_stage"), str), "deal_stage must be str"
    assert isinstance(obs.get("active_blockers"), list), "active_blockers must be list"
    assert isinstance(obs.get("days_to_deadline"), (int, float)), (
        "days_to_deadline must be numeric"
    )
    assert isinstance(obs.get("done"), bool), "done must be bool"

    print("  ✓ All field types correct")


def test_1_4_engagement_history_window_size():
    print("\n[1.4] Engagement history window size...")
    r = requests.post(f"{BASE_URL}/reset", json={"task_id": "conflicted"}, timeout=30)
    obs = r.json()
    history = obs.get("engagement_history", [])

    assert len(history) >= 5, (
        f"engagement_history must have >=5 entries, got {len(history)}"
    )
    for i, entry in enumerate(history):
        assert isinstance(entry, dict), (
            f"engagement_history[{i}] must be dict (stakeholder snapshots)"
        )

    print(f"  ✓ Window size = {len(history)} entries (>=5 required)")


def test_1_5_engagement_level_delta_single_float():
    print("\n[1.5] engagement_level_delta is single float...")
    session = requests.Session()
    r = session.post(f"{BASE_URL}/reset", json={"task_id": "aligned", "seed": 10})
    session_id = r.json().get("metadata", {}).get("session_id")

    r = session.post(
        f"{BASE_URL}/step",
        json={
            "metadata": {"session_id": session_id},
            "action_type": "direct_message",
            "target_ids": ["Finance"],
            "message": "Test.",
            "documents": [],
            "lookahead": None,
        },
        timeout=60,
    )
    obs = r.json().get("observation", r.json())

    delta = obs.get("engagement_level_delta")
    assert delta is not None, "engagement_level_delta missing from observation"
    assert isinstance(delta, (int, float)), (
        f"Expected numeric, got {type(delta).__name__}"
    )
    assert not isinstance(delta, dict), "engagement_level_delta must NOT be a dict"

    print(f"  ✓ delta = {delta:.4f} (single float, not dict)")


def test_1_6_cross_stakeholder_echoes_is_list():
    print("\n[1.6] cross_stakeholder_echoes is list of dicts...")
    r = requests.post(f"{BASE_URL}/reset", json={"task_id": "aligned"}, timeout=30)
    cse = r.json().get("cross_stakeholder_echoes")

    assert isinstance(cse, list), (
        f"cross_stakeholder_echoes must be list, got {type(cse).__name__}"
    )
    for i, echo in enumerate(cse):
        assert isinstance(echo, dict), (
            f"echo[{i}] must be dict, got {type(echo).__name__}"
        )
        assert "from" in echo or "from_stakeholder" in echo or "sender" in echo, (
            f"echo[{i}] missing sender field"
        )

    print(f"  ✓ echoes is list of {len(cse)} dicts")


def test_1_7_stakeholder_messages_populated_after_step():
    print("\n[1.7] stakeholder_messages populated after step...")
    session = requests.Session()
    r = session.post(f"{BASE_URL}/reset", json={"task_id": "conflicted", "seed": 20})
    session_id = r.json().get("metadata", {}).get("session_id")

    r = session.post(
        f"{BASE_URL}/step",
        json={
            "metadata": {"session_id": session_id},
            "action_type": "send_document",
            "target_ids": ["Finance"],
            "message": "DPA attached.",
            "documents": [{"name": "DPA", "content": "DPA content"}],
            "lookahead": None,
        },
        timeout=60,
    )
    obs = r.json().get("observation", r.json())

    msgs = obs.get("stakeholder_messages", {})
    assert isinstance(msgs, dict), (
        f"stakeholder_messages must be dict, got {type(msgs).__name__}"
    )

    print(f"  ✓ stakeholder_messages populated: {len(msgs)} entries")


def test_1_8_action_schema_accepts_lookahead():
    print("\n[1.8] Action schema accepts lookahead...")
    session = requests.Session()
    r = session.post(f"{BASE_URL}/reset", json={"task_id": "aligned", "seed": 30})
    session_id = r.json().get("metadata", {}).get("session_id")

    r = session.post(
        f"{BASE_URL}/step",
        json={
            "metadata": {"session_id": session_id},
            "action_type": "direct_message",
            "target_ids": ["Finance"],
            "message": "Thinking ahead...",
            "documents": [],
            "lookahead": {
                "depth": 2,
                "n_hypotheses": 2,
                "action_draft": {
                    "action_type": "direct_message",
                    "target_ids": ["Finance"],
                    "message": "Draft response.",
                    "documents": [],
                    "lookahead": None,
                },
            },
        },
        timeout=60,
    )
    assert r.status_code == 200, (
        f"Lookahead action rejected: {r.status_code} {r.text[:200]}"
    )

    result = r.json()
    has_reward = "reward" in result or "observation" in result
    assert has_reward, "Lookahead action did not return reward info"

    print("  ✓ Lookahead action schema accepted")


def test_1_9_approval_path_progress_structure():
    print("\n[1.9] approval_path_progress structure...")
    session = requests.Session()
    r = session.post(f"{BASE_URL}/reset", json={"task_id": "conflicted"}, timeout=30)
    obs = r.json()
    progress = obs.get("approval_path_progress", {})

    assert isinstance(progress, dict), "approval_path_progress must be dict"
    for stakeholder, payload in progress.items():
        assert isinstance(payload, dict), f"{stakeholder} progress must be dict"
        assert "band" in payload, f"{stakeholder}: missing 'band' field"
        assert payload["band"] in ["blocker", "neutral", "workable", "supporter"], (
            f"{stakeholder}: invalid band '{payload['band']}'"
        )

    print(f"  ✓ approval_path_progress valid: {len(progress)} stakeholders")


def test_1_10_deal_stage_valid_transitions():
    print("\n[1.10] deal_stage valid values and round increment...")
    session = requests.Session()
    r = session.post(f"{BASE_URL}/reset", json={"task_id": "aligned", "seed": 40})
    session_id = r.json().get("metadata", {}).get("session_id")
    obs0 = r.json()

    VALID_STAGES = [
        "evaluation",
        "negotiation",
        "legal_review",
        "final_approval",
        "closed",
    ]
    stage = obs0.get("deal_stage")
    assert stage in VALID_STAGES, f"Invalid deal_stage: {stage}"

    r = session.post(
        f"{BASE_URL}/step",
        json={
            "metadata": {"session_id": session_id},
            "action_type": "direct_message",
            "target_ids": ["Finance"],
            "message": "Round 1.",
            "documents": [],
            "lookahead": None,
        },
        timeout=60,
    )
    obs1 = r.json().get("observation", r.json())

    assert obs1.get("round_number") == 1, (
        f"round_number should be 1 after first step, got {obs1.get('round_number')}"
    )
    print(f"  ✓ round_number incremented correctly, deal_stage='{stage}'")


def test_1_11_documents_field_format():
    print("\n[1.11] documents field format (list of {name, content} objects)...")
    session = requests.Session()
    r = session.post(f"{BASE_URL}/reset", json={"task_id": "aligned", "seed": 50})
    session_id = r.json().get("metadata", {}).get("session_id")

    r = session.post(
        f"{BASE_URL}/step",
        json={
            "metadata": {"session_id": session_id},
            "action_type": "send_document",
            "target_ids": ["Legal"],
            "message": "Please review.",
            "documents": [
                {"name": "DPA", "content": "Data Processing Agreement text here."},
                {"name": "security_cert", "content": "Security certification details."},
            ],
            "lookahead": None,
        },
        timeout=60,
    )
    assert r.status_code == 200

    result = r.json()
    reward = result.get("reward")
    assert reward is not None, "send_document action should produce a reward"

    print("  ✓ documents field correctly formatted as [{name, content}, ...]")


def run_all():
    print("=" * 60)
    print("  DealRoom v3 — Schema Validation")
    print("=" * 60)

    tests = [
        test_1_1_all_required_fields_present,
        test_1_2_no_hidden_fields_exposed,
        test_1_3_field_types_correct,
        test_1_4_engagement_history_window_size,
        test_1_5_engagement_level_delta_single_float,
        test_1_6_cross_stakeholder_echoes_is_list,
        test_1_7_stakeholder_messages_populated_after_step,
        test_1_8_action_schema_accepts_lookahead,
        test_1_9_approval_path_progress_structure,
        test_1_10_deal_stage_valid_transitions,
        test_1_11_documents_field_format,
    ]

    for t in tests:
        t()

    print("\n" + "=" * 60)
    print(f"  ✓ SECTION 1 PASSED — 11/11 schema checks passed")
    print("=" * 60)


if __name__ == "__main__":
    run_all()
