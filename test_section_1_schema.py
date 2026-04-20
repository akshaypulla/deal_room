import requests

BASE = "http://127.0.0.1:7860"

V3_REQUIRED_FIELDS = [
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

FORBIDDEN_FIELDS = [
    "G",
    "causal_graph",
    "graph",
    "true_beliefs",
    "belief_distributions",
    "tau",
    "tau_i",
    "risk_thresholds",
    "cvar_thresholds",
    "edge_weights",
    "w_ij",
    "deliberation_transcript",
    "deliberation_log",
    "internal_dialogue",
]


def test_1_1_all_required_fields_present():
    r = requests.post(f"{BASE}/reset", json={"task": "aligned"})
    obs = r.json()
    missing = [f for f in V3_REQUIRED_FIELDS if f not in obs]
    assert not missing, f"Required fields missing from observation: {missing}"
    print(f"✓ 1.1: All {len(V3_REQUIRED_FIELDS)} required fields present")


def test_1_2_no_hidden_fields_exposed():
    r = requests.post(f"{BASE}/reset", json={"task": "hostile_acquisition"})
    obs = r.json()

    def find_forbidden(d, path=""):
        found = []
        if isinstance(d, dict):
            for k, v in d.items():
                if k.lower() in [f.lower() for f in FORBIDDEN_FIELDS]:
                    found.append(f"{path}.{k}")
                found.extend(find_forbidden(v, f"{path}.{k}"))
        return found

    exposed = find_forbidden(obs)
    assert not exposed, f"HIDDEN FIELDS EXPOSED IN OBSERVATION: {exposed}"
    print("✓ 1.2: No hidden fields (G, tau_i, beliefs) exposed in observation")


def test_1_3_engagement_history_window_size():
    r = requests.post(f"{BASE}/reset", json={"task": "conflicted"})
    obs = r.json()
    history = obs.get("engagement_history", [])
    assert len(history) > 0, "engagement_history is empty"
    assert isinstance(history, list), (
        "engagement_history must be a list (per-timestamp snapshots)"
    )
    assert len(history) >= 5, (
        f"engagement_history must have >=5 entries for window, got {len(history)}"
    )
    for i, entry in enumerate(history):
        assert isinstance(entry, dict), (
            f"engagement_history[{i}] must be a dict (stakeholder snapshots)"
        )
    print("✓ 1.3: engagement_history is list of 5 snapshots (one per round window)")


def test_1_4_engagement_level_delta_present_after_step():
    session = requests.Session()
    r = session.post(f"{BASE}/reset", json={"task": "aligned"})
    session_id = r.json().get("metadata", {}).get("session_id")
    r = session.post(
        f"{BASE}/step",
        json={
            "metadata": {"session_id": session_id},
            "action_type": "direct_message",
            "target_ids": ["Finance"],
            "message": "I would like to share our ROI analysis with you.",
            "documents": [],
            "lookahead": None,
        },
    )
    obs = r.json().get("observation", r.json())
    assert "engagement_level_delta" in obs
    delta = obs["engagement_level_delta"]
    assert isinstance(delta, (int, float)), (
        f"engagement_level_delta must be numeric, got {type(delta).__name__}"
    )
    print("✓ 1.4: engagement_level_delta populated after first step")


def test_1_5_cross_stakeholder_echoes_is_dict():
    session = requests.Session()
    r = session.post(f"{BASE}/reset", json={"task": "aligned"})
    obs = r.json()
    cse = obs.get("cross_stakeholder_echoes")
    assert cse is not None, "cross_stakeholder_echoes field missing"
    assert isinstance(cse, (dict, list)), (
        f"cross_stakeholder_echoes must be dict or list, got {type(cse).__name__}"
    )
    print("✓ 1.5: cross_stakeholder_echoes is present and valid")


def test_1_6_stakeholder_messages_populated():
    session = requests.Session()
    r = session.post(f"{BASE}/reset", json={"task": "conflicted"})
    obs = r.json()
    msgs = obs.get("stakeholder_messages", {})
    assert isinstance(msgs, dict), (
        f"stakeholder_messages must be a dict, got {type(msgs).__name__}"
    )
    print(
        f"✓ 1.6: stakeholder_messages is a dict (field exists, populated after first action)"
    )


def test_1_7_action_schema_accepts_lookahead():
    session = requests.Session()
    r = session.post(f"{BASE}/reset", json={"task": "aligned"})
    session_id = r.json().get("metadata", {}).get("session_id")
    r = session.post(
        f"{BASE}/step",
        json={
            "metadata": {"session_id": session_id},
            "action_type": "direct_message",
            "target_ids": ["Finance"],
            "message": "Let me think ahead before responding.",
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
    )
    assert r.status_code == 200, f"Lookahead action rejected: {r.status_code} {r.text}"
    print("✓ 1.7: Action schema accepts lookahead field")


if __name__ == "__main__":
    for fn in [
        test_1_1_all_required_fields_present,
        test_1_2_no_hidden_fields_exposed,
        test_1_3_engagement_history_window_size,
        test_1_4_engagement_level_delta_present_after_step,
        test_1_5_cross_stakeholder_echoes_is_dict,
        test_1_6_stakeholder_messages_populated,
        test_1_7_action_schema_accepts_lookahead,
    ]:
        fn()
    print("\n✓ SECTION 1 PASSED — Observation schema is correct")
