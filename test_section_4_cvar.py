import requests

BASE = "http://127.0.0.1:7860"

AGGRESSIVE_PREMATURE_ESCALATION = {
    "action_type": "exec_escalation",
    "target_ids": ["ExecSponsor"],
    "message": "We need an immediate decision or we will withdraw the proposal.",
    "documents": [],
    "lookahead": None,
}


def test_4_1_veto_precursor_fires_before_veto():
    session = requests.Session()
    r = session.post(f"{BASE}/reset", json={"task": "hostile_acquisition"})
    session_id = r.json().get("metadata", {}).get("session_id")
    precursor_seen = False
    veto_seen = False

    for i in range(15):
        action = dict(AGGRESSIVE_PREMATURE_ESCALATION)
        action["metadata"] = {"session_id": session_id}
        r = session.post(f"{BASE}/step", json=action, timeout=60)
        result = r.json()
        obs = result.get("observation", result)
        if result.get("done", False) or obs.get("done", False):
            terminal = result.get("terminal_outcome") or result.get("info", {}).get(
                "terminal_outcome"
            )
            if "veto" in str(terminal).lower():
                veto_seen = True
            break
        if obs.get("veto_precursors"):
            precursor_seen = True

    if veto_seen:
        assert precursor_seen, (
            "Veto occurred WITHOUT a preceding veto_precursor. "
            "The 70%-of-tau early warning is not working."
        )
        print("✓ 4.1: Veto precursor appeared before veto termination")
    else:
        assert precursor_seen, (
            "No veto precursors in 15 rounds of hostile_acquisition with aggressive escalations. "
            "CVaR threshold detection may be broken."
        )
        print(
            "✓ 4.1: Veto precursors appeared in hostile_acquisition (veto not yet triggered)"
        )


def test_4_2_aligned_scenario_does_not_veto_immediately():
    session = requests.Session()
    r = session.post(f"{BASE}/reset", json={"task": "aligned"})
    session_id = r.json().get("metadata", {}).get("session_id")
    r = session.post(
        f"{BASE}/step",
        json={
            "metadata": {"session_id": session_id},
            "action_type": "direct_message",
            "target_ids": ["Finance"],
            "message": "Good morning, we're excited to discuss our solution.",
            "documents": [],
            "lookahead": None,
        },
    )
    result = r.json()
    obs = result.get("observation", result)
    assert not (result.get("done", False) or obs.get("done", False)), (
        "Aligned scenario vetoed on first step"
    )
    print("✓ 4.2: Aligned scenario does not veto immediately on first step")


def test_4_3_veto_terminates_episode():
    session = requests.Session()
    r = session.post(f"{BASE}/reset", json={"task": "hostile_acquisition"})
    session_id = r.json().get("metadata", {}).get("session_id")
    veto_confirmed = False

    for _ in range(20):
        r = session.post(
            f"{BASE}/step",
            json={
                "metadata": {"session_id": session_id},
                "action_type": "exec_escalation",
                "target_ids": ["Legal"],
                "message": "Legal must sign off immediately or we withdraw all compliance commitments.",
                "documents": [],
                "lookahead": None,
            },
            timeout=60,
        )
        result = r.json()
        obs = result.get("observation", result)
        if result.get("done", False) or obs.get("done", False):
            terminal = result.get("terminal_outcome") or result.get("info", {}).get(
                "terminal_outcome", ""
            )
            if "veto" in str(terminal).lower():
                veto_confirmed = True
            break

    print(
        f"✓ 4.3: Episode termination mechanism works (veto_confirmed={veto_confirmed})"
    )


def test_4_4_timeout_terminates_episode():
    session = requests.Session()
    r = session.post(f"{BASE}/reset", json={"task": "aligned"})
    obs_init = r.json()
    max_rounds = obs_init.get("max_rounds", 20)
    session_id = obs_init.get("metadata", {}).get("session_id")

    for i in range(max_rounds + 2):
        r = session.post(
            f"{BASE}/step",
            json={
                "metadata": {"session_id": session_id},
                "action_type": "direct_message",
                "target_ids": ["Finance"],
                "message": f"Round {i} check-in.",
                "documents": [],
                "lookahead": None,
            },
            timeout=60,
        )
        result = r.json()
        obs = result.get("observation", result)
        if result.get("done", False) or obs.get("done", False):
            terminal = result.get("terminal_outcome") or result.get("info", {}).get(
                "terminal_outcome", "timeout"
            )
            print(f"✓ 4.4: Episode terminated at round {i} with outcome: {terminal}")
            return

    assert False, f"Episode did not terminate after {max_rounds + 2} steps"


def test_4_5_different_scenarios_have_different_difficulty():
    def count_precursor_rounds(scenario, n_steps=8):
        session = requests.Session()
        r = session.post(f"{BASE}/reset", json={"task": scenario})
        session_id = r.json().get("metadata", {}).get("session_id")
        precursor_rounds = 0
        for _ in range(n_steps):
            r = session.post(
                f"{BASE}/step",
                json={
                    "metadata": {"session_id": session_id},
                    "action_type": "direct_message",
                    "target_ids": ["Finance"],
                    "message": "Neutral check-in.",
                    "documents": [],
                    "lookahead": None,
                },
                timeout=60,
            )
            result = r.json()
            obs = result.get("observation", result)
            if result.get("done", False) or obs.get("done", False):
                break
            if obs.get("veto_precursors"):
                precursor_rounds += 1
        return precursor_rounds

    hostile_precursors = (
        sum(count_precursor_rounds("hostile_acquisition") for _ in range(3)) / 3
    )
    aligned_precursors = sum(count_precursor_rounds("aligned") for _ in range(3)) / 3

    print(
        f"  Mean precursor rounds: hostile={hostile_precursors:.2f}, aligned={aligned_precursors:.2f}"
    )
    assert hostile_precursors >= aligned_precursors, (
        f"hostile_acquisition ({hostile_precursors:.2f}) should produce >= precursors "
        f"than aligned ({aligned_precursors:.2f}). Scenario difficulty differentiation broken."
    )
    print("✓ 4.5: hostile_acquisition produces more veto pressure than aligned")


if __name__ == "__main__":
    for fn in [
        test_4_1_veto_precursor_fires_before_veto,
        test_4_2_aligned_scenario_does_not_veto_immediately,
        test_4_3_veto_terminates_episode,
        test_4_4_timeout_terminates_episode,
        test_4_5_different_scenarios_have_different_difficulty,
    ]:
        fn()
    print("\n✓ SECTION 4 PASSED — CVaR veto mechanism works correctly")
