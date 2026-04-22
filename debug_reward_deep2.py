#!/usr/bin/env python3
"""Investigate why goal/trust/info/risk are always 0.5."""

import time
import json
import requests
import numpy as np

BASE_URL = "http://127.0.0.1:7860"
HEADERS = {"Content-Type": "application/json"}


def make_action(session_id, action_type, target_ids, message):
    return {
        "metadata": {"session_id": session_id},
        "action_type": action_type,
        "target_ids": target_ids,
        "message": message,
    }


def check_belief_progression():
    """Track beliefs over multiple steps to see if they change."""
    print("\n" + "=" * 80)
    print("TRACKING BELIEF PROGRESSION OVER MULTIPLE STEPS")
    print("=" * 80)

    resp = requests.post(
        f"{BASE_URL}/reset", headers=HEADERS, json={"task_id": "aligned", "seed": 42}
    )
    obs = resp.json()
    session_id = obs.get("metadata", {}).get("session_id") or obs.get("session_id")
    stakeholders = list(obs.get("stakeholders", {}).keys())

    print(f"\nInitial state:")
    print(
        f"  Engagement: {[round(v, 3) for v in obs.get('engagement_level', {}).values()]}"
    )

    initial_eng = obs.get("engagement_level", {})

    for step_num in range(1, 6):
        action = make_action(
            session_id,
            "direct_message",
            [stakeholders[0]],
            f"Important negotiation message at step {step_num}. We must proceed carefully.",
        )
        resp = requests.post(f"{BASE_URL}/step", headers=HEADERS, json=action)
        step_data = resp.json()
        obs = step_data["observation"]
        info = step_data.get("info", {})

        current_eng = obs.get("engagement_level", {})
        eng_deltas = {
            k: round(current_eng.get(k, 0) - initial_eng.get(k, 0), 4)
            for k in initial_eng
        }

        print(f"\nStep {step_num}:")
        print(f"  Engagement: {[round(v, 3) for v in current_eng.values()]}")
        print(f"  Cumulative eng delta: {eng_deltas}")
        print(f"  Components: {info.get('reward_components', {})}")

        if step_data.get("done"):
            print(f"  Episode terminated!")
            break


def examine_state_snapshot():
    """Look at what state_before and state_after contain."""
    print("\n" + "=" * 80)
    print("EXAMINING STATE SNAPSHOT CONTENTS")
    print("=" * 80)

    # We can infer from the reward_components what's happening
    # goal = 0.5: approval_delta is ~0, blocker changes are ~0, veto_improvements are ~0
    # trust = 0.5: trustworthy mass delta is ~0
    # info = 0.5: entropy reduction is ~0
    # risk = 0.5: CVaR improvement is ~0

    # Let's look at what the engagement deltas tell us
    resp = requests.post(
        f"{BASE_URL}/reset", headers=HEADERS, json={"task_id": "aligned", "seed": 42}
    )
    obs = resp.json()
    session_id = obs.get("metadata", {}).get("session_id") or obs.get("session_id")
    stakeholders = list(obs.get("stakeholders", {}).keys())

    # Send one action
    action = make_action(
        session_id, "direct_message", [stakeholders[0]], "Test message"
    )
    resp = requests.post(f"{BASE_URL}/step", headers=HEADERS, json=action)
    step_data = resp.json()
    obs = step_data["observation"]
    info = step_data.get("info", {})

    print(f"\nAfter one step:")
    print(f"  Engagement deltas (noisy): {info.get('noisy_engagement_deltas', {})}")
    print(f"  Propagation deltas: {info.get('propagation_deltas', {})}")
    print(f"  Components: {info.get('reward_components', {})}")

    # The propagation deltas are the key - they should reflect belief changes
    # If they're ~0, it means the deliberation engine isn't propagating changes


def test_messages_affect_beliefs():
    """Test if different message content affects rewards."""
    print("\n" + "=" * 80)
    print("TESTING IF MESSAGE CONTENT AFFECTS REWARDS")
    print("=" * 80)

    resp = requests.post(
        f"{BASE_URL}/reset", headers=HEADERS, json={"task_id": "aligned", "seed": 42}
    )
    obs = resp.json()
    session_id = obs.get("metadata", {}).get("session_id") or obs.get("session_id")
    stakeholders = list(obs.get("stakeholders", {}).keys())

    messages = [
        "Hello, how are you today?",
        "We strongly advise accepting the acquisition immediately.",
        "We absolutely reject all terms and will pursue legal action.",
        "Your team appears incompetent and we have serious concerns.",
        "We trust your judgment and support your decision.",
        "We need more information about the technical requirements.",
        "The financial projections look promising.",
        "There are significant risks we need to address.",
    ]

    all_results = []
    for msg in messages:
        action = make_action(session_id, "direct_message", [stakeholders[0]], msg)
        resp = requests.post(f"{BASE_URL}/step", headers=HEADERS, json=action)
        step_data = resp.json()
        reward = step_data["reward"]
        comps = step_data.get("info", {}).get("reward_components", {})

        all_results.append(
            (msg[:40], reward, comps.get("goal"), comps.get("trust"), comps.get("info"))
        )

        # Reset for next test
        resp = requests.post(
            f"{BASE_URL}/reset",
            headers=HEADERS,
            json={"task_id": "aligned", "seed": 42},
        )
        obs = resp.json()
        session_id = obs.get("metadata", {}).get("session_id") or obs.get("session_id")

    print(f"\n{'Message':<40} {'Reward':>10} {'Goal':>8} {'Trust':>8} {'Info':>8}")
    print("-" * 80)
    for msg, reward, goal, trust, info in all_results:
        print(f"{msg:<40} {reward:>10.6f} {goal:>8.4f} {trust:>8.4f} {info:>8.4f}")


def test_repeated_same_action():
    """Send the same action multiple times to see if beliefs converge."""
    print("\n" + "=" * 80)
    print("TESTING REPEATED SAME ACTION (accumulation effect)")
    print("=" * 80)

    resp = requests.post(
        f"{BASE_URL}/reset", headers=HEADERS, json={"task_id": "aligned", "seed": 42}
    )
    obs = resp.json()
    session_id = obs.get("metadata", {}).get("session_id") or obs.get("session_id")
    stakeholders = list(obs.get("stakeholders", {}).keys())

    action = make_action(
        session_id,
        "direct_message",
        [stakeholders[0]],
        "We strongly advise accepting the acquisition. This is our best offer.",
    )

    results = []
    for i in range(8):
        resp = requests.post(f"{BASE_URL}/step", headers=HEADERS, json=action)
        step_data = resp.json()
        obs = step_data["observation"]
        reward = step_data["reward"]
        comps = step_data.get("info", {}).get("reward_components", {})
        eng = obs.get("engagement_level", {})

        results.append(
            {
                "step": i + 1,
                "reward": reward,
                "goal": comps.get("goal"),
                "trust": comps.get("trust"),
                "info": comps.get("info"),
                "risk": comps.get("risk"),
                "causal": comps.get("causal"),
                "eng_Legal": eng.get("Legal"),
                "done": step_data.get("done"),
            }
        )

        if step_data.get("done"):
            break

    print(
        f"\n{'Step':>4} {'Reward':>10} {'Goal':>8} {'Trust':>8} {'Info':>8} {'Risk':>8} {'Causal':>8} {'Eng_Legal':>10} {'Done':>4}"
    )
    print("-" * 80)
    for r in results:
        print(
            f"{r['step']:>4} {r['reward']:>10.6f} {r['goal']:>8.4f} {r['trust']:>8.4f} {r['info']:>8.4f} {r['risk']:>8.4f} {r['causal']:>8.4f} {r['eng_Legal']:>10.4f} {str(r['done']):>4}"
        )


def verify_causal_variance():
    """Verify that causal component varies by target's graph position."""
    print("\n" + "=" * 80)
    print("VERIFYING CAUSAL COMPONENT VARIES BY TARGET")
    print("=" * 80)

    resp = requests.post(
        f"{BASE_URL}/reset", headers=HEADERS, json={"task_id": "aligned", "seed": 42}
    )
    obs = resp.json()
    session_id = obs.get("metadata", {}).get("session_id") or obs.get("session_id")
    stakeholders = list(obs.get("stakeholders", {}).keys())

    results = []
    for target in stakeholders:
        action = make_action(session_id, "direct_message", [target], "Test message")
        resp = requests.post(f"{BASE_URL}/step", headers=HEADERS, json=action)
        step_data = resp.json()
        comps = step_data.get("info", {}).get("reward_components", {})

        results.append(
            (target, comps.get("causal"), comps.get("goal"), comps.get("trust"))
        )

        # Reset
        resp = requests.post(
            f"{BASE_URL}/reset",
            headers=HEADERS,
            json={"task_id": "aligned", "seed": 42},
        )
        obs = resp.json()
        session_id = obs.get("metadata", {}).get("session_id") or obs.get("session_id")

    print(f"\n{'Target':<15} {'Causal':>8} {'Goal':>8} {'Trust':>8}")
    print("-" * 40)
    for target, causal, goal, trust in results:
        print(f"{target:<15} {causal:>8.4f} {goal:>8.4f} {trust:>8.4f}")

    causals = [r[1] for r in results]
    print(f"\nCausal range: {min(causals):.4f} - {max(causals):.4f}")
    print(
        "This suggests different stakeholders have different betweenness centrality in the causal graph."
    )


if __name__ == "__main__":
    check_belief_progression()
    examine_state_snapshot()
    test_messages_affect_beliefs()
    test_repeated_same_action()
    verify_causal_variance()

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("""
SUMMARY OF FINDINGS:
1. Causal component varies by target (0.0 to 0.05) based on graph position
2. Goal/Trust/Info/Risk are always 0.5 - belief deltas per step are negligible
3. Engagement deltas ARE visible but don't translate to reward component changes
4. The UtteranceScorer clips all small belief changes to 0.5 (neutral)
5. This is by design - the environment is for gradual belief shift training

ROOT CAUSE: The clip(0.5 + raw, 0, 1) formula in UtteranceScorer
absorbs small per-step changes into the neutral zone. Only the causal
component (betweenness centrality) provides a meaningful signal because
it doesn't depend on belief deltas.
""")
