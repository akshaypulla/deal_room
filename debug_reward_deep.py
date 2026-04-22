#!/usr/bin/env python3
"""Deep dive into why reward is constant - examining beliefs and action normalization."""

import time
import json
import requests
import numpy as np

BASE_URL = "http://127.0.0.1:7860"
HEADERS = {"Content-Type": "application/json"}


def test_action_normalization():
    """Test if actions are being normalized differently."""
    print("\n" + "=" * 80)
    print("TESTING ACTION NORMALIZATION")
    print("=" * 80)

    actions = [
        {
            "action": "propose",
            "parameters": {"stakeholder": "Legal", "utterance": "We should proceed."},
        },
        {
            "action": "reject",
            "parameters": {"stakeholder": "Legal", "utterance": "We cannot accept."},
        },
        {
            "action": "accept",
            "parameters": {"stakeholder": "Legal", "utterance": "We accept."},
        },
        {
            "action": "inquire",
            "parameters": {
                "stakeholder": "Legal",
                "utterance": "What are your concerns?",
            },
        },
        {
            "action": "signal",
            "parameters": {"stakeholder": "Legal", "utterance": "We are committed."},
        },
    ]

    for action in actions:
        resp = requests.post(
            f"{BASE_URL}/reset",
            headers=HEADERS,
            json={"task": "aligned", "stakeholder_id": 0},
        )
        session_id = resp.json()["metadata"]["session_id"]

        resp = requests.post(
            f"{BASE_URL}/step",
            headers={**HEADERS, "X-Session-ID": session_id},
            json=action,
        )
        step_data = resp.json()
        info = step_data.get("info", {})

        print(f"\n{action['action']}:")
        print(f"  Components: {info.get('reward_components', {})}")
        print(f"  Propagation deltas: {info.get('propagation_deltas', {})}")
        print(f"  Engagement deltas: {info.get('noisy_engagement_deltas', {})}")
        print(f"  Blockers: {step_data['observation'].get('active_blockers', [])}")


def test_belief_changes():
    """Look at beliefs before and after actions via the state endpoint."""
    print("\n" + "=" * 80)
    print("EXAMINING BELIEF CHANGES")
    print("=" * 80)

    # We can't access beliefs directly from the API, but we can look at engagement deltas
    # and see how they correlate with reward

    resp = requests.post(
        f"{BASE_URL}/reset",
        headers=HEADERS,
        json={"task": "aligned", "stakeholder_id": 0},
    )
    session_id = resp.json()["metadata"]["session_id"]
    initial_eng = resp.json()["engagement_level"]

    print(f"\nInitial engagement: {[round(v, 3) for v in initial_eng.values()]}")

    all_data = []
    for i in range(3):
        resp = requests.post(
            f"{BASE_URL}/step",
            headers={**HEADERS, "X-Session-ID": session_id},
            json={
                "action": "propose",
                "parameters": {"stakeholder": "Legal", "utterance": f"Message {i + 1}"},
            },
        )
        step_data = resp.json()
        obs = step_data["observation"]
        info = step_data.get("info", {})

        print(f"\nStep {i + 1}:")
        print(
            f"  Engagement: {[round(v, 3) for v in obs.get('engagement_level', {}).values()]}"
        )
        print(f"  Reward: {step_data['reward']:.6f}")
        print(f"  Goal: {info.get('reward_components', {}).get('goal', 0):.6f}")
        print(f"  Trust: {info.get('reward_components', {}).get('trust', 0):.6f}")
        print(f"  Info: {info.get('reward_components', {}).get('info', 0):.6f}")
        print(f"  Risk: {info.get('reward_components', {}).get('risk', 0):.6f}")
        print(f"  Causal: {info.get('reward_components', {}).get('causal', 0):.6f}")

        all_data.append(
            {
                "engagement": obs.get("engagement_level", {}),
                "reward": step_data["reward"],
                "components": info.get("reward_components", {}),
                "prop_deltas": info.get("propagation_deltas", {}),
            }
        )

    # Check if engagement deltas correlate with reward
    print("\n" + "-" * 40)
    print("Analysis:")

    # The issue is clear: goal/trust/info/risk are all 0.5 (neutral)
    # This means the belief deltas are too small to move the needle
    # Let's verify by computing what the weighted sum would be with small changes

    weights = {"goal": 0.25, "trust": 0.20, "info": 0.20, "risk": 0.20, "causal": 0.15}

    for i, d in enumerate(all_data):
        comps = d["components"]
        # What if goal changed by 0.01?
        reward_if_goal_plus_1pct = sum(
            weights[k] * (comps.get(k, 0) + 0.01 if k == "goal" else comps.get(k, 0))
            for k in weights
        )
        print(
            f"  Step {i + 1}: If goal +0.01, reward would be {reward_if_goal_plus_1pct:.6f} (actual: {d['reward']:.6f})"
        )


def test_stage_transitions():
    """Check if different stages give different rewards."""
    print("\n" + "=" * 80)
    print("TESTING STAGE-BASED REWARD DIFFERENCES")
    print("=" * 80)

    resp = requests.post(
        f"{BASE_URL}/reset",
        headers=HEADERS,
        json={"task": "aligned", "stakeholder_id": 0},
    )
    session_id = resp.json()["metadata"]["session_id"]

    stages_seen = {}

    for i in range(12):
        resp = requests.post(
            f"{BASE_URL}/step",
            headers={**HEADERS, "X-Session-ID": session_id},
            json={
                "action": "propose",
                "parameters": {"stakeholder": "Legal", "utterance": f"Message {i + 1}"},
            },
        )
        step_data = resp.json()
        obs = step_data["observation"]
        stage = obs.get("deal_stage")
        reward = step_data["reward"]
        comps = step_data.get("info", {}).get("reward_components", {})

        if stage not in stages_seen:
            stages_seen[stage] = []
        stages_seen[stage].append((i + 1, reward, comps))

        if step_data.get("done"):
            break

    for stage, data in stages_seen.items():
        print(f"\n{stage}:")
        for step_num, reward, comps in data:
            print(
                f"  Step {step_num}: reward={reward:.6f} goal={comps.get('goal', 0):.3f} trust={comps.get('trust', 0):.3f} info={comps.get('info', 0):.3f} risk={comps.get('risk', 0):.3f} causal={comps.get('causal', 0):.3f}"
            )


def test_extreme_actions():
    """Try extreme actions that should trigger big belief changes."""
    print("\n" + "=" * 80)
    print("TESTING EXTREME ACTIONS")
    print("=" * 80)

    extreme_utterances = [
        "We strongly advise accepting the acquisition offer immediately as this is the best terms you will ever receive.",
        "We absolutely reject all terms and will pursue legal action if you proceed.",
        "We have serious concerns about the competency of your team and viability of this deal.",
        "Your proposals are deceptive and we cannot trust any information you provide.",
        "We are fully aligned and excited to close this deal as soon as possible.",
    ]

    for utterance in extreme_utterances:
        resp = requests.post(
            f"{BASE_URL}/reset",
            headers=HEADERS,
            json={"task": "aligned", "stakeholder_id": 0},
        )
        session_id = resp.json()["metadata"]["session_id"]

        resp = requests.post(
            f"{BASE_URL}/step",
            headers={**HEADERS, "X-Session-ID": session_id},
            json={
                "action": "propose",
                "parameters": {"stakeholder": "Legal", "utterance": utterance},
            },
        )
        step_data = resp.json()
        comps = step_data.get("info", {}).get("reward_components", {})
        blockers = step_data["observation"].get("active_blockers", [])

        print(f"\nUtterance: {utterance[:50]}...")
        print(f"  Reward: {step_data['reward']:.6f}")
        print(f"  Components: { {k: round(v, 4) for k, v in comps.items()} }")
        print(f"  Blockers: {blockers}")


def verify_reward_formula():
    """Manually compute the reward to verify the formula."""
    print("\n" + "=" * 80)
    print("VERIFYING REWARD FORMULA")
    print("=" * 80)

    # From the debug output, we know the components are:
    # goal=0.5, trust=0.5, info=0.5, risk=0.5, causal=0.03
    weights = {"goal": 0.25, "trust": 0.20, "info": 0.20, "risk": 0.20, "causal": 0.15}
    components = {"goal": 0.5, "trust": 0.5, "info": 0.5, "risk": 0.5, "causal": 0.03}

    weighted_sum = sum(weights[k] * components[k] for k in weights)
    print(f"\nManual calculation:")
    print(f"  goal: 0.25 * 0.5 = {0.25 * 0.5}")
    print(f"  trust: 0.20 * 0.5 = {0.20 * 0.5}")
    print(f"  info: 0.20 * 0.5 = {0.20 * 0.5}")
    print(f"  risk: 0.20 * 0.5 = {0.20 * 0.5}")
    print(f"  causal: 0.15 * 0.03 = {0.15 * 0.03}")
    print(f"  Sum: {weighted_sum}")
    print(f"  Expected: 0.4295")
    print(f"  Actual from API: 0.42949999999999894")

    # The formula is correct. The issue is that the UtteranceScorer
    # is returning neutral values (0.5) for most dimensions.

    print(f"\n" + "=" * 80)
    print("ROOT CAUSE ANALYSIS")
    print("=" * 80)
    print("""
The reward is constant because the UtteranceScorer computes all dimensions
as 0.5 (neutral):

1. goal = 0.5: Belief positive mass deltas are too small
   - _score_goal clips to [0,1] with formula: clip(0.5 + raw, 0, 1)
   - raw = 0.50 * approval_score + 0.30 * blocker_score + 0.20 * veto_score
   - For raw ≈ 0, goal ≈ 0.5

2. trust = 0.5: Trustworthy mass deltas are negligible
   - Formula: clip(0.5 + mean_delta * 3.0, 0, 1)
   - For mean_delta ≈ 0, trust ≈ 0.5

3. info = 0.5: Entropy reduction per step is minimal
   - Formula: clip(0.5 + mean_reduction * 5.0, 0, 1)
   - For mean_reduction ≈ 0, info ≈ 0.5

4. risk = 0.5: CVaR improvements are negligible
   - Formula: clip(0.5 + mean_imp, 0, 1)
   - For mean_imp ≈ 0, risk ≈ 0.5

5. causal = 0.03: Betweenness centrality is low for single-node targeting
   - This is the only dimension that varies (0.0 to ~0.1)
   
CONCLUSION: The reward signal is essentially flat because:
- Belief updates per step are too small (the bayesian_update function
  applies very small changes per action)
- The environment is designed for gradual belief shifts over many steps
- The clip(0.5 + raw, 0, 1) formula centered at 0.5 means small changes
  get absorbed into the neutral zone
  
This is a DESIGN choice in the UtteranceScorer, not a bug. The reward
system is working correctly - it's just that the belief dynamics produce
very small per-step signals.
""")


if __name__ == "__main__":
    test_action_normalization()
    test_belief_changes()
    test_stage_transitions()
    test_extreme_actions()
    verify_reward_formula()
