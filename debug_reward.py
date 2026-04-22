#!/usr/bin/env python3
"""Debug reward computation across episodes."""

import time
import json
import requests

BASE_URL = "http://127.0.0.1:7860"
HEADERS = {"Content-Type": "application/json"}


def run_debug_episode(task, num_steps=5):
    print(f"\n{'=' * 80}")
    print(f"DEBUG EPISODE: task={task}")
    print(f"{'=' * 80}")

    # Reset
    resp = requests.post(
        f"{BASE_URL}/reset", headers=HEADERS, json={"task": task, "stakeholder_id": 0}
    )
    obs = resp.json()
    session_id = obs["metadata"]["session_id"]
    print(f"\n[RESET] Session: {session_id}")
    print(f"Initial stage: {obs['deal_stage']}")
    print(
        f"Initial engagement: {[round(v, 3) for v in obs['engagement_level'].values()]}"
    )
    print(f"Initial blockers: {obs.get('active_blockers', [])}")

    for step_num in range(1, num_steps + 1):
        action_type = "propose" if step_num % 2 == 1 else "inquire"

        resp = requests.post(
            f"{BASE_URL}/step",
            headers={**HEADERS, "X-Session-ID": session_id},
            json={
                "action": action_type,
                "parameters": {
                    "stakeholder": "Legal",
                    "utterance": f"Test message {step_num}",
                },
            },
        )
        step_data = resp.json()
        obs = step_data["observation"]
        reward = step_data["reward"]
        info = step_data.get("info", {})
        reward_components = info.get("reward_components", {})

        print(f"\n[STEP {step_num}] action={action_type}")
        print(
            f"  Round: {obs.get('round_number')} | Stage: {obs.get('deal_stage')} | Done: {step_data.get('done')}"
        )
        print(f"  Reward (total): {reward}")
        print(f"  Reward components:")
        for k, v in reward_components.items():
            print(f"    {k}: {v:.6f}")

        # Compute weighted sum manually
        weights = {
            "goal": 0.25,
            "trust": 0.20,
            "info": 0.20,
            "risk": 0.20,
            "causal": 0.15,
        }
        manual_sum = sum(weights[k] * reward_components.get(k, 0) for k in weights)
        print(f"  Manual weighted sum: {manual_sum:.6f}")
        print(f"  Difference: {abs(reward - manual_sum):.10f}")

        print(f"  Blockers: {obs.get('active_blockers', [])}")
        print(
            f"  Engagement: {[round(v, 3) for v in obs.get('engagement_level', {}).values()]}"
        )
        print(f"  Momentum: {obs.get('deal_momentum')}")

        if step_data.get("done"):
            print(f"\n  >> EPISODE TERMINATED")
            print(f"  Terminal reward: {info.get('terminal_reward', 0)}")
            print(f"  Terminal outcome: {info.get('terminal_outcome', '')}")
            break


def test_different_scenarios():
    print("\n" + "=" * 80)
    print("TESTING DIFFERENT SCENARIOS FOR REWARD VARIANCE")
    print("=" * 80)

    scenarios = [
        ("aligned", "Legal", "propose", "We should proceed with the acquisition."),
        ("aligned", "Legal", "reject", "We cannot accept these terms."),
        ("aligned", "Legal", "accept", "We accept the proposed terms."),
        ("aligned", "Finance", "propose", "We should proceed with the acquisition."),
        ("aligned", "TechLead", "propose", "We should proceed with the acquisition."),
        (
            "hostile_acquisition",
            "Legal",
            "propose",
            "We should proceed with the acquisition.",
        ),
        ("conflicted", "Legal", "propose", "We should proceed with the acquisition."),
    ]

    results = []

    for task, stakeholder, action, utterance in scenarios:
        resp = requests.post(
            f"{BASE_URL}/reset",
            headers=HEADERS,
            json={"task": task, "stakeholder_id": 0},
        )
        session_id = resp.json()["metadata"]["session_id"]

        resp = requests.post(
            f"{BASE_URL}/step",
            headers={**HEADERS, "X-Session-ID": session_id},
            json={
                "action": action,
                "parameters": {"stakeholder": stakeholder, "utterance": utterance},
            },
        )
        step_data = resp.json()
        obs = step_data["observation"]
        reward = step_data["reward"]
        info = step_data.get("info", {})
        reward_components = info.get("reward_components", {})

        results.append(
            {
                "task": task,
                "stakeholder": stakeholder,
                "action": action,
                "reward": reward,
                "components": reward_components,
            }
        )

        print(f"\n{task} | {stakeholder} | {action}")
        print(f"  Reward: {reward}")
        print(f"  Components: {reward_components}")

    print(f"\n{'=' * 80}")
    print("REWARD VARIANCE ANALYSIS")
    print(f"{'=' * 80}")

    all_rewards = [r["reward"] for r in results]
    print(f"Rewards: {[round(r, 6) for r in all_rewards]}")
    print(
        f"Min: {min(all_rewards):.6f}  Max: {max(all_rewards):.6f}  Range: {max(all_rewards) - min(all_rewards):.6f}"
    )

    # Check component variance
    for dim in ["goal", "trust", "info", "risk", "causal"]:
        vals = [r["components"].get(dim, 0) for r in results]
        print(
            f"  {dim}: min={min(vals):.4f} max={max(vals):.4f} range={max(vals) - min(vals):.4f}"
        )


def test_belief_deltas():
    print("\n" + "=" * 80)
    print("TESTING BELIEF DELTAS OVER EPISODE")
    print("=" * 80)

    resp = requests.post(
        f"{BASE_URL}/reset",
        headers=HEADERS,
        json={"task": "aligned", "stakeholder_id": 0},
    )
    session_id = resp.json()["metadata"]["session_id"]

    beliefs_sequence = []

    for step_num in range(1, 6):
        resp = requests.post(
            f"{BASE_URL}/step",
            headers={**HEADERS, "X-Session-ID": session_id},
            json={
                "action": "propose",
                "parameters": {
                    "stakeholder": "Legal",
                    "utterance": f"Message {step_num}",
                },
            },
        )
        step_data = resp.json()
        obs = step_data["observation"]

        # The beliefs are not exposed directly in observation, but engagement levels are
        print(f"\nStep {step_num}:")
        print(f"  Stage: {obs.get('deal_stage')}")
        print(
            f"  Engagement: {[round(v, 3) for v in obs.get('engagement_level', {}).values()]}"
        )
        print(f"  Reward: {step_data['reward']}")
        print(f"  Components: {step_data.get('info', {}).get('reward_components', {})}")

        if step_data.get("done"):
            break


if __name__ == "__main__":
    test_different_scenarios()
    test_belief_deltas()
    run_debug_episode("aligned", num_steps=5)
    run_debug_episode("hostile_acquisition", num_steps=5)
