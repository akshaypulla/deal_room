#!/usr/bin/env python3
"""Final verification with correct action types."""

import requests
import numpy as np

BASE_URL = "http://127.0.0.1:7860"


def make_action(session_id, action_type, target_ids, message):
    return {
        "metadata": {"session_id": session_id},
        "action_type": action_type,
        "target_ids": target_ids,
        "message": message,
    }


def run_full_episode_test():
    print("=" * 80)
    print("FINAL VERIFICATION WITH CORRECT ACTION TYPES")
    print("=" * 80)

    all_rewards = []
    all_components = []

    # Test aligned scenario with varied actions
    resp = requests.post(
        f"{BASE_URL}/reset",
        headers={"Content-Type": "application/json"},
        json={"task_id": "aligned", "seed": 42},
    )
    obs = resp.json()
    session_id = obs.get("metadata", {}).get("session_id") or obs.get("session_id")
    stakeholders = list(obs.get("stakeholders", {}).keys())

    action_types = [
        "direct_message",
        "group_proposal",
        "backchannel",
        "send_document",
        "concession",
    ]

    print(f"\nAligned episode (correct action types):")
    for i in range(10):
        action_type = action_types[i % len(action_types)]
        target = stakeholders[i % len(stakeholders)]
        action = make_action(session_id, action_type, [target], f"Test message {i}")
        resp = requests.post(
            f"{BASE_URL}/step",
            headers={"Content-Type": "application/json"},
            json=action,
        )
        step_data = resp.json()
        reward = step_data["reward"]
        comps = step_data.get("info", {}).get("reward_components", {})
        done = step_data.get("done")

        all_rewards.append(reward)
        all_components.append(comps)

        print(
            f"  Step {i + 1}: reward={reward:.4f} goal={comps.get('goal', 0):.4f} trust={comps.get('trust', 0):.4f} info={comps.get('info', 0):.4f} done={done}"
        )

        if done:
            break

    print(f"\nHostile acquisition episode:")
    resp = requests.post(
        f"{BASE_URL}/reset",
        headers={"Content-Type": "application/json"},
        json={"task_id": "hostile_acquisition", "seed": 999},
    )
    session_id = resp.json().get("metadata", {}).get("session_id") or resp.json().get(
        "session_id"
    )
    stakeholders = list(resp.json().get("stakeholders", {}).keys())

    for i in range(5):
        action = make_action(
            session_id, "direct_message", [stakeholders[0]], "We reject all terms"
        )
        resp = requests.post(
            f"{BASE_URL}/step",
            headers={"Content-Type": "application/json"},
            json=action,
        )
        step_data = resp.json()
        reward = step_data["reward"]
        done = step_data.get("done")
        print(f"  Step {i + 1}: reward={reward:.4f} done={done}")
        if done:
            print(
                f"  Terminal: {step_data.get('info', {}).get('terminal_outcome', 'N/A')}"
            )
            break

    # Summary stats
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total rewards: {len(all_rewards)}")
    print(f"Reward std: {np.std(all_rewards):.4f}")
    print(f"Reward range: {min(all_rewards):.4f} - {max(all_rewards):.4f}")

    goals = [c.get("goal", 0.5) for c in all_components]
    trusts = [c.get("trust", 0.5) for c in all_components]
    infos = [c.get("info", 0.5) for c in all_components]
    print(f"Goal std: {np.std(goals):.4f}")
    print(f"Trust std: {np.std(trusts):.4f}")
    print(f"Info std: {np.std(infos):.4f}")


if __name__ == "__main__":
    run_full_episode_test()
