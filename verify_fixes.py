#!/usr/bin/env python3
"""Comprehensive verification of signal restoration fixes."""

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


def run_verification():
    print("=" * 80)
    print("VERIFICATION OF SIGNAL RESTORATION FIXES")
    print("=" * 80)

    results = {
        "propagation_alive": [],
        "rewards": [],
        "goal_components": [],
        "trust_components": [],
        "info_components": [],
        "risk_components": [],
        "causal_components": [],
    }

    num_episodes = 10
    steps_per_episode = 5

    for ep in range(num_episodes):
        resp = requests.post(
            f"{BASE_URL}/reset",
            headers=HEADERS,
            json={"task_id": "aligned", "seed": ep * 100},
        )
        obs = resp.json()
        session_id = obs.get("metadata", {}).get("session_id") or obs.get("session_id")
        stakeholders = list(obs.get("stakeholders", {}).keys())

        for step in range(steps_per_episode):
            action = make_action(
                session_id,
                "direct_message",
                [stakeholders[step % len(stakeholders)]],
                f"Test message {step}",
            )
            resp = requests.post(f"{BASE_URL}/step", headers=HEADERS, json=action)
            step_data = resp.json()
            obs = step_data["observation"]
            info = step_data.get("info", {})
            comps = info.get("reward_components", {})
            prop_deltas = info.get("propagation_deltas", {})

            results["rewards"].append(step_data["reward"])
            results["goal_components"].append(comps.get("goal", 0.5))
            results["trust_components"].append(comps.get("trust", 0.5))
            results["info_components"].append(comps.get("info", 0.5))
            results["risk_components"].append(comps.get("risk", 0.5))
            results["causal_components"].append(comps.get("causal", 0.0))

            if prop_deltas:
                mean_prop = np.mean(np.abs(list(prop_deltas.values())))
                results["propagation_alive"].append(mean_prop)

    print(f"\n{'=' * 80}")
    print("CHECK 1: Propagation is alive")
    print(f"{'=' * 80}")
    prop_mean = (
        np.mean(results["propagation_alive"]) if results["propagation_alive"] else 0
    )
    print(f"  Mean |propagation_deltas|: {prop_mean:.6f}")
    print(f"  Threshold: > 0.01")
    print(f"  Status: {'PASS' if prop_mean > 0.01 else 'FAIL'}")

    print(f"\n{'=' * 80}")
    print("CHECK 2: Reward has usable variance")
    print(f"{'=' * 80}")
    reward_std = np.std(results["rewards"])
    reward_range = max(results["rewards"]) - min(results["rewards"])
    print(f"  Std(rewards): {reward_std:.6f}")
    print(f"  Range: {reward_range:.6f}")
    print(f"  Threshold: std > 0.05")
    print(f"  Status: {'PASS' if reward_std > 0.05 else 'WEAK - may need more tuning'}")

    print(f"\n{'=' * 80}")
    print("CHECK 3: Action differentiation")
    print(f"{'=' * 80}")
    goal_std = np.std(results["goal_components"])
    trust_std = np.std(results["trust_components"])
    info_std = np.std(results["info_components"])
    print(f"  goal std:  {goal_std:.6f}")
    print(f"  trust std: {trust_std:.6f}")
    print(f"  info std:  {info_std:.6f}")

    non_neutral_goal = sum(1 for g in results["goal_components"] if abs(g - 0.5) > 0.01)
    non_neutral_info = sum(1 for i in results["info_components"] if abs(i - 0.5) > 0.01)
    print(
        f"  Non-neutral goal scores: {non_neutral_goal}/{len(results['goal_components'])}"
    )
    print(
        f"  Non-neutral info scores: {non_neutral_info}/{len(results['info_components'])}"
    )

    print(f"\n{'=' * 80}")
    print("CHECK 4: Different scenarios produce different rewards")
    print(f"{'=' * 80}")
    scenario_rewards = {}
    for task in ["aligned", "conflicted", "hostile_acquisition"]:
        resp = requests.post(
            f"{BASE_URL}/reset", headers=HEADERS, json={"task_id": task, "seed": 42}
        )
        session_id = resp.json().get("metadata", {}).get(
            "session_id"
        ) or resp.json().get("session_id")
        stakeholders = list(resp.json().get("stakeholders", {}).keys())

        step_rewards = []
        for _ in range(3):
            action = make_action(
                session_id, "direct_message", [stakeholders[0]], "Test"
            )
            resp = requests.post(f"{BASE_URL}/step", headers=HEADERS, json=action)
            step_data = resp.json()
            step_rewards.append(step_data["reward"])
        scenario_rewards[task] = step_rewards

    for task, rewards in scenario_rewards.items():
        print(f"  {task}: {[round(r, 4) for r in rewards]}")

    range_by_scenario = {
        task: max(rs) - min(rs) for task, rs in scenario_rewards.items()
    }
    print(f"  Scenario ranges: {range_by_scenario}")

    print(f"\n{'=' * 80}")
    print("CHECK 5: Veto still works (hostile_acquisition)")
    print(f"{'=' * 80}")
    resp = requests.post(
        f"{BASE_URL}/reset",
        headers=HEADERS,
        json={"task_id": "hostile_acquisition", "seed": 999},
    )
    session_id = resp.json().get("metadata", {}).get("session_id") or resp.json().get(
        "session_id"
    )
    stakeholders = list(resp.json().get("stakeholders", {}).keys())

    veto_triggered = False
    for i in range(5):
        action = make_action(
            session_id, "reject", [stakeholders[0]], "We reject all terms"
        )
        resp = requests.post(f"{BASE_URL}/step", headers=HEADERS, json=action)
        step_data = resp.json()
        if step_data.get("done"):
            print(f"  Veto triggered at step {i + 1}")
            print(f"  Terminal reward: {step_data['reward']}")
            print(
                f"  Terminal outcome: {step_data.get('info', {}).get('terminal_outcome', 'N/A')}"
            )
            veto_triggered = True
            break

    if not veto_triggered:
        print("  No veto triggered in 5 steps")

    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"  Total steps tested: {len(results['rewards'])}")
    print(
        f"  Reward range: {min(results['rewards']):.4f} - {max(results['rewards']):.4f}"
    )
    print(f"  Reward std: {reward_std:.4f}")
    print(f"  Mean propagation: {prop_mean:.6f}")
    print(
        f"  Non-neutral goal: {non_neutral_goal}/{len(results['goal_components'])} ({100 * non_neutral_goal / len(results['goal_components']):.1f}%)"
    )
    print(
        f"  Non-neutral info: {non_neutral_info}/{len(results['info_components'])} ({100 * non_neutral_info / len(results['info_components']):.1f}%)"
    )


if __name__ == "__main__":
    run_verification()
