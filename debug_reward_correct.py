#!/usr/bin/env python3
"""Debug reward computation - CORRECT action format."""

import time
import json
import requests
import numpy as np

BASE_URL = "http://127.0.0.1:7860"
HEADERS = {"Content-Type": "application/json"}


# Correct action format based on DealRoomAction model
def make_action(session_id, action_type, target_ids, message):
    return {
        "metadata": {"session_id": session_id},
        "action_type": action_type,
        "target_ids": target_ids,
        "message": message,
    }


def run_episode_correct(task, max_steps=10):
    print(f"\n{'=' * 80}")
    print(f"CORRECT FORMAT EPISODE: task={task}")
    print(f"{'=' * 80}")

    resp = requests.post(
        f"{BASE_URL}/reset", headers=HEADERS, json={"task_id": task, "seed": 42}
    )
    obs = resp.json()
    session_id = obs.get("metadata", {}).get("session_id") or obs.get("session_id")
    stakeholders = list(obs.get("stakeholders", {}).keys())

    print(f"\n[RESET] Session: {session_id}")
    print(f"Initial stage: {obs['deal_stage']}")
    print(f"Stakeholders: {stakeholders}")

    results = []

    for step_num in range(1, max_steps + 1):
        # Cycle through different action types and messages
        action_types = [
            "direct_message",
            "group_proposal",
            "backchannel",
            "send_document",
            "concession",
        ]
        action_type = action_types[step_num % len(action_types)]

        # Different stakeholders and messages
        target = stakeholders[step_num % len(stakeholders)]
        messages = [
            f"We should proceed with the acquisition at step {step_num}.",
            f"What are the key concerns from {target}?",
            f"The terms appear favorable for both parties.",
            f"Please review the attached proposal document.",
            f"We are prepared to make concessions on pricing.",
            f"Let's schedule a follow-up discussion.",
        ]
        message = messages[step_num % len(messages)]

        action = make_action(session_id, action_type, [target], message)

        resp = requests.post(f"{BASE_URL}/step", headers=HEADERS, json=action)
        step_data = resp.json()
        obs = step_data["observation"]
        reward = step_data["reward"]
        info = step_data.get("info", {})
        reward_components = info.get("reward_components", {})

        results.append(
            {
                "step": step_num,
                "action_type": action_type,
                "target": target,
                "reward": reward,
                "components": reward_components,
                "stage": obs.get("deal_stage"),
                "done": step_data.get("done"),
                "blockers": obs.get("active_blockers", []),
            }
        )

        print(f"\n[STEP {step_num}] action={action_type} target={target}")
        print(f"  Stage: {obs.get('deal_stage')} | Done: {step_data.get('done')}")
        print(f"  Reward: {reward:.6f}")
        print(
            f"  Components: { {k: round(v, 4) for k, v in reward_components.items()} }"
        )
        print(f"  Blockers: {obs.get('active_blockers', [])}")

        if step_data.get("done"):
            print(f"\n  >> EPISODE TERMINATED")
            break

    return results


def test_action_types():
    """Test different action types with CORRECT format."""
    print("\n" + "=" * 80)
    print("TESTING DIFFERENT ACTION TYPES (CORRECT FORMAT)")
    print("=" * 80)

    resp = requests.post(
        f"{BASE_URL}/reset", headers=HEADERS, json={"task_id": "aligned", "seed": 42}
    )
    obs = resp.json()
    session_id = obs.get("metadata", {}).get("session_id") or obs.get("session_id")
    stakeholders = list(obs.get("stakeholders", {}).keys())

    action_types = [
        ("direct_message", "We should proceed with the acquisition."),
        ("group_proposal", "We propose moving forward together as a group."),
        ("backchannel", "Let's have an off-the-record discussion."),
        ("send_document", "Please review the attached proposal."),
        ("concession", "We can offer better terms and pricing."),
        ("walkaway_signal", "We may need to reconsider this partnership."),
        ("reframe_value_prop", "Let me reposition the value proposition."),
        ("exec_escalation", "Let's bring in executive leadership."),
    ]

    results = []
    for action_type, message in action_types:
        action = make_action(session_id, action_type, [stakeholders[0]], message)
        resp = requests.post(f"{BASE_URL}/step", headers=HEADERS, json=action)
        step_data = resp.json()
        reward = step_data["reward"]
        info = step_data.get("info", {})
        comps = info.get("reward_components", {})

        results.append((action_type, reward, comps))
        print(f"\n{action_type}:")
        print(f"  Reward: {reward:.6f}")
        print(f"  Components: { {k: round(v, 4) for k, v in comps.items()} }")

    # Check variance
    rewards = [r[1] for r in results]
    print(
        f"\nReward range: {min(rewards):.6f} - {max(rewards):.6f} (diff: {max(rewards) - min(rewards):.6f})"
    )


def test_targets():
    """Test different target stakeholders."""
    print("\n" + "=" * 80)
    print("TESTING DIFFERENT TARGETS (CORRECT FORMAT)")
    print("=" * 80)

    resp = requests.post(
        f"{BASE_URL}/reset", headers=HEADERS, json={"task_id": "aligned", "seed": 42}
    )
    obs = resp.json()
    session_id = obs.get("metadata", {}).get("session_id") or obs.get("session_id")
    stakeholders = list(obs.get("stakeholders", {}).keys())

    for target in stakeholders:
        action = make_action(
            session_id,
            "direct_message",
            [target],
            "We should proceed with the acquisition.",
        )
        resp = requests.post(f"{BASE_URL}/step", headers=HEADERS, json=action)
        step_data = resp.json()
        reward = step_data["reward"]
        comps = step_data.get("info", {}).get("reward_components", {})

        print(f"\n{target}:")
        print(f"  Reward: {reward:.6f}")
        print(f"  Components: { {k: round(v, 4) for k, v in comps.items()} }")


def test_scenarios():
    """Test different scenarios."""
    print("\n" + "=" * 80)
    print("TESTING DIFFERENT SCENARIOS (CORRECT FORMAT)")
    print("=" * 80)

    for task in ["aligned", "conflicted", "hostile_acquisition"]:
        resp = requests.post(
            f"{BASE_URL}/reset", headers=HEADERS, json={"task_id": task, "seed": 42}
        )
        obs = resp.json()
        session_id = obs.get("metadata", {}).get("session_id") or obs.get("session_id")
        stakeholders = list(obs.get("stakeholders", {}).keys())

        action = make_action(
            session_id,
            "direct_message",
            [stakeholders[0]],
            "We should proceed with the acquisition.",
        )
        resp = requests.post(f"{BASE_URL}/step", headers=HEADERS, json=action)
        step_data = resp.json()
        reward = step_data["reward"]
        comps = step_data.get("info", {}).get("reward_components", {})
        blockers = step_data["observation"].get("active_blockers", [])

        print(f"\n{task}:")
        print(f"  Reward: {reward:.6f}")
        print(f"  Components: { {k: round(v, 4) for k, v in comps.items()} }")
        print(f"  Blockers: {blockers}")


if __name__ == "__main__":
    test_action_types()
    test_targets()
    test_scenarios()

    print("\n" + "=" * 80)
    print("FULL EPISODES")
    print("=" * 80)

    run_episode_correct("aligned", max_steps=10)
    run_episode_correct("hostile_acquisition", max_steps=10)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
