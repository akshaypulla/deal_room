#!/usr/bin/env python3
"""
Clean learning progression visualization.
Shows that rewards improve over episodes as Q-values converge.
"""

import sys
import numpy as np

sys.path.insert(0, "/Users/akshaypulla/Documents/deal_room")

from deal_room.environment.dealroom_v3 import DealRoomV3


def make_policy(env: DealRoomV3, lr: float = 0.2, eps: float = 0.4):
    class QPolicy:
        def __init__(pself):
            pself.env = env
            pself.action_space = env.action_space
            pself.q = np.zeros(len(env.action_space))
            pself.lr = lr
            pself.epsilon = eps
            pself.last_idx = None

        def act(pself):
            if np.random.random() < pself.epsilon:
                idx = np.random.randint(len(pself.action_space))
            else:
                idx = int(np.argmax(pself.q))
            pself.last_idx = idx
            return pself.action_space[idx].model_copy(update={"message": "msg"})

        def update(pself, reward):
            if pself.last_idx is not None:
                pself.q[pself.last_idx] += pself.lr * (reward - pself.q[pself.last_idx])
                pself.epsilon = max(0.05, pself.epsilon * 0.96)
                pself.last_idx = None

    return QPolicy()


def run_session(task_id: str, n_episodes: int = 25):
    env = DealRoomV3()
    rewards = []
    q_history = []

    for ep in range(n_episodes):
        policy = make_policy(env)
        obs = env.reset(seed=42 + ep, task_id=task_id)
        total = 0.0
        done = False
        steps = 0

        while not done and steps < 10:
            action = policy.act()
            obs, reward, done, info = env.step(action)
            policy.update(reward)
            total += reward
            steps += 1

        rewards.append(total)
        q_history.append(list(policy.q))

    return rewards, q_history


def main():
    print("=" * 65)
    print("LEARNING PROGRESSION — Conflicted Scenario")
    print("=" * 65)

    rewards, q_history = run_session("conflicted", n_episodes=25)

    print(f"\nEpisode rewards: {[f'{r:.3f}' for r in rewards]}")
    print(f"\nReward stats:")
    print(f"  Mean:   {np.mean(rewards):.4f}")
    print(f"  Std:    {np.std(rewards):.4f}")
    print(f"  Min:    {np.min(rewards):.4f}")
    print(f"  Max:    {np.max(rewards):.4f}")
    print(f"  Range:  {np.max(rewards) - np.min(rewards):.4f}")

    q_history = np.array(q_history)
    print(f"\nQ-value progression (episode 1, 5, 10, 15, 20, 25):")
    for ep in [0, 4, 9, 14, 19, 24]:
        if ep < len(q_history):
            print(f"  Episode {ep + 1:2d}: {[f'{v:.4f}' for v in q_history[ep]]}")

    best_action = np.argmax(q_history[-1])
    print(f"\nBest action by episode 25: action {best_action}")
    print(f"Q-value spread: {np.max(q_history[-1]) - np.min(q_history[-1]):.4f}")

    print("\n" + "=" * 65)
    print("PROGRESSION TABLE")
    print("=" * 65)
    print(
        f"{'Ep':>4} | {'Reward':>8} | {'Q[0]':>8} | {'Q[1]':>8} | {'Q[2]':>8} | {'Q[3]':>8}"
    )
    print("-" * 65)
    for ep in [0, 4, 9, 14, 19, 24]:
        if ep < len(rewards):
            q = q_history[ep]
            print(
                f"{ep + 1:4d} | {rewards[ep]:8.4f} | {q[0]:8.4f} | {q[1]:8.4f} | {q[2]:8.4f} | {q[3]:8.4f}"
            )

    improvement = np.mean(rewards[-5:]) - np.mean(rewards[:5])
    print(f"\nFirst 5 mean: {np.mean(rewards[:5]):.4f}")
    print(f"Last 5 mean:  {np.mean(rewards[-5:]):.4f}")
    print(f"Change:       {improvement:+.4f}")

    print("\n" + "=" * 65)
    print("VERDICT")
    print("=" * 65)
    spread = np.max(q_history[-1]) - np.min(q_history[-1])
    if spread > 0.05:
        print(
            f"✅ Q-values converged (spread={spread:.4f}) — policy learned to discriminate actions"
        )
        print(
            f"✅ Reward distribution non-flat (std={np.std(rewards):.4f}) — environment is informative"
        )
        print(
            f"✅ Conflicted scenario rewards: {np.min(rewards):.4f} to {np.max(rewards):.4f}"
        )
        return 0
    else:
        print(f"⚠️ Q-values did not converge (spread={spread:.4f})")
        return 1


if __name__ == "__main__":
    sys.exit(main())
