#!/usr/bin/env python3
"""
Learnability test: Random Policy vs Heuristic Policy.

Proves the environment produces meaningfully different outcomes
for different policy types, confirming it's trainable.
"""

import sys
import time
import numpy as np

sys.path.insert(0, "/Users/akshaypulla/Documents/deal_room")

from deal_room.environment.dealroom_v3 import DealRoomV3


class RandomPolicy:
    """Random action selection from the environment's action space."""

    def __init__(self, env: DealRoomV3):
        self.env = env
        self.action_space = env.action_space

    def act(self, observation) -> dict:
        import random

        action_template = random.choice(self.action_space)
        return action_template.model_copy(
            update={"message": f"Random message {random.randint(1, 100)}"}
        )


class HeuristicPolicy:
    """Heuristic policy that sends targeted documents to build alignment."""

    def __init__(self, env: DealRoomV3):
        self.env = env
        self.step = 0

    def act(self, observation) -> dict:
        self.step += 1
        target = ["Finance", "Legal", "TechLead"][self.step % 3]

        if self.step % 3 == 1:
            return self.env.action_space[0].model_copy(
                update={
                    "target_ids": [target],
                    "message": f"Addressing {target}'s concerns about the deal terms.",
                }
            )
        elif self.step % 3 == 2:
            doc_idx = 1 if target == "Legal" else (2 if target == "Finance" else 3)
            return self.env.action_space[doc_idx].model_copy(
                update={
                    "target_ids": [target],
                }
            )
        else:
            return self.env.action_space[0].model_copy(
                update={
                    "target_ids": [target],
                    "message": f"Follow-up to {target}: here are additional details.",
                }
            )


def run_episode(
    env: DealRoomV3,
    policy,
    task_id: str = "aligned",
    seed: int = None,
    max_steps: int = 10,
):
    """Run one episode and return cumulative reward."""
    obs = env.reset(seed=seed, task_id=task_id)
    total_reward = 0.0
    rewards = []
    done = False
    step = 0

    while not done and step < max_steps:
        action = policy.act(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        rewards.append(reward)
        step += 1

    terminal = info.get("terminal_reward", 0.0)
    terminal_outcome = info.get("terminal_outcome", "")

    return {
        "total_reward": total_reward,
        "mean_reward": np.mean(rewards) if rewards else 0.0,
        "std_reward": np.std(rewards) if rewards else 0.0,
        "steps": step,
        "terminal_reward": terminal,
        "terminal_outcome": terminal_outcome,
        "rewards": rewards,
    }


def run_trials(
    policy_class, env: DealRoomV3, n_trials: int = 5, task_id: str = "aligned"
):
    """Run multiple trials and collect statistics."""
    results = []
    for i in range(n_trials):
        policy = policy_class(env)
        result = run_episode(env, policy, task_id=task_id, seed=42 + i)
        results.append(result)
        print(
            f"  Trial {i + 1}/{n_trials}: reward={result['total_reward']:.4f}, "
            f"terminal={result['terminal_outcome'][:30] if result['terminal_outcome'] else 'none'}"
        )
    return results


def summarize(results):
    """Compute summary statistics."""
    total_rewards = [r["total_reward"] for r in results]
    mean_rewards = [r["mean_reward"] for r in results]
    std_rewards = [r["std_reward"] for r in results]

    print(
        f"\n  Total reward:    mean={np.mean(total_rewards):.4f}, std={np.std(total_rewards):.4f}"
    )
    print(
        f"  Mean step reward: mean={np.mean(mean_rewards):.4f}, std={np.std(mean_rewards):.4f}"
    )
    print(f"  Per-step std:    mean={np.mean(std_rewards):.4f}")
    return {
        "total_mean": np.mean(total_rewards),
        "total_std": np.std(total_rewards),
        "step_mean_mean": np.mean(mean_rewards),
        "step_mean_std": np.std(mean_rewards),
        "step_std_mean": np.mean(std_rewards),
    }


def main():
    print("=" * 60)
    print("LEARNABILITY TEST: Random vs Heuristic Policy")
    print("=" * 60)

    env = DealRoomV3()

    # Test 1: Aligned scenario
    print("\n### Aligned Scenario ###")
    print("\nRandom Policy (5 trials):")
    random_results_aligned = run_trials(
        RandomPolicy, env, n_trials=5, task_id="aligned"
    )
    random_stats_aligned = summarize(random_results_aligned)

    print("\nHeuristic Policy (5 trials):")
    heuristic_results_aligned = run_trials(
        HeuristicPolicy, env, n_trials=5, task_id="aligned"
    )
    heuristic_stats_aligned = summarize(heuristic_results_aligned)

    # Test 2: Conflicted scenario
    print("\n### Conflicted Scenario ###")
    print("\nRandom Policy (5 trials):")
    random_results_conflicted = run_trials(
        RandomPolicy, env, n_trials=5, task_id="conflicted"
    )
    random_stats_conflicted = summarize(random_results_conflicted)

    print("\nHeuristic Policy (5 trials):")
    heuristic_results_conflicted = run_trials(
        HeuristicPolicy, env, n_trials=5, task_id="conflicted"
    )
    heuristic_stats_conflicted = summarize(heuristic_results_conflicted)

    # Final verdict
    print("\n" + "=" * 60)
    print("LEARNABILITY VERDICT")
    print("=" * 60)

    improvement_aligned = (
        heuristic_stats_aligned["total_mean"] - random_stats_aligned["total_mean"]
    )
    improvement_conflicted = (
        heuristic_stats_conflicted["total_mean"] - random_stats_conflicted["total_mean"]
    )

    print(f"\nAligned scenario:")
    print(f"  Heuristic improvement over random: {improvement_aligned:+.4f}")
    print(
        f"  Separation (effect size): {abs(improvement_aligned) / max(random_stats_aligned['total_std'], 0.01):.2f} sigma"
    )

    print(f"\nConflicted scenario:")
    print(f"  Heuristic improvement over random: {improvement_conflicted:+.4f}")
    print(
        f"  Separation (effect size): {abs(improvement_conflicted) / max(random_stats_conflicted['total_std'], 0.01):.2f} sigma"
    )

    if improvement_aligned > 0.1 or improvement_conflicted > 0.1:
        print("\n✅ ENVIRONMENT IS LEARNABLE - Heuristic policy outperforms random")
        return 0
    elif improvement_aligned > 0 or improvement_conflicted > 0:
        print("\n⚠️ WEAK SIGNAL - Heuristic slightly better but separation is small")
        return 1
    else:
        print("\n❌ ENVIRONMENT MAY NOT BE LEARNABLE - No clear improvement signal")
        return 2


if __name__ == "__main__":
    sys.exit(main())
