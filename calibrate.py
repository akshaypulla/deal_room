"""
Calibration script — run before submission.
Target: heuristic policy should beat random on the canonical v3 runtime.
"""

from __future__ import annotations

import numpy as np

from deal_room.environment.dealroom_v3 import DealRoomV3
from deal_room.training.grpo_trainer import HeuristicPolicyAdapter, RandomPolicyAdapter
from models import DealRoomAction


class RandomAgent:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.adapter = RandomPolicyAdapter()

    def act(self, obs) -> DealRoomAction:
        return self.adapter.act(obs, self.rng)


class StrategicAgent:
    """Benchmark policy aligned with the trainer's heuristic baseline."""

    def __init__(self):
        self.adapter = HeuristicPolicyAdapter()
        self.rng = np.random.default_rng(0)

    def act(self, obs) -> DealRoomAction:
        return self.adapter.act(obs, self.rng)


def run_episodes(task_id: str, agent_class, n: int = 50) -> list[float]:
    scores = []
    for i in range(n):
        rng = np.random.default_rng(i)
        agent = agent_class(rng) if agent_class == RandomAgent else agent_class()
        env = DealRoomV3()
        obs = env.reset(seed=i, task_id=task_id)
        total_reward = 0.0
        for _ in range(20):
            action = agent.act(obs)
            obs, reward, done, _ = env.step(action)
            total_reward += float(reward)
            if done:
                break
        scores.append(total_reward)
    return scores


if __name__ == "__main__":
    tasks = ["aligned", "conflicted", "hostile_acquisition"]
    print("DealRoom v3 Calibration (50 episodes per agent per task)\n")
    all_pass = True
    for task in tasks:
        rand_scores = run_episodes(task, RandomAgent, n=50)
        strat_scores = run_episodes(task, StrategicAgent, n=50)
        rand_avg = sum(rand_scores) / len(rand_scores)
        strat_avg = sum(strat_scores) / len(strat_scores)
        spread = strat_avg - rand_avg
        status = "PASS" if spread >= 0.15 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"{task}:")
        print(f"  Random agent:    {rand_avg:.3f}")
        print(f"  Strategic agent: {strat_avg:.3f}")
        print(f"  Spread:          {spread:.3f}  [{status}]")
        print()

    if all_pass:
        print("All calibration targets met. Ready to submit.")
    else:
        print("Calibration below target. Inspect benchmark artifacts and veto frequency.")
