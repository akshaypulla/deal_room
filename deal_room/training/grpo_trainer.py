"""
GRPO trainer for DealRoom v3 - Group Relative Policy Optimization for multi-dimensional rewards.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class TrainingMetrics:
    goal_reward: float = 0.0
    trust_reward: float = 0.0
    info_reward: float = 0.0
    risk_reward: float = 0.0
    causal_reward: float = 0.0
    lookahead_usage_rate: float = 0.0
    prediction_accuracy: float = 0.0
    total_reward: float = 0.0


@dataclass
class EpisodeTrajectory:
    observations: List
    actions: List
    rewards: List[List[float]]  # 5D rewards per step
    lookahead_used: List[bool]
    prediction_accuracies: List[float]


class GRPOTrainer:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-3B-Instruct",
        learning_rate: float = 1e-5,
        grpo_clip: float = 0.2,
        entropy_coef: float = 0.01,
        reward_weights: List[float] = None,
    ):
        self.model_id = model_id
        self.learning_rate = learning_rate
        self.grpo_clip = grpo_clip
        self.entropy_coef = entropy_coef
        self.reward_weights = reward_weights or [0.25, 0.20, 0.20, 0.20, 0.15]
        self.rng = np.random.default_rng(42)

    def compute_group_relative_advantage(
        self, episode_rewards: List[List[float]], group_rewards: List[List[float]]
    ) -> List[float]:
        if not group_rewards:
            return [0.0] * len(episode_rewards)

        aggregated = []
        for rewards in episode_rewards:
            weighted = sum(r * w for r, w in zip(rewards, self.reward_weights))
            aggregated.append(weighted)

        group_aggregated = []
        for rewards in group_rewards:
            weighted = sum(r * w for r, w in zip(rewards, self.reward_weights))
            group_aggregated.append(weighted)

        mean = np.mean(group_aggregated)
        std = np.std(group_aggregated) + 1e-8

        advantages = [(a - mean) / std for a in aggregated]
        return advantages

    def run_self_play_episode(
        self,
        env,  # DealRoomV3
        policy_fn=None,
        max_steps: int = 10,
    ) -> EpisodeTrajectory:
        obs = env.reset(seed=int(self.rng.integers(0, 2**31)), task_id="aligned")

        observations = []
        actions = []
        rewards = []
        lookahead_used = []
        prediction_accuracies = []

        for step in range(max_steps):
            observations.append(obs)

            if policy_fn:
                action = policy_fn(obs)
            else:
                action = self._default_policy(obs)

            actions.append(action)
            lookahead_used.append(action.lookahead is not None)

            obs, reward, done, info = env.step(action)

            if isinstance(reward, list):
                rewards.append(reward)
            else:
                rewards.append([reward] * 5)

            if info and "prediction_accuracy" in info:
                prediction_accuracies.append(info["prediction_accuracy"])
            else:
                prediction_accuracies.append(0.0)

            if done:
                break

        return EpisodeTrajectory(
            observations=observations,
            actions=actions,
            rewards=rewards,
            lookahead_used=lookahead_used,
            prediction_accuracies=prediction_accuracies,
        )

    def _default_policy(self, obs) -> "DealRoomAction":
        from models import DealRoomAction

        stakeholders = list(obs.stakeholders.keys())
        if not stakeholders:
            stakeholders = [
                "Legal",
                "Finance",
                "TechLead",
                "Procurement",
                "Operations",
                "ExecSponsor",
            ]

        target = stakeholders[self.rng.integers(0, len(stakeholders))]
        messages = [
            "I understand your concerns and will address them in our proposal.",
            "Let me provide more details on how we can meet your requirements.",
            "Can you help me understand your specific constraints here?",
            "We are committed to finding a solution that works for all parties.",
        ]
        message = messages[self.rng.integers(0, len(messages))]

        return DealRoomAction(
            action_type="direct_message",
            target=target,
            target_ids=[target],
            message=message,
        )

    def compute_training_metrics(
        self, trajectories: List[EpisodeTrajectory]
    ) -> TrainingMetrics:
        if not trajectories:
            return TrainingMetrics()

        all_goal = []
        all_trust = []
        all_info = []
        all_risk = []
        all_causal = []
        total_steps = 0
        lookahead_count = 0
        prediction_acc_sum = 0.0

        for traj in trajectories:
            for step_rewards in traj.rewards:
                if len(step_rewards) >= 5:
                    all_goal.append(step_rewards[0])
                    all_trust.append(step_rewards[1])
                    all_info.append(step_rewards[2])
                    all_risk.append(step_rewards[3])
                    all_causal.append(step_rewards[4])
                total_steps += 1

            lookahead_count += sum(traj.lookahead_used)
            prediction_acc_sum += sum(traj.prediction_accuracies)

        return TrainingMetrics(
            goal_reward=np.mean(all_goal) if all_goal else 0.0,
            trust_reward=np.mean(all_trust) if all_trust else 0.0,
            info_reward=np.mean(all_info) if all_info else 0.0,
            risk_reward=np.mean(all_risk) if all_risk else 0.0,
            causal_reward=np.mean(all_causal) if all_causal else 0.0,
            lookahead_usage_rate=lookahead_count / max(total_steps, 1),
            prediction_accuracy=prediction_acc_sum / max(total_steps, 1),
            total_reward=sum(
                np.mean([r[0] for r in rewards]) for rewards in [[t.rewards]]
            )
            / len(trajectories)
            if trajectories
            else 0.0,
        )

    def run_training_loop(
        self,
        env,  # DealRoomV3
        n_episodes: int = 50,
        episodes_per_batch: int = 4,
        max_steps: int = 10,
        verbose: bool = True,
    ) -> List[TrainingMetrics]:
        all_metrics = []

        for episode in range(n_episodes):
            batch_trajectories = []
            for _ in range(episodes_per_batch):
                traj = self.run_self_play_episode(env, max_steps=max_steps)
                batch_trajectories.append(traj)

            metrics = self.compute_training_metrics(batch_trajectories)
            all_metrics.append(metrics)

            if verbose and (episode + 1) % 10 == 0:
                print(
                    f"Episode {episode + 1}/{n_episodes} | "
                    f"Goal: {metrics.goal_reward:.3f} | "
                    f"Trust: {metrics.trust_reward:.3f} | "
                    f"Info: {metrics.info_reward:.3f} | "
                    f"Risk: {metrics.risk_reward:.3f} | "
                    f"Causal: {metrics.causal_reward:.3f} | "
                    f"Lookahead: {metrics.lookahead_usage_rate:.2%}"
                )

        return all_metrics
