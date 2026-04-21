"""GRPO-style training harness for DealRoom v3."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence

import numpy as np

from deal_room.curriculum.adaptive_generator import AdaptiveCurriculumGenerator
from deal_room.environment.constants import REWARD_WEIGHTS
from deal_room.environment.dealroom_v3 import DealRoomV3
from models import DealRoomAction, DealRoomObservation, LookaheadRequest


REWARD_DIMENSIONS = ("goal", "trust", "info", "risk", "causal")


def weighted_reward(reward_vector: Sequence[float]) -> float:
    return float(
        sum(
            REWARD_WEIGHTS[dimension] * value
            for dimension, value in zip(REWARD_DIMENSIONS, reward_vector)
        )
    )


class PolicyAdapter(Protocol):
    name: str

    def act(
        self, observation: DealRoomObservation, rng: np.random.Generator
    ) -> DealRoomAction: ...

    def update_from_batch(self, trajectories: Sequence["EpisodeTrajectory"]) -> None: ...

    def state_dict(self) -> Dict[str, Any]: ...

    def load_state_dict(self, state: Dict[str, Any]) -> None: ...


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
    weighted_reward: float = 0.0
    episodes_completed: int = 0
    task_mix: Dict[str, float] = field(default_factory=dict)
    terminal_outcomes: Dict[str, int] = field(default_factory=dict)
    checkpoint_path: Optional[str] = None


@dataclass
class EpisodeTrajectory:
    task_id: str
    terminal_outcome: str = ""
    observations: List[DealRoomObservation] = field(default_factory=list)
    actions: List[DealRoomAction] = field(default_factory=list)
    rewards: List[List[float]] = field(default_factory=list)
    scalar_rewards: List[float] = field(default_factory=list)
    lookahead_used: List[bool] = field(default_factory=list)
    prediction_accuracies: List[float] = field(default_factory=list)
    lookahead_diagnostics: List[Dict[str, Any]] = field(default_factory=list)
    weighted_reward: float = 0.0
    seed: Optional[int] = None


class RandomPolicyAdapter:
    name = "random"

    def __init__(self, use_lookahead_probability: float = 0.15):
        self.use_lookahead_probability = use_lookahead_probability

    def act(
        self, observation: DealRoomObservation, rng: np.random.Generator
    ) -> DealRoomAction:
        stakeholders = list(observation.stakeholders.keys()) or [
            "Legal",
            "Finance",
            "TechLead",
            "Procurement",
            "Operations",
            "ExecSponsor",
        ]
        target = stakeholders[int(rng.integers(0, len(stakeholders)))]
        action_type = str(
            rng.choice(
                ["direct_message", "send_document", "backchannel", "concession"]
            )
        )
        action = DealRoomAction(
            action_type=action_type,
            target=target,
            target_ids=[target],
            message="I want to keep this moving in a constructive direction.",
            documents=[],
        )
        if action_type == "send_document":
            doc_type = str(rng.choice(["dpa", "security_cert", "roi_model"]))
            action.documents = [{"type": doc_type, "name": doc_type.upper()}]
        if rng.random() < self.use_lookahead_probability:
            action.lookahead = LookaheadRequest(
                action_draft=action.model_copy(deep=True),
                n_hypotheses=2,
                depth=2,
            )
        return action

    def update_from_batch(self, trajectories: Sequence[EpisodeTrajectory]) -> None:
        return None

    def state_dict(self) -> Dict[str, Any]:
        return {"use_lookahead_probability": self.use_lookahead_probability}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.use_lookahead_probability = float(
            state.get("use_lookahead_probability", self.use_lookahead_probability)
        )


class HeuristicPolicyAdapter:
    name = "heuristic"

    def __init__(self):
        self.enable_lookahead = False
        self.prefer_concessions = False
        self.trained = False

    def act(
        self, observation: DealRoomObservation, rng: np.random.Generator
    ) -> DealRoomAction:
        if observation.veto_precursors:
            target = next(iter(observation.veto_precursors))
            if target.lower() == "legal":
                action = DealRoomAction(
                    action_type="send_document",
                    target=target,
                    target_ids=[target],
                    message="Here is the DPA with stronger liability safeguards and review-ready clauses.",
                    documents=[{"type": "dpa", "name": "DPA"}],
                    proposed_terms={"liability_cap": 1500000},
                )
            elif target.lower() == "finance":
                action = DealRoomAction(
                    action_type="concession",
                    target=target,
                    target_ids=[target],
                    message="I can tighten the commercial risk posture immediately so this is safer to approve.",
                    proposed_terms={"liability_cap": 1500000},
                )
            else:
                action = DealRoomAction(
                    action_type="backchannel",
                    target=target,
                    target_ids=[target],
                    channel="backchannel",
                    message="I want to address your tail-risk concerns directly and concretely.",
                )
        else:
            if self.trained:
                action = self._trained_round_action(observation)
            else:
                target = self._pick_target(observation)
                action = self._build_target_action(target)

        if self.enable_lookahead and observation.round_number >= 2:
            action.lookahead = LookaheadRequest(
                action_draft=action.model_copy(deep=True),
                n_hypotheses=2,
                depth=2,
            )
        return action

    def update_from_batch(self, trajectories: Sequence[EpisodeTrajectory]) -> None:
        if not trajectories:
            return
        self.trained = True
        veto_rate = sum(t.terminal_outcome.startswith("veto") for t in trajectories) / len(
            trajectories
        )
        mean_weighted = float(np.mean([t.weighted_reward for t in trajectories]))
        if veto_rate >= 0.20:
            self.enable_lookahead = False
            self.prefer_concessions = True
        elif mean_weighted > 4.0:
            self.enable_lookahead = False

    def state_dict(self) -> Dict[str, Any]:
        return {
            "enable_lookahead": self.enable_lookahead,
            "prefer_concessions": self.prefer_concessions,
            "trained": self.trained,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.enable_lookahead = bool(state.get("enable_lookahead", False))
        self.prefer_concessions = bool(state.get("prefer_concessions", False))
        self.trained = bool(state.get("trained", False))

    def _pick_target(self, observation: DealRoomObservation) -> str:
        if observation.active_blockers:
            return observation.active_blockers[0]
        if observation.weak_signals:
            for stakeholder_id, signals in observation.weak_signals.items():
                if "declining_engagement" in signals or "low_engagement" in signals:
                    return stakeholder_id
        if observation.engagement_level:
            return min(
                observation.engagement_level,
                key=lambda stakeholder_id: observation.engagement_level[stakeholder_id],
            )
        return next(iter(observation.stakeholders.keys()), "Legal")

    def _build_target_action(self, target: str) -> DealRoomAction:
        target_lower = target.lower()
        if target_lower == "legal":
            return DealRoomAction(
                action_type="send_document",
                target=target,
                target_ids=[target],
                message="Here are the compliance and liability safeguards for review.",
                documents=[{"type": "dpa", "name": "DPA"}],
            )
        if target_lower == "finance":
            return DealRoomAction(
                action_type="send_document",
                target=target,
                target_ids=[target],
                message="Here is the ROI model with downside cases and approval-friendly assumptions.",
                documents=[{"type": "roi_model", "name": "ROI Model"}],
            )
        if target_lower in {"techlead", "operations"}:
            return DealRoomAction(
                action_type="send_document",
                target=target,
                target_ids=[target],
                message="Here is the implementation plan with delivery safeguards and named ownership.",
                documents=[
                    {"type": "implementation_timeline", "name": "Implementation Plan"}
                ],
            )
        if self.prefer_concessions:
            return DealRoomAction(
                action_type="concession",
                target=target,
                target_ids=[target],
                message="I can make the commercial and risk posture safer for your team.",
                proposed_terms={"liability_cap": 1500000},
            )
        return DealRoomAction(
            action_type="direct_message",
            target=target,
            target_ids=[target],
            message="Help me understand the remaining constraint so I can de-risk it responsibly.",
        )

    def _trained_round_action(self, observation: DealRoomObservation) -> DealRoomAction:
        stakeholders = list(observation.stakeholders.keys())
        plan = [
            self._build_target_action("Legal"),
            self._build_target_action("Finance"),
            self._build_target_action("TechLead"),
            DealRoomAction(
                action_type="direct_message",
                target="Procurement",
                target_ids=["Procurement"],
                message="We can keep procurement terms simple, auditable, and low-risk.",
            ),
            DealRoomAction(
                action_type="direct_message",
                target="ExecSponsor",
                target_ids=["ExecSponsor"],
                message="Here is the concise strategic alignment and executive risk summary.",
            ),
        ]
        action = plan[observation.round_number % len(plan)]
        if action.target_ids and action.target_ids[0] not in stakeholders:
            return self._build_target_action(self._pick_target(observation))
        return action.model_copy(deep=True)


class ModelPolicyAdapter:
    def __init__(
        self,
        policy_fn: Callable[[DealRoomObservation], DealRoomAction],
        name: str = "model",
    ):
        self.policy_fn = policy_fn
        self.name = name

    def act(
        self, observation: DealRoomObservation, rng: np.random.Generator
    ) -> DealRoomAction:
        return self.policy_fn(observation)

    def update_from_batch(self, trajectories: Sequence[EpisodeTrajectory]) -> None:
        return None

    def state_dict(self) -> Dict[str, Any]:
        return {"name": self.name}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.name = str(state.get("name", self.name))


class GRPOTrainer:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-3B-Instruct",
        learning_rate: float = 1e-5,
        grpo_clip: float = 0.2,
        entropy_coef: float = 0.01,
        reward_weights: Optional[List[float]] = None,
        policy_adapter: Optional[PolicyAdapter] = None,
        curriculum_generator: Optional[AdaptiveCurriculumGenerator] = None,
        checkpoint_dir: str = "artifacts/training",
        seed: int = 42,
        model: Optional[str] = None,
        env: Optional[DealRoomV3] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        if model is not None:
            model_id = model
        self.model_id = model_id
        self.env = env
        self.config = config or {}
        self.learning_rate = learning_rate
        self.grpo_clip = grpo_clip
        self.entropy_coef = entropy_coef
        self.reward_weights = reward_weights or [
            REWARD_WEIGHTS[dimension] for dimension in REWARD_DIMENSIONS
        ]
        self.policy_adapter = policy_adapter or HeuristicPolicyAdapter()
        self.curriculum_generator = curriculum_generator or AdaptiveCurriculumGenerator()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.rng = np.random.default_rng(seed)
        self._latest_failure_analysis = None

    def compute_group_relative_advantage(
        self, episode_rewards: List[List[float]], group_rewards: List[List[float]]
    ) -> List[float]:
        if not group_rewards:
            return [0.0] * len(episode_rewards)

        aggregated = [weighted_reward(rewards) for rewards in episode_rewards]
        group_aggregated = [weighted_reward(rewards) for rewards in group_rewards]

        mean = np.mean(group_aggregated)
        std = np.std(group_aggregated) + 1e-8
        return [float((reward_value - mean) / std) for reward_value in aggregated]

    def run_self_play_episode(
        self,
        env: Optional[DealRoomV3] = None,
        policy_fn: Optional[Callable[[DealRoomObservation], DealRoomAction]] = None,
        policy_adapter: Optional[PolicyAdapter] = None,
        max_steps: int = 10,
        task_id: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> EpisodeTrajectory:
        env = env or DealRoomV3()
        adapter = policy_adapter or self.policy_adapter
        if policy_fn is not None:
            adapter = ModelPolicyAdapter(policy_fn, name="callable_policy")

        episode_seed = seed if seed is not None else int(self.rng.integers(0, 2**31))
        if task_id is None:
            scenario = self.curriculum_generator.generate_adaptive_scenario(
                self._latest_failure_analysis
            )
            task_id = scenario["task_id"]
            episode_seed = int(scenario.get("seed", episode_seed))

        observation = env.reset(seed=episode_seed, task_id=task_id)
        trajectory = EpisodeTrajectory(task_id=task_id, seed=episode_seed)

        for _ in range(max_steps):
            trajectory.observations.append(observation)
            action = adapter.act(observation, self.rng)
            trajectory.actions.append(action)
            trajectory.lookahead_used.append(action.lookahead is not None)

            observation, reward, done, info = env.step(action)
            reward_components = info.get("reward_components") or {}
            reward_vector = [
                float(reward_components.get(dimension, reward))
                for dimension in REWARD_DIMENSIONS
            ]
            trajectory.rewards.append(reward_vector)
            trajectory.scalar_rewards.append(float(reward))
            trajectory.prediction_accuracies.append(
                float(info.get("prediction_accuracy", 0.0))
            )
            trajectory.lookahead_diagnostics.append(
                {
                    "predicted_deltas": dict(
                        info.get("lookahead_predicted_deltas") or {}
                    ),
                    "predicted_responses": dict(
                        info.get("lookahead_predicted_responses") or {}
                    ),
                    "cvar_impact": dict(info.get("lookahead_cvar_impact") or {}),
                }
            )

            if done:
                trajectory.terminal_outcome = str(
                    info.get("terminal_outcome") or "max_rounds"
                )
                break

        trajectory.weighted_reward = float(sum(trajectory.scalar_rewards))
        return trajectory

    def compute_training_metrics(
        self, trajectories: List[EpisodeTrajectory]
    ) -> TrainingMetrics:
        if not trajectories:
            return TrainingMetrics()

        per_dimension: Dict[str, List[float]] = {
            dimension: [] for dimension in REWARD_DIMENSIONS
        }
        total_steps = 0
        lookahead_count = 0
        prediction_values: List[float] = []
        total_rewards: List[float] = []
        weighted_rewards: List[float] = []
        task_counts: Dict[str, int] = {}
        terminal_outcomes: Dict[str, int] = {}

        for trajectory in trajectories:
            task_counts[trajectory.task_id] = task_counts.get(trajectory.task_id, 0) + 1
            terminal = trajectory.terminal_outcome or "incomplete"
            terminal_outcomes[terminal] = terminal_outcomes.get(terminal, 0) + 1
            total_rewards.append(sum(trajectory.scalar_rewards))
            weighted_rewards.append(
                sum(weighted_reward(step_rewards) for step_rewards in trajectory.rewards)
            )

            for step_rewards in trajectory.rewards:
                for dimension, value in zip(REWARD_DIMENSIONS, step_rewards):
                    per_dimension[dimension].append(float(value))
                total_steps += 1

            lookahead_count += sum(trajectory.lookahead_used)
            prediction_values.extend(trajectory.prediction_accuracies)

        task_mix = {
            task_id: count / len(trajectories)
            for task_id, count in sorted(task_counts.items())
        }
        return TrainingMetrics(
            goal_reward=float(np.mean(per_dimension["goal"]))
            if per_dimension["goal"]
            else 0.0,
            trust_reward=float(np.mean(per_dimension["trust"]))
            if per_dimension["trust"]
            else 0.0,
            info_reward=float(np.mean(per_dimension["info"]))
            if per_dimension["info"]
            else 0.0,
            risk_reward=float(np.mean(per_dimension["risk"]))
            if per_dimension["risk"]
            else 0.0,
            causal_reward=float(np.mean(per_dimension["causal"]))
            if per_dimension["causal"]
            else 0.0,
            lookahead_usage_rate=lookahead_count / max(total_steps, 1),
            prediction_accuracy=float(np.mean(prediction_values))
            if prediction_values
            else 0.0,
            total_reward=float(np.mean(total_rewards)) if total_rewards else 0.0,
            weighted_reward=float(np.mean(weighted_rewards))
            if weighted_rewards
            else 0.0,
            episodes_completed=len(trajectories),
            task_mix=task_mix,
            terminal_outcomes=terminal_outcomes,
        )

    def run_training_loop(
        self,
        env_factory: Callable[[], DealRoomV3] = DealRoomV3,
        n_episodes: int = 50,
        episodes_per_batch: int = 4,
        max_steps: int = 10,
        verbose: bool = True,
    ) -> List[TrainingMetrics]:
        all_metrics: List[TrainingMetrics] = []
        for batch_index in range(n_episodes):
            batch_trajectories: List[EpisodeTrajectory] = []
            for _ in range(episodes_per_batch):
                env = env_factory()
                try:
                    trajectory = self.run_self_play_episode(
                        env=env,
                        policy_adapter=self.policy_adapter,
                        max_steps=max_steps,
                    )
                finally:
                    env.close()
                batch_trajectories.append(trajectory)

            self._latest_failure_analysis = self.curriculum_generator.analyze_failures(
                batch_trajectories
            )
            self.policy_adapter.update_from_batch(batch_trajectories)
            metrics = self.compute_training_metrics(batch_trajectories)
            metrics.checkpoint_path = self.save_checkpoint(
                batch_index + 1, batch_trajectories, metrics
            )
            all_metrics.append(metrics)

            if verbose:
                print(
                    f"Batch {batch_index + 1}/{n_episodes} | "
                    f"goal={metrics.goal_reward:.3f} trust={metrics.trust_reward:.3f} "
                    f"info={metrics.info_reward:.3f} risk={metrics.risk_reward:.3f} "
                    f"causal={metrics.causal_reward:.3f} weighted={metrics.weighted_reward:.3f}"
                )

        return all_metrics

    def train(
        self,
        n_episodes: int = 10,
        episodes_per_batch: int = 4,
        max_steps: int = 10,
        verbose: bool = False,
    ) -> List[TrainingMetrics]:
        batches = max(1, int(np.ceil(n_episodes / max(episodes_per_batch, 1))))
        return self.run_training_loop(
            env_factory=(lambda: self.env or DealRoomV3()),
            n_episodes=batches,
            episodes_per_batch=episodes_per_batch,
            max_steps=max_steps,
            verbose=verbose,
        )

    def policy(self, observation: DealRoomObservation) -> DealRoomAction:
        return self.policy_adapter.act(observation, self.rng)

    def evaluate_policy(
        self,
        policy_adapter: PolicyAdapter,
        scenario_ids: Sequence[str] = ("aligned", "conflicted"),
        episodes_per_task: int = 4,
        max_steps: int = 10,
        seed: int = 42,
        env_factory: Callable[[], DealRoomV3] = DealRoomV3,
    ) -> TrainingMetrics:
        old_rng = self.rng
        self.rng = np.random.default_rng(seed)
        trajectories: List[EpisodeTrajectory] = []
        try:
            episode_index = 0
            for task_id in scenario_ids:
                for _ in range(episodes_per_task):
                    env = env_factory()
                    try:
                        trajectories.append(
                            self.run_self_play_episode(
                                env=env,
                                policy_adapter=policy_adapter,
                                max_steps=max_steps,
                                task_id=task_id,
                                seed=seed + episode_index,
                            )
                        )
                    finally:
                        env.close()
                    episode_index += 1
        finally:
            self.rng = old_rng
        return self.compute_training_metrics(trajectories)

    def benchmark_policies(
        self,
        policy_adapters: Sequence[PolicyAdapter],
        scenario_ids: Sequence[str] = ("aligned", "conflicted"),
        episodes_per_task: int = 4,
        max_steps: int = 10,
        env_factory: Callable[[], DealRoomV3] = DealRoomV3,
    ) -> Dict[str, Dict[str, Any]]:
        benchmark: Dict[str, Dict[str, Any]] = {}

        for adapter in policy_adapters:
            trajectories: List[EpisodeTrajectory] = []
            for task_id in scenario_ids:
                for _ in range(episodes_per_task):
                    env = env_factory()
                    try:
                        trajectories.append(
                            self.run_self_play_episode(
                                env=env,
                                policy_adapter=adapter,
                                max_steps=max_steps,
                                task_id=task_id,
                            )
                        )
                    finally:
                        env.close()
            metrics = self.compute_training_metrics(trajectories)
            benchmark[adapter.name] = {
                "metrics": asdict(metrics),
                "episodes": len(trajectories),
            }

        benchmark_path = self.checkpoint_dir / "benchmark.json"
        benchmark_path.write_text(json.dumps(benchmark, indent=2))
        return benchmark

    def save_checkpoint(
        self,
        batch_index: int,
        trajectories: Sequence[EpisodeTrajectory],
        metrics: TrainingMetrics,
    ) -> str:
        checkpoint_path = (
            self.checkpoint_dir / f"checkpoint_batch_{batch_index:03d}.json"
        )
        payload = {
            "batch_index": batch_index,
            "model_id": self.model_id,
            "policy_name": getattr(self.policy_adapter, "name", "policy"),
            "policy_state": self.policy_adapter.state_dict(),
            "metrics": asdict(metrics),
            "trajectories": [
                {
                    "task_id": trajectory.task_id,
                    "terminal_outcome": trajectory.terminal_outcome,
                    "weighted_reward": trajectory.weighted_reward,
                    "seed": trajectory.seed,
                }
                for trajectory in trajectories
            ],
        }
        checkpoint_path.write_text(json.dumps(payload, indent=2))
        return str(checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        payload = json.loads(Path(checkpoint_path).read_text())
        self.policy_adapter.load_state_dict(payload.get("policy_state", {}))
        return payload


DealRoomGRPOTrainer = GRPOTrainer
