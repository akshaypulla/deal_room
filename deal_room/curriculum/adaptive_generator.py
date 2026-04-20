"""
Curriculum module for DealRoom v3 - adaptive curriculum generator.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class FailureAnalysis:
    failure_modes: Dict[str, float] = field(default_factory=dict)
    worst_graph_configs: List = field(default_factory=list)
    worst_cvar_configs: List = field(default_factory=list)
    agent_capability_estimate: float = 0.0


@dataclass
class CurriculumConfig:
    analysis_batch_size: int = 10
    easy_ratio: float = 0.20
    frontier_ratio: float = 0.60
    hard_ratio: float = 0.20
    max_graph_variation: float = 0.3


FAILURE_MODE_DESCRIPTIONS = {
    "F1": "CVaR veto despite positive expected outcome",
    "F2": "Trust collapse mid-episode",
    "F3": "Failed graph inference",
    "F4": "Timeout without coalition formation",
    "F5": "Single-dimension reward hacking",
    "F6": "Authority shift blindness",
}


class AdaptiveCurriculumGenerator:
    def __init__(self, config: CurriculumConfig = None):
        self.config = config or CurriculumConfig()
        self._scenario_pool: List[Dict] = []
        self._difficulty_distribution = [
            self.config.easy_ratio,
            self.config.frontier_ratio,
            self.config.hard_ratio,
        ]
        self._rng = np.random.default_rng(42)
        self._initialize_scenario_pool()

    def _initialize_scenario_pool(self):
        base_configs = [
            {"task_id": "aligned", "difficulty": "easy"},
            {"task_id": "conflicted", "difficulty": "frontier"},
            {"task_id": "hostile_acquisition", "difficulty": "hard"},
        ]
        for _ in range(5):
            for config in base_configs:
                variant = dict(config)
                variant["seed"] = int(self._rng.integers(0, 2**31))
                self._scenario_pool.append(variant)

    def analyze_failures(self, trajectories: List) -> FailureAnalysis:
        failure_counts: Dict[str, int] = {}
        for traj in trajectories:
            detected = self._detect_failures(traj)
            for failure_id in detected:
                failure_counts[failure_id] = failure_counts.get(failure_id, 0) + 1

        total = len(trajectories)
        failure_modes = (
            {k: v / total for k, v in failure_counts.items()} if total > 0 else {}
        )

        recent_rewards = []
        for t in trajectories[-5:]:
            if hasattr(t, "rewards"):
                step_rewards = t.rewards[-1] if t.rewards else [0.0]
                weighted = sum(
                    r * w for r, w in zip(step_rewards, [0.25, 0.20, 0.20, 0.20, 0.15])
                )
                recent_rewards.append(weighted)
        capability = float(np.mean(recent_rewards)) if recent_rewards else 0.5

        return FailureAnalysis(
            failure_modes=failure_modes,
            worst_graph_configs=[],
            worst_cvar_configs=[],
            agent_capability_estimate=capability,
        )

    def _detect_failures(self, traj) -> Dict[str, int]:
        failures: Dict[str, int] = {}

        if hasattr(traj, "terminal_outcome"):
            if traj.terminal_outcome == "veto":
                failures["F1"] = 1

        if hasattr(traj, "rewards") and len(traj.rewards) >= 7:
            trust_rewards = [r[1] if len(r) > 1 else 0.0 for r in traj.rewards[6:10]]
            if len(trust_rewards) >= 2:
                max_drop = max(trust_rewards) - min(trust_rewards)
                if max_drop > 0.20:
                    failures["F2"] = 1

        if hasattr(traj, "rewards"):
            causal_rewards = [r[4] if len(r) > 4 else 0.0 for r in traj.rewards]
            if all(0.15 <= c <= 0.30 for c in causal_rewards):
                failures["F3"] = 1

        return failures

    def select_next_scenario(
        self, failure_analysis: Optional[FailureAnalysis] = None
    ) -> Dict:
        if failure_analysis is None or failure_analysis.agent_capability_estimate < 0.3:
            return self._rng.choice(self._scenario_pool)

        capability = failure_analysis.agent_capability_estimate

        if capability < 0.5:
            difficulty = "easy"
        elif capability < 0.75:
            difficulty = "frontier"
        else:
            difficulty = "hard"

        candidates = [s for s in self._scenario_pool if s["difficulty"] == difficulty]
        if not candidates:
            candidates = self._scenario_pool

        selected = self._rng.choice(candidates)
        return dict(selected)

    def generate_adaptive_scenario(
        self, failure_analysis: Optional[FailureAnalysis] = None
    ) -> Dict:
        scenario = self.select_next_scenario(failure_analysis)

        if failure_analysis and failure_analysis.failure_modes:
            for failure_id, freq in failure_analysis.failure_modes.items():
                if failure_id == "F1" and freq > 0.3:
                    scenario["difficulty"] = "easy"
                    scenario["reduce_cvar_tension"] = True

        return scenario


def create_curriculum_generator(
    config: Optional[CurriculumConfig] = None,
) -> AdaptiveCurriculumGenerator:
    return AdaptiveCurriculumGenerator(config=config)
