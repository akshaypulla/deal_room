"""
DealRoom v3 — TRL-Compatible Text Environment Wrapper.

Wraps DealRoomV3 as a text-in/text-out RL environment compatible with
TRL's GRPOTrainer reward_function interface.

Usage with TRL GRPOTrainer:
    def reward_function(prompts, completions):
        return [env.execute(p, c) for p, c in zip(prompts, completions)]

The environment maintains episode state internally. Each call to step()
processes one action within the current episode.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from deal_room.environment.dealroom_v3 import DealRoomV3
from deal_room.environment.prompts import build_situation_prompt, parse_action_text
from models import DealRoomAction, DealRoomObservation


class DealRoomTextEnv:
    """
    TRL-compatible text-in/text-out wrapper for DealRoomV3.

    Key design:
    - reset() returns the initial situation prompt for an episode
    - step(action_text) executes one LLM-generated action and returns
      (next_prompt, reward, done, info)
    - State persists across steps within an episode

    For TRL GRPOTrainer: each episode = one prompt/completion pair at each
    decision step. The reward_function calls env.execute(prompt, completion)
    which runs the full episode step internally.
    """

    def __init__(
        self,
        task_id: str = "hostile_acquisition",
        seed: int = 42,
        use_llm_stakeholders: bool = False,
    ):
        self.task_id = task_id
        self.seed = seed
        self.use_llm_stakeholders = use_llm_stakeholders
        self.env: Optional[DealRoomV3] = None
        self._initialized = False
        self.current_obs: Optional[DealRoomObservation] = None
        self._step_count: int = 0
        self._episode_reward: float = 0.0

    def reset(self) -> str:
        """Reset environment and return initial situation prompt."""
        self.env = DealRoomV3(use_llm_stakeholders=self.use_llm_stakeholders)
        self.current_obs = self.env.reset(seed=self.seed, task_id=self.task_id)
        self._step_count = 0
        self._episode_reward = 0.0
        return build_situation_prompt(self.current_obs)

    def step(self, action_text: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        """
        Execute one text action within the current episode.

        Args:
            action_text: LLM-generated action string (e.g., "send_document Finance roi_model ...")

        Returns:
            Tuple of (next_prompt, reward, done, info)
        """
        if self.env is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        action = parse_action_text(action_text)

        self.current_obs, reward, done, info = self.env.step(action)

        self._step_count += 1
        self._episode_reward += reward

        if done:
            next_prompt = _build_terminal_prompt(
                self.current_obs,
                self._episode_reward,
                info,
            )
        else:
            next_prompt = build_situation_prompt(self.current_obs)

        return next_prompt, float(reward), bool(done), info

    def close(self) -> None:
        """Clean up environment resources."""
        if self.env is not None:
            self.env.close()
            self.env = None

    def execute(self, prompt: str, completion: str) -> float:
        """
        Convenience method for TRL reward_function interface.
        Executes one action and returns the scalar reward.

        Args:
            prompt: Current situation prompt (unused, state is internal)
            completion: LLM-generated action text

        Returns:
            Scalar reward for this action
        """
        _, reward, done, _ = self.step(completion)
        if done:
            return float(self._episode_reward)
        return float(reward)


def _build_terminal_prompt(
    obs: DealRoomObservation,
    total_reward: float,
    info: Dict[str, Any],
) -> str:
    """Build terminal prompt shown to LLM when episode ends."""
    terminal_outcome = info.get("terminal_outcome", "unknown")
    pareto_eff = info.get("pareto_efficiency", 0.0)
    causal_acc = info.get("causal_accuracy", 0.0)
    veto = info.get("veto_stakeholder", None)

    lines = [
        "=== EPISODE COMPLETE ===",
        f"Terminal Outcome: {terminal_outcome}",
        f"Total Reward: {total_reward:.3f}",
        f"Pareto Efficiency: {pareto_eff:.3f}",
        f"Causal Graph Accuracy: {causal_acc:.3f}",
    ]
    if veto:
        lines.append(f"Veto triggered by: {veto}")

    return "\n".join(lines)


def build_reward_function(
    task_id: str = "hostile_acquisition",
    seed: int = 42,
    use_llm_stakeholders: bool = False,
) -> callable:
    """
    Factory that builds a TRL-compatible reward_function.

    Usage:
        from trl import GRPOTrainer, GRPOConfig

        reward_fn = build_reward_function(task_id="hostile_acquisition", seed=42)

        trainer = GRPOTrainer(
            model=model,
            tokenizer=tokenizer,
            args=GRPOConfig(...),
            reward_function=reward_fn,
        )

    The returned function matches TRL's reward_function signature:
        def reward_function(prompts: List[str], completions: List[str]) -> List[float]

    Args:
        task_id: Scenario to use for training
        seed: Random seed for environment initialization
        use_llm_stakeholders: Whether to use GPT-4o-mini for stakeholder responses

    Returns:
        A reward_function compatible with TRL GRPOTrainer
    """
    env = DealRoomTextEnv(
        task_id=task_id, seed=seed, use_llm_stakeholders=use_llm_stakeholders
    )

    def reward_function(prompts: List[str], completions: List[str]) -> List[float]:
        """
        TRL GRPOTrainer calls this after each generation batch.

        Each completion runs a full episode (up to 8 steps) internally.
        The returned reward is the cumulative episode reward, which provides
        richer signal for GRPO advantage estimation.
        """
        rewards = []
        for prompt, completion in zip(prompts, completions):
            env.reset()
            total_reward = 0.0
            done = False
            for _ in range(8):
                _, reward, done, _ = env.step(completion)
                total_reward += reward
                if done:
                    break
            rewards.append(float(total_reward))
        return rewards

    return reward_function


def run_episode_with_text_actions(
    task_id: str = "hostile_acquisition",
    seed: int = 42,
    max_steps: int = 10,
    policy_fn=None,
    use_llm_stakeholders: bool = False,
) -> Dict[str, Any]:
    """
    Run a full episode given a policy function that maps prompts to action texts.

    Args:
        task_id: Scenario ID
        seed: Environment seed
        max_steps: Maximum steps per episode
        policy_fn: Callable that takes a prompt str and returns action text.
                   If None, uses a random policy.
        use_llm_stakeholders: Whether to use GPT-4o-mini for stakeholder responses

    Returns:
        Dict with episode trajectory, rewards, and final metrics
    """
    env = DealRoomTextEnv(
        task_id=task_id, seed=seed, use_llm_stakeholders=use_llm_stakeholders
    )
    prompt = env.reset()

    observations = []
    actions = []
    rewards = []
    infos = []

    if policy_fn is None:
        import random

        _fallback_actions = [
            "direct_message Finance I want to understand your concerns about the proposal.",
            "send_document Legal dpa Here is our DPA with enhanced liability safeguards.",
            "direct_message TechLead Can you share your technical requirements?",
            "group_proposal I propose we move to final contract review with agreed terms.",
            "concession Finance liability_cap=1500000",
        ]
        policy_fn = lambda p: random.choice(_fallback_actions)

    for step in range(max_steps):
        action_text = policy_fn(prompt)
        observations.append(prompt)
        actions.append(action_text)

        prompt, reward, done, info = env.step(action_text)
        rewards.append(reward)
        infos.append(info)

        if done:
            break

    total_reward = sum(rewards)
    final_info = infos[-1] if infos else {}

    return {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "infos": infos,
        "total_reward": total_reward,
        "pareto_efficiency": final_info.get("pareto_efficiency", 0.0),
        "terminal_outcome": final_info.get("terminal_outcome", "unknown"),
        "steps": len(actions),
    }
