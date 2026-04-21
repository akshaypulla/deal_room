"""Reproducible benchmark entrypoint for DealRoom v3 policies."""

from __future__ import annotations

import argparse
import json

from deal_room.training.grpo_trainer import (
    GRPOTrainer,
    HeuristicPolicyAdapter,
    RandomPolicyAdapter,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DealRoom v3 policy benchmark.")
    parser.add_argument("--episodes-per-task", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["aligned", "conflicted"],
        choices=["aligned", "conflicted", "hostile_acquisition"],
    )
    args = parser.parse_args()

    trainer = GRPOTrainer()
    benchmark = trainer.benchmark_policies(
        policy_adapters=[RandomPolicyAdapter(), HeuristicPolicyAdapter()],
        scenario_ids=args.tasks,
        episodes_per_task=args.episodes_per_task,
        max_steps=args.max_steps,
    )
    print(json.dumps(benchmark, indent=2))


if __name__ == "__main__":
    main()
