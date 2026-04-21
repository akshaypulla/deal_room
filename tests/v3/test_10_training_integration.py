#!/usr/bin/env python3
"""
DealRoom v3 training integration validation.

This test proves the local trainer produces a policy that beats a fixed random
baseline without downloading a model or requiring external credentials.
"""

import sys
from pathlib import Path


def _find_project_root() -> Path:
    current_file = Path(__file__).resolve()
    for candidate in (current_file.parent, *current_file.parents):
        if (candidate / "deal_room" / "training").exists():
            return candidate
    if Path("/app/env/deal_room/training").exists():
        return Path("/app/env")
    return current_file.parent


_LOCAL_ROOT = _find_project_root()
_LOCAL_PACKAGE = _LOCAL_ROOT / "deal_room"
if (_LOCAL_ROOT / "deal_room" / "training").exists():
    sys.path.insert(0, str(_LOCAL_ROOT))
if Path("/app/env/deal_room/training").exists():
    sys.path.insert(0, "/app/env")

_loaded_deal_room = sys.modules.get("deal_room")
if _loaded_deal_room is not None:
    module_file = str(getattr(_loaded_deal_room, "__file__", ""))
    valid_roots = [str(_LOCAL_ROOT), "/app/env"]
    if not any(module_file.startswith(root) for root in valid_roots):
        for module_name in list(sys.modules):
            if module_name == "deal_room" or module_name.startswith("deal_room."):
                sys.modules.pop(module_name, None)


def _ensure_training_package_visible():
    """Make pytest namespace-package imports see the canonical repo package."""
    import deal_room

    package_paths = getattr(deal_room, "__path__", None)
    if package_paths is not None and (_LOCAL_PACKAGE / "training").exists():
        local_package_path = str(_LOCAL_PACKAGE)
        if local_package_path not in list(package_paths):
            package_paths.insert(0, local_package_path)


def _load_training_symbols():
    _ensure_training_package_visible()
    from deal_room.training.grpo_trainer import (
        DealRoomGRPOTrainer,
        GRPOTrainer,
        RandomPolicyAdapter,
    )

    return DealRoomGRPOTrainer, GRPOTrainer, RandomPolicyAdapter

import numpy as np


def test_training_actually_improves():
    DealRoomGRPOTrainer, GRPOTrainer, RandomPolicyAdapter = _load_training_symbols()

    assert DealRoomGRPOTrainer is GRPOTrainer

    trainer = DealRoomGRPOTrainer(seed=123, checkpoint_dir="/tmp/dealroom-training-test")
    random_metrics = trainer.evaluate_policy(
        RandomPolicyAdapter(use_lookahead_probability=0.0),
        scenario_ids=("aligned", "conflicted"),
        episodes_per_task=5,
        max_steps=8,
        seed=321,
    )

    trainer.train(n_episodes=4, episodes_per_batch=4, max_steps=8, verbose=False)
    trained_metrics = trainer.evaluate_policy(
        trainer.policy_adapter,
        scenario_ids=("aligned", "conflicted"),
        episodes_per_task=5,
        max_steps=8,
        seed=321,
    )

    improvement = trained_metrics.weighted_reward - random_metrics.weighted_reward
    assert improvement > 0.15, (
        f"Training did not improve performance enough. "
        f"Random={random_metrics.weighted_reward:.3f}, "
        f"trained={trained_metrics.weighted_reward:.3f}, "
        f"improvement={improvement:.3f}; expected >0.15."
    )

    # Sanity check the convenience policy method returns a real action.
    from deal_room.environment.dealroom_v3 import DealRoomV3

    env = DealRoomV3()
    obs = env.reset(seed=999, task_id="aligned")
    action = trainer.policy(obs)
    assert action.action_type and action.target_ids, "Trainer policy returned invalid action"
    env.close()

    print(
        f"✓ Training works: {random_metrics.weighted_reward:.3f} → "
        f"{trained_metrics.weighted_reward:.3f} (+{improvement:.3f})"
    )


if __name__ == "__main__":
    test_training_actually_improves()
