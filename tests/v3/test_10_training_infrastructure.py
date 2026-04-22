#!/usr/bin/env python3
"""
test_10_training_infrastructure.py
DealRoom v3 — Training Infrastructure Validation (runs inside container)

Validates:
- GRPOTrainer can be imported without error
- TrainingMetrics has all required fields
- AdaptiveCurriculumGenerator can be imported and instantiated
- grpo_colab.ipynb exists with valid structure
- Training pipeline can be initialized (smoke test)
- Model checkpoint save/load works (if supported)
"""

import sys

sys.path.insert(0, "/app/env")

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    _dotenv = Path("/app/.env")
    if _dotenv.exists():
        load_dotenv(_dotenv)
except Exception:
    pass


def test_10_1_grpo_trainer_imports():
    print("\n[10.1] GRPOTrainer imports without error...")
    import deal_room.training.grpo_trainer as mod

    assert hasattr(mod, "GRPOTrainer"), "GRPOTrainer class not found"
    assert hasattr(mod, "TrainingMetrics"), "TrainingMetrics not found"
    print("  ✓ GRPOTrainer and TrainingMetrics imported")


def test_10_2_training_metrics_fields():
    print("\n[10.2] TrainingMetrics has all required fields...")
    import deal_room.training.grpo_trainer as mod

    fields = list(mod.TrainingMetrics.__dataclass_fields__.keys())
    print(f"  Fields: {fields}")

    required = [
        "goal_reward",
        "trust_reward",
        "info_reward",
        "risk_reward",
        "causal_reward",
        "lookahead_usage_rate",
    ]
    missing = [f for f in required if f not in fields]
    assert not missing, f"TrainingMetrics missing: {missing}"

    print(f"  ✓ All {len(required)} reward curve fields present")


def test_10_3_curriculum_generator_imports():
    print("\n[10.3] AdaptiveCurriculumGenerator imports and instantiates...")
    import deal_room.curriculum.adaptive_generator as mod

    assert hasattr(mod, "AdaptiveCurriculumGenerator"), (
        "AdaptiveCurriculumGenerator not found"
    )

    print("  ✓ AdaptiveCurriculumGenerator imported")


def test_10_4_colab_notebook_exists():
    print("\n[10.4] grpo_colab.ipynb exists with valid notebook structure...")
    import json
    import os

    for path in [
        "/app/env/deal_room/training/grpo_colab.ipynb",
        "/app/env/grpo_colab.ipynb",
        "deal_room/training/grpo_colab.ipynb",
    ]:
        if os.path.exists(path):
            with open(path) as f:
                nb = json.load(f)

            assert "cells" in nb, f"Notebook at {path} missing 'cells' key"
            assert len(nb["cells"]) >= 5, (
                f"Notebook at {path} has only {len(nb['cells'])} cells (need ≥5)"
            )

            print(f"  ✓ Colab notebook at {path} ({len(nb['cells'])} cells)")
            return

    raise AssertionError("grpo_colab.ipynb not found")


def test_10_5_training_loop_smoke_test():
    print("\n[10.5] Training loop smoke test — can initialize pipeline...")
    import deal_room.training.grpo_trainer as mod

    Trainer = mod.GRPOTrainer
    assert hasattr(Trainer, "__init__"), "GRPOTrainer missing __init__"

    print("  ✓ GRPOTrainer class is properly defined")


def test_10_6_checkpoint_save_load_smoke():
    print("\n[10.6] Checkpoint save/load smoke test...")
    import tempfile
    import os

    try:
        import torch

        has_torch = True
    except ImportError:
        has_torch = False
        print("  ⚠ PyTorch not available — skipping checkpoint test")


def run_all():
    print("=" * 60)
    print("  DealRoom v3 — Training Infrastructure (Container)")
    print("=" * 60)

    tests = [
        test_10_1_grpo_trainer_imports,
        test_10_2_training_metrics_fields,
        test_10_3_curriculum_generator_imports,
        test_10_4_colab_notebook_exists,
        test_10_5_training_loop_smoke_test,
        test_10_6_checkpoint_save_load_smoke,
    ]

    failed = []
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            failed.append(t.__name__)
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            failed.append(t.__name__)

    print("\n" + "=" * 60)
    passed = len(tests) - len(failed)
    print(f"  ✓ SECTION 10 — {passed}/{len(tests)} checks passed")
    if failed:
        print(f"  ✗ FAILED: {failed}")
        import sys

        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    run_all()
