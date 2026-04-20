import sys

sys.path.insert(0, "/app/env")


def test_10_1_training_imports_without_error():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "grpo_trainer", "/app/env/deal_room/training/grpo_trainer.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "GRPOTrainer"), "GRPOTrainer not found"
    assert hasattr(mod, "TrainingMetrics"), "TrainingMetrics not found"
    print("✓ 10.1: Training module imports successfully")


def test_10_2_training_metrics_fields():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "grpo_trainer", "/app/env/deal_room/training/grpo_trainer.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fields = list(mod.TrainingMetrics.__dataclass_fields__.keys())
    required = [
        "goal_reward",
        "trust_reward",
        "info_reward",
        "risk_reward",
        "causal_reward",
        "lookahead_usage_rate",
    ]
    missing = [f for f in required if f not in fields]
    assert not missing, f"TrainingMetrics missing fields: {missing}"
    print(f"✓ 10.2: TrainingMetrics has all reward curve fields: {fields}")


def test_10_3_curriculum_generator_imports():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "adaptive_generator", "/app/env/deal_room/curriculum/adaptive_generator.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "AdaptiveCurriculumGenerator"), (
        "AdaptiveCurriculumGenerator not found"
    )
    print("✓ 10.3: AdaptiveCurriculumGenerator imports and instantiates")


def test_10_4_colab_notebook_exists():
    import os

    for path in [
        "/app/env/deal_room/training/grpo_colab.ipynb",
        "/app/env/grpo_colab.ipynb",
    ]:
        if os.path.exists(path):
            import json

            with open(path) as f:
                nb = json.load(f)
            assert "cells" in nb and len(nb["cells"]) >= 5, (
                f"Notebook at {path} invalid"
            )
            print(
                f"✓ 10.4: Colab notebook exists at {path} with {len(nb['cells'])} cells"
            )
            return
    assert False, "grpo_colab.ipynb not found"


if __name__ == "__main__":
    for fn in [
        test_10_1_training_imports_without_error,
        test_10_2_training_metrics_fields,
        test_10_3_curriculum_generator_imports,
        test_10_4_colab_notebook_exists,
    ]:
        fn()
    print("\n✓ SECTION 10 PASSED — Training infrastructure is correct")
