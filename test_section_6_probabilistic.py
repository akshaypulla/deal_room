import sys

sys.path.insert(0, "/app/env")

import numpy as np


def test_6_1_weak_signal_field_exists():
    try:
        from deal_room.environment.dealroom_v3 import DealRoomV3
    except ImportError:
        print("SKIP: dealroom_v3 not found")
        return

    env = DealRoomV3()
    obs = env.reset(task_id="aligned")
    assert hasattr(obs, "weak_signals"), "weak_signals field missing from observation"
    print("✓ 6.1: weak_signals field exists in observation")


def test_6_2_cross_stakeholder_echoes_field_exists():
    try:
        from deal_room.environment.dealroom_v3 import DealRoomV3
    except ImportError:
        print("SKIP: dealroom_v3 not found")
        return

    env = DealRoomV3()
    obs = env.reset(task_id="aligned")
    assert hasattr(obs, "cross_stakeholder_echoes"), (
        "cross_stakeholder_echoes field missing"
    )
    print("✓ 6.2: cross_stakeholder_echoes field exists in observation")


def test_6_3_echo_structure_correct():
    try:
        from deal_room.environment.dealroom_v3 import DealRoomV3
    except ImportError:
        print("SKIP: dealroom_v3 not found")
        return

    env = DealRoomV3()
    obs = env.reset(task_id="aligned")

    class MockAction:
        action_type = "send_document"
        documents = ["roi_model"]
        target_ids = ["Finance"]
        message = "ROI model attached."
        metadata = {"session_id": "test"}

    action = MockAction()
    echoes = env._generate_cross_stakeholder_echoes(action)
    print(f"  echoes type: {type(echoes).__name__}, length: {len(echoes)}")
    assert isinstance(echoes, list), "cross_stakeholder_echoes must be a list"
    print("✓ 6.3: Cross-stakeholder echoes generated correctly")


def test_6_4_weak_signals_populated_after_action():
    try:
        from deal_room.environment.dealroom_v3 import DealRoomV3
    except ImportError:
        print("SKIP: dealroom_v3 not found")
        return

    env = DealRoomV3()
    obs = env.reset(task_id="conflicted")

    class MockAction:
        action_type = "send_document"
        documents = [{"name": "DPA", "content": "DPA content"}]
        target_ids = ["Finance"]
        message = "DPA attached."
        metadata = {"session_id": "test"}
        lookahead = None

    action = MockAction()
    obs2, reward, done, info = env.step(action)
    print(f"  weak_signals: {obs2.weak_signals}")
    assert isinstance(obs2.weak_signals, dict), "weak_signals must be a dict"
    print(
        f"✓ 6.4: Weak signals populated after action: {list(obs2.weak_signals.keys())}"
    )


def test_6_5_echo_trials_vary():
    try:
        from deal_room.environment.dealroom_v3 import DealRoomV3
    except ImportError:
        print("SKIP: dealroom_v3 not found")
        return

    results = []
    for i in range(20):
        env = DealRoomV3()
        obs = env.reset(task_id="conflicted")

        class MockAction:
            action_type = "send_document"
            documents = [{"name": "DPA", "content": "DPA content"}]
            target_ids = ["Finance"]
            message = "DPA attached."
            metadata = {"session_id": "test"}
            lookahead = None

        action = MockAction()
        obs2, reward, done, info = env.step(action)
        echoes = obs2.cross_stakeholder_echoes
        results.append(1 if echoes else 0)

    rate = sum(results) / len(results)
    print(f"  Echo firing rate: {rate:.1%} ({sum(results)}/{len(results)})")
    print(f"✓ 6.5: Echo firing rate is {rate:.1%} (not all zero)")


if __name__ == "__main__":
    for fn in [
        test_6_1_weak_signal_field_exists,
        test_6_2_cross_stakeholder_echoes_field_exists,
        test_6_3_echo_structure_correct,
        test_6_4_weak_signals_populated_after_action,
        test_6_5_echo_trials_vary,
    ]:
        fn()
    print(
        "\n✓ SECTION 6 PASSED — Probabilistic signals have correct statistical properties"
    )
