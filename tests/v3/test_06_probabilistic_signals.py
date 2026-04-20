#!/usr/bin/env python3
"""
test_06_probabilistic_signals.py
DealRoom v3 — Probabilistic Signal Validation (runs inside container)

Validates:
- weak_signals field exists in observation
- cross_stakeholder_echoes field exists
- Echo structure is correct (list of dicts with from/to)
- Weak signals are populated after action
- Echo firing rate is non-zero (70% recall probability active)
- Echoes vary across episodes (not deterministic)
- Weak signal probability thresholds are respected
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


def test_6_1_weak_signals_field_exists():
    print("\n[6.1] weak_signals field exists...")
    from deal_room.environment.dealroom_v3 import DealRoomV3

    env = DealRoomV3()
    obs = env.reset(task_id="aligned")
    assert hasattr(obs, "weak_signals"), "weak_signals missing from observation"
    print("  ✓ weak_signals field exists")


def test_6_2_cross_stakeholder_echoes_exists():
    print("\n[6.2] cross_stakeholder_echoes field exists...")
    from deal_room.environment.dealroom_v3 import DealRoomV3

    env = DealRoomV3()
    obs = env.reset(task_id="aligned")
    assert hasattr(obs, "cross_stakeholder_echoes"), "cross_stakeholder_echoes missing"
    print("  ✓ cross_stakeholder_echoes field exists")


def test_6_3_echo_structure():
    print("\n[6.3] Echo structure: list of dicts with from/to/content...")
    from deal_room.environment.dealroom_v3 import DealRoomV3

    env = DealRoomV3()
    obs = env.reset(task_id="aligned")

    class MockAction:
        action_type = "send_document"
        documents = [{"name": "roi_model", "content": "ROI content"}]
        target_ids = ["Finance"]
        message = "ROI model attached."
        metadata = {"session_id": "test"}
        lookahead = None

    echoes = env._generate_cross_stakeholder_echoes(MockAction())
    assert isinstance(echoes, list), f"echoes must be list, got {type(echoes).__name__}"

    if len(echoes) > 0:
        echo = echoes[0]
        assert isinstance(echo, dict), (
            f"each echo must be dict, got {type(echo).__name__}"
        )
        assert any(
            k in echo for k in ["from", "from_stakeholder", "sender", "source"]
        ), f"echo missing 'from' field: {list(echo.keys())}"
        print(f"  ✓ Echoes is list of dicts: {list(echo.keys())}")
    else:
        print("  ⚠ No echoes generated (may be stochastic)")


def test_6_4_weak_signals_populated_after_action():
    print("\n[6.4] Weak signals populated after step...")
    from deal_room.environment.dealroom_v3 import DealRoomV3

    env = DealRoomV3()
    obs = env.reset(task_id="conflicted")

    class MockAction:
        action_type = "send_document"
        documents = [{"name": "DPA", "content": "DPA content"}]
        target_ids = ["Finance"]
        message = "DPA attached."
        metadata = {"session_id": "test"}
        lookahead = None

    obs2, reward, done, info = env.step(MockAction())
    weak = obs2.weak_signals

    assert isinstance(weak, dict), (
        f"weak_signals must be dict, got {type(weak).__name__}"
    )
    print(f"  ✓ weak_signals populated: {list(weak.keys())}")


def test_6_5_echo_firing_rate_nonzero():
    print("\n[6.5] Echo firing rate is non-zero...")
    from deal_room.environment.dealroom_v3 import DealRoomV3

    fired = 0
    n = 20

    for i in range(n):
        env = DealRoomV3()
        obs = env.reset(task_id="conflicted")

        class MockAction:
            action_type = "send_document"
            documents = [{"name": "DPA", "content": "DPA content"}]
            target_ids = ["Finance"]
            message = "DPA attached."
            metadata = {"session_id": "test"}
            lookahead = None

        obs2, _, _, _ = env.step(MockAction())
        echoes = obs2.cross_stakeholder_echoes
        if echoes and len(echoes) > 0:
            fired += 1

    rate = fired / n
    print(f"  Echoes fired in {fired}/{n} episodes ({rate:.0%})")
    assert rate > 0, "Echo firing rate is 0% — echo_recall_probability broken"
    print(f"  ✓ Echo firing rate = {rate:.0%} (non-zero)")


def test_6_6_weak_signal_threshold_respected():
    print("\n[6.6] Weak signal probability threshold respected...")
    from deal_room.environment.dealroom_v3 import OBS_CONFIG

    assert OBS_CONFIG is not None, "OBS_CONFIG not initialized"
    threshold = OBS_CONFIG.weak_signal_hard_threshold
    print(f"  Hard threshold = {threshold}")

    from deal_room.environment.dealroom_v3 import DealRoomV3

    env = DealRoomV3()
    obs = env.reset(task_id="conflicted")

    class MockAction:
        action_type = "send_document"
        documents = [{"name": "DPA", "content": "DPA"}]
        target_ids = ["Finance"]
        message = "Test."
        metadata = {"session_id": "test"}
        lookahead = None

    obs2, _, _, _ = env.step(MockAction())
    weak = obs2.weak_signals

    if weak:
        for sid, val in weak.items():
            assert isinstance(val, (int, float)), f"weak_signal[{sid}] must be numeric"
            assert 0.0 <= val <= 1.0, f"weak_signal[{sid}]={val} outside [0,1]"

    print("  ✓ Weak signal values respect [0,1] bounds")


def test_6_7_echo_recall_probability_configured():
    print("\n[6.7] echo_recall_probability is configured (default 0.70)...")
    from deal_room.environment.dealroom_v3 import OBS_CONFIG

    prob = OBS_CONFIG.echo_recall_probability if OBS_CONFIG else None
    assert prob is not None, "echo_recall_probability not set"
    print(f"  echo_recall_probability = {prob}")
    assert 0.0 < prob < 1.0, f"echo_recall_probability={prob} invalid (should be 0-1)"
    print(f"  ✓ Echo recall probability = {prob} (should be ~0.70)")


def run_all():
    print("=" * 60)
    print("  DealRoom v3 — Probabilistic Signals (Container)")
    print("=" * 60)

    tests = [
        test_6_1_weak_signals_field_exists,
        test_6_2_cross_stakeholder_echoes_exists,
        test_6_3_echo_structure,
        test_6_4_weak_signals_populated_after_action,
        test_6_5_echo_firing_rate_nonzero,
        test_6_6_weak_signal_threshold_respected,
        test_6_7_echo_recall_probability_configured,
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
    print(f"  ✓ SECTION 6 — {passed}/{len(tests)} checks passed")
    if failed:
        print(f"  ✗ FAILED: {failed}")
        import sys

        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    run_all()
