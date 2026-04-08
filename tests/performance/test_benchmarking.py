"""
Performance benchmarking tests with realistic data volumes.
Tests throughput, latency, and scalability.
"""

import pytest
import time
import numpy as np
from models import DealRoomAction
from server.deal_room_environment import DealRoomEnvironment


class TestEnvironmentPerformance:
    """Test environment performance characteristics."""

    def test_reset_performance(self):
        """Test reset performance is acceptable."""
        env = DealRoomEnvironment()

        start = time.perf_counter()
        for _ in range(100):
            env.reset(seed=42, task_id="aligned")
        elapsed = time.perf_counter() - start

        # 100 resets should complete in under 1 second
        assert elapsed < 1.0, f"100 resets took {elapsed:.2f}s, too slow"

    def test_step_performance(self):
        """Test step performance is acceptable."""
        env = DealRoomEnvironment()
        env.reset(seed=42, task_id="aligned")

        action = DealRoomAction(
            action_type="direct_message", target="all", message="Test"
        )

        start = time.perf_counter()
        for _ in range(100):
            env.step(action)
        elapsed = time.perf_counter() - start

        # 100 steps should complete in under 1 second
        assert elapsed < 1.0, f"100 steps took {elapsed:.2f}s, too slow"

    def test_episode_throughput(self):
        """Test complete episode throughput."""
        env = DealRoomEnvironment()
        env.reset(seed=42, task_id="aligned")

        start = time.perf_counter()
        steps = 0
        for _ in range(20):
            action = DealRoomAction(
                action_type="direct_message", target="all", message="Test"
            )
            obs, reward, done, info = env.step(action)
            steps += 1
            if done:
                break
        elapsed = time.perf_counter() - start

        # Should handle at least 20 steps per second
        assert elapsed < 2.0, f"20 steps took {elapsed:.2f}s"

    def test_memory_efficiency(self):
        """Test that repeated episodes don't leak memory."""
        import sys

        env = DealRoomEnvironment()

        # Get baseline
        initial_size = sys.getsizeof(env.state)

        # Run many episodes
        for _ in range(100):
            env.reset(seed=42, task_id="aligned")
            for _ in range(5):
                action = DealRoomAction(
                    action_type="direct_message", target="all", message="Test"
                )
                env.step(action)

        # State should not grow significantly
        final_size = sys.getsizeof(env.state)
        assert final_size < initial_size * 2, "Memory usage grew significantly"


class TestScenariosPerformance:
    """Test performance across different scenarios."""

    def test_all_scenarios_performance(self):
        """Test that all scenarios have acceptable performance."""
        for task_id in ["aligned", "conflicted", "hostile_acquisition"]:
            env = DealRoomEnvironment()
            env.reset(seed=42, task_id=task_id)

            start = time.perf_counter()
            for _ in range(50):
                action = DealRoomAction(
                    action_type="direct_message", target="all", message="Test"
                )
                env.step(action)
            elapsed = time.perf_counter() - start

            assert elapsed < 1.0, f"{task_id} 50 steps took {elapsed:.2f}s"


class TestScaling:
    """Test scaling characteristics."""

    def test_large_number_of_steps(self):
        """Test environment handles large number of steps."""
        env = DealRoomEnvironment()
        env.reset(seed=42, task_id="aligned")

        start = time.perf_counter()
        for _ in range(200):
            action = DealRoomAction(
                action_type="direct_message", target="all", message="Test"
            )
            obs, reward, done, info = env.step(action)
            if done and reward == 0.0:  # Only count timeout terminations
                break
        elapsed = time.perf_counter() - start

        # 200 steps should complete in reasonable time
        assert elapsed < 5.0, f"200 steps took {elapsed:.2f}s"


class TestLatencySimulation:
    """Test network latency simulation (placeholder for actual implementation)."""

    def test_latency_config_exists(self):
        """Test that latency simulation configuration exists."""
        from tests.conftest import LATENCY_SIMULATION

        assert "none" in LATENCY_SIMULATION
        assert "medium" in LATENCY_SIMULATION
        assert "high" in LATENCY_SIMULATION

    @pytest.mark.skip(reason="Latency simulation requires actual implementation")
    def test_simulated_latency(self):
        """Test that simulated latency is applied."""
        pass


class TestDeterminism:
    """Test deterministic behavior for performance testing."""

    def test_deterministic_performance(self):
        """Test that same actions produce same timing patterns."""
        times1 = []
        times2 = []

        for _ in range(10):
            env1 = DealRoomEnvironment()
            env1.reset(seed=42, task_id="aligned")

            start = time.perf_counter()
            for _ in range(5):
                action = DealRoomAction(
                    action_type="direct_message", target="all", message="Test"
                )
                env1.step(action)
            times1.append(time.perf_counter() - start)

        for _ in range(10):
            env2 = DealRoomEnvironment()
            env2.reset(seed=42, task_id="aligned")

            start = time.perf_counter()
            for _ in range(5):
                action = DealRoomAction(
                    action_type="direct_message", target="all", message="Test"
                )
                env2.step(action)
            times2.append(time.perf_counter() - start)

        # Both should have similar timing characteristics
        avg1 = sum(times1) / len(times1)
        avg2 = sum(times2) / len(times2)
        assert abs(avg1 - avg2) < avg1 * 0.5  # Within 50% of each other
