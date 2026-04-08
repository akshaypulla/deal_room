"""
Pytest configuration and fixtures for DealRoom test suite.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
import numpy as np

# Ensure the project root is on the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Environment modes for testing
ENV_MODES = {
    "development": {
        "log_level": "DEBUG",
        "enable_verbose_errors": True,
        "mock_external_apis": True,
    },
    "staging": {
        "log_level": "INFO",
        "enable_verbose_errors": False,
        "mock_external_apis": False,
    },
    "production": {
        "log_level": "WARNING",
        "enable_verbose_errors": False,
        "mock_external_apis": False,
    },
}

# Network latency simulation settings (in seconds)
LATENCY_SIMULATION = {
    "none": 0.0,
    "low": 0.05,
    "medium": 0.15,
    "high": 0.5,
    "very_high": 1.0,
}


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def test_data_dir(project_root: Path) -> Path:
    return project_root / "tests" / "fixtures"


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def test_seed() -> int:
    return 42


@pytest.fixture
def all_task_ids() -> list:
    return ["aligned", "conflicted", "hostile_acquisition"]


@pytest.fixture
def environment_config() -> dict:
    return {
        "environment": "test",
        "log_level": "DEBUG",
        "latency_simulation": "none",
    }


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response for testing without real API calls."""
    return {
        "choices": [
            {
                "message": {
                    "content": '{"action_type": "direct_message", "target": "CFO", "message": "Test message"}'
                }
            }
        ]
    }


@pytest.fixture
def mock_api_client():
    """Mock external API client for integration testing."""
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = MagicMock()
    return client


@pytest.fixture
def test_logger(caplog):
    """Configure test logger with specified level."""
    caplog.set_level(logging.DEBUG)
    return caplog


@pytest.fixture
def stakeholder_ids() -> list:
    return ["CFO", "CTO", "Legal", "Procurement", "Ops"]


@pytest.fixture
def action_types() -> list:
    return [
        "direct_message",
        "group_proposal",
        "backchannel",
        "send_document",
        "concession",
        "walkaway_signal",
        "reframe_value_prop",
        "exec_escalation",
    ]


@pytest.fixture
def valid_targets() -> list:
    return [
        "CFO",
        "CTO",
        "Legal",
        "Procurement",
        "Ops",
        "all",
        "cto_cfo",
        "legal_procurement",
    ]


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "e2e: End-to-end tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "security: Security tests")
    config.addinivalue_line("markers", "slow: Tests that take longer to run")
    config.addinivalue_line(
        "markers", "requires_db: Tests requiring database connection"
    )
    config.addinivalue_line("markers", "requires_api: Tests requiring external API")


def pytest_collection_modifyitems(config, items):
    """Auto-mark tests based on their directory location."""
    for item in items:
        if "unit" in item.nodeid:
            item.add_marker(pytest.mark.unit)
        elif "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        elif "e2e" in item.nodeid:
            item.add_marker(pytest.mark.e2e)
        elif "performance" in item.nodeid:
            item.add_marker(pytest.mark.performance)
        elif "security" in item.nodeid:
            item.add_marker(pytest.mark.security)


# Pytest collection hooks for better test organization
def pytest_addoption(parser):
    parser.addoption(
        "--env-mode",
        action="store",
        default="development",
        choices=["development", "staging", "production"],
        help="Test environment mode",
    )
    parser.addoption(
        "--latency",
        action="store",
        default="none",
        choices=["none", "low", "medium", "high", "very_high"],
        help="Simulated network latency",
    )
    parser.addoption(
        "--skip-slow",
        action="store_true",
        default=False,
        help="Skip slow tests",
    )
