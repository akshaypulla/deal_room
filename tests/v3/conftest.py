"""
Shared test configuration and utilities for DealRoom v3 test suite.
Loads environment variables from .env file and provides common helpers.
"""

import os
import sys
from pathlib import Path

# Attempt to load .env file if it exists
_dotenv_paths = [
    Path(__file__).parent / ".env",  # tests/v3/.env (primary)
    Path(__file__).parent.parent.parent / ".env",  # project_root/.env
]
for _dotenv_path in _dotenv_paths:
    if _dotenv_path.exists():
        try:
            from dotenv import load_dotenv

            load_dotenv(_dotenv_path)
            break
        except ImportError:
            pass

BASE_URL = os.getenv("DEALROOM_BASE_URL", "http://127.0.0.1:7860")
CONTAINER_NAME = os.getenv("DEALROOM_CONTAINER_NAME", "dealroom-v3-test")

OPTIONAL_ENV_VARS = ["MINIMAX_API_KEY", "OPENAI_API_KEY"]


def validate_api_keys():
    """Report API keys, but do not fail because the runtime is fail-soft."""
    configured = [v for v in OPTIONAL_ENV_VARS if os.getenv(v)]
    missing = [v for v in OPTIONAL_ENV_VARS if not os.getenv(v)]
    if configured:
        print(f"Configured optional API keys: {configured}")
    if missing:
        print(f"Missing optional API keys (fail-soft mode): {missing}")


def check_container_running():
    """Verify the Docker container is running."""
    import subprocess

    result = subprocess.run(
        ["docker", "ps", "--filter", f"name={CONTAINER_NAME}", "-q"],
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def ensure_container():
    """Start the container if it is not running."""
    import subprocess

    if check_container_running():
        return
    print(f"Container '{CONTAINER_NAME}' not running. Starting...")
    minimax_key = os.getenv("MINIMAX_API_KEY", "")
    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "-d",
            "-p",
            "7860:7860",
            "-e",
            f"MINIMAX_API_KEY={minimax_key}",
            "--name",
            CONTAINER_NAME,
            "dealroom-v3-test:latest",
        ]
    )
    import time

    print("Waiting 15s for container startup...")
    time.sleep(15)


def get_session(task="aligned", seed=None):
    """Get a fresh requests Session and initial observation."""
    import requests

    payload = {"task_id": task}
    if seed is not None:
        payload["seed"] = seed
    session = requests.Session()
    r = session.post(f"{BASE_URL}/reset", json=payload, timeout=30)
    r.raise_for_status()
    obs = r.json()
    session_id = obs.get("metadata", {}).get("session_id") or obs.get("session_id")
    return session, session_id


def make_action(
    session_id, action_type, target_ids, message="", documents=None, lookahead=None
):
    return {
        "metadata": {"session_id": session_id},
        "action_type": action_type,
        "target_ids": target_ids,
        "message": message,
        "documents": documents or [],
        "lookahead": lookahead,
    }


def step(session, session_id, action, timeout=30):
    """Execute a step and return the parsed result."""
    import requests

    r = session.post(f"{BASE_URL}/step", json=action, timeout=timeout)
    r.raise_for_status()
    return r.json()


def get_reward(result):
    """Extract reward from step result (single float)."""
    reward = result.get("reward")
    if reward is None:
        reward = result.get("observation", {}).get("reward")
    return float(reward) if reward is not None else None


def get_obs(result):
    """Extract observation dict from step result."""
    if isinstance(result, dict) and "observation" in result:
        return result["observation"]
    return result


def assert_near(value, target, tol=0.05, msg=None):
    import numpy as np

    diff = abs(float(value) - target)
    if diff > tol:
        raise AssertionError(
            (msg or f"Value {value} not near target {target} (diff={diff:.4f})")
        )


def assert_in_range(value, lo=0.0, hi=1.0, msg=None):
    v = float(value)
    if not (lo <= v <= hi):
        raise AssertionError((msg or f"Value {v} outside range [{lo}, {hi}]"))
