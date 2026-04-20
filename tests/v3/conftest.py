"""
Shared test configuration and utilities for DealRoom v3 test suite.
Loads environment variables from .env file and provides common helpers.
"""

import os
import sys
from pathlib import Path

# Attempt to load .env file if it exists
_dotenv_path = Path(__file__).parent.parent.parent / ".env"
if _dotenv_path.exists():
    try:
        from dotenv import load_dotenv

        load_dotenv(_dotenv_path)
    except ImportError:
        pass  # dotenv not installed — rely on exported env vars

BASE_URL = os.getenv("DEALROOM_BASE_URL", "http://127.0.0.1:7860")
CONTAINER_NAME = os.getenv("DEALROOM_CONTAINER_NAME", "dealroom-v3-test")

REQUIRED_ENV_VARS = ["MINIMAX_API_KEY", "OPENAI_API_KEY"]


def validate_api_keys():
    """Fail fast if required API keys are not set."""
    missing = [v for v in REQUIRED_ENV_VARS if not os.getenv(v)]
    if missing:
        print("=" * 62)
        print("ERROR: Required environment variables are not set:")
        for v in missing:
            print(f"  - {v}")
        print()
        print("Copy .env.example to .env and fill in your keys:")
        print("  cp .env.example .env")
        print()
        print("Or export them directly:")
        for v in missing:
            print(f"  export {v}=your_key_here")
        print("=" * 62)
        sys.exit(1)


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
    openai_key = os.getenv("OPENAI_API_KEY", "")
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
            "-e",
            f"OPENAI_API_KEY={openai_key}",
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

    payload = {"task": task}
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


def step(session, session_id, action, timeout=60):
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
