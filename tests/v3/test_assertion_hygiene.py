#!/usr/bin/env python3
"""Guard against print-only critical v3 tests."""

import ast
from pathlib import Path


CRITICAL_TEST_FILES = [
    "test_02_reward_integrity.py",
    "test_03_causal_inference.py",
    "test_04_cvar_veto.py",
    "test_09_full_episode_e2e.py",
]


def _has_failure_signal(node: ast.FunctionDef) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Assert):
            return True
        if isinstance(child, ast.Raise):
            return True
        if isinstance(child, ast.Call):
            func = child.func
            if isinstance(func, ast.Attribute) and func.attr in {"fail", "skip"}:
                if isinstance(func.value, ast.Name) and func.value.id == "pytest":
                    return func.attr == "fail"
    return False


def test_critical_v3_tests_have_assertions_or_explicit_failures():
    base = Path(__file__).parent
    offenders = []

    for filename in CRITICAL_TEST_FILES:
        tree = ast.parse((base / filename).read_text())
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                if not _has_failure_signal(node):
                    offenders.append(f"{filename}::{node.name}")

    assert not offenders, (
        "Critical v3 tests must not be print-only. Missing assert/raise/pytest.fail: "
        + ", ".join(offenders)
    )
