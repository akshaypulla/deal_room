#!/usr/bin/env python3
"""Smoke-test the live DealRoom container routes."""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request


BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:7860"
TASKS = ("aligned", "conflicted", "hostile_acquisition")
VALID_STAGES = {"evaluation", "negotiation", "legal_review", "final_approval", "closed"}
VALID_MOMENTUM = {"progressing", "stalling", "critical"}


def request(
    path: str,
    method: str = "GET",
    payload: dict | None = None,
    expected: int = 200,
    parse_json: bool = True,
):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(f"{BASE_URL}{path}", data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode("utf-8")
            status = resp.status
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        status = exc.code
    if status != expected:
        raise SystemExit(f"{method} {path} expected {expected}, got {status}: {body}")
    if not parse_json:
        return body
    return json.loads(body) if body else None


def assert_stage_payload(observation: dict) -> None:
    assert observation["deal_stage"] in VALID_STAGES
    assert observation["deal_momentum"] in VALID_MOMENTUM
    assert isinstance(observation["active_blockers"], list)
    assert isinstance(observation["known_constraints"], list)
    assert isinstance(observation["requested_artifacts"], dict)
    assert isinstance(observation["stakeholders"], dict)


def assert_state_payload(state: dict, expected_task: str) -> None:
    assert state["task_id"] == expected_task
    assert 2 <= len(state["stakeholders"]) <= 4
    assert len(state["relationship_edges"]) <= 2
    assert len(state["hidden_constraints"]) <= 2
    assert state["deal_stage"] in VALID_STAGES


def step(payload: dict, expected: int = 200) -> dict:
    return request("/step", "POST", payload, expected=expected)


def run_task_flow(task_id: str, seed: int) -> None:
    obs = request("/reset", "POST", {"task_id": task_id, "seed": seed})
    assert 2 <= len(obs["stakeholders"]) <= 4
    assert_stage_payload(obs)

    state = request("/state")
    assert_state_payload(state, task_id)
    roster = list(obs["stakeholders"])
    first = roster[0]
    second = roster[1] if len(roster) > 1 else roster[0]

    direct = step(
        {
            "action_type": "direct_message",
            "target": first,
            "target_ids": [first],
            "message": "Help me understand the real approval constraint we need to respect here.",
        }
    )
    assert set(direct) == {"observation", "reward", "done", "info"}
    assert 0.0 <= float(direct["reward"]) <= 1.0
    assert isinstance(direct["done"], bool)
    assert_stage_payload(direct["observation"])

    backchannel = step(
        {
            "action_type": "backchannel",
            "target_ids": [first, second],
            "message": "Quiet alignment check: what internal concern is most likely to slow approval?",
            "channel": "backchannel",
        }
    )
    assert_stage_payload(backchannel["observation"])

    document = step(
        {
            "action_type": "send_document",
            "target": first,
            "target_ids": [first],
            "message": "Sharing a concrete artifact to reduce review friction.",
            "documents": [{"type": "roi_model", "specificity": "high"}],
        }
    )
    assert_stage_payload(document["observation"])

    proposal = step(
        {
            "action_type": "group_proposal",
            "target": "all",
            "message": "Proposing terms that stay within implementation capacity and reduce adoption risk.",
            "proposed_terms": {
                "price": 90000,
                "timeline_weeks": 10,
                "support_level": "priority",
            },
        }
    )
    assert_stage_payload(proposal["observation"])

    malformed = step(
        {
            "action_type": "direct_message",
            "target": "unknown_target",
            "message": "Testing malformed target handling.",
        }
    )
    assert malformed["info"]["last_action_error"] in {"unknown_target:unknown_target", None}
    assert_stage_payload(malformed["observation"])

    unresolved = step(
        {
            "action_type": "exec_escalation",
            "target": "all",
            "message": "Requesting immediate approval before all blockers are cleared.",
        }
    )
    assert_stage_payload(unresolved["observation"])

    state_after_steps = request("/state")
    assert_state_payload(state_after_steps, task_id)


def main():
    web_page = request("/web", parse_json=False)
    assert "iframe" in web_page.lower()
    assert "/ui/" in web_page
    assert "dealroom" in web_page.lower()

    web_page_slash = request("/web/", parse_json=False)
    assert "iframe" in web_page_slash.lower()
    assert "/ui/" in web_page_slash

    ui_page = request("/ui/", parse_json=False)
    assert "Playground" in ui_page
    assert "Custom" in ui_page

    health = request("/health")
    assert health["status"] == "ok"
    assert health["service"] == "deal-room"
    assert set(health["tasks"]) == set(TASKS)

    metadata = request("/metadata")
    assert metadata["name"] == "deal-room"
    assert metadata["version"] == "1.0.0"
    assert set(metadata["tasks"]) == set(TASKS)

    initial_state = request("/state")
    assert isinstance(initial_state["stakeholders"], dict)
    assert initial_state["deal_stage"] in VALID_STAGES

    for task_id, seed in zip(TASKS, (42, 7, 99), strict=True):
        run_task_flow(task_id, seed)

    invalid = request("/reset", "POST", {"task_id": "not_real"}, expected=400)
    assert "Unknown task_id" in invalid["detail"]

    print("container route smoke tests passed")


if __name__ == "__main__":
    main()
