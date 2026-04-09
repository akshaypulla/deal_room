from fastapi.testclient import TestClient

from server.app import app


def test_state_requires_active_session():
    client = TestClient(app)

    response = client.get("/state")

    assert response.status_code == 400
    assert "Call /reset first" in response.json()["detail"]


def test_sessions_are_isolated_by_client_cookie():
    client_a = TestClient(app)
    client_b = TestClient(app)

    reset_a = client_a.post("/reset", json={"task_id": "aligned", "seed": 42})
    reset_b = client_b.post("/reset", json={"task_id": "hostile_acquisition", "seed": 99})

    assert reset_a.status_code == 200
    assert reset_b.status_code == 200

    state_a = client_a.get("/state")
    state_b = client_b.get("/state")

    assert state_a.status_code == 200
    assert state_b.status_code == 200
    assert state_a.json()["task_id"] == "aligned"
    assert state_b.json()["task_id"] == "hostile_acquisition"

    target_a = next(iter(reset_a.json()["stakeholders"]))
    step_a = client_a.post(
        "/step",
        json={
            "action_type": "direct_message",
            "target": target_a,
            "target_ids": [target_a],
            "message": "Checking that client A keeps its own session.",
        },
    )

    assert step_a.status_code == 200
    state_b_after = client_b.get("/state")
    assert state_b_after.status_code == 200
    assert state_b_after.json()["task_id"] == "hostile_acquisition"
