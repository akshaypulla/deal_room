from fastapi.testclient import TestClient

from server.app import app


def test_root_redirects_to_web():
    client = TestClient(app)
    response = client.get("/", follow_redirects=False)
    assert response.status_code in {302, 307}
    assert response.headers["location"] == "/web"


def test_web_page_exposes_playground_and_custom_tabs():
    client = TestClient(app)
    response = client.get("/web")
    assert response.status_code == 200
    body = response.text
    assert "Playground" in body
    assert "Custom" in body
    assert "deal-room" in body.lower()
