from fastapi.testclient import TestClient

from server.app import app


def test_root_redirects_to_web():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "iframe" in response.text.lower()
    assert "/ui/" in response.text


def test_web_page_exposes_wrapper_without_redirect():
    client = TestClient(app)
    response = client.get("/web")
    assert response.status_code == 200
    body = response.text
    assert "iframe" in body.lower()
    assert "/ui/" in body
    assert "dealroom" in body.lower()


def test_web_slash_page_exposes_same_wrapper():
    client = TestClient(app)
    response = client.get("/web/")
    assert response.status_code == 200
    body = response.text
    assert "iframe" in body.lower()
    assert "/ui/" in body


def test_ui_shell_contains_custom_round_table_labels():
    client = TestClient(app)
    response = client.get("/ui/")
    assert response.status_code == 200
    body = response.text
    assert "Playground" in body
    assert "Custom" in body
    assert "Quick Start" in body
    assert "Raw JSON response" in body
    assert "Move Type" in body
    assert "Get state" in body
    assert "DealRoom Custom Lab" in body
    assert "Simple Round" in body
    assert "Medium Round" in body
    assert "Hard Round" in body
