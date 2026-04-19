from fastapi.testclient import TestClient

from server.app import app


def test_root_redirects_to_web():
    client = TestClient(app, follow_redirects=False)
    response = client.get("/")
    assert response.status_code == 302
    assert response.headers["location"] == "/web"


def test_web_page_exposes_wrapper_without_redirect():
    client = TestClient(app)
    response = client.get("/web")
    assert response.status_code == 200
    body = response.text
    assert "iframe" in body.lower()
    assert "/__gradio_ui__/" in body
    assert "dealroom" in body.lower()


def test_web_slash_page_redirects_to_web():
    client = TestClient(app, follow_redirects=False)
    response = client.get("/web/")
    assert response.status_code == 302
    assert response.headers["location"] == "/web"


def test_ui_blocked_direct_access():
    client = TestClient(app, follow_redirects=False)
    response = client.get("/ui/")
    assert response.status_code == 302
    assert response.headers["location"] == "/web"


def test_health_endpoint_still_works():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert "deal-room" in response.text
