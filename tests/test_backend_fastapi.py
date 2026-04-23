"""Additional tests for FastAPI backend skeleton.

Covers app settings propagation, global exception handler, and websocket echo.
"""
from fastapi.testclient import TestClient

from src.app.app import create_app
from src.app.core.settings import Settings


def test_create_app_with_settings():
    settings = Settings(app_name="TestApp", app_version="9.9.9", debug=True)
    app = create_app(settings=settings)
    assert app.title == "TestApp"
    assert app.version == "9.9.9"


def test_global_exception_handler():
    app = create_app()

    # Add a route that raises an exception to exercise the global handler
    def _boom():
        raise RuntimeError("boom")

    app.add_api_route("/boom", _boom, methods=["GET"])
    client = TestClient(app, raise_server_exceptions=False)

    r = client.get("/boom") 
    assert r.status_code == 500
    assert r.json().get("detail") == "Internal server error"


def test_websocket_echo():
    app = create_app()
    client = TestClient(app)

    with client.websocket_connect("/ws/echo") as ws:
        ws.send_text("hello")
        msg = ws.receive_text()
        assert msg == "echo: hello"
