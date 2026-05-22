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


def test_database_url_auto_generated_from_fields(monkeypatch):
    """database_url is built from individual DB fields when not explicitly set."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("DB_HOST", raising=False)
    monkeypatch.delenv("DB_PORT", raising=False)
    monkeypatch.delenv("DB_USER", raising=False)
    monkeypatch.delenv("DB_PASSWORD", raising=False)
    monkeypatch.delenv("POSTGRES_DB", raising=False)
    s = Settings(
        _env_file=None,
        db_user="alice",
        db_password="secret",
        db_host="pghost",
        db_port=5433,
        postgres_db="mydb",
    )
    assert s.database_url == "postgresql://alice:secret@pghost:5433/mydb"


def test_database_url_explicit_overrides_auto_generation(monkeypatch):
    """Explicit DATABASE_URL takes precedence over individual DB fields."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("DB_HOST", raising=False)
    monkeypatch.delenv("DB_PORT", raising=False)
    monkeypatch.delenv("DB_USER", raising=False)
    monkeypatch.delenv("DB_PASSWORD", raising=False)
    monkeypatch.delenv("POSTGRES_DB", raising=False)
    explicit_url = "postgresql://other:pass@otherhost:5432/otherdb"
    s = Settings(
        _env_file=None,
        db_user="alice",
        db_password="secret",
        db_host="pghost",
        db_port=5433,
        postgres_db="mydb",
        database_url=explicit_url,
    )
    assert s.database_url == explicit_url


def test_default_settings_db_fields(monkeypatch):
    """Default DB settings are present and consistent."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("DB_HOST", raising=False)
    monkeypatch.delenv("DB_PORT", raising=False)
    monkeypatch.delenv("DB_USER", raising=False)
    monkeypatch.delenv("DB_PASSWORD", raising=False)
    monkeypatch.delenv("POSTGRES_DB", raising=False)
    s = Settings(_env_file=None)
    assert s.db_host == "localhost"
    assert s.db_port == 5432
    assert s.db_user == "hbr_user"
    assert s.db_password == "hbr_password"
    assert s.postgres_db == "hbr_db"
    assert s.database_url == "postgresql://hbr_user:hbr_password@localhost:5432/hbr_db"
