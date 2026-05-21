## Linked Issue
Closes #82 

# Sprint 3 Implementation Summary: Database Persistence

## 1. Scope of Work Delivered
* **Database Schema & Models**: Implemented a hybrid storage strategy mapping `EventPayload` to relational tables.
* **Pipeline Integration**: Linked the background inference worker thread with a thread-safe database session.
* **API Development**: Created REST endpoints for paginated and filtered historical query lookups.
* **Testing & Verification**: Built a fast automated test suite using an in-memory database and verified Docker state connectivity.
* **Linter & Security Improvements**: Resolved all outstanding linter and security-related issues.

## 2. Codebase Changelog

### Database Layer
* **`requirements.txt`**: Added `SQLAlchemy==2.0.28` and `psycopg2-binary==2.9.9`.
* **`src/app/db/models.py`**: Defined `DBEvent` model (`events` table) with indexed columns (`event_id`, `camera_id`, `event_type`) and a `JSON` column for the raw `payload`.
* **`src/app/db/session.py`**:
  * Implemented lazy initialization for `SessionLocal`, data engine setup, `init_db()` table creator, and the `get_db()` FastAPI dependency.
  * Added type annotations for `*args: Any` and `**kwargs: Any` in `LazySessionmaker.__call__` and added a docstring.
  * Redacted connection passwords from logging output by formatting connection strings using SQLAlchemy's `make_url().render_as_string(hide_password=True)`.
* **`src/app/db/repository.py`**:
  * Added `save_event` (with safe write-path validation and failure logging), `get_events` (with pagination/filtering queries), and `get_event_by_id`.
  * Removed unused exception object `e` from database rollback handler to fix `F841` linting.

### Pipeline & Application Integration
* **`src/app/services/session_manager.py`**: Created the `handle_pipeline_event` orchestrator within `_run_session_task`. Incoming payloads are now split concurrently into the `websocket_manager.broadcast_sync` pipeline and a thread-safe DB context (`with SessionLocal()`).
* **`src/app/app.py`**: Registered `init_db()` execution within the FastAPI application `startup` event handler (`@app.on_event("startup")`).
* **`src/inference/offline_runtime.py`**:
  * Constrained bare Pillow-related exception handling block inside the animated image reader to catch only `OSError` and `ValueError`, logging the fallback to static image reading rather than swallowing all exceptions silently.

### API Endpoints & Routes
* **`src/app/endpoints/events.py`**:
  * Created query routes: `GET /api/events/` (supports optional parameters: `event_type`, `camera_id`, `limit`, `offset`) and `GET /api/events/{event_id}`.
  * Replaced detailed exception tracebacks in 500 error responses with generic API parsing messages to prevent internal detail leaks to clients.
* **`src/app/api/routes_impl.py`**: Registered `events_router` under the `/events` prefix.
* **`docs/backend.md`**: Appended Database Persistence Layer architecture and schema reference documentation.

### Verification
* **`tests/app/test_persistence.py`**:
  * Added automated unit and integration tests utilizing isolated `sqlite:///:memory:` configurations.
  * Cleaned up unused SQLAlchemy engine and sessionmaker imports.
* **`tests/test_backend_fastapi.py`**:
  * Isolated database settings tests from host-defined database env variables via `monkeypatch.delenv` and `_env_file=None`.
* **Manual Verification**: Successfully mapped Docker port configurations and verified live PostgreSQL persistence state using the **Database Client** extension in VS Code.
