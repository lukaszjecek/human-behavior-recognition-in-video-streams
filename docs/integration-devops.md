# Integration and DevOps
## Author: [Łukasz Jęcek](https://github.com/lukaszjecek)
## Updated by: [Szymon Kaźmierczak](https://github.com/Szymon110903) - Sprint 3

[Back to README](../README.md)

###
This file is the module documentation template for Integration and DevOps.
This section should later describe Docker, compose setup, environment wiring, and integration flow between modules.
For now, it is a placeholder waiting for the module owner to complete it. 
###


## Docker Compose Orchestration

This section describes the Docker Compose setup for local development, covering the orchestration of API, database, and inference services.

### Architecture Overview

The project uses Docker Compose to orchestrate three services running on a private Docker network (`hbr-network`):

1. **Database (db)** - PostgreSQL 15 for persistent data storage
2. **API (api)** - FastAPI backend service providing REST endpoints
3. **Inference (inference)** - Model inference service for video processing

### Network Configuration

All services communicate through a private Docker bridge network named `hbr-network`. This isolates the application stack and allows services to communicate by container name rather than IP addresses.

```yaml
networks:
  hbr-network:
    driver: bridge
```

### Service Details

#### Database Service

- **Image**: `postgres:15-alpine`
- **Container Name**: `hbr_db`
- **Port**: 5432 (configurable via `DB_PORT`)
- **Network**: `hbr-network`
- **Volumes**: `db_data` (persistent PostgreSQL data)

**Health Check**: The database service includes a health check that verifies PostgreSQL is ready to accept connections. The API service depends on this health check to ensure the database is running before startup.

**Environment Variables**:
```
POSTGRES_USER=hbr_user (configurable via DB_USER)
POSTGRES_PASSWORD=hbr_password (configurable via DB_PASSWORD)
POSTGRES_DB=hbr_db (configurable via POSTGRES_DB)
```

#### API Service

- **Image**: Built from `docker/app/Dockerfile`
- **Container Name**: `hbr_api`
- **Port**: 8000 by default (override with `PORT` in `.env`)
- **Network**: `hbr-network`
- **Command**: `uvicorn src.app.app:create_app --host <HOST> --port <PORT> --factory` (values substituted at container startup from `HOST` and `PORT` env vars)
- **Depends On**: `db` service (with health check condition)

**Environment Variables**:
```
DB_HOST=db (DNS name on hbr-network)
DB_PORT=5432 (must match database service port)
DB_USER=hbr_user (must match POSTGRES_USER)
DB_PASSWORD=hbr_password (must match POSTGRES_PASSWORD)
DATABASE_URL=postgresql://hbr_user:hbr_password@db:5432/hbr_db
DATA_DIR=/app/data/raw
LOG_DIR=/app/data/logs
HOST=0.0.0.0
PORT=8000
PYTHONUNBUFFERED=1
```

#### Inference Service

- **Image**: Built from `docker/inference/Dockerfile`
- **Container Name**: `hbr_inference`
- **Network**: `hbr-network`
- **Volumes**: Raw data, subset data (read-only), logs, configs (read-only)
- **Entrypoint**: `docker/inference/entrypoint.sh`
- **Depends On**: `db` service (with health check condition)

The inference container uses a dedicated Dockerfile (`docker/inference/Dockerfile`) and a shell
entrypoint instead of a raw `CMD`. On boot the entrypoint:
1. Logs all wired environment variables for traceability.
2. Warns if `INFERENCE_CHECKPOINT` is not set.
3. Runs `python -m src.main` in startup-summary mode (non-fatal — container stays alive even on error).
4. Keeps the container alive with `exec tail -f /dev/null` so ad-hoc jobs can be dispatched with
   `docker compose run --rm inference ...`.

The inference service shares the same Docker image base and source volume as the API service.
The backend calls `run_inference(InferenceServiceRequest(...))` **in-process** — no HTTP hop is
required. `API_HOST` and `API_PORT` are wired as env vars for completeness and future result
push-back use cases.

**Environment Variables**:
```
DATA_DIR=/app/data/raw
LOG_DIR=/app/data/logs
INFERENCE_CHECKPOINT=   # path to .pth checkpoint inside container (required for model inference)
INFERENCE_CONFIG=configs/data_pipeline.yml  # runtime YAML config path (relative to /app)
INFERENCE_DEVICE=auto   # device override: auto | cpu | cuda | mps
INFERENCE_LOG_LEVEL=INFO # structured logging level (INFO, DEBUG, WARNING, ERROR)
INFERENCE_LOG_DETAIL=standard # logging detail: minimal | standard | verbose
API_HOST=api             # DNS name of the API container on hbr-network
API_PORT=8000            # API port (matches PORT env var)
```

### Inference Runtime Logging

The inference runtime emits **structured JSON logs** to stdout for lifecycle visibility and
integration troubleshooting. Each log line includes correlation fields:

- `event`: short event identifier (e.g., `inference_session_started`)
- `session_id`: per-run identifier used across all runtime logs
- `source_type` / `source_ref`: file or RTSP source metadata
  
Optional build metadata (`BUILD_SHA`, `IMAGE_TAG`, `APP_VERSION`) is included when set.

You can control verbosity via `INFERENCE_LOG_LEVEL` (or `LOG_LEVEL` as a fallback). Example:

```json
{"timestamp":"2026-05-20T12:07:16.120000+00:00","level":"INFO","logger":"src.inference.service","message":"Inference session started.","event":"inference_session_started","session_id":"8f1d7c1f7c8b4e2c9d6c0a2f44d3b6c2","source_type":"file","source_ref":"/app/data/raw/sample.mp4","checkpoint_path":"/app/data/logs/checkpoints/baseline.pth","config_path":"/app/configs/data_pipeline.yml","device_request":"auto"}
```

#### Log Detail Levels

Use `INFERENCE_LOG_DETAIL` (or `LOG_DETAIL` for API logs) to control how much metadata is emitted:

- `minimal` — only core correlation fields and error phase/type.
- `standard` — includes counts, timings, config paths, and request metadata.
- `verbose` — includes all fields plus full exception stack traces.

This allows reducing log volume when storage is tight while keeping the option to
increase detail for investigations.

To centralize logs, ship container stdout to your preferred log backend (e.g. Docker
logging driver, Fluent Bit, or a hosted log platform) — the JSON format is compatible
with most collectors.

### API Request Correlation

The API generates a per-request `X-Request-ID` (or honors the incoming header) and
returns it in the response headers. This ID is also logged as `session_id` in
structured logs, so one request can be traced end-to-end.

### Investigation Quick Steps

1. Find the `session_id` (or `X-Request-ID`) from the failing request.
2. Filter logs for that ID and look for `runtime_failed`, `inference_session_failed`,
   or `http_request_failed` to locate the failing phase.
3. If details are insufficient, temporarily switch `INFERENCE_LOG_DETAIL=verbose`
   to capture full stack traces and detailed runtime metrics.

### Environment Wiring

Environment variables are passed to services through the `compose.yaml` file. You can override defaults by creating a `.env` file in the repository root:

```bash
# .env
DB_USER=my_user
DB_PASSWORD=secure_password
POSTGRES_DB=my_db
DB_PORT=5432
DATABASE_URL=postgresql://my_user:secure_password@db:5432/my_db
```

**Note**: The `DATABASE_URL` follows the format: `postgresql://[user]:[password]@[host]:[port]/[database]`

When using compose, service names (like `db`, `api`) resolve to their container addresses automatically on the private network.

### Service Startup Order

Docker Compose ensures correct startup order through the `depends_on` directive with health checks:

1. **Database starts first** and performs health checks.
2. **API service waits** for the database health check to pass before starting.
3. **Inference service waits** for the database health check to pass before starting.

Both the API and inference services share the same `depends_on: db: condition: service_healthy`
configuration. The inference container does **not** wait for the API to be healthy; if in-process
calls to `run_inference` are made from the API, the caller must ensure the API has fully started
before dispatching inference work.

### Running the Stack

#### Quick Start

```bash
# Start all services
docker compose up --build

# Or in background
docker compose up -d --build
```

#### Development Workflow

```bash
# View logs for all services
docker compose logs -f

# View logs for specific service
docker compose logs -f api
docker compose logs -f db

# Execute command in running container
docker compose exec api bash
docker compose exec db psql -U hbr_user -d hbr_db

# Stop services
docker compose down

# Stop and remove volumes (careful!)
docker compose down -v
```

#### Custom Environment

```bash
# Create .env with custom settings
echo "DB_PASSWORD=my_secure_password" > .env

# Start with custom environment
docker compose up --build
```

### Service Communication

Services can communicate by container name on the `hbr-network`:

| Direction | Host | Port | Notes |
|-----------|------|------|-------|
| API → Database | `db` | 5432 | Resolves to `hbr_db` container |
| Inference → API | `api` | `API_PORT` (default 8000) | Resolves to `hbr_api` container; `API_HOST=api` is wired in compose |
| External client → API | `localhost` | `PORT` (default 8000) | Published port on the host |

Within the compose stack the inference container reaches the API at `http://api:8000`. This
resolution is guaranteed by the shared `hbr-network` bridge and the `API_HOST` / `API_PORT`
environment variables wired in `compose.yaml`.

### Running One-Off Inference Jobs

The inference container stays alive after boot. Use `docker compose run --rm` to dispatch a
one-off MP4 inference job without restarting the stack:

```bash
docker compose run --rm \
  -e INFERENCE_CHECKPOINT=/app/data/logs/checkpoints/baseline_epoch_50.pth \
  inference \
  python -m src.main \
    --input /app/data/raw/walking/sample.mp4 \
    --checkpoint /app/data/logs/checkpoints/baseline_epoch_50.pth \
    --config /app/configs/data_pipeline.yml \
    --output /app/data/logs/actions.json \
    --device auto
```

If `INFERENCE_CHECKPOINT` is set in `.env`, the `-e` override above can be omitted.

To confirm the inference container can reach the API over `hbr-network`:

```bash
docker compose exec inference curl -sf http://api:8000/health
```

### Volumes

The compose setup uses named volumes for persistent data:

- **db_data**: PostgreSQL data directory (persists across container restarts)
- **./data/logs**: Application logs (mounted from host)
- **./data/raw**: Raw video data (mounted from host)

### Settings Integration

The FastAPI application settings are managed by `src/app/core/settings.py` using Pydantic V2 and environment variables. The Settings class loads configuration from:

1. Environment variables (highest priority)
2. `.env` file (if present)
3. Built-in defaults (lowest priority)

**Available Settings**:
```python
app_name: str = "HBR Backend"
app_version: str = "0.1.0"
debug: bool = False
host: str = "0.0.0.0"
port: int = 8000
db_host: str = "localhost"
db_port: int = 5432
db_user: str = "hbr_user"
db_password: str = "hbr_password"
postgres_db: str = "hbr_db"
database_url: str = "postgresql://hbr_user:hbr_password@localhost:5432/hbr_db"
data_dir: Path = "/app/data/raw"
log_dir: Path = "/app/data/logs"
```

### Troubleshooting

**Problem**: API fails to connect to database
- **Solution**: Check that `db` service is healthy (`docker compose ps`). View logs: `docker compose logs db`

**Problem**: Port 8000 already in use
- **Solution**: Set `PORT=<free_port>` in `.env` (e.g., `PORT=8001`), then recreate containers with `docker compose down && docker compose up`. The Compose configuration automatically uses the same port for both the published port mapping and uvicorn startup.

**Problem**: Permission denied on `/app/data` volumes
- **Solution**: Ensure directories exist: `mkdir -p data/raw data/logs data/subset`

**Problem**: Database initialization fails
- **Solution**: Remove volume and restart: `docker compose down -v && docker compose up --build`
