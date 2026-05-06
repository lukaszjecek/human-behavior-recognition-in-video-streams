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
- **Port**: 8000 (configurable via PORT)
- **Network**: `hbr-network`
- **Command**: `uvicorn src.app.app:create_app --host 0.0.0.0 --port 8000 --factory`
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

- **Image**: Built from `docker/app/Dockerfile`
- **Container Name**: `hbr_inference`
- **Network**: `hbr-network`
- **Volumes**: Raw data, subset data (read-only), logs
- **Command**: `python -m src.main && tail -f /dev/null`

The inference service runs independently and can process video files using the trained model.

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

1. **Database starts first** and performs health checks
2. **API service waits** for database health check to pass before starting
3. **Inference service** starts independently (no dependencies)

This ensures the database is ready to accept connections before the API tries to connect.

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

- **From API to Database**: Use host `db` (resolves to `hbr_db` container)
- **From Inference to API**: Use host `api` (resolves to `hbr_api` container)
- **From external client**: Use `localhost:8000` for API

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
- **Solution**: Changing `PORT` in `.env` alone will not fix this in the current Docker Compose setup. Stop the conflicting process or run `docker compose down` before starting. If you need a different port, update the Compose port mapping and the API startup port together.

**Problem**: Permission denied on `/app/data` volumes
- **Solution**: Ensure directories exist: `mkdir -p data/raw data/logs data/subset`

**Problem**: Database initialization fails
- **Solution**: Remove volume and restart: `docker compose down -v && docker compose up --build`
