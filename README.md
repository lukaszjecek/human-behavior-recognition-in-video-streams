# Human Behavior Recognition in Video Streams

End-to-end software engineering project focused on human behavior recognition from video sequences, with emphasis on temporal modeling, reproducible data handling, modular inference logic, and containerized development.

## Quick Start

### Requirements
- Docker
- local clone of this repository
- dataset subset in `./data/raw`

### Run

Start all services (API, database, and inference):

```bash
docker compose up --build
```

On Windows PowerShell, create the folders manually if they do not already exist:

```powershell
mkdir data\raw
mkdir data\logs
mkdir data\subset
docker compose up --build
```

### Expected Output

After startup, the services will initialize:

- **API Service** - FastAPI backend running on `http://localhost:8000`
  - Health endpoint: `GET http://localhost:8000/health`
  - API documentation: `http://localhost:8000/docs`

- **Database Service** - PostgreSQL running on `localhost:5432`
  - Credentials: `hbr_user` / `hbr_password`
  - Database: `hbr_db`

- **Inference Service** - Model inference service
  - Logs: `./data/logs/startup_summary.json`
  - If dataset subset is mounted correctly, logs include discovered video files and classes

- **Frontend Service** - React (Vite) frontend application
  - URL: `http://localhost:5173`

## Backend (Sprint 3)

### API Development

The FastAPI backend is located in `src/app/`:

```
src/app/
├── __init__.py
├── app.py           # Application factory
├── api/
│   ├── routes.py    # API endpoints
│   └── websocket.py # WebSocket handlers
├── core/
│   └── settings.py  # Configuration (Pydantic V2)
└── endpoints/
    └── health.py    # Health check endpoint
```

### Running Backend Locally

```bash
# Start all services
docker compose up --build

# Or start the API service and its declared dependencies (for example, the database)
docker compose up api --build
```

### Testing API Endpoints

```bash
# Health check
curl http://localhost:8000/health

# API documentation (Swagger UI)
open http://localhost:8000/docs

# API documentation (ReDoc)
open http://localhost:8000/redoc
```

### Inference Session REST Endpoints

The API now exposes endpoints for managing offline inference sessions asynchronously:
- `POST /api/sessions/` - Starts a new inference session in a background thread.
- `GET /api/sessions/{session_id}` - Retrieves the current status (`pending`, `running`, `completed`, `failed`, `stopped`).
- `POST /api/sessions/{session_id}/stop` - Gracefully interrupts and stops an ongoing running session.

These endpoints use `asyncio.to_thread` for non-blocking execution and native `threading.Event` triggers down to the underlying inference loops to allow safe and clean interruptions without blocking the FastAPI event loop.

### Configuration

Backend configuration is managed through environment variables in `src/app/core/settings.py`:

- `HOST` - API host (default: `0.0.0.0`)
- `PORT` - API port (default: `8000`)
- `DB_HOST` - Database host (default: `db` in Docker, `localhost` locally)
- `DB_PORT` - Database port (default: `5432`)
- `DB_USER` - Database user (default: `hbr_user`)
- `DB_PASSWORD` - Database password (default: `hbr_password`)
- `DATABASE_URL` - Full connection string (auto-generated if not provided)
- `DEBUG` - Debug mode (default: `False`)

Create a `.env` file to override defaults:

```bash
# .env
DB_USER=my_user
DB_PASSWORD=secure_password
DEBUG=true
```

For detailed integration and DevOps documentation, see [Integration and DevOps](docs/integration-devops.md).

## Sprint 2 MP4 to JSON CLI

Run action inference from a single MP4 file and write a JSON event log:

```powershell
python -m src.main `
  --input data\raw\walking\sample.mp4 `
  --checkpoint data\logs\checkpoints\baseline_epoch_10.pth `
  --config configs\data_pipeline.yml `
  --output data\logs\actions.json `
  --device auto
```

Run the same flow in Docker:

```bash
docker compose run --rm inference python -m src.main \
  --input /app/data/raw/walking/sample.mp4 \
  --checkpoint /app/data/logs/checkpoints/baseline_epoch_10.pth \
  --config /app/configs/data_pipeline.yml \
  --output /app/data/logs/actions.json
```

Inference mode requires `--input` and `--checkpoint` together. Without these arguments,
`src.main` keeps the startup summary behavior and writes `startup_summary.json`.

## Documentation
- [Data Pipeline](docs/data-pipeline.md)
- [Inference](docs/inference.md)
- [ML Baseline](docs/ml-baseline.md)
- [Backend](docs/backend.md)
- [Frontend](docs/frontend.md)
- [Integration and DevOps](docs/integration-devops.md)
- [Contributing](docs/contributing.md)
- [CI Workflows](docs/ci-workflows.md)

