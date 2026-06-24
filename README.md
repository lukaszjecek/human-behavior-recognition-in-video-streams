# Human Behavior Recognition in Video Streams

End-to-end human behavior recognition system for video streams and MP4 clips. The project combines a FastAPI backend, React operator dashboard, PostgreSQL event history, reusable inference pipeline, Docker Compose orchestration, and documentation for demo, benchmark, and module workflows.

This documentation pass is a draft while issue #130 is still pending. The final checkpoint release link and final performance numbers are not available yet.

## Quick Start

Requirements:

- Docker Desktop or Docker Engine with Docker Compose
- Local clone of this repository
- Local data folders used by Compose

Create the expected local folders when starting from a clean clone:

```powershell
mkdir data\raw
mkdir data\logs
mkdir data\uploads
mkdir data\subset
mkdir data\logs\checkpoints
```

Start the CPU-safe stack:

```bash
docker compose up --build
```

For local NVIDIA GPU testing, layer the optional override:

```bash
docker compose -f compose.yaml -f compose.gpu.yaml up --build
```

CI and the default local stack use `compose.yaml`, so GPU access is optional.

## Local Services

- [Frontend](http://localhost:5173)
- [Swagger UI / OpenAPI docs](http://localhost:8000/docs)
- [ReDoc API docs](http://localhost:8000/redoc)
- [API health](http://localhost:8000/health)

If `.env` overrides `PORT`, use that API port instead of `8000`.

## Documentation Index

- [Architecture](docs/architecture.md) - Compose topology, Mermaid diagram, API/WebSocket surface, demo paths, checkpoint status, and limitations.
- [Final Demo Runbook](docs/final-demo-runbook.md) - Infrastructure smoke verification and final MP4 fallback runbook.
- [Performance Benchmark](docs/performance-benchmark.md) - Repeatable MP4 benchmark path and current non-final smoke measurements.
- [Backend](docs/backend.md) - API contracts, WebSocket contracts, event schema, persistence, and logging details.
- [Frontend](docs/frontend.md) - React dashboard, live camera mode, MP4 session mode, and operator workflow.
- [Integration and DevOps](docs/integration-devops.md) - Docker Compose wiring, logs, environment variables, and smoke test flow.
- [Inference](docs/inference.md) - MP4 inference CLI, runtime service API, checkpoint metadata, tracking, and context module.
- [Data Pipeline](docs/data-pipeline.md) - Dataset preparation and visualization utilities.
- [ML Baseline](docs/ml-baseline.md) - Baseline training, validation, checkpoint output, and selected classes.
- [CI Workflows](docs/ci-workflows.md) - Automated validation paths.
- [Contributing](docs/contributing.md) - Contributor workflow.

## Demo Paths

Primary operator path:

- Live camera mode in the [Frontend](docs/frontend.md) streams browser camera frames to `WS /ws/camera`.
- The backend returns live detection/status messages and broadcasts generated events through `WS /ws/live`.
- Live camera behavior depends on a valid local checkpoint and config path. Treat final verification as pending until issue #130 publishes the final checkpoint.

Fallback demo path:

- MP4 session mode uploads an operator-selected `.mp4` to `POST /api/videos/upload`.
- The frontend starts processing with `POST /api/sessions/`, polls `GET /api/sessions/{session_id}`, and then reads `GET /api/events/sessions/{session_id}`.
- The final runbook also supports direct MP4-to-JSON fallback inference with `scripts/run_mp4_inference.ps1`.

Infrastructure smoke path:

```powershell
.\scripts\final_demo_smoke.ps1
```

The smoke script verifies API health, session creation, WebSocket event flow, database persistence, and logs using a generated dummy checkpoint and dummy MP4. It is not final model validation.

## Checkpoint Status

Place model checkpoints under:

```text
data/logs/checkpoints/<checkpoint>.pth
```

Inside Docker, the same file is visible as:

```text
/app/data/logs/checkpoints/<checkpoint>.pth
```

The Compose inference service can read `INFERENCE_CHECKPOINT`, and the demo/benchmark scripts accept checkpoint paths through their `-Checkpoint` arguments. The frontend defaults to `data/logs/checkpoints/baseline_epoch_50.pth`, but that checkpoint is only suitable for local smoke work and must not be presented as final.

The final GitHub Release checkpoint link remains blocked by issue #130.

## Performance Status

See [Performance Benchmark](docs/performance-benchmark.md) for the repeatable benchmark command and current smoke measurements. The current numbers use `baseline_epoch_50.pth` and are non-final.

Final performance must be re-run after issue #130 publishes the final checkpoint. Do not claim final 15 FPS or <= 2 s compliance unless the final benchmark proves it.

## Known Limitations

- Final checkpoint and final release link are pending issue #130.
- Final performance is pending a benchmark re-run on the final checkpoint.
- Live camera mode is implemented, but final demo verification depends on the final checkpoint and target runtime conditions.
- CPU mode may be slower than GPU mode.
- Smoke benchmarks are not accuracy, model-quality, or final latency validation.
