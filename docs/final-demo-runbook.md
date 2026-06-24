# Final Demo Runbook

[Back to README](../README.md)

## Purpose

This runbook covers the final demo and operational verification flow for the Docker Compose stack. It separates infrastructure smoke verification, which uses the existing dummy smoke path, from final MP4 inference, which requires a real model checkpoint.

The scope is operational demo readiness only. It does not train a model, benchmark performance, or finalize checkpoint release information.

## Requirements

- Docker Desktop or Docker Engine with Docker Compose
- PowerShell
- A local clone of this repository
- For final MP4 inference only: a real `.pth` checkpoint and an MP4 demo video

## Clean Clone Setup

From the repository root, create the local data folders used by Compose:

```powershell
mkdir data\raw
mkdir data\logs
mkdir data\uploads
mkdir data\subset
mkdir data\logs\checkpoints
```

Put raw or demo MP4 files under `data\raw\`. A useful convention is:

```text
data\raw\<video>.mp4
```

Put model checkpoints under:

```text
data\logs\checkpoints\<checkpoint>.pth
```

The default runtime config is:

```text
configs\data_pipeline.yml
```

## Final Checkpoint Setup

The final model checkpoint is expected to come from issue #130 or a GitHub Release asset once it is available. The current final model scope is expected to use a 21-class checkpoint, but the checkpoint may not exist yet.

Smoke verification can still run before the final checkpoint exists. The smoke path creates a dummy MP4 and dummy checkpoint, then verifies the API, WebSocket event flow, database persistence, and log/event plumbing. It is not final model validation.

## Start The Stack

Start the CPU-safe default stack:

```powershell
docker compose up --build
```

For a background run:

```powershell
docker compose up -d --build
```

The repository also includes an optional NVIDIA GPU override for local machines with compatible Docker GPU support:

```powershell
docker compose -f compose.yaml -f compose.gpu.yaml up --build
```

CI and the default demo smoke path use `compose.yaml` and do not require GPU access.

## Infrastructure Smoke Verification

Use the PowerShell wrapper to create required folders, start the stack, wait for API health, and run the existing integration smoke test inside the API container:

```powershell
.\scripts\final_demo_smoke.ps1
```

Useful options:

```powershell
.\scripts\final_demo_smoke.ps1 -SkipBuild
.\scripts\final_demo_smoke.ps1 -NoStart
.\scripts\final_demo_smoke.ps1 -ApiUrl http://localhost:8000
```

This verifies:

- API health at `GET /health`
- session creation through the REST API
- in-process inference using a dummy checkpoint
- WebSocket event streaming at `/ws/live`
- event persistence in PostgreSQL
- generated backend, inference, and audit logs under `data\logs`

The dummy smoke path requires session completion, at least one persisted `DETECTION` event, and WebSocket event reception. `ALERT` events are optional for this path: the script prints a warning when no alert is produced, because alert generation depends on danger-label configuration and state-machine persistence.

It does not verify the final model checkpoint or final alert behavior. Check final alerts separately with a checkpoint, config, and video that trigger a configured danger label.

## Final MP4 Inference Demo

After the real checkpoint is available, run fallback MP4 inference with:

```powershell
.\scripts\run_mp4_inference.ps1 `
  -Input data\raw\<video>.mp4 `
  -Checkpoint data\logs\checkpoints\<checkpoint>.pth
```

Optional parameters:

```powershell
.\scripts\run_mp4_inference.ps1 `
  -Input data\raw\<video>.mp4 `
  -Checkpoint data\logs\checkpoints\<checkpoint>.pth `
  -Config configs\data_pipeline.yml `
  -Output data\logs\actions.json `
  -Device auto
```

The script validates that the input video and checkpoint exist on the host, creates the output directory, converts repository paths to container paths, and runs:

```powershell
docker compose run --rm inference python -m src.main ...
```

The output is a JSON event log, by default:

```text
data\logs\actions.json
```

## API And Frontend Locations

- API health: [http://localhost:8000/health](http://localhost:8000/health)
- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- Frontend: [http://localhost:5173](http://localhost:5173)

If `.env` overrides `PORT`, use that port for API URLs.

## Logs And Event History

The Compose stack writes backend, inference, and audit logs under:

```text
data\logs
```

Useful files:

- `data\logs\backend.log`
- `data\logs\inference.log`
- `data\logs\audit.log`
- `data\logs\actions.json` after MP4 fallback inference

After the smoke script passes, inspect persisted event counts with:

```powershell
docker compose exec db psql -U hbr_user -d hbr_db -c "SELECT count(*), event_type FROM events GROUP BY event_type;"
```

You can also inspect event history through the API:

```powershell
curl.exe http://localhost:8000/api/events/
```

## Troubleshooting

### Docker Not Running

Start Docker Desktop and retry:

```powershell
docker compose version
```

### Port 8000 Busy

Set a free API port in `.env`, then recreate the stack:

```text
PORT=8001
```

```powershell
docker compose down
docker compose up -d --build
.\scripts\final_demo_smoke.ps1 -NoStart -ApiUrl http://localhost:8001
```

### Missing Checkpoint

The smoke script does not need the final checkpoint. Final MP4 inference does. Put the checkpoint under `data\logs\checkpoints\` and pass it with `-Checkpoint`.

### Missing Input Video

Put the MP4 under `data\raw\` and pass it with `-Input`. The MP4 helper fails before starting Docker if the file does not exist.

### API Not Healthy

Check container status and logs:

```powershell
docker compose ps
docker compose logs api
docker compose logs db
```

Then verify health manually:

```powershell
curl http://localhost:8000/health
```

### Stale DB Volume

If old database state is interfering with verification, remove the Compose volume and rebuild:

```powershell
docker compose down -v
docker compose up -d --build
```

This deletes the local PostgreSQL volume for the Compose stack.

## Final Checklist Before Demo

- Docker Compose stack starts cleanly.
- API health returns `{"status":"ok"}`.
- Frontend opens at `http://localhost:5173`.
- `.\scripts\final_demo_smoke.ps1` passes.
- Real checkpoint from issue #130 or a release asset is present when available.
- `.\scripts\run_mp4_inference.ps1` produces `data\logs\actions.json` with the real checkpoint.
- Final alert behavior is checked with a checkpoint/config/video that triggers a configured danger label.
- Backend, inference, audit logs, and database events can be inspected.
- Known limitation is stated clearly: final checkpoint verification remains pending until issue #130 publishes the checkpoint.
