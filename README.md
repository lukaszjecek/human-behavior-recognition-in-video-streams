# Human Behavior Recognition in Video Streams

End-to-end software engineering project focused on human behavior recognition from video sequences, with emphasis on temporal modeling, reproducible data handling, modular inference logic, and containerized development.

## Quick Start

### Requirements
- Docker
- local clone of this repository
- dataset subset in `./data/raw`

### Run
```bash
docker compose up --build
```

### Expected output
After startup, the container should create:
- `./data/logs/startup_summary.json`

If the dataset subset is mounted correctly, the logs should include the number of discovered video files and classes.

### Windows note
On Windows PowerShell, create the folders manually if they do not already exist:

```powershell
mkdir data\raw
mkdir data\logs
docker compose up --build
```

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
