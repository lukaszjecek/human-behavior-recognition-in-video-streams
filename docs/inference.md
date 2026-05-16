# Inference
## Coauthor: [Aleksander Kaźmierczak](https://github.com/blanqtoja)
## Coauthor: [Ireneusz Bartoszek](https://github.com/bartoszir)
## Coauthor: [Łukasz Murza](https://github.com/XEN00000)

[Back to README](../README.md)

## Sprint 2 CLI: MP4 to JSON

To run in inference mode from the compose stack:
```bash
docker compose run --rm inference python -m src.main \
  --input /app/data/raw/car_drops_off_person/0BD540FB-26D7-4814-8229-5572B9132328-306-00000008A9AAB259_1.mp4 \
  --checkpoint /app/data/logs/checkpoints/baseline_epoch_50.pth \
  --config /app/configs/data_pipeline.yml \
  --output /app/data/logs/actions.json \
  --device auto
```

If `INFERENCE_CHECKPOINT` and `INFERENCE_CONFIG` are already set in `.env`, you can also override
them inline without repeating the flags:
```bash
docker compose run --rm \
  -e INFERENCE_CHECKPOINT=/app/data/logs/checkpoints/baseline_epoch_50.pth \
  inference \
  python -m src.main \
    --input /app/data/raw/car_drops_off_person/sample.mp4 \
    --checkpoint "${INFERENCE_CHECKPOINT}" \
    --config "${INFERENCE_CONFIG:-configs/data_pipeline.yml}" \
    --output /app/data/logs/actions.json \
    --device auto
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--input` | Yes (in inference mode) | Input `.mp4` path |
| `--checkpoint` | Yes (in inference mode) | Model checkpoint (`.pth`) |
| `--config` | No | YAML config with runtime options (default: `configs/data_pipeline.yml`) |
| `--output` | No | JSON output path (default: `data/logs/actions.json`) |
| `--device` | No | Device override: `auto`, `cpu`, `cuda`, `mps` (CLI override has priority over config) |

`--input` and `--checkpoint` must be provided together.
If neither is provided, `src.main` runs startup summary mode.

## Runtime flow

1. Load YAML settings (`pipeline`, optional `inference`, optional `tracking`).
2. Load model checkpoint.
3. Read MP4 frames with offline runtime.
4. Run `InferenceEngine` windows through model + tensorizer adapter.
5. Convert `InferenceResult` objects to `ActionEvent` records.
6. Save action log JSON with `ActionEventWriter`.

## Reusable service entrypoint

For programmatic integrations (for example upcoming backend routes), use the
service API instead of the CLI wrapper:

```python
from pathlib import Path

from src.inference.service import InferenceServiceRequest, run_inference

file_result = run_inference(
    InferenceServiceRequest(
        checkpoint_path=Path("data/logs/checkpoints/baseline_epoch_10.pth"),
        config_path=Path("configs/data_pipeline.yml"),
        video_path=Path("data/raw/walking/sample.mp4"),  # file source
        device="auto",
    )
)

rtsp_result = run_inference(
    InferenceServiceRequest(
        checkpoint_path=Path("data/logs/checkpoints/baseline_epoch_10.pth"),
        config_path=Path("configs/data_pipeline.yml"),
        video_path=None,
        source_type="rtsp",
        source_uri="rtsp://user:password@camera-host:554/stream",
        device="auto",
    )
)

print(file_result.event_count, rtsp_result.event_count)
```

For MP4-only offline callers, `run_offline_mp4_inference` is kept as a compatibility wrapper.
Use this wrapper only for legacy MP4/offline integrations (including the MP4 CLI wrapper).
It validates `source_type="file"`, requires `video_path`, and requires an `.mp4` suffix.

The service returns a typed in-memory result with:
- processed frame count
- inference window count
- expanded `InferenceResult` items
- generated `ActionEvent` records
- resolved runtime settings and selected torch device

`run_mp4_to_json_action_inference` (CLI wrapper) now delegates runtime execution to
this service entrypoint and handles only output-file serialization.

Shared runtime primitives are implemented in `src/inference/runtime.py` and reused
by both the service and CLI layers (`load_runtime_settings`, device resolution,
checkpoint loading, `WindowModelAdapter`, track-id building, and
`expand_batched_inference_results`).

Input-source adapters are implemented in `src/inference/source_adapters.py`:
- `FileSourceAdapter` for local file paths
- `RtspSourceAdapter` for `rtsp://` and `rtsps://` streams

The reusable service selects adapters via `InferenceServiceRequest.source_type`
(`file` by default) and `video_path`/`source_uri`.

### Container Networking and Compose Wiring

Within the Sprint 3 compose stack the inference container uses an **in-process** integration
model - no HTTP hop between containers:

- The backend (`api`) and inference container share the **same Docker image base** and source
  tree (mounted at `/app` via volume).
- `run_inference(InferenceServiceRequest(...))` is called **directly** inside the Python process;
  there is no remote RPC or network request to the `hbr_inference` container from the API.
- The `hbr_inference` container exists as a **companion service** that stays alive for on-demand
  `docker compose run --rm inference ...` dispatch and for future in-process extension.

**Environment variables wired by compose**:

| Variable | Default | Purpose |
|----------|---------|----------|
| `INFERENCE_CHECKPOINT` | _(empty)_ | Path to `.pth` checkpoint inside the container |
| `INFERENCE_CONFIG` | `configs/data_pipeline.yml` | Path to runtime YAML config (relative to `/app`) |
| `INFERENCE_DEVICE` | `auto` | Device override: `auto` / `cpu` / `cuda` / `mps` |
| `API_HOST` | `api` | DNS name of the API container on `hbr-network` |
| `API_PORT` | `8000` | Port of the API container |

The inference container resolves `api` by DNS on `hbr-network`. To verify connectivity:
```bash
docker compose exec inference curl -sf http://api:8000/health
```

Set `INFERENCE_CHECKPOINT` in `.env` (or pass `-e INFERENCE_CHECKPOINT=...` to
`docker compose run`) before dispatching model inference jobs.

### Offline runtime details

The offline runtime processes video frames using a producer-consumer pattern:

- A producer thread reads frames from the input MP4 file in source order.
- A consumer thread feeds frames into the `InferenceEngine`.
- Frame buffering and windowing are handled internally by the engine.

The runtime guarantees:
- deterministic frame ordering
- safe shutdown using an EOF sentinel
- propagation of frame indices and timestamps in `InferenceResult`

## Supported config keys

```yaml
pipeline:
  target_resolution: [224, 224]
  temporal_window: 16

inference:
  stride: 1
  class_labels: []  # optional list of labels by class index
  device: auto      # optional: auto/cpu/cuda/mps

tracking:
  default_track_id: null  # optional integer
```

If `tracking.default_track_id` is set, that track ID is attached to every emitted event.

Device resolution order:
1. `--device` (CLI override)
2. `inference.device` in YAML config
3. automatic fallback: `cuda` -> `mps` -> `cpu`

## Checkpoint metadata requirements

Inference expects checkpoint metadata fields:
- `model_name` (supported: `baseline`, `dummy`)
- `model_state_dict`

## Tracking

Tracking is implemented through a simple abstraction layer:

- `BaseTracker` defines the interface for assigning track IDs
- `SingleTrackTracker` is the initial backend implementation

### Current behavior

- A single persistent `track_id` is assigned to all inference results
- Track IDs are propagated into `ActionEvent` records
- Tracking operates on inference windows, not raw frames

### Integration in pipeline

1. `InferenceEngine` produces `InferenceResult` objects
2. Tracker assigns `track_id` values to each result
3. `ActionEventWriter` includes `track_id` in output events

### Limitations

- Assumes a single continuous subject or identity
- No multi-object tracking support
- No spatial matching (no bounding boxes or IoU-based association)
- No re-identification across disjoint segments

This implementation serves as a baseline for future multi-object tracking extensions.

## Sprint 3: Minimal Context Module

### Overview
To support context-aware alerting in Sprint 3, a lightweight **Context Module** has been integrated into the inference pipeline. This module identifies the environmental setting of a video clip without requiring retraining of the primary action-recognition baseline.

### Technical Implementation
- **Model:** Pre-trained MobileNetV2 (ImageNetV2 weights).
- **Approach:** Zero-shot scene classification via index mapping. The module analyzes the first frame of each video to determine the global context.
- **Performance:** Deterministic output confirmed via local verification tests (`tests/inference/test_context.py`).

### Output Contract
The `ContextModule` extends the `ActionEvent` schema. Every event produced in `actions.json` now includes a `context` object:

```json
"context": {
  "scene_tag": "string",
  "confidence": 0.85
}
```

**Supported Tags:**
- `outdoor`: Parks, streets, open areas.
- `indoor`: Rooms, hallways, office spaces.
- `vehicle_setting`: Car interiors or immediate transport surroundings.
- `unknown`: Fallback for low-confidence or ambiguous scenes.

### Integration Assumptions for Alerting
The alerting logic (downstream in Sprint 3) is expected to consume the `scene_tag` to apply conditional filters:
1. **Contextual Thresholds:** Alerts for "running" might have a higher priority in `indoor` settings compared to `outdoor`.
2. **First-Frame Context Assumption (Global Context):** 
   - **Crucial:** The `scene_tag` is extracted ONLY from the first valid frame of the video and applied uniformly to all events in the stream.
   - It does NOT support dynamic scene changes (e.g., transitioning from indoor to outdoor within a single clip). 
   - Downstream logic must treat this as a "Global Scene Label" for the entire inference session.
3. **Fallback Logic:** If `scene_tag` is `unknown`, the alerting system should default to the most restrictive safety policy.
