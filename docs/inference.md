# Inference
## Coauthor: [Aleksander Kaźmierczak](https://github.com/blanqtoja)
## Coauthor: [Ireneusz Bartoszek](https://github.com/bartoszir)
## Coauthor: [Łukasz Murza](https://github.com/XEN00000)

[Back to README](../README.md)

## Sprint 2 CLI: MP4 to JSON

To run in inference mode:
```
docker compose run --rm inference python -m src.main \
  --input /app/data/raw/car_drops_off_person/0BD540FB-26D7-4814-8229-5572B9132328-306-00000008A9AAB259_1.mp4 \
  --checkpoint /app/data/logs/checkpoints/baseline_epoch_50.pth \
  --config /app/configs/data_pipeline.yml \
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
2. **Deterministic Mapping:** The alerting system can rely on the `scene_tag` being consistent across the entire video duration as it is derived from the initial state.
3. **Fallback Logic:** If `scene_tag` is `unknown`, the alerting system should default to the most restrictive safety policy.