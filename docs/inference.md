# Inference
## Author: [Aleksander Kaźmierczak](https://github.com/blanqtoja)

[Back to README](../README.md)

## Sprint 2 CLI: MP4 to JSON

The Sprint 2 milestone is exposed through `src.main` inference mode:


(the commands below are for bash, below commands are for bash on Windows, if you are using Powershel, remove 'MSYS_NO_PATHCONV=1')

Run in this order:
```
MSYS_NO_PATHCONV=1 docker compose run --rm inference python -m src.data.sample --config /app/configs/data_pipeline.yml --output manifest.jsonl
```

If the first step prints "No videos found", it means that there is no data in the data/raw on the host and it needs to be placed there first.

Check if the file has been created:
```
MSYS_NO_PATHCONV=1 docker compose run --rm inference sh -lc "ls -l /app/data/manifests/manifest.jsonl"
```

Then training:
```
MSYS_NO_PATHCONV=1 docker compose run --rm inference python -m scripts.train
```

At the end, the inference:
```
MSYS_NO_PATHCONV=1 docker compose run --rm inference python -m src.main \
  --input /app/data/raw/walking/sample.mp4 \
  --checkpoint /app/data/logs/checkpoints/baseline_epoch_10.pth \
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
- `num_classes` (positive integer)
- `model_state_dict`

The loader uses `num_classes` from metadata and does not infer class count from specific layer names (for example `fc`, `head`, or `classifier`).
