# Performance Benchmark

## Purpose

This benchmark provides a repeatable MP4 inference measurement path for the current system. It reports measured wall-clock throughput and latency-style values from an actual run instead of estimated performance claims.

The benchmark is intended for smoke-level performance reporting. It does not train a model, optimize runtime, or prove that the project meets a particular FPS or latency target by itself.

## Run The Benchmark

From Windows PowerShell:

```powershell
.\scripts\benchmark_mp4_inference.ps1 `
  -Input .\data\raw\car_makes_u_turn\0A2BF1E8-55E5-4D3D-9B7D-59929025D9CC_0.mp4 `
  -Checkpoint .\data\logs\checkpoints\baseline_epoch_50.pth `
  -Output .\data\logs\benchmark_summary.json `
  -Device auto
```

For the NVIDIA GPU Compose override, pass `-Gpu` and request CUDA:

```powershell
.\scripts\benchmark_mp4_inference.ps1 `
  -Input .\data\raw\car_makes_u_turn\0A2BF1E8-55E5-4D3D-9B7D-59929025D9CC_0.mp4 `
  -Checkpoint .\data\logs\checkpoints\baseline_epoch_50.pth `
  -Output .\data\logs\benchmark_summary_gpu.json `
  -Device cuda `
  -Gpu
```

Optional arguments:

- `-Config` defaults to `configs/data_pipeline.yml`.
- `-Output` defaults to `data/logs/benchmark_summary.json`.
- `-Device` accepts `auto`, `cpu`, `cuda`, or `mps`.
- `-Gpu` uses `docker compose -f compose.yaml -f compose.gpu.yaml run --rm inference ...`.

Without `-Gpu`, the wrapper uses the default CPU-safe Compose path: `docker compose run --rm inference ...`.

The wrapper validates the input MP4, checkpoint, and config paths, creates the output directory, runs the benchmark through the Docker Compose `inference` service, and prints the JSON output path.

You can also run the Python script directly inside an environment with the project dependencies installed:

```powershell
python .\scripts\benchmark_mp4_inference.py `
  --input .\data\raw\car_makes_u_turn\0A2BF1E8-55E5-4D3D-9B7D-59929025D9CC_0.mp4 `
  --checkpoint .\data\logs\checkpoints\baseline_epoch_50.pth `
  --config .\configs\data_pipeline.yml `
  --output .\data\logs\benchmark_summary.json `
  --device auto
```

## Output Values

The JSON summary is machine-readable and includes:

- `timestamp`: UTC time when the benchmark summary was produced.
- `host`: operating system and platform metadata.
- `python_version`: Python runtime version used by the benchmark process.
- `cpu`: CPU metadata available from Python and the host environment.
- `cuda`: PyTorch/CUDA availability and CUDA device names when available.
- `requested_device`: value passed with `--device` or `-Device`.
- `resolved_runtime_device`: device selected by the inference runtime.
- `input_video_path`, `checkpoint_path`, `config_path`: files used for the run.
- `window_size`: temporal window size loaded from the config.
- `stride`: inference stride loaded from the config.
- `target_resolution`: frame resolution used by the runtime tensorizer.
- `frame_count`: frames read and processed from the MP4.
- `inference_count`: number of inference windows produced.
- `event_count`: number of action events emitted by the pipeline.
- `total_wall_clock_duration_s`: measured duration for one complete offline MP4 inference run.
- `approx_processed_fps`: `frame_count / total_wall_clock_duration_s`.
- `approx_inference_windows_per_second`: `inference_count / total_wall_clock_duration_s`.
- `average_time_per_inference_window_s`: `total_wall_clock_duration_s / inference_count`.
- `limitations`: notes copied into the result to keep interpretation attached to the numbers.

## Limitations

This benchmark is a single-run offline MP4 wall-clock measurement. The duration includes model loading, video decode, preprocessing, inference, event pipeline work, and any enabled context or bounding-box enrichment in the current config.

It does not measure live camera capture-to-display latency, frontend rendering latency, network latency, or database persistence latency. For final reporting, run the benchmark more than once on the target machine and report the hardware, checkpoint, config, and device alongside the numbers.

The final benchmark must be re-run after issue #130 publishes the final checkpoint. Until then, measurements from `baseline_epoch_50.pth` are smoke measurements only and must not be treated as final project performance.

## Current Non-Final Smoke Measurements

These measurements were captured locally on 2026-06-24 using `data/logs/checkpoints/baseline_epoch_50.pth`. That checkpoint is not the final checkpoint from issue #130, so every result in this table is a non-final smoke measurement only.

All three runs used the same MP4 input: `data/raw/car_makes_u_turn/0A2BF1E8-55E5-4D3D-9B7D-59929025D9CC_0.mp4`.

| Run | Device/path | Frames | Inference windows | Events | Wall-clock duration | Approx processed FPS | Approx inference windows/s | Avg time/window | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Docker CPU smoke | CPU in Docker, default Compose | `369` | `23` | `23` | `35.162069 s` | `10.494263` | `0.654114` | `1.528786 s` | Non-final smoke measurement using `baseline_epoch_50.pth`. Includes Docker/container initialization and model startup behavior. |
| Local Python CPU smoke | CPU in local `.venv` | `369` | `23` | `23` | `26.020006 s` | `14.181396` | `0.883935` | `1.131305 s` | Not directly comparable with Docker because local Python used BBoxEnricher fallback behavior without `ultralytics` installed. |
| Docker GPU smoke | CUDA in Docker with `compose.gpu.yaml` | `369` | `23` | `23` | `11.127545 s` | `33.160954` | `2.066943` | `0.483806 s` | Non-final smoke measurement using `baseline_epoch_50.pth`. Confirms the Docker CUDA path works, but final performance must be re-measured after #130. |

Final performance must be re-measured after issue #130 publishes the final checkpoint. Do not use these smoke measurements to claim that the project meets 15 FPS or <= 2 s end-to-end latency.
