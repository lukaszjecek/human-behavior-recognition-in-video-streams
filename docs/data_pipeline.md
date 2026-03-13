# Data Pipeline Specification

This document defines the data standards and layout for the Human Behavior Recognition project, with a specific focus on the PIP 370k dataset.

## 1. Directory Layout
All data is stored inside the container within the `/app/data/` directory. According to the repository rules, the `data/*` directory is ignored in version control.

The directory structure is defined as follows:
- `data/raw/` - Raw, original video files downloaded from the source.
- `data/processed/` - Preprocessed video files (e.g., resized, cropped, decoded into frames).
- `data/manifests/` - Manifest files defining the mapping of video files to labels and splits.
- `data/logs/` - Reports, logs, EDA visualizations, and sample annotation overlays.

## 2. Manifest Format
The manifest is the primary file used by the Data Loader. We use the **JSON Lines (.jsonl)** format due to its flexibility in storing complex nested data (e.g., bounding boxes) for individual frames.

**Required fields:**
- `video_id` (string): Unique identifier for the recording.
- `path` (string): Relative path to the processed video file (relative to `data/processed/`).
- `label` (string/int): Human behavior class.
- `split` (string): Split assignment (`train`, `val`, `test`).

**Optional fields:**
- `timestamps` (list of floats): Time markers defining the start and end of a specific action.
- `bboxes` (list of dicts): Bounding box coordinates for specific frames formatted as `[x_min, y_min, x_max, y_max]`.

**Example manifest line:**
`{"video_id": "pip_001", "path": "walking/pip_001.mp4", "label": "walking", "split": "train", "timestamps": [1.5, 4.2], "bboxes": {"frame_10": [10, 20, 100, 200]}}`

## 3. Configuration & Seed Policy
The main configuration is located in the `configs/data_pipeline.yaml` file. 
To ensure full reproducibility of experiments, it is **strictly required** to pass a constant seed value (default `42` defined in the `.yaml` file) to pseudo-random number generators in the `random`, `numpy` modules, and ML frameworks during split creation and data sampling.

## 4. Example Commands (CLI)
The following commands will be available once the full pipeline is implemented (defined in subsequent sprints):

- Download data: `python -m src.data.download --config configs/data_pipeline.yaml`
- Generate subset: `python -m src.data.sample --fraction 0.05 --output data/manifests/subset_manifest.jsonl`

### Visual Validation
To overlay labels (and bounding boxes, if present) onto a video and export the result to an MP4 file, run the visualizer tool:
```bash
# Pick a random video from the manifest
docker compose run --rm inference python -m scripts.visualize

# Or specify an exact video by ID and choose a custom output path
docker compose run --rm inference python -m scripts.visualize --video_id "20200524_191348-983363826_0" --output "/app/data/logs/my_custom_overlay.mp4"