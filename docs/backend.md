# Backend
## Author: [Szymon Kaźmierczak](https://github.com/Szymon110903)

[Back to README](../README.md)

# JSON Output Schema for Action Events

## Schema Definition

### ActionEvent Record

Each action event represents a detected action/behavior with the following JSON structure:

```json
{
  "start_frame_index": integer,
  "end_frame_index": integer,
  "label": string,
  "confidence": float,
  "start_timestamp": float (optional),
  "end_timestamp": float (optional),
  "track_id": integer (optional)
}
```

#### Field Descriptions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `start_frame_index` | integer | Yes | Starting frame index of the detection window (0-indexed) |
| `end_frame_index` | integer | Yes | Ending frame index of the detection window (inclusive) |
| `label` | string | Yes | Label of the detected action/behavior class (e.g., "walking", "running") |
| `confidence` | float | Yes | Confidence score of the prediction (0.0 to 1.0) |
| `start_timestamp` | float | No | Starting timestamp in seconds |
| `end_timestamp` | float | No | Ending timestamp in seconds |
| `track_id` | integer | No | Tracking ID for multi-object tracking |

### ActionEventLog Format

The complete output JSON file structure:

```json
{
  "event_count": integer,
  "events": [
    { ActionEvent record 1 },
    { ActionEvent record 2 },
    ...
  ]
}
```

## Example Output

```json
{
  "event_count": 3,
  "events": [
    {
      "start_frame_index": 0,
      "end_frame_index": 15,
      "label": "walking",
      "confidence": 0.95,
      "start_timestamp": 0.0,
      "end_timestamp": 0.5,
      "track_id": 1
    },
    {
      "start_frame_index": 16,
      "end_frame_index": 31,
      "label": "running",
      "confidence": 0.87,
      "start_timestamp": 0.533,
      "end_timestamp": 1.033,
      "track_id": 1
    },
    {
      "start_frame_index": 32,
      "end_frame_index": 47,
      "label": "jumping",
      "confidence": 0.92,
      "start_timestamp": 1.067,
      "end_timestamp": 1.567
    }
  ]
}
```

## Validation Rules

The schema enforces:

1. `start_frame_index >= 0`
2. `end_frame_index >= start_frame_index`
3. `0.0 <= confidence <= 1.0`
4. `label` is non-empty string
5. `start_timestamp <= end_timestamp` (if both provided)

## Implementation Files

- **Schema**: `src/inference/action_event.py` - ActionEvent and ActionEventLog classes
- **Writer**: `src/inference/json_writer.py` - ActionEventWriter for serialization
- **Tests**: `tests/inference/test_action_event.py` - Comprehensive test suite
- **Validation Script**: `tests/inference/test_modules.py` - Standalone validation and sample generation
