from src.inference.engine import InferenceResult
from src.inference.tracker import SingleTrackTracker


def test_single_track_tracker_assigns_consistent_track_id():
    tracker = SingleTrackTracker(track_id=1)

    results = [
        InferenceResult(
            window=("f1", "f2"),
            start_frame_index=1,
            end_frame_index=2,
            start_timestamp=1.0,
            end_timestamp=2.0,
            prediction={"label": "action", "confidence": 0.9},
        ),
        InferenceResult(
            window=("f2", "f3"),
            start_frame_index=2,
            end_frame_index=3,
            start_timestamp=2.0,
            end_timestamp=3.0,
            prediction={"label": "action", "confidence": 0.9},
        ),
        InferenceResult(
            window=("f3", "f4"),
            start_frame_index=3,
            end_frame_index=4,
            start_timestamp=3.0,
            end_timestamp=4.0,
            prediction={"label": "action", "confidence": 0.9},
        ),
    ]

    track_ids = tracker.assign_track_ids(results)

    assert track_ids == [1, 1, 1]