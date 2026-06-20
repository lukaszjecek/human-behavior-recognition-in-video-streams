"""Tests for bbox_detector module (issue #119).

Covers:
- ACTION_LABEL_TO_OBJECT_CLASSES mapping: known / unknown labels
- BBoxEnricher: filtering, multi-bbox, frame selector variants
- BBoxEnricher: BoundingBox field values (box_format, coordinate_space, dimensions)
- BBoxEnricher: frame_index computation per selector
- BBoxEnricher: empty window safety, bboxes=None contract
- BBoxEnricher: call signature matches BBoxHook contract
- Integration: BBoxEnricher as bbox_hook= in InferenceEventPipeline
- YoloObjectDetector: missing ultralytics raises RuntimeError
- YoloObjectDetector: invalid confidence_threshold raises ValueError
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from src.app.schemas.action_event import ActionEvent, EventType
from src.inference.alert_state_machine import AlertStateMachine
from src.inference.bbox_detector import (
    ACTION_LABEL_TO_OBJECT_CLASSES,
    BBoxEnricher,
    RawDetection,
    YoloObjectDetector,
)
from src.inference.engine import InferenceEngine, InferenceResult
from src.inference.json_writer import ActionEventWriter
from src.inference.pipeline import InferenceEventPipeline

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class _FakeDetector:
    def __init__(self, detections: list[RawDetection]) -> None:
        self._detections = detections

    def detect(self, frame: np.ndarray) -> list[RawDetection]:
        return self._detections


class _TrackingDetector:
    """Fake detector that records the frame it received."""

    def __init__(self, detections: list[RawDetection]) -> None:
        self._detections = detections
        self.received_frame: np.ndarray | None = None

    def detect(self, frame: np.ndarray) -> list[RawDetection]:
        self.received_frame = frame
        return self._detections


def _make_frame(h: int = 4, w: int = 6) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _make_result(
    frames: list[np.ndarray],
    start_frame_index: int = 0,
    label: str = "car_drops_off_person",
    confidence: float = 0.9,
) -> InferenceResult:
    return InferenceResult(
        window=tuple(frames),
        start_frame_index=start_frame_index,
        end_frame_index=start_frame_index + len(frames) - 1,
        start_timestamp=None,
        end_timestamp=None,
        prediction={"label": label, "confidence": confidence},
    )


def _make_event(
    label: str = "car_drops_off_person",
    start: int = 0,
    end: int = 2,
) -> ActionEvent:
    return ActionEvent(
        start_frame_index=start,
        end_frame_index=end,
        label=label,
        confidence=0.9,
    )


def _car_detection() -> RawDetection:
    return RawDetection(label="car", confidence=0.85, 
                        x_min=10.0, y_min=20.0, x_max=100.0, y_max=80.0)


def _person_detection() -> RawDetection:
    return RawDetection(label="person", confidence=0.75, 
                        x_min=50.0, y_min=30.0, x_max=90.0, y_max=120.0)


def _dog_detection() -> RawDetection:
    return RawDetection(label="dog", confidence=0.6, x_min=5.0, y_min=5.0, x_max=40.0, y_max=40.0)


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------


def test_label_to_classes_known_label():
    assert ACTION_LABEL_TO_OBJECT_CLASSES["car_drops_off_person"] == ["car", "person"]


def test_label_to_classes_unknown_label_returns_event_unchanged():
    enricher = BBoxEnricher(detector=_FakeDetector([_car_detection()]))
    event = _make_event(label="unknown_action_xyz")
    result = _make_result([_make_frame()], label="unknown_action_xyz")
    out = enricher(event, result)
    assert out.bboxes is None
    assert out is event


# ---------------------------------------------------------------------------
# Detection filtering
# ---------------------------------------------------------------------------


def test_filters_irrelevant_object_classes():
    enricher = BBoxEnricher(
        detector=_FakeDetector([_car_detection(), _dog_detection(), _person_detection()])
    )
    event = _make_event(label="car_drops_off_person")
    result = _make_result([_make_frame()])
    out = enricher(event, result)
    assert out.bboxes is not None
    labels = {b.label for b in out.bboxes}
    assert labels == {"car", "person"}
    assert "dog" not in labels


def test_no_relevant_detections_returns_bboxes_none():
    enricher = BBoxEnricher(
        detector=_FakeDetector([_dog_detection()])
    )
    event = _make_event(label="car_drops_off_person")
    result = _make_result([_make_frame()])
    out = enricher(event, result)
    assert out.bboxes is None


def test_multiple_bboxes_per_event_supported():
    person1 = RawDetection(label="person", confidence=0.9, 
                           x_min=0.0, y_min=0.0, x_max=50.0, y_max=100.0)
    person2 = RawDetection(label="person", confidence=0.8, 
                           x_min=60.0, y_min=0.0, x_max=110.0, y_max=100.0)
    enricher = BBoxEnricher(detector=_FakeDetector([person1, person2]))
    event = _make_event(label="person_sits_down")
    result = _make_result([_make_frame()], label="person_sits_down")
    out = enricher(event, result)
    assert out.bboxes is not None
    assert len(out.bboxes) == 2


# ---------------------------------------------------------------------------
# BoundingBox field values
# ---------------------------------------------------------------------------


def test_bbox_fields_populated_correctly():
    det = RawDetection(label="car", confidence=0.88, 
                       x_min=10.0, y_min=20.0, x_max=100.0, y_max=80.0)
    frame = np.zeros((60, 120, 3), dtype=np.uint8)  # h=60, w=120
    enricher = BBoxEnricher(detector=_FakeDetector([det]), frame_selector="first")
    event = _make_event(label="car_drops_off_person")
    result = _make_result([frame])
    out = enricher(event, result)
    assert out.bboxes is not None
    bbox = out.bboxes[0]
    assert bbox.box_format == "xyxy"
    assert bbox.coordinate_space == "source_pixels"
    assert bbox.source_width == 120
    assert bbox.source_height == 60
    assert bbox.x_min == pytest.approx(10.0)
    assert bbox.y_min == pytest.approx(20.0)
    assert bbox.x_max == pytest.approx(100.0)
    assert bbox.y_max == pytest.approx(80.0)
    assert bbox.label == "car"
    assert bbox.confidence == pytest.approx(0.88)


def test_frame_index_matches_selected_frame():
    frames = [_make_frame() for _ in range(5)]
    start = 10
    det = _car_detection()

    for selector, expected_offset in [("first", 0), ("middle", 2), ("last", 4)]:
        enricher = BBoxEnricher(
            detector=_FakeDetector([det]),
            frame_selector=selector,
        )
        event = _make_event(label="car_drops_off_person")
        result = _make_result(frames, start_frame_index=start)
        out = enricher(event, result)
        assert out.bboxes is not None, f"selector={selector!r}: expected bboxes"
        assert out.bboxes[0].frame_index == start + expected_offset, (
            f"selector={selector!r}: expected frame_index={start + expected_offset}"
        )


# ---------------------------------------------------------------------------
# Frame selector
# ---------------------------------------------------------------------------


def test_frame_selector_first():
    frames = [_make_frame() for _ in range(3)]
    tracker = _TrackingDetector([_car_detection()])
    enricher = BBoxEnricher(detector=tracker, frame_selector="first")
    event = _make_event(label="car_drops_off_person")
    result = _make_result(frames)
    enricher(event, result)
    assert tracker.received_frame is frames[0]


def test_frame_selector_middle():
    frames = [_make_frame() for _ in range(3)]
    tracker = _TrackingDetector([_car_detection()])
    enricher = BBoxEnricher(detector=tracker, frame_selector="middle")
    event = _make_event(label="car_drops_off_person")
    result = _make_result(frames)
    enricher(event, result)
    assert tracker.received_frame is frames[1]  # 3 // 2 == 1


def test_frame_selector_last():
    frames = [_make_frame() for _ in range(3)]
    tracker = _TrackingDetector([_car_detection()])
    enricher = BBoxEnricher(detector=tracker, frame_selector="last")
    event = _make_event(label="car_drops_off_person")
    result = _make_result(frames)
    enricher(event, result)
    assert tracker.received_frame is frames[2]


# ---------------------------------------------------------------------------
# Backward-compatibility / safety
# ---------------------------------------------------------------------------


def test_empty_window_returns_event_unchanged():
    enricher = BBoxEnricher(detector=_FakeDetector([_car_detection()]))
    event = _make_event(label="car_drops_off_person")
    result = _make_result([])  # empty window
    out = enricher(event, result)
    assert out is event
    assert out.bboxes is None


def test_event_without_bboxes_still_valid():
    event = ActionEvent(
        start_frame_index=0,
        end_frame_index=1,
        label="person_sits_down",
        confidence=0.9,
    )
    assert event.bboxes is None
    # Round-trip through Pydantic: should not raise
    dumped = event.model_dump()
    reloaded = ActionEvent(**dumped)
    assert reloaded.bboxes is None


def test_call_matches_bbox_hook_signature():
    import inspect
    import typing
    enricher = BBoxEnricher(detector=_FakeDetector([]))
    sig = inspect.signature(enricher.__call__)
    params = list(sig.parameters.keys())
    # Should accept exactly (event, result)
    assert "event" in params
    assert "result" in params
    # get_type_hints resolves forward references (PEP 563 string annotations)
    hints = typing.get_type_hints(enricher.__call__)
    assert hints["return"] is ActionEvent


# ---------------------------------------------------------------------------
# Integration with InferenceEventPipeline
# ---------------------------------------------------------------------------


def _make_pipeline_with_enricher(
    bbox_hook: BBoxEnricher,
    label: str = "car_drops_off_person",
    window_size: int = 3,
) -> InferenceEventPipeline:
    def _predict(_window):
        return {"label": label, "confidence": 0.9}

    engine = InferenceEngine(window_size=window_size, stride=1, model=_predict)
    writer = ActionEventWriter(class_labels=[])
    alert_sm = AlertStateMachine(persistence_threshold=10, danger_labels=[label])
    return InferenceEventPipeline(
        engine=engine,
        writer=writer,
        alert_processor=alert_sm,
        bbox_hook=bbox_hook,
    )


def test_bbox_enricher_as_pipeline_hook():
    det = RawDetection(label="car", confidence=0.9, x_min=5.0, y_min=5.0, x_max=50.0, y_max=40.0)
    enricher = BBoxEnricher(
        detector=_FakeDetector([det]),
        label_to_classes={"car_drops_off_person": ["car", "person"]},
    )
    pipeline = _make_pipeline_with_enricher(enricher, label="car_drops_off_person", window_size=3)

    frame = _make_frame()
    payloads = []
    for _ in range(3):
        payloads.extend(pipeline.push_frame(frame))

    detections = [p for p in payloads if p.event_type == EventType.DETECTION]
    assert len(detections) >= 1
    event = detections[0].data
    assert isinstance(event, ActionEvent)
    assert event.bboxes is not None
    assert len(event.bboxes) >= 1
    assert event.bboxes[0].label == "car"


# ---------------------------------------------------------------------------
# YoloObjectDetector
# ---------------------------------------------------------------------------


def test_yolo_detector_raises_clear_error_without_ultralytics(monkeypatch):
    # Simulate missing ultralytics by setting sys.modules entry to None,
    # which causes ImportError when the module is imported inside __init__.
    monkeypatch.setitem(sys.modules, "ultralytics", None)
    with pytest.raises(RuntimeError, match="ultralytics"):
        YoloObjectDetector()


def test_yolo_detector_invalid_confidence_threshold_raises():
    with pytest.raises(ValueError, match="confidence_threshold"):
        YoloObjectDetector(confidence_threshold=1.5)

    with pytest.raises(ValueError, match="confidence_threshold"):
        YoloObjectDetector(confidence_threshold=-0.1)
