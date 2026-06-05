"""Tests for InferenceEventPipeline.

Covers:
- push_frame() emits DETECTION before source EOF
- push_frame() returns empty list when window not yet full / stride not met
- Context enrichment: fallback when context_module=None
- Context enrichment: fallback when context_module.get_context() raises
- Context cadence: context evaluated every N windows (not every window)
- BBox hook: called after context, result propagated to EventPayload
- BBox hook: bad return value is ignored gracefully
- BBox hook: exception is swallowed, inference continues
- Alert processor: ALERT EventPayload emitted when state machine fires
- ContextAwareAlertProcessor variant accepted
- Thread-safety: concurrent push_frame() calls are serialised
- reset() clears engine + context cache
- get_metrics() returns expected keys
- Constructor validation (TypeError / ValueError)
"""

from __future__ import annotations

import threading
import threading
from unittest.mock import MagicMock
from uuid import UUID

import numpy as np
import pytest

from src.app.schemas.action_event import ActionEvent, ContextData, EventPayload, EventType
from src.inference.alert_state_machine import AlertStateMachine
from src.inference.context_policy import ContextAwareAlertProcessor, ContextPolicy
from src.inference.engine import InferenceEngine
from src.inference.json_writer import ActionEventWriter
from src.inference.pipeline import InferenceEventPipeline


# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------


def _make_frame(h: int = 4, w: int = 4) -> np.ndarray:
    """Return a minimal valid BGR uint8 frame."""
    return np.zeros((h, w, 3), dtype=np.uint8)


def _make_dict_model(label: str = "fight", confidence: float = 0.9):
    """Return a callable that always predicts the same label/confidence dict."""

    def _predict(_window):
        return {"label": label, "confidence": confidence}

    return _predict


def _make_pipeline(
    window_size: int = 3,
    stride: int = 1,
    label: str = "fight",
    confidence: float = 0.9,
    danger_labels: list[str] | None = None,
    context_module=None,
    context_eval_every_n_windows: int = 1,
    bbox_hook=None,
    persistence_threshold: int = 1,
) -> InferenceEventPipeline:
    """Build a pipeline with a deterministic in-process model."""
    engine = InferenceEngine(
        window_size=window_size,
        stride=stride,
        model=_make_dict_model(label, confidence),
    )
    writer = ActionEventWriter(class_labels=[])
    alert_sm = AlertStateMachine(
        persistence_threshold=persistence_threshold,
        danger_labels=danger_labels or [label],
    )
    return InferenceEventPipeline(
        engine=engine,
        writer=writer,
        alert_processor=alert_sm,
        context_module=context_module,
        context_eval_every_n_windows=context_eval_every_n_windows,
        bbox_hook=bbox_hook,
    )


def _push_n(pipeline: InferenceEventPipeline, n: int) -> list[EventPayload]:
    """Push *n* identical blank frames and collect all emitted EventPayloads."""
    payloads: list[EventPayload] = []
    for _ in range(n):
        payloads.extend(pipeline.push_frame(_make_frame()))
    return payloads


# ---------------------------------------------------------------------------
# push_frame — basic emission
# ---------------------------------------------------------------------------


class TestPushFrameBasicEmission:
    def test_no_events_before_window_full(self):
        pipeline = _make_pipeline(window_size=4, stride=1)
        # Push 3 frames — window_size=4 → should never trigger
        payloads = _push_n(pipeline, 3)
        assert payloads == []

    def test_detection_emitted_before_eof(self):
        """Core DoD item: DETECTION must be emitted before source EOF."""
        pipeline = _make_pipeline(window_size=3, stride=1)
        payloads = _push_n(pipeline, 3)  # exactly window_size frames
        detections = [p for p in payloads if p.event_type == EventType.DETECTION]
        assert len(detections) >= 1, "Expected at least one DETECTION before EOF"

    def test_returns_list_of_event_payloads(self):
        pipeline = _make_pipeline(window_size=2, stride=1)
        result = _push_n(pipeline, 2)
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, EventPayload)

    def test_empty_list_between_stride_triggers(self):
        pipeline = _make_pipeline(window_size=2, stride=3)
        # Fill window (frame 1-2) → trigger
        _push_n(pipeline, 2)
        # Frames 3-4: stride=3 means next trigger at frame 5
        for _ in range(2):
            assert pipeline.push_frame(_make_frame()) == []

    def test_multiple_detections_over_many_frames(self):
        pipeline = _make_pipeline(window_size=2, stride=2)
        payloads = _push_n(pipeline, 10)
        detections = [p for p in payloads if p.event_type == EventType.DETECTION]
        # 10 frames, stride=2, window=2 → triggers at frames 2,4,6,8,10 = 5
        assert len(detections) == 5

    def test_frame_must_be_ndarray(self):
        pipeline = _make_pipeline()
        with pytest.raises(TypeError, match="numpy ndarray"):
            pipeline.push_frame("not-a-frame")  # type: ignore[arg-type]

    def test_push_frame_with_timestamp(self):
        pipeline = _make_pipeline(window_size=1, stride=1)
        payloads = pipeline.push_frame(_make_frame(), timestamp=12345.6)
        assert len(payloads) >= 1
        event = payloads[0].data
        assert isinstance(event, ActionEvent)
        assert event.start_timestamp == 12345.6

    def test_session_id_and_camera_id_propagate(self):
        import uuid
        test_uuid = uuid.uuid4()
        engine = InferenceEngine(window_size=1, stride=1, model=_make_dict_model())
        writer = ActionEventWriter(class_labels=[])
        alert_sm = AlertStateMachine(persistence_threshold=1, danger_labels=["fight"])
        pipeline = InferenceEventPipeline(
            engine=engine, writer=writer, alert_processor=alert_sm,
            camera_id="cam-99", session_id=test_uuid
        )
        payloads = pipeline.push_frame(_make_frame())
        assert len(payloads) == 2  # DETECTION and ALERT
        for p in payloads:
            assert p.camera_id == "cam-99"
            assert p.session_id == test_uuid


# ---------------------------------------------------------------------------
# Context enrichment — fallback behaviour
# ---------------------------------------------------------------------------


class TestContextFallback:
    def test_context_unknown_when_no_module(self):
        pipeline = _make_pipeline(context_module=None)
        payloads = _push_n(pipeline, 3)
        detection = next(p for p in payloads if p.event_type == EventType.DETECTION)
        event = detection.data
        assert isinstance(event, ActionEvent)
        assert event.context is None

    def test_context_unknown_when_module_raises(self):
        bad_module = MagicMock()
        bad_module.get_context.side_effect = RuntimeError("GPU not available")
        pipeline = _make_pipeline(context_module=bad_module, context_eval_every_n_windows=1)
        payloads = _push_n(pipeline, 3)
        detection = next(p for p in payloads if p.event_type == EventType.DETECTION)
        event = detection.data
        assert isinstance(event, ActionEvent)
        assert event.context is None

    def test_context_unknown_when_module_returns_bad_dict(self):
        """get_context() returning unexpected keys should default gracefully."""
        bad_module = MagicMock()
        bad_module.get_context.return_value = {}  # no scene_tag key
        pipeline = _make_pipeline(context_module=bad_module, context_eval_every_n_windows=1)
        payloads = _push_n(pipeline, 3)
        detection = next(p for p in payloads if p.event_type == EventType.DETECTION)
        event = detection.data
        assert isinstance(event, ActionEvent)
        # None is the default when context fails
        assert event.context is None

    def test_valid_context_attached_when_module_works(self):
        good_module = MagicMock()
        good_module.get_context.return_value = {"scene_tag": "indoor", "confidence": 0.85}
        pipeline = _make_pipeline(context_module=good_module, context_eval_every_n_windows=1)
        payloads = _push_n(pipeline, 3)
        detection = next(p for p in payloads if p.event_type == EventType.DETECTION)
        event = detection.data
        assert isinstance(event, ActionEvent)
        assert event.context.scene_tag == "indoor"
        assert pytest.approx(event.context.confidence, abs=1e-6) == 0.85

    def test_evaluate_context_converts_numpy_to_pil(self):
        mock_module = MagicMock()
        mock_module.get_context.return_value = {"scene_tag": "indoor", "confidence": 0.9}
        pipeline = _make_pipeline(window_size=1, stride=1, context_module=mock_module, context_eval_every_n_windows=1)
        _push_n(pipeline, 1)
        args, _ = mock_module.get_context.call_args
        from PIL.Image import Image as PilImage
        assert isinstance(args[0], PilImage)


# ---------------------------------------------------------------------------
# Context cadence
# ---------------------------------------------------------------------------


class TestContextCadence:
    def test_context_evaluated_every_n_windows(self):
        """Context module called once per N inference windows, not every window."""
        mock_module = MagicMock()
        mock_module.get_context.return_value = {"scene_tag": "outdoor", "confidence": 0.7}

        # window=2, stride=2 → triggers at frame 2, 4, 6, 8, 10 (5 inference windows)
        # context_eval_every_n_windows=3 → evaluated at windows 3 and (eventually) 6
        pipeline = _make_pipeline(
            window_size=2,
            stride=2,
            context_module=mock_module,
            context_eval_every_n_windows=3,
        )
        _push_n(pipeline, 10)  # 5 inference windows

        # Evaluation should happen at windows 3 (counter reaches 3) and not again
        # before window 6 (5 total — only windows 3 and possibly 6).
        # At least 1 call, no more than ceil(5/3) = 2 calls.
        call_count = mock_module.get_context.call_count
        assert 1 <= call_count <= 2, (
            f"Expected 1-2 context evaluations over 5 windows "
            f"(every_n=3), got {call_count}"
        )

    def test_context_evaluated_every_window_when_n_equals_1(self):
        mock_module = MagicMock()
        mock_module.get_context.return_value = {"scene_tag": "outdoor", "confidence": 0.6}

        pipeline = _make_pipeline(
            window_size=2,
            stride=2,
            context_module=mock_module,
            context_eval_every_n_windows=1,
        )
        _push_n(pipeline, 10)  # 5 inference windows
        assert mock_module.get_context.call_count == 5

    def test_cached_context_reused_between_evaluations(self):
        """All detection events between two context evaluations share the same context.

        With context_eval_every_n_windows=3:
        - windows 1-2: counter increments (1, 2) → cached "unknown" reused
        - window 3:    counter reaches 3 → evaluation fires → "outdoor" cached
        - windows 4-5: counter increments (1, 2) → "outdoor" reused
        - window 6:    counter reaches 3 → evaluation fires → "indoor" cached
        """
        call_results = [
            {"scene_tag": "outdoor", "confidence": 0.9},
            {"scene_tag": "indoor", "confidence": 0.8},
        ]
        mock_module = MagicMock()
        mock_module.get_context.side_effect = call_results

        # window=1, stride=1 → triggers every frame; context every 3 windows
        pipeline = _make_pipeline(
            window_size=1,
            stride=1,
            context_module=mock_module,
            context_eval_every_n_windows=3,
        )
        payloads = _push_n(pipeline, 6)  # 6 inference windows
        detections = [p for p in payloads if p.event_type == EventType.DETECTION]
        tags = [d.data.context.scene_tag if d.data.context else "unknown" for d in detections if isinstance(d.data, ActionEvent)]

        # Windows 1-2: counter hasn't reached 3 → cached fallback "unknown"
        # Window 3:    counter hits 3 → first evaluation → "outdoor"
        # Windows 4-5: "outdoor" cached and reused
        # Window 6:    counter hits 3 again → second evaluation → "indoor"
        assert tags[0] == "unknown"   # window 1: not yet evaluated
        assert tags[1] == "unknown"   # window 2: not yet evaluated
        assert tags[2] == "outdoor"   # window 3: first evaluation
        assert tags[3] == "outdoor"   # window 4: cached
        assert tags[4] == "outdoor"   # window 5: cached
        assert tags[5] == "indoor"    # window 6: second evaluation

# ---------------------------------------------------------------------------
# BBox hook
# ---------------------------------------------------------------------------


class TestBBoxHook:
    def _make_bbox_hook(self, bbox_label: str = "person"):
        """Hook that returns the event with a bboxes list of length 1."""
        from src.app.schemas.action_event import BoundingBox

        def hook(event: ActionEvent) -> ActionEvent:
            bbox = BoundingBox(label=bbox_label, confidence=0.95)
            return event.model_copy(update={"bboxes": [bbox]})

        return hook

    def test_bbox_hook_result_propagated(self):
        pipeline = _make_pipeline(bbox_hook=self._make_bbox_hook("person"))
        payloads = _push_n(pipeline, 3)
        detection = next(p for p in payloads if p.event_type == EventType.DETECTION)
        event = detection.data
        assert isinstance(event, ActionEvent)
        assert event.bboxes is not None
        assert len(event.bboxes) == 1
        assert event.bboxes[0].label == "person"

    def test_bbox_hook_exception_swallowed(self):
        """A crashing bbox_hook must not prevent DETECTION from being emitted."""

        def bad_hook(event: ActionEvent) -> ActionEvent:
            raise RuntimeError("bbox model unavailable")

        pipeline = _make_pipeline(bbox_hook=bad_hook)
        payloads = _push_n(pipeline, 3)
        detections = [p for p in payloads if p.event_type == EventType.DETECTION]
        assert len(detections) >= 1, "DETECTION must still be emitted when bbox_hook fails"
        # bboxes should be None (original event unchanged)
        event = detections[0].data
        assert isinstance(event, ActionEvent)
        assert event.bboxes is None

    def test_bbox_hook_bad_return_type_ignored(self):
        """Hook returning non-ActionEvent must be silently ignored."""

        def bad_hook(_event: ActionEvent):
            return "not-an-action-event"

        pipeline = _make_pipeline(bbox_hook=bad_hook)
        payloads = _push_n(pipeline, 3)
        detections = [p for p in payloads if p.event_type == EventType.DETECTION]
        assert len(detections) >= 1
        assert isinstance(detections[0].data, ActionEvent)

    def test_no_bbox_hook_bboxes_is_none(self):
        pipeline = _make_pipeline(bbox_hook=None)
        payloads = _push_n(pipeline, 3)
        detection = next(p for p in payloads if p.event_type == EventType.DETECTION)
        event = detection.data
        assert isinstance(event, ActionEvent)
        assert event.bboxes is None


# ---------------------------------------------------------------------------
# Alert processing
# ---------------------------------------------------------------------------


class TestAlertProcessing:
    def test_alert_emitted_when_threshold_reached(self):
        """ALERT payload must appear once persistence_threshold windows fire."""
        # persistence_threshold=2: first danger window → CANDIDATE, second → ACTIVE+ALERT
        pipeline = _make_pipeline(
            window_size=1,
            stride=1,
            label="fight",
            danger_labels=["fight"],
            persistence_threshold=2,
        )
        payloads = _push_n(pipeline, 3)  # 3 inference windows
        alerts = [p for p in payloads if p.event_type == EventType.ALERT]
        assert len(alerts) >= 1

    def test_no_alert_when_label_not_dangerous(self):
        pipeline = _make_pipeline(
            window_size=1,
            stride=1,
            label="walk",
            danger_labels=["fight"],  # "walk" is not dangerous
            persistence_threshold=1,
        )
        payloads = _push_n(pipeline, 5)
        alerts = [p for p in payloads if p.event_type == EventType.ALERT]
        assert alerts == []

    def test_context_aware_alert_processor_accepted(self):
        """ContextAwareAlertProcessor can be injected in place of AlertStateMachine."""
        engine = InferenceEngine(window_size=1, stride=1, model=_make_dict_model("fight"))
        writer = ActionEventWriter(class_labels=[])
        policy = ContextPolicy(default_persistence_threshold=3, default_resolve_threshold=1)
        cap = ContextAwareAlertProcessor(policy=policy, danger_labels=["fight"])
        pipeline = InferenceEventPipeline(engine=engine, writer=writer, alert_processor=cap)
        payloads = _push_n(pipeline, 5)
        assert isinstance(payloads, list)
        alerts = [p for p in payloads if p.event_type == EventType.ALERT]
        assert len(alerts) >= 1


# ---------------------------------------------------------------------------
# Thread-safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_push_frame_no_exception(self):
        """Multiple threads calling push_frame() concurrently must not raise."""
        pipeline = _make_pipeline(window_size=2, stride=1)
        errors: list[Exception] = []

        def worker():
            try:
                for _ in range(20):
                    pipeline.push_frame(_make_frame())
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread safety violations: {errors}"

    def test_concurrent_push_frame_and_reset(self):
        pipeline = _make_pipeline(window_size=2, stride=1)
        errors: list[Exception] = []

        def worker_push():
            try:
                for _ in range(50):
                    pipeline.push_frame(_make_frame())
            except Exception as exc:
                errors.append(exc)

        def worker_reset():
            try:
                for _ in range(10):
                    pipeline.reset()
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=worker_push),
            threading.Thread(target=worker_reset),
            threading.Thread(target=worker_push)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []


# ---------------------------------------------------------------------------
# reset() and get_metrics()
# ---------------------------------------------------------------------------


class TestResetAndMetrics:
    def test_reset_clears_engine_state(self):
        pipeline = _make_pipeline(window_size=3, stride=1)
        _push_n(pipeline, 3)
        pipeline.reset()
        # After reset the engine frame_count should be 0
        assert pipeline.get_metrics()["total_frames_processed"] == 0

    def test_reset_clears_context_cache(self):
        mock_module = MagicMock()
        mock_module.get_context.return_value = {"scene_tag": "indoor", "confidence": 0.9}
        pipeline = _make_pipeline(context_module=mock_module, context_eval_every_n_windows=1)
        _push_n(pipeline, 3)
        pipeline.reset()
        assert pipeline.get_metrics()["cached_context_scene_tag"] is None

    def test_reset_clears_alert_processor_state(self):
        pipeline = _make_pipeline(window_size=1, stride=1, persistence_threshold=2)
        payloads = _push_n(pipeline, 3)
        assert any(p.event_type == EventType.ALERT for p in payloads)
        pipeline.reset()
        # Next push should NOT alert because state was cleared.
        payloads2 = _push_n(pipeline, 1)
        assert not any(p.event_type == EventType.ALERT for p in payloads2)

    def test_get_metrics_returns_expected_keys(self):
        pipeline = _make_pipeline(window_size=2, stride=1, context_eval_every_n_windows=5)
        _push_n(pipeline, 4)
        metrics = pipeline.get_metrics()
        expected_keys = {
            "total_frames_processed",
            "total_inferences",
            "total_frames_skipped",
            "total_windows_processed",
            "cached_context_scene_tag",
            "cached_context_confidence",
            "context_eval_every_n_windows",
        }
        assert expected_keys.issubset(metrics.keys())
        assert metrics["total_frames_processed"] == 4
        assert metrics["context_eval_every_n_windows"] == 5

    def test_reset_resets_window_counter(self):
        pipeline = _make_pipeline(window_size=1, stride=1)
        _push_n(pipeline, 5)
        pipeline.reset()
        assert pipeline.get_metrics()["total_windows_processed"] == 0


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


class TestConstructorValidation:
    def _base_args(self):
        engine = InferenceEngine(window_size=2, stride=1, model=_make_dict_model())
        writer = ActionEventWriter()
        alert_sm = AlertStateMachine(persistence_threshold=1, danger_labels=["fight"])
        return engine, writer, alert_sm

    def test_invalid_engine_type(self):
        _, writer, alert_sm = self._base_args()
        with pytest.raises(TypeError, match="engine must be an InferenceEngine"):
            InferenceEventPipeline(engine="not-engine", writer=writer, alert_processor=alert_sm)

    def test_invalid_writer_type(self):
        engine, _, alert_sm = self._base_args()
        with pytest.raises(TypeError, match="writer must be an ActionEventWriter"):
            InferenceEventPipeline(engine=engine, writer=42, alert_processor=alert_sm)

    def test_invalid_context_eval_zero(self):
        engine, writer, alert_sm = self._base_args()
        with pytest.raises(ValueError, match="context_eval_every_n_windows must be >= 1"):
            InferenceEventPipeline(
                engine=engine,
                writer=writer,
                alert_processor=alert_sm,
                context_eval_every_n_windows=0,
            )

    def test_invalid_context_eval_negative(self):
        engine, writer, alert_sm = self._base_args()
        with pytest.raises(ValueError, match="context_eval_every_n_windows must be >= 1"):
            InferenceEventPipeline(
                engine=engine,
                writer=writer,
                alert_processor=alert_sm,
                context_eval_every_n_windows=-3,
            )

    def test_invalid_bbox_hook_not_callable(self):
        engine, writer, alert_sm = self._base_args()
        with pytest.raises(TypeError, match="bbox_hook must be callable"):
            InferenceEventPipeline(
                engine=engine,
                writer=writer,
                alert_processor=alert_sm,
                bbox_hook="not-callable",
            )

    def test_invalid_track_id_negative(self):
        engine, writer, alert_sm = self._base_args()
        with pytest.raises(ValueError, match="track_id must be >= 0"):
            InferenceEventPipeline(
                engine=engine,
                writer=writer,
                alert_processor=alert_sm,
                track_id=-1,
            )

    def test_valid_construction_with_all_defaults(self):
        engine, writer, alert_sm = self._base_args()
        pipeline = InferenceEventPipeline(engine=engine, writer=writer, alert_processor=alert_sm)
        assert pipeline is not None

    def test_session_id_auto_generated_when_none(self):
        engine, writer, alert_sm = self._base_args()
        pipeline = InferenceEventPipeline(engine=engine, writer=writer, alert_processor=alert_sm)
        assert isinstance(pipeline._session_id, UUID)
