"""
Tests for action event schema and JSON writer.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from src.inference.action_event import ActionEvent, ActionEventLog
from src.inference.engine import InferenceResult
from src.inference.json_writer import ActionEventWriter


class TestActionEvent:
    """Test ActionEvent schema validation and serialization."""

    def test_valid_action_event(self):
        """Test creating a valid ActionEvent."""
        event = ActionEvent(
            start_frame_index=0,
            end_frame_index=15,
            label="walking",
            confidence=0.95,
            start_timestamp=0.0,
            end_timestamp=0.5,
            track_id=1,
        )
        assert event.start_frame_index == 0
        assert event.end_frame_index == 15
        assert event.label == "walking"
        assert event.confidence == 0.95

    def test_action_event_invalid_frames(self):
        """Test ActionEvent validation for invalid frame indices."""
        with pytest.raises(ValueError, match="end_frame_index must be >= start_frame_index"):
            ActionEvent(
                start_frame_index=10,
                end_frame_index=5,
                label="walking",
                confidence=0.95,
            )

    def test_action_event_invalid_confidence(self):
        """Test ActionEvent validation for invalid confidence."""
        with pytest.raises(ValueError, match="confidence must be between"):
            ActionEvent(
                start_frame_index=0,
                end_frame_index=15,
                label="walking",
                confidence=1.5,
            )

        with pytest.raises(ValueError, match="confidence must be between"):
            ActionEvent(
                start_frame_index=0,
                end_frame_index=15,
                label="walking",
                confidence=-0.1,
            )

    def test_action_event_empty_label(self):
        """Test ActionEvent validation for empty label."""
        with pytest.raises(ValueError, match="label must be a non-empty string"):
            ActionEvent(
                start_frame_index=0,
                end_frame_index=15,
                label="",
                confidence=0.95,
            )

    def test_action_event_negative_frame_index(self):
        """Test ActionEvent validation for negative frame index."""
        with pytest.raises(ValueError, match="start_frame_index must be >= 0"):
            ActionEvent(
                start_frame_index=-1,
                end_frame_index=15,
                label="walking",
                confidence=0.95,
            )

    def test_action_event_invalid_timestamps(self):
        """Test ActionEvent validation for invalid timestamps."""
        with pytest.raises(ValueError, match="end_timestamp must be >= start_timestamp"):
            ActionEvent(
                start_frame_index=0,
                end_frame_index=15,
                label="walking",
                confidence=0.95,
                start_timestamp=1.0,
                end_timestamp=0.5,
            )

    def test_action_event_to_dict(self):
        """Test ActionEvent serialization to dict."""
        event = ActionEvent(
            start_frame_index=0,
            end_frame_index=15,
            label="walking",
            confidence=0.95,
            start_timestamp=0.0,
            end_timestamp=0.5,
            track_id=1,
        )
        data = event.to_dict()
        assert data["start_frame_index"] == 0
        assert data["end_frame_index"] == 15
        assert data["label"] == "walking"
        assert data["confidence"] == 0.95
        assert data["track_id"] == 1

    def test_action_event_to_dict_omits_none(self):
        """Test ActionEvent serialization omits None values."""
        event = ActionEvent(
            start_frame_index=0,
            end_frame_index=15,
            label="walking",
            confidence=0.95,
        )
        data = event.to_dict()
        assert "track_id" not in data
        assert "start_timestamp" not in data
        assert "end_timestamp" not in data

    def test_action_event_to_json(self):
        """Test ActionEvent JSON serialization."""
        event = ActionEvent(
            start_frame_index=0,
            end_frame_index=15,
            label="walking",
            confidence=0.95,
        )
        json_str = event.to_json()
        data = json.loads(json_str)
        assert data["label"] == "walking"
        assert data["confidence"] == 0.95

    def test_action_event_from_dict(self):
        """Test creating ActionEvent from dict."""
        data = {
            "start_frame_index": 0,
            "end_frame_index": 15,
            "label": "walking",
            "confidence": 0.95,
        }
        event = ActionEvent.from_dict(data)
        assert event.label == "walking"
        assert event.confidence == 0.95


class TestActionEventLog:
    """Test ActionEventLog container and file I/O."""

    def test_action_event_log_add_event(self):
        """Test adding events to log."""
        log = ActionEventLog()
        event = ActionEvent(
            start_frame_index=0,
            end_frame_index=15,
            label="walking",
            confidence=0.95,
        )
        log.add_event(event)
        assert len(log.events) == 1

    def test_action_event_log_add_events(self):
        """Test adding multiple events to log."""
        log = ActionEventLog()
        events = [
            ActionEvent(0, 15, "walking", 0.95),
            ActionEvent(16, 31, "running", 0.87),
        ]
        log.add_events(events)
        assert len(log.events) == 2

    def test_action_event_log_to_dict(self):
        """Test log serialization to dict."""
        log = ActionEventLog()
        log.add_event(ActionEvent(0, 15, "walking", 0.95))
        log.add_event(ActionEvent(16, 31, "running", 0.87))

        data = log.to_dict()
        assert data["event_count"] == 2
        assert len(data["events"]) == 2

    def test_action_event_log_to_json(self):
        """Test log JSON serialization."""
        log = ActionEventLog()
        log.add_event(ActionEvent(0, 15, "walking", 0.95))
        json_str = log.to_json()
        data = json.loads(json_str)
        assert data["event_count"] == 1

    def test_action_event_log_save_and_load(self):
        """Test saving and loading log from file."""
        log = ActionEventLog()
        log.add_event(ActionEvent(0, 15, "walking", 0.95))
        log.add_event(ActionEvent(16, 31, "running", 0.87))

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.json"
            log.save_to_file(str(filepath))

            assert filepath.exists()

            loaded_log = ActionEventLog.load_from_file(str(filepath))
            assert len(loaded_log.events) == 2
            assert loaded_log.events[0].label == "walking"
            assert loaded_log.events[1].label == "running"

    def test_action_event_log_clear(self):
        """Test clearing events from log."""
        log = ActionEventLog()
        log.add_event(ActionEvent(0, 15, "walking", 0.95))
        assert len(log.events) == 1

        log.clear()
        assert len(log.events) == 0


class TestActionEventWriter:
    """Test ActionEventWriter for converting inference results."""

    def test_writer_with_dict_prediction(self):
        """Test writer with dictionary prediction format."""
        writer = ActionEventWriter(class_labels=["walking", "running"])
        result = InferenceResult(
            window=tuple(),
            start_frame_index=0,
            end_frame_index=15,
            start_timestamp=0.0,
            end_timestamp=0.5,
            prediction={"class": 0, "confidence": 0.95},
        )
        event = writer.process_inference_result(result)

        assert event is not None
        assert event.label == "walking"
        assert event.confidence == 0.95

    def test_writer_with_logits(self):
        """Test writer with logits array."""
        writer = ActionEventWriter(class_labels=["walking", "running", "jumping"])
        logits = np.array([0.1, 2.5, 0.3])  # argmax=1 (running)
        result = InferenceResult(
            window=tuple(),
            start_frame_index=0,
            end_frame_index=15,
            start_timestamp=0.0,
            end_timestamp=0.5,
            prediction=logits,
        )
        event = writer.process_inference_result(result)

        assert event is not None
        assert event.label == "running"
        assert 0.0 <= event.confidence <= 1.0

    def test_writer_with_torch_tensor(self):
        """Test writer with PyTorch tensor."""
        writer = ActionEventWriter(class_labels=["walking", "running"])
        logits = torch.tensor([1.0, 2.5])
        result = InferenceResult(
            window=tuple(),
            start_frame_index=0,
            end_frame_index=15,
            start_timestamp=0.0,
            end_timestamp=0.5,
            prediction=logits,
        )
        event = writer.process_inference_result(result)

        assert event is not None
        assert event.label == "running"

    def test_writer_with_none_prediction(self):
        """Test writer gracefully handles None prediction."""
        writer = ActionEventWriter()
        result = InferenceResult(
            window=tuple(),
            start_frame_index=0,
            end_frame_index=15,
            start_timestamp=0.0,
            end_timestamp=0.5,
            prediction=None,
        )
        event = writer.process_inference_result(result)

        assert event is None

    def test_writer_add_result(self):
        """Test adding single result to writer."""
        writer = ActionEventWriter(class_labels=["walking", "running"])
        result = InferenceResult(
            window=tuple(),
            start_frame_index=0,
            end_frame_index=15,
            start_timestamp=0.0,
            end_timestamp=0.5,
            prediction={"class": 0, "confidence": 0.95},
        )
        success = writer.add_result(result, track_id=1)

        assert success is True
        assert len(writer.log.events) == 1
        assert writer.log.events[0].track_id == 1

    def test_writer_add_results_multiple(self):
        """Test adding multiple results."""
        writer = ActionEventWriter(class_labels=["walking", "running"])
        results = [
            InferenceResult(
                window=tuple(),
                start_frame_index=0,
                end_frame_index=15,
                start_timestamp=0.0,
                end_timestamp=0.5,
                prediction={"class": 0, "confidence": 0.95},
            ),
            InferenceResult(
                window=tuple(),
                start_frame_index=16,
                end_frame_index=31,
                start_timestamp=0.5,
                end_timestamp=1.0,
                prediction={"class": 1, "confidence": 0.88},
            ),
        ]
        count = writer.add_results(results, track_ids=[1, 1])

        assert count == 2
        assert len(writer.log.events) == 2

    def test_writer_save_to_file(self):
        """Test saving writer log to file."""
        writer = ActionEventWriter(class_labels=["walking", "running"])
        result = InferenceResult(
            window=tuple(),
            start_frame_index=0,
            end_frame_index=15,
            start_timestamp=0.0,
            end_timestamp=0.5,
            prediction={"class": 0, "confidence": 0.95},
        )
        writer.add_result(result)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "output.json"
            writer.save(str(filepath))

            assert filepath.exists()
            data = json.loads(filepath.read_text())
            assert data["event_count"] == 1
            assert data["events"][0]["label"] == "walking"

    def test_writer_clear(self):
        """Test clearing writer log."""
        writer = ActionEventWriter(class_labels=["walking", "running"])
        result = InferenceResult(
            window=tuple(),
            start_frame_index=0,
            end_frame_index=15,
            start_timestamp=0.0,
            end_timestamp=0.5,
            prediction={"class": 0, "confidence": 0.95},
        )
        writer.add_result(result)
        assert len(writer.log.events) == 1

        writer.clear()
        assert len(writer.log.events) == 0

    def test_writer_default_labels(self):
        """Test writer uses numeric labels when class_labels not provided."""
        writer = ActionEventWriter()
        result = InferenceResult(
            window=tuple(),
            start_frame_index=0,
            end_frame_index=15,
            start_timestamp=0.0,
            end_timestamp=0.5,
            prediction={"class": 5, "confidence": 0.85},
        )
        event = writer.process_inference_result(result)

        assert event is not None
        assert event.label == "class_5"

    def test_writer_invalid_dict_prediction(self):
        """Test writer handles invalid dictionary predictions."""
        writer = ActionEventWriter()
        result = InferenceResult(
            window=tuple(),
            start_frame_index=0,
            end_frame_index=15,
            start_timestamp=0.0,
            end_timestamp=0.5,
            prediction={"invalid": "format"},
        )
        event = writer.process_inference_result(result)

        assert event is None

    def test_writer_single_value_confidence(self):
        """Test writer with single confidence value."""
        writer = ActionEventWriter()
        result = InferenceResult(
            window=tuple(),
            start_frame_index=0,
            end_frame_index=15,
            start_timestamp=0.0,
            end_timestamp=0.5,
            prediction=0.75,
        )
        event = writer.process_inference_result(result)

        assert event is not None
        assert event.confidence == 0.75


class TestActionEventValidation:
    """Test new validation features in ActionEvent."""

    def test_action_event_type_check_confidence_string(self):
        """Test ActionEvent rejects string confidence (issue #48)."""
        with pytest.raises(TypeError, match="confidence must be a float"):
            ActionEvent(
                start_frame_index=0,
                end_frame_index=15,
                label="walking",
                confidence="0.95",
            )

    def test_action_event_type_check_confidence_dict(self):
        """Test ActionEvent rejects dict confidence."""
        with pytest.raises(TypeError, match="confidence must be a float"):
            ActionEvent(
                start_frame_index=0,
                end_frame_index=15,
                label="walking",
                confidence={"value": 0.95},
            )

    def test_action_event_type_check_label(self):
        """Test ActionEvent rejects non-string label."""
        with pytest.raises(TypeError, match="label must be a string"):
            ActionEvent(
                start_frame_index=0,
                end_frame_index=15,
                label=123,
                confidence=0.95,
            )

    def test_action_event_type_check_start_frame(self):
        """Test ActionEvent rejects non-int start_frame_index."""
        with pytest.raises(TypeError, match="start_frame_index must be an integer"):
            ActionEvent(
                start_frame_index=0.5,
                end_frame_index=15,
                label="walking",
                confidence=0.95,
            )

    def test_action_event_type_check_end_frame(self):
        """Test ActionEvent rejects non-int end_frame_index."""
        with pytest.raises(TypeError, match="end_frame_index must be an integer"):
            ActionEvent(
                start_frame_index=0,
                end_frame_index="15",
                label="walking",
                confidence=0.95,
            )

    def test_action_event_negative_start_timestamp(self):
        """Test ActionEvent rejects negative start_timestamp (issue #48)."""
        with pytest.raises(ValueError, match="start_timestamp must be >= 0"):
            ActionEvent(
                start_frame_index=0,
                end_frame_index=15,
                label="walking",
                confidence=0.95,
                start_timestamp=-1.0,
            )

    def test_action_event_negative_end_timestamp(self):
        """Test ActionEvent rejects negative end_timestamp (issue #48)."""
        with pytest.raises(ValueError, match="end_timestamp must be >= 0"):
            ActionEvent(
                start_frame_index=0,
                end_frame_index=15,
                label="walking",
                confidence=0.95,
                end_timestamp=-0.5,
            )

    def test_action_event_type_check_start_timestamp(self):
        """Test ActionEvent rejects non-float start_timestamp."""
        with pytest.raises(TypeError, match="start_timestamp must be a float"):
            ActionEvent(
                start_frame_index=0,
                end_frame_index=15,
                label="walking",
                confidence=0.95,
                start_timestamp="0.0",
            )

    def test_action_event_type_check_end_timestamp(self):
        """Test ActionEvent rejects non-float end_timestamp."""
        with pytest.raises(TypeError, match="end_timestamp must be a float"):
            ActionEvent(
                start_frame_index=0,
                end_frame_index=15,
                label="walking",
                confidence=0.95,
                end_timestamp=[0.5],
            )

    def test_action_event_type_check_track_id(self):
        """Test ActionEvent rejects non-int track_id."""
        with pytest.raises(TypeError, match="track_id must be an integer"):
            ActionEvent(
                start_frame_index=0,
                end_frame_index=15,
                label="walking",
                confidence=0.95,
                track_id="1",
            )

    def test_action_event_rejects_bool_confidence(self):
        """Test ActionEvent rejects bool as confidence (bool is subclass of int in Python)."""
        with pytest.raises(TypeError, match="confidence must be a float"):
            ActionEvent(
                start_frame_index=0,
                end_frame_index=15,
                label="walking",
                confidence=True,
            )

    def test_action_event_rejects_bool_timestamps(self):
        """Test ActionEvent rejects bool as timestamps."""
        with pytest.raises(TypeError, match="start_timestamp must be a float"):
            ActionEvent(
                start_frame_index=0,
                end_frame_index=15,
                label="walking",
                confidence=0.95,
                start_timestamp=False,
            )


class TestActionEventLogValidation:
    """Test new validation features in ActionEventLog."""

    def test_action_event_log_add_event_type_check(self):
        """Test add_event rejects non-ActionEvent objects (issue #48)."""
        log = ActionEventLog()
        with pytest.raises(TypeError, match="event must be an ActionEvent instance"):
            log.add_event({"label": "zaq1"})

    def test_action_event_log_add_event_rejects_string(self):
        """Test add_event rejects string."""
        log = ActionEventLog()
        with pytest.raises(TypeError, match="event must be an ActionEvent instance"):
            log.add_event("invalid")

    def test_action_event_log_add_event_rejects_dict(self):
        """Test add_event rejects dict."""
        log = ActionEventLog()
        with pytest.raises(TypeError, match="event must be an ActionEvent instance"):
            log.add_event({
                "start_frame_index": 0,
                "end_frame_index": 15,
                "label": "walking",
                "confidence": 0.95,
            })

    def test_action_event_log_add_event_rejects_none(self):
        """Test add_event rejects None."""
        log = ActionEventLog()
        with pytest.raises(TypeError, match="event must be an ActionEvent instance"):
            log.add_event(None)

    def test_action_event_log_load_valid_event_count(self):
        """Test load_from_file validates event_count consistency (issue #48)."""
        log = ActionEventLog()
        log.add_event(ActionEvent(0, 15, "walking", 0.95))
        log.add_event(ActionEvent(16, 31, "running", 0.87))

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.json"
            log.save_to_file(str(filepath))

            loaded_log = ActionEventLog.load_from_file(str(filepath))
            assert len(loaded_log.events) == 2

    def test_action_event_log_load_event_count_mismatch(self):
        """Test load_from_file raises error on event_count mismatch (issue #48)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "corrupted.json"

            bad_data = {
                "event_count": 5,
                "events": [
                    {
                        "start_frame_index": 0,
                        "end_frame_index": 15,
                        "label": "walking",
                        "confidence": 0.95,
                    },
                    {
                        "start_frame_index": 16,
                        "end_frame_index": 31,
                        "label": "running",
                        "confidence": 0.87,
                    },
                ]
            }

            with open(filepath, "w") as f:
                json.dump(bad_data, f)

            with pytest.raises(ValueError, match="event_count mismatch"):
                ActionEventLog.load_from_file(str(filepath))

    def test_action_event_log_load_no_event_count(self):
        """Test load_from_file works without event_count field."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "no_count.json"

            data = {
                "events": [
                    {
                        "start_frame_index": 0,
                        "end_frame_index": 15,
                        "label": "walking",
                        "confidence": 0.95,
                    },
                ]
            }

            with open(filepath, "w") as f:
                json.dump(data, f)

            loaded_log = ActionEventLog.load_from_file(str(filepath))
            assert len(loaded_log.events) == 1


class TestJsonWriterConfidenceHandling:
    """Test fixed confidence handling in ActionEventWriter (issue #48)."""

    def test_writer_single_confidence_clamping_positive(self):
        """Test writer clamps confidence > 1.0 to valid range."""
        writer = ActionEventWriter()
        result = InferenceResult(
            window=tuple(),
            start_frame_index=0,
            end_frame_index=15,
            start_timestamp=0.0,
            end_timestamp=0.5,
            prediction=1.5,
        )
        event = writer.process_inference_result(result)

        assert event is not None
        assert event.confidence == 1.0
        assert 0.0 <= event.confidence <= 1.0

    def test_writer_single_confidence_clamping_negative(self):
        """Test writer clamps negative confidence to valid range."""
        writer = ActionEventWriter()
        result = InferenceResult(
            window=tuple(),
            start_frame_index=0,
            end_frame_index=15,
            start_timestamp=0.0,
            end_timestamp=0.5,
            prediction=-0.5,
        )
        event = writer.process_inference_result(result)

        assert event is not None
        assert event.confidence == 0.0
        assert 0.0 <= event.confidence <= 1.0

    def test_writer_confidence_label_consistency(self):
        """Test label selection is based on clamped confidence (issue #48)."""
        writer = ActionEventWriter()

        result_high = InferenceResult(
            window=tuple(),
            start_frame_index=0,
            end_frame_index=15,
            start_timestamp=0.0,
            end_timestamp=0.5,
            prediction=0.75,
        )
        event_high = writer.process_inference_result(result_high)
        assert event_high.label == "action"
        assert event_high.confidence == 0.75

        result_low = InferenceResult(
            window=tuple(),
            start_frame_index=0,
            end_frame_index=15,
            start_timestamp=0.5,
            end_timestamp=1.0,
            prediction=0.25,
        )
        event_low = writer.process_inference_result(result_low)
        assert event_low.label == "no_action"
        assert event_low.confidence == 0.25

    def test_writer_no_abs_applied_to_confidence(self):
        """Test that abs() is not applied to confidence (issue #48)."""
        writer = ActionEventWriter()

        result = InferenceResult(
            window=tuple(),
            start_frame_index=0,
            end_frame_index=15,
            start_timestamp=0.0,
            end_timestamp=0.5,
            prediction=0.3,
        )
        event = writer.process_inference_result(result)

        assert event.confidence == 0.3
        assert event.label == "no_action"

    def test_writer_all_confidence_values_valid(self):
        """Test that all returned confidence values are in valid range."""
        writer = ActionEventWriter()

        test_cases = [
            (-2.0, 0.0),
            (-0.5, 0.0),
            (0.0, 0.0),
            (0.3, 0.3),
            (0.5, 0.5),
            (0.75, 0.75),
            (1.0, 1.0),
            (1.5, 1.0),
            (2.0, 1.0),
        ]

        for input_confidence, expected_clamped in test_cases:
            result = InferenceResult(
                window=tuple(),
                start_frame_index=0,
                end_frame_index=15,
                start_timestamp=0.0,
                end_timestamp=0.5,
                prediction=input_confidence,
            )
            event = writer.process_inference_result(result)
            assert event.confidence == expected_clamped, \
                f"Input {input_confidence} should clamp to {expected_clamped}"
            assert 0.0 <= event.confidence <= 1.0


class TestJsonWriterTrackIdsType:
    """Test fixed track_ids type hint in ActionEventWriter (issue #48)."""

    def test_writer_add_results_with_mixed_track_ids(self):
        """Test add_results handles list with None and int values (issue #48)."""
        writer = ActionEventWriter(class_labels=["walking", "running"])
        results = [
            InferenceResult(
                window=tuple(),
                start_frame_index=0,
                end_frame_index=15,
                start_timestamp=0.0,
                end_timestamp=0.5,
                prediction={"class": 0, "confidence": 0.95},
            ),
            InferenceResult(
                window=tuple(),
                start_frame_index=16,
                end_frame_index=31,
                start_timestamp=0.5,
                end_timestamp=1.0,
                prediction={"class": 1, "confidence": 0.88},
            ),
            InferenceResult(
                window=tuple(),
                start_frame_index=32,
                end_frame_index=47,
                start_timestamp=1.0,
                end_timestamp=1.5,
                prediction={"class": 0, "confidence": 0.92},
            ),
        ]

        track_ids = [1, None, 2]
        count = writer.add_results(results, track_ids=track_ids)

        assert count == 3
        assert writer.log.events[0].track_id == 1
        assert writer.log.events[1].track_id is None
        assert writer.log.events[2].track_id == 2

    def test_writer_add_results_all_none_track_ids(self):
        """Test add_results with all None track_ids."""
        writer = ActionEventWriter(class_labels=["walking", "running"])
        results = [
            InferenceResult(
                window=tuple(),
                start_frame_index=0,
                end_frame_index=15,
                start_timestamp=0.0,
                end_timestamp=0.5,
                prediction={"class": 0, "confidence": 0.95},
            ),
            InferenceResult(
                window=tuple(),
                start_frame_index=16,
                end_frame_index=31,
                start_timestamp=0.5,
                end_timestamp=1.0,
                prediction={"class": 1, "confidence": 0.88},
            ),
        ]

        track_ids = [None, None]
        count = writer.add_results(results, track_ids=track_ids)

        assert count == 2
        assert writer.log.events[0].track_id is None
        assert writer.log.events[1].track_id is None
