from typing import Optional

import pytest

from src.inference.action_event import ActionEvent
from src.inference.aggregator import BusinessEvent, EventAggregator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _evt(
    label: str,
    track_id: Optional[int] = None,
    start: int = 0,
    end: int = 15,
    confidence: float = 0.9,
    start_ts: Optional[float] = None,
    end_ts: Optional[float] = None,
) -> ActionEvent:
    return ActionEvent(
        start_frame_index=start,
        end_frame_index=end,
        label=label,
        confidence=confidence,
        track_id=track_id,
        start_timestamp=start_ts,
        end_timestamp=end_ts,
    )


# ---------------------------------------------------------------------------
# Single / double window — no close
# ---------------------------------------------------------------------------


def test_single_window_returns_none():
    agg = EventAggregator()
    assert agg.update(_evt("walk")) is None


def test_two_windows_same_label_returns_none():
    agg = EventAggregator()
    agg.update(_evt("walk", start=0, end=15))
    result = agg.update(_evt("walk", start=16, end=31))
    assert result is None


# ---------------------------------------------------------------------------
# Label change
# ---------------------------------------------------------------------------


def test_label_change_closes_event():
    agg = EventAggregator()
    agg.update(_evt("walk", start=0, end=15))
    agg.update(_evt("walk", start=16, end=31))
    result = agg.update(_evt("fight", start=32, end=47))
    assert isinstance(result, BusinessEvent)
    assert result.label == "walk"


def test_label_change_starts_new_event():
    agg = EventAggregator()
    agg.update(_evt("walk", start=0, end=15))
    agg.update(_evt("fight", start=16, end=31))  # closes "walk", opens "fight"
    result = agg.update(_evt("run", start=32, end=47))  # closes "fight"
    assert isinstance(result, BusinessEvent)
    assert result.label == "fight"


# ---------------------------------------------------------------------------
# Continuity rules
# ---------------------------------------------------------------------------


def test_gap_between_windows_closes_event():
    agg = EventAggregator()
    agg.update(_evt("walk", start=0, end=15))
    # 20 > 15 + 1 → gap detected
    result = agg.update(_evt("walk", start=20, end=35))
    assert isinstance(result, BusinessEvent)
    assert result.label == "walk"


def test_overlapping_windows_are_continuous():
    agg = EventAggregator()
    agg.update(_evt("walk", start=0, end=15))
    # 8 <= 15 + 1 → continuous
    result = agg.update(_evt("walk", start=8, end=23))
    assert result is None


def test_adjacent_windows_are_continuous():
    agg = EventAggregator()
    agg.update(_evt("walk", start=0, end=15))
    # 16 == 15 + 1 → continuous
    result = agg.update(_evt("walk", start=16, end=31))
    assert result is None


# ---------------------------------------------------------------------------
# BusinessEvent field correctness
# ---------------------------------------------------------------------------


def test_business_event_start_end_frames():
    agg = EventAggregator()
    agg.update(_evt("walk", start=10, end=25))
    agg.update(_evt("walk", start=26, end=41))
    result = agg.update(_evt("fight", start=42, end=57))
    assert result is not None
    assert result.start_frame_index == 10
    assert result.end_frame_index == 41


def test_business_event_duration_windows():
    agg = EventAggregator()
    agg.update(_evt("walk", start=0, end=15))
    agg.update(_evt("walk", start=16, end=31))
    agg.update(_evt("walk", start=32, end=47))
    result = agg.update(_evt("fight", start=48, end=63))
    assert result is not None
    assert result.duration_windows == 3


def test_business_event_mean_confidence():
    agg = EventAggregator()
    agg.update(_evt("walk", start=0, end=15, confidence=0.6))
    agg.update(_evt("walk", start=16, end=31, confidence=0.8))
    agg.update(_evt("walk", start=32, end=47, confidence=1.0))
    result = agg.update(_evt("fight", start=48, end=63))
    assert result is not None
    assert abs(result.mean_confidence - 0.8) < 1e-9


def test_business_event_timestamps():
    agg = EventAggregator()
    agg.update(_evt("walk", start=0, end=15, start_ts=0.0, end_ts=0.5))
    agg.update(_evt("walk", start=16, end=31, start_ts=0.5, end_ts=1.0))
    result = agg.update(_evt("fight", start=32, end=47, start_ts=1.0, end_ts=1.5))
    assert result is not None
    assert result.start_timestamp == 0.0
    assert result.end_timestamp == 1.0


# ---------------------------------------------------------------------------
# Per-track isolation
# ---------------------------------------------------------------------------


def test_track_isolation():
    agg = EventAggregator()
    agg.update(_evt("walk", track_id=1, start=0, end=15))
    agg.update(_evt("walk", track_id=1, start=16, end=31))
    agg.update(_evt("fight", track_id=2, start=0, end=15))

    result1 = agg.update(_evt("fight", track_id=1, start=32, end=47))
    assert isinstance(result1, BusinessEvent)
    assert result1.label == "walk"
    assert result1.track_id == 1

    result2 = agg.update(_evt("fight", track_id=2, start=16, end=31))
    assert result2 is None


def test_none_track_id_handled():
    agg = EventAggregator()
    agg.update(_evt("walk", track_id=None, start=0, end=15))
    result = agg.update(_evt("fight", track_id=None, start=16, end=31))
    assert isinstance(result, BusinessEvent)
    assert result.track_id is None
    assert result.label == "walk"


# ---------------------------------------------------------------------------
# flush
# ---------------------------------------------------------------------------


def test_flush_all_returns_open_events():
    agg = EventAggregator()
    agg.update(_evt("walk", track_id=1, start=0, end=15))
    agg.update(_evt("fight", track_id=2, start=0, end=15))
    results = agg.flush()
    assert len(results) == 2
    labels = {r.label for r in results}
    assert labels == {"walk", "fight"}


def test_flush_clears_buffered_state():
    agg = EventAggregator()
    agg.update(_evt("walk", start=0, end=15))
    agg.flush()
    # After flush, no open event — next update opens a fresh one
    result = agg.update(_evt("walk", start=16, end=31))
    assert result is None
    results = agg.flush()
    assert len(results) == 1
    assert results[0].duration_windows == 1


def test_flush_single_track():
    agg = EventAggregator()
    agg.update(_evt("walk", track_id=1, start=0, end=15))
    agg.update(_evt("fight", track_id=2, start=0, end=15))

    results = agg.flush(track_id=1)
    assert len(results) == 1
    assert results[0].track_id == 1
    assert results[0].label == "walk"

    # Track 2 still open
    remaining = agg.flush()
    assert len(remaining) == 1
    assert remaining[0].track_id == 2


def test_flush_empty_returns_empty_list():
    agg = EventAggregator()
    assert agg.flush() == []


def test_flush_deterministic_order():
    agg = EventAggregator()
    agg.update(_evt("a", track_id=3, start=0, end=15))
    agg.update(_evt("b", track_id=1, start=0, end=15))
    agg.update(_evt("c", track_id=None, start=0, end=15))
    agg.update(_evt("d", track_id=2, start=0, end=15))

    results = agg.flush()
    assert len(results) == 4
    assert [r.track_id for r in results] == [1, 2, 3, None]


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------


def test_reset_discards_without_emitting():
    agg = EventAggregator()
    agg.update(_evt("walk", start=0, end=15))
    agg.reset()
    assert agg.flush() == []


# ---------------------------------------------------------------------------
# Sequences
# ---------------------------------------------------------------------------


def test_multiple_splits_in_sequence():
    agg = EventAggregator()

    agg.update(_evt("walk", start=0, end=15))
    agg.update(_evt("walk", start=16, end=31))
    e1 = agg.update(_evt("fight", start=32, end=47))
    assert e1 is not None and e1.label == "walk" and e1.duration_windows == 2

    agg.update(_evt("fight", start=48, end=63))
    e2 = agg.update(_evt("run", start=64, end=79))
    assert e2 is not None and e2.label == "fight" and e2.duration_windows == 2

    results = agg.flush()
    assert len(results) == 1
    assert results[0].label == "run"


def test_single_window_flush():
    agg = EventAggregator()
    agg.update(_evt("walk", start=0, end=15))
    results = agg.flush()
    assert len(results) == 1
    assert results[0].duration_windows == 1
    assert results[0].label == "walk"
    assert results[0].start_frame_index == 0
    assert results[0].end_frame_index == 15


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_invalid_event_raises():
    agg = EventAggregator()
    with pytest.raises(TypeError, match="ActionEvent"):
        agg.update("not an event")  # type: ignore[arg-type]
