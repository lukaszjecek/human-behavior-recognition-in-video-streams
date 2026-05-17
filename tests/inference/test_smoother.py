from typing import Optional

import pytest

from src.inference.action_event import ActionEvent
from src.inference.alert_state_machine import AlertState, AlertStateMachine
from src.inference.smoother import MajorityVoteSmoother

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _evt(
    label: str,
    track_id: Optional[int] = None,
    confidence: float = 0.9,
    start: int = 0,
    end: int = 1,
) -> ActionEvent:
    return ActionEvent(
        start_frame_index=start,
        end_frame_index=end,
        label=label,
        confidence=confidence,
        track_id=track_id,
    )


# ---------------------------------------------------------------------------
# Buffer fill behaviour
# ---------------------------------------------------------------------------


def test_returns_none_until_buffer_full():
    smoother = MajorityVoteSmoother(window_size=3)
    assert smoother.update(_evt("fight")) is None
    assert smoother.update(_evt("fight")) is None


def test_returns_event_when_buffer_full():
    smoother = MajorityVoteSmoother(window_size=3)
    smoother.update(_evt("fight"))
    smoother.update(_evt("fight"))
    result = smoother.update(_evt("fight"))
    assert isinstance(result, ActionEvent)


# ---------------------------------------------------------------------------
# Majority vote label selection
# ---------------------------------------------------------------------------


def test_majority_label_wins():
    smoother = MajorityVoteSmoother(window_size=5)
    result = None
    for label in ["fight", "fight", "fight", "walk", "walk"]:
        result = smoother.update(_evt(label))
    assert result is not None
    assert result.label == "fight"


def test_minority_label_does_not_win():
    smoother = MajorityVoteSmoother(window_size=5)
    result = None
    for label in ["fight", "walk", "walk", "walk", "walk"]:
        result = smoother.update(_evt(label))
    assert result is not None
    assert result.label == "walk"


def test_tie_resolved_by_most_recent():
    # Buffer: walk, fight, walk, fight — tie 2:2; most recent is "fight"
    smoother = MajorityVoteSmoother(window_size=4)
    result = None
    for label in ["walk", "fight", "walk", "fight"]:
        result = smoother.update(_evt(label))
    assert result is not None
    assert result.label == "fight"


# ---------------------------------------------------------------------------
# Sliding window
# ---------------------------------------------------------------------------


def test_sliding_window_updates_on_each_event():
    smoother = MajorityVoteSmoother(window_size=3)
    for _ in range(3):
        smoother.update(_evt("fight"))          # fill: fight, fight, fight

    result = smoother.update(_evt("walk"))      # window: fight, fight, walk → fight
    assert result is not None
    assert result.label == "fight"

    result = smoother.update(_evt("walk"))      # window: fight, walk, walk → walk
    assert result is not None
    assert result.label == "walk"


# ---------------------------------------------------------------------------
# Confidence averaging
# ---------------------------------------------------------------------------


def test_confidence_averaged_over_winning_label_only():
    smoother = MajorityVoteSmoother(window_size=3)
    smoother.update(_evt("fight", confidence=0.8))
    smoother.update(_evt("fight", confidence=0.6))
    result = smoother.update(_evt("walk", confidence=0.9))
    # fight wins 2–1; avg confidence = (0.8 + 0.6) / 2 = 0.7
    assert result is not None
    assert result.label == "fight"
    assert abs(result.confidence - 0.7) < 1e-9


# ---------------------------------------------------------------------------
# Per-track isolation
# ---------------------------------------------------------------------------


def test_track_isolation():
    smoother = MajorityVoteSmoother(window_size=3)
    for _ in range(3):
        smoother.update(_evt("fight", track_id=1))
    for _ in range(3):
        smoother.update(_evt("walk", track_id=2))

    result1 = smoother.update(_evt("fight", track_id=1))
    result2 = smoother.update(_evt("walk", track_id=2))

    assert result1 is not None and result1.label == "fight"
    assert result2 is not None and result2.label == "walk"


def test_none_track_id_handled():
    smoother = MajorityVoteSmoother(window_size=2)
    smoother.update(_evt("fight", track_id=None))
    result = smoother.update(_evt("fight", track_id=None))
    assert result is not None
    assert result.track_id is None
    assert result.label == "fight"


# ---------------------------------------------------------------------------
# reset
# ---------------------------------------------------------------------------


def test_reset_single_track_clears_buffer():
    smoother = MajorityVoteSmoother(window_size=3)
    for _ in range(3):
        smoother.update(_evt("fight", track_id=1))
    smoother.update(_evt("walk", track_id=2))
    smoother.update(_evt("walk", track_id=2))

    smoother.reset(track_id=1)

    # Track 1 buffer cleared — two subsequent events still return None
    assert smoother.update(_evt("fight", track_id=1)) is None
    assert smoother.update(_evt("fight", track_id=1)) is None

    # Track 2 unaffected — one more event completes its buffer
    result = smoother.update(_evt("walk", track_id=2))
    assert result is not None
    assert result.label == "walk"


def test_reset_all_clears_all_tracks():
    smoother = MajorityVoteSmoother(window_size=2)
    smoother.update(_evt("fight", track_id=1))
    smoother.update(_evt("walk", track_id=2))

    smoother.reset()

    assert smoother.update(_evt("fight", track_id=1)) is None
    assert smoother.update(_evt("walk", track_id=2)) is None


# ---------------------------------------------------------------------------
# is_ready
# ---------------------------------------------------------------------------


def test_is_ready_false_before_full():
    smoother = MajorityVoteSmoother(window_size=3)
    smoother.update(_evt("fight"))
    smoother.update(_evt("fight"))
    assert smoother.is_ready() is False


def test_is_ready_true_when_full():
    smoother = MajorityVoteSmoother(window_size=3)
    smoother.update(_evt("fight"))
    smoother.update(_evt("fight"))
    smoother.update(_evt("fight"))
    assert smoother.is_ready() is True


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_window_size_1_returns_immediately():
    smoother = MajorityVoteSmoother(window_size=1)
    result = smoother.update(_evt("fight"))
    assert result is not None
    assert result.label == "fight"


def test_output_event_metadata_from_most_recent():
    smoother = MajorityVoteSmoother(window_size=3)
    smoother.update(_evt("fight", track_id=5, start=0, end=10))
    smoother.update(_evt("fight", track_id=5, start=10, end=20))
    result = smoother.update(_evt("fight", track_id=5, start=20, end=30))
    assert result is not None
    assert result.start_frame_index == 20
    assert result.end_frame_index == 30
    assert result.track_id == 5


def test_invalid_window_size_raises():
    with pytest.raises(ValueError):
        MajorityVoteSmoother(window_size=0)
    with pytest.raises(TypeError):
        MajorityVoteSmoother(window_size=2.5)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Sequence stability
# ---------------------------------------------------------------------------


def test_noisy_then_stable_sequence():
    smoother = MajorityVoteSmoother(window_size=5)
    result = None
    # Noisy: fight wins 3–2
    for label in ["fight", "walk", "fight", "walk", "fight"]:
        result = smoother.update(_evt(label))
    assert result is not None
    assert result.label == "fight"

    # Stable: all walk — after 5 events walk dominates completely
    for _ in range(5):
        result = smoother.update(_evt("walk"))
    assert result is not None
    assert result.label == "walk"


def test_stable_then_noisy_sequence():
    smoother = MajorityVoteSmoother(window_size=5)
    result = None
    # Stable: all walk
    for _ in range(5):
        result = smoother.update(_evt("walk"))
    assert result is not None
    assert result.label == "walk"

    # Noisy: fight gradually takes over — after 5 noisy events fight wins 3–2
    for label in ["fight", "walk", "fight", "walk", "fight"]:
        result = smoother.update(_evt(label))
    assert result is not None
    assert result.label == "fight"


# ---------------------------------------------------------------------------
# Mini test flow
# ---------------------------------------------------------------------------


def test_smoother_output_feeds_alert_state_machine_alert_raised():
    """Noisy sequence that majority-vote stabilises to danger → alert raised."""
    smoother = MajorityVoteSmoother(window_size=3)
    sm = AlertStateMachine(persistence_threshold=2, danger_labels=["fight"])

    # window 1: ["fight", "walk", "fight"] → majority "fight" → CANDIDATE (hits=1)
    smoother.update(_evt("fight", track_id=1))
    smoother.update(_evt("walk", track_id=1))
    smoothed = smoother.update(_evt("fight", track_id=1))
    assert smoothed is not None and smoothed.label == "fight"
    sm.process_event(smoothed)
    assert sm.get_state(track_id=1) == AlertState.CANDIDATE

    # window 2: ["walk", "fight", "fight"] → majority "fight" → ACTIVE (hits=2)
    smoother.update(_evt("walk", track_id=1))
    smoother.update(_evt("fight", track_id=1))
    smoothed = smoother.update(_evt("fight", track_id=1))
    assert smoothed is not None and smoothed.label == "fight"
    alert = sm.process_event(smoothed)
    assert alert is not None
    assert sm.get_state(track_id=1) == AlertState.ACTIVE


def test_smoother_suppresses_noise_and_alert_not_raised():
    """Noisy sequence that majority-vote stabilises to safe → no alert raised."""
    smoother = MajorityVoteSmoother(window_size=3)
    sm = AlertStateMachine(persistence_threshold=2, danger_labels=["fight"])

    # window 1: ["fight", "walk", "walk"] → majority "walk" → sm stays INACTIVE
    smoother.update(_evt("fight", track_id=1))
    smoother.update(_evt("walk", track_id=1))
    smoothed = smoother.update(_evt("walk", track_id=1))
    assert smoothed is not None and smoothed.label == "walk"
    sm.process_event(smoothed)
    assert sm.get_state(track_id=1) == AlertState.INACTIVE

    # window 2: ["walk", "walk", "fight"] → majority "walk" → sm stays INACTIVE
    smoother.update(_evt("walk", track_id=1))
    smoother.update(_evt("walk", track_id=1))
    smoothed = smoother.update(_evt("fight", track_id=1))
    assert smoothed is not None and smoothed.label == "walk"
    sm.process_event(smoothed)
    assert sm.get_state(track_id=1) == AlertState.INACTIVE