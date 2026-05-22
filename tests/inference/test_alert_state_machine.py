from typing import Optional

import pytest

from src.app.schemas.action_event import ActionEvent
from src.inference.alert_state_machine import (
    AlertRaisedEvent,
    AlertState,
    AlertStateMachine,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _evt(label: str, track_id: Optional[int] = None) -> ActionEvent:
    return ActionEvent(
        start_frame_index=0,
        end_frame_index=1,
        label=label,
        confidence=0.9,
        track_id=track_id,
    )


def _danger_sm(threshold: int = 3, resolve: int = 1) -> AlertStateMachine:
    return AlertStateMachine(
        persistence_threshold=threshold,
        danger_labels=["fight"],
        resolve_threshold=resolve,
    )


# ---------------------------------------------------------------------------
# Basic state transitions
# ---------------------------------------------------------------------------


def test_inactive_to_candidate_on_first_danger():
    sm = _danger_sm(threshold=3)
    result = sm.process_event(_evt("fight"))
    assert result is None
    assert sm.get_state() == AlertState.CANDIDATE


def test_candidate_to_inactive_on_miss():
    sm = _danger_sm(threshold=3)
    sm.process_event(_evt("fight"))
    sm.process_event(_evt("walk"))
    assert sm.get_state() == AlertState.INACTIVE


def test_candidate_to_active_at_threshold():
    sm = _danger_sm(threshold=3)
    results = [sm.process_event(_evt("fight")) for _ in range(3)]
    assert results[0] is None
    assert results[1] is None
    alert = results[2]
    assert isinstance(alert, AlertRaisedEvent)
    assert alert.state == AlertState.ACTIVE
    assert alert.consecutive_hits == 3
    assert sm.get_state() == AlertState.ACTIVE


def test_active_stays_active_on_continued_danger():
    sm = _danger_sm(threshold=2)
    sm.process_event(_evt("fight"))
    sm.process_event(_evt("fight"))  # → ACTIVE
    result = sm.process_event(_evt("fight"))
    assert result is None
    assert sm.get_state() == AlertState.ACTIVE


def test_active_to_resolved_after_resolve_threshold():
    sm = _danger_sm(threshold=2, resolve=2)
    sm.process_event(_evt("fight"))
    sm.process_event(_evt("fight"))  # → ACTIVE
    sm.process_event(_evt("walk"))   # misses=1, resolve=2 → still ACTIVE
    assert sm.get_state() == AlertState.ACTIVE
    sm.process_event(_evt("walk"))   # misses=2 >= 2 → RESOLVED
    assert sm.get_state() == AlertState.RESOLVED


def test_resolved_to_candidate_on_new_danger():
    sm = _danger_sm(threshold=2, resolve=1)
    sm.process_event(_evt("fight"))
    sm.process_event(_evt("fight"))  # → ACTIVE
    sm.process_event(_evt("walk"))   # → RESOLVED
    sm.process_event(_evt("fight"))  # → CANDIDATE (threshold=2 needs another hit)
    assert sm.get_state() == AlertState.CANDIDATE


def test_resolved_to_inactive_on_miss():
    sm = _danger_sm(threshold=2, resolve=1)
    sm.process_event(_evt("fight"))
    sm.process_event(_evt("fight"))  # → ACTIVE
    sm.process_event(_evt("walk"))   # → RESOLVED
    sm.process_event(_evt("walk"))   # → INACTIVE
    assert sm.get_state() == AlertState.INACTIVE


# ---------------------------------------------------------------------------
# Threshold edge cases
# ---------------------------------------------------------------------------


def test_persistence_threshold_1_goes_directly_to_active():
    sm = AlertStateMachine(persistence_threshold=1, danger_labels=["fight"])
    alert = sm.process_event(_evt("fight"))
    assert isinstance(alert, AlertRaisedEvent)
    assert sm.get_state() == AlertState.ACTIVE


def test_resolve_threshold_1_resolves_on_single_miss():
    sm = _danger_sm(threshold=1, resolve=1)
    sm.process_event(_evt("fight"))  # → ACTIVE immediately
    sm.process_event(_evt("walk"))   # → RESOLVED
    assert sm.get_state() == AlertState.RESOLVED


# ---------------------------------------------------------------------------
# Per-track isolation
# ---------------------------------------------------------------------------


def test_track_isolation():
    sm = _danger_sm(threshold=2)
    # Push track 1 to ACTIVE
    sm.process_event(_evt("fight", track_id=1))
    sm.process_event(_evt("fight", track_id=1))
    # Track 2 receives a miss — must not affect track 1
    sm.process_event(_evt("walk", track_id=2))
    assert sm.get_state(track_id=1) == AlertState.ACTIVE
    assert sm.get_state(track_id=2) == AlertState.INACTIVE


def test_none_track_id_handled():
    sm = _danger_sm(threshold=1)
    alert = sm.process_event(_evt("fight", track_id=None))
    assert isinstance(alert, AlertRaisedEvent)
    assert alert.track_id is None
    assert sm.get_state(track_id=None) == AlertState.ACTIVE


def test_none_track_id_isolated_from_int_track_id():
    sm = _danger_sm(threshold=2)
    sm.process_event(_evt("fight", track_id=None))
    sm.process_event(_evt("fight", track_id=None))  # → ACTIVE for track None
    # Integer track 0 should be INACTIVE
    assert sm.get_state(track_id=0) == AlertState.INACTIVE
    assert sm.get_state(track_id=None) == AlertState.ACTIVE


# ---------------------------------------------------------------------------
# danger_labels filter
# ---------------------------------------------------------------------------


def test_danger_labels_filter_safe_label_is_miss():
    sm = AlertStateMachine(persistence_threshold=3, danger_labels=["fight", "steal"])
    sm.process_event(_evt("fight"))   # hit → CANDIDATE
    sm.process_event(_evt("steal"))   # hit → CANDIDATE (hits=2)
    sm.process_event(_evt("walk"))    # miss → INACTIVE
    assert sm.get_state() == AlertState.INACTIVE


def test_danger_labels_none_treats_all_labels_as_danger():
    sm = AlertStateMachine(persistence_threshold=2, danger_labels=None)
    sm.process_event(_evt("walk"))    # any label is danger → CANDIDATE
    alert = sm.process_event(_evt("sit"))  # second → ACTIVE
    assert isinstance(alert, AlertRaisedEvent)
    assert sm.get_state() == AlertState.ACTIVE


def test_danger_labels_empty_list_treats_all_labels_as_danger():
    sm = AlertStateMachine(persistence_threshold=2, danger_labels=[])
    sm.process_event(_evt("walk"))
    alert = sm.process_event(_evt("sit"))
    assert isinstance(alert, AlertRaisedEvent)


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------


def test_process_events_batch():
    sm = _danger_sm(threshold=2)
    events = [_evt("fight"), _evt("fight"), _evt("walk"), _evt("fight")]
    alerts = sm.process_events(events)
    # Only the 2nd event triggers an alert
    assert len(alerts) == 1
    assert alerts[0].consecutive_hits == 2


def test_process_events_returns_empty_when_no_alerts():
    sm = _danger_sm(threshold=3)
    alerts = sm.process_events([_evt("fight"), _evt("fight")])
    assert alerts == []


# ---------------------------------------------------------------------------
# active_track_ids
# ---------------------------------------------------------------------------


def test_active_track_ids_returns_only_active_tracks():
    sm = _danger_sm(threshold=1)
    sm.process_event(_evt("fight", track_id=1))   # → ACTIVE
    sm.process_event(_evt("fight", track_id=2))   # → ACTIVE
    sm.process_event(_evt("fight", track_id=3))   # → ACTIVE
    sm.process_event(_evt("walk", track_id=2))    # → RESOLVED (resolve=1)
    active = sm.active_track_ids()
    assert set(active) == {1, 3}


def test_active_track_ids_empty_when_no_active_tracks():
    sm = _danger_sm(threshold=3)
    sm.process_event(_evt("fight", track_id=5))
    assert sm.active_track_ids() == []


# ---------------------------------------------------------------------------
# reset_all
# ---------------------------------------------------------------------------


def test_reset_all_clears_state():
    sm = _danger_sm(threshold=1)
    sm.process_event(_evt("fight", track_id=1))
    sm.process_event(_evt("fight", track_id=2))
    assert sm.get_state(track_id=1) == AlertState.ACTIVE

    sm.reset_all()

    assert sm.get_state(track_id=1) == AlertState.INACTIVE
    assert sm.get_state(track_id=2) == AlertState.INACTIVE
    assert sm.get_record(track_id=1) is None
    assert sm.active_track_ids() == []


# ---------------------------------------------------------------------------
# get_record
# ---------------------------------------------------------------------------


def test_get_record_returns_none_for_unseen_track():
    sm = _danger_sm()
    assert sm.get_record(track_id=99) is None


def test_get_record_reflects_hits_and_state():
    sm = _danger_sm(threshold=3)
    sm.process_event(_evt("fight"))
    sm.process_event(_evt("fight"))
    record = sm.get_record()
    assert record is not None
    assert record.consecutive_hits == 2
    assert record.state == AlertState.CANDIDATE


# ---------------------------------------------------------------------------
# AlertRaisedEvent content
# ---------------------------------------------------------------------------


def test_alert_raised_event_contains_triggering_event():
    sm = _danger_sm(threshold=1)
    ev = _evt("fight", track_id=7)
    alert = sm.process_event(ev)
    assert alert is not None
    assert alert.triggering_event is ev
    assert alert.label == "fight"
    assert alert.track_id == 7


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_invalid_persistence_threshold_raises():
    with pytest.raises(ValueError):
        AlertStateMachine(persistence_threshold=0)
    with pytest.raises(TypeError):
        AlertStateMachine(persistence_threshold="3")  # type: ignore[arg-type]


def test_invalid_resolve_threshold_raises():
    with pytest.raises(ValueError):
        AlertStateMachine(resolve_threshold=0)
    with pytest.raises(TypeError):
        AlertStateMachine(resolve_threshold=1.5)  # type: ignore[arg-type]


def test_invalid_danger_labels_raises():
    with pytest.raises(TypeError):
        AlertStateMachine(danger_labels="fight")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        AlertStateMachine(danger_labels=[1, 2])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# set_thresholds
# ---------------------------------------------------------------------------


def test_set_thresholds_updates_persistence():
    # Start with persistence=3; lower to 1 mid-sequence — next hit should activate.
    sm = AlertStateMachine(persistence_threshold=3, danger_labels=["fight"])
    sm.process_event(_evt("fight"))  # hit 1 → CANDIDATE
    assert sm.get_state() == AlertState.CANDIDATE

    sm.set_thresholds(persistence_threshold=1, resolve_threshold=1)
    # Threshold is now 1; one more danger event triggers ACTIVE.
    alert = sm.process_event(_evt("fight"))  # hit 2, threshold=1 already met → ACTIVE
    assert alert is not None
    assert sm.get_state() == AlertState.ACTIVE


def test_set_thresholds_preserves_track_state():
    sm = AlertStateMachine(persistence_threshold=3, danger_labels=["fight"])
    sm.process_event(_evt("fight", track_id=1))  # hit 1
    sm.process_event(_evt("fight", track_id=1))  # hit 2

    record_before = sm.get_record(track_id=1)
    assert record_before is not None
    hits_before = record_before.consecutive_hits
    state_before = record_before.state

    sm.set_thresholds(persistence_threshold=5, resolve_threshold=2)

    record_after = sm.get_record(track_id=1)
    assert record_after is not None
    assert record_after.consecutive_hits == hits_before
    assert record_after.state == state_before


def test_set_thresholds_invalid_raises():
    sm = AlertStateMachine()
    with pytest.raises(ValueError):
        sm.set_thresholds(persistence_threshold=0, resolve_threshold=1)
    with pytest.raises(ValueError):
        sm.set_thresholds(persistence_threshold=1, resolve_threshold=0)
    with pytest.raises(TypeError):
        sm.set_thresholds(persistence_threshold="2", resolve_threshold=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        sm.set_thresholds(persistence_threshold=1, resolve_threshold=1.5)  # type: ignore[arg-type]
