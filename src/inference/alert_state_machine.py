"""Per-track alert state machine for behavior detection pipelines.

Transitions action events through INACTIVE → CANDIDATE → ACTIVE → RESOLVED
states with configurable persistence and resolve thresholds.
"""

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from src.inference.action_event import ActionEvent

logger = logging.getLogger(__name__)


class AlertState(Enum):
    """States of the per-track alert state machine."""

    INACTIVE = auto()
    CANDIDATE = auto()
    ACTIVE = auto()
    RESOLVED = auto()


@dataclass
class TrackAlertRecord:
    """Mutable alert state record for a single track.

    Attributes:
        state: Current alert state for this track.
        consecutive_hits: Number of consecutive danger events since last reset.
        consecutive_misses: Number of consecutive non-danger events since last hit.
    """

    state: AlertState = AlertState.INACTIVE
    consecutive_hits: int = 0
    consecutive_misses: int = 0

@dataclass(frozen=True)
class AlertRaisedEvent:
    """Emitted when a track transitions to ACTIVE state.

    Attributes:
        track_id: Track identifier that triggered the alert.
        label: Danger label from the triggering event.
        state: Alert state at emission time (always ACTIVE).
        consecutive_hits: Number of consecutive danger events that triggered activation.
        triggering_event: The ActionEvent that caused the transition to ACTIVE.
    """

    track_id: Optional[int]
    label: str
    state: AlertState
    consecutive_hits: int
    triggering_event: ActionEvent


class AlertStateMachine:
    """Per-track alert state machine for detecting persistent dangerous behaviors.

    Each track_id evolves independently through:
        INACTIVE → CANDIDATE → ACTIVE → RESOLVED → INACTIVE

    The machine is deterministic: identical event sequences always produce
    identical state transitions and outputs.

    Example:
        >>> sm = AlertStateMachine(persistence_threshold=2, danger_labels=["fighting"])
        >>> event = ActionEvent(0, 1, "fighting", 0.9, track_id=1)
        >>> sm.process_event(event)  # first hit → CANDIDATE
        >>> sm.process_event(event)  # second hit → ACTIVE, returns AlertRaisedEvent
    """

    def __init__(
        self,
        persistence_threshold: int = 3,
        danger_labels: Optional[list[str]] = None,
        resolve_threshold: int = 1,
    ) -> None:
        """Initialize the state machine with threshold parameters.

        Args:
            persistence_threshold: Number of consecutive danger events required
                to transition from INACTIVE/CANDIDATE to ACTIVE. Must be >= 1.
            danger_labels: Labels considered dangerous. None or empty list means
                every label is treated as danger (miss is never triggered).
            resolve_threshold: Number of consecutive non-danger events required
                to transition from ACTIVE to RESOLVED. Must be >= 1.

        Raises:
            TypeError: If threshold arguments are not integers.
            ValueError: If threshold arguments are less than 1.
        """
        if not isinstance(persistence_threshold, int) or isinstance(
            persistence_threshold, bool
        ):
            raise TypeError("persistence_threshold must be an integer")
        if persistence_threshold < 1:
            raise ValueError("persistence_threshold must be >= 1")

        if danger_labels is not None:
            if not isinstance(danger_labels, list):
                raise TypeError("danger_labels must be a list of strings or None")
            for label in danger_labels:
                if not isinstance(label, str):
                    raise TypeError("danger_labels must contain only strings")
                if not label.strip():
                    raise ValueError("danger_labels must not contain empty strings")

        if not isinstance(resolve_threshold, int) or isinstance(resolve_threshold, bool):
            raise TypeError("resolve_threshold must be an integer")
        if resolve_threshold < 1:
            raise ValueError("resolve_threshold must be >= 1")

        self._persistence_threshold = persistence_threshold
        self._danger_labels: Optional[list[str]] = danger_labels
        self._resolve_threshold = resolve_threshold
        self._tracks: dict[Optional[int], TrackAlertRecord] = {}

    def process_event(self, event: ActionEvent) -> Optional[AlertRaisedEvent]:
        """Process a single action event and update the corresponding track state.

        Args:
            event: ActionEvent to process. The track_id field identifies which
                track record to update.

        Returns:
            AlertRaisedEvent when a track transitions to ACTIVE, None otherwise.

        Raises:
            TypeError: If event is not an ActionEvent instance.
        """
        if not isinstance(event, ActionEvent):
            raise TypeError("event must be an ActionEvent instance")

        track_id = event.track_id
        if track_id not in self._tracks:
            self._tracks[track_id] = TrackAlertRecord()

        record = self._tracks[track_id]
        return self._apply_transition(record, event)

    def process_events(self, events: list[ActionEvent]) -> list[AlertRaisedEvent]:
        """Process a batch of action events and collect all raised alerts.

        Args:
            events: Ordered list of ActionEvent objects to process.

        Returns:
            List of AlertRaisedEvent objects emitted during processing,
            in the order they were triggered.

        Raises:
            TypeError: If events is not a list.
        """
        if not isinstance(events, list):
            raise TypeError("events must be a list of ActionEvent objects")

        raised: list[AlertRaisedEvent] = []
        for event in events:
            result = self.process_event(event)
            if result is not None:
                raised.append(result)
        return raised

    def get_state(self, track_id: Optional[int] = None) -> AlertState:
        """Return the current alert state for a track.

        Args:
            track_id: Track identifier to query. Defaults to None.

        Returns:
            Current AlertState for the track, INACTIVE if never seen.
        """
        record = self._tracks.get(track_id)
        return record.state if record is not None else AlertState.INACTIVE

    def get_record(self, track_id: Optional[int] = None) -> Optional[TrackAlertRecord]:
        """Return the full alert record for a track.

        Args:
            track_id: Track identifier to query. Defaults to None.

        Returns:
            TrackAlertRecord if the track has been seen, None otherwise.
        """
        return self._tracks.get(track_id)

    def active_track_ids(self) -> list[Optional[int]]:
        """Return track IDs whose current state is ACTIVE.

        Returns:
            List of track_id values (may include None) in ACTIVE state.
        """
        return [
            tid
            for tid, record in self._tracks.items()
            if record.state == AlertState.ACTIVE
        ]

    def reset_all(self) -> None:
        """Reset all track records, returning every track to INACTIVE.

        Clears all accumulated hit/miss counts and removes all track entries.
        """
        self._tracks.clear()
        logger.debug("AlertStateMachine: all tracks reset to INACTIVE")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_danger(self, label: str) -> bool:
        """Return True when label counts as a danger event.

        Args:
            label: Action label from an ActionEvent.

        Returns:
            True if the label is considered dangerous given current config.
        """
        if not self._danger_labels:
            return True
        return label in self._danger_labels

    def _apply_transition(
        self,
        record: TrackAlertRecord,
        event: ActionEvent,
    ) -> Optional[AlertRaisedEvent]:
        """Apply one state-machine transition and return an alert if raised.

        Args:
            record: Mutable per-track state record (mutated in place).
            event: ActionEvent that drives the transition.

        Returns:
            AlertRaisedEvent if the track just became ACTIVE, None otherwise.
        """
        is_danger = self._is_danger(event.label)
        track_id = event.track_id

        if record.state == AlertState.INACTIVE:
            if is_danger:
                record.consecutive_hits = 1
                record.consecutive_misses = 0
                if record.consecutive_hits >= self._persistence_threshold:
                    record.state = AlertState.ACTIVE
                    logger.info(
                        "Track %s: INACTIVE → ACTIVE (hits=%d)",
                        track_id,
                        record.consecutive_hits,
                    )
                    return self._make_alert(record, event)
                record.state = AlertState.CANDIDATE
                logger.debug(
                    "Track %s: INACTIVE → CANDIDATE (hits=%d)",
                    track_id,
                    record.consecutive_hits,
                )
            # miss in INACTIVE → stay INACTIVE (no log noise)

        elif record.state == AlertState.CANDIDATE:
            if is_danger:
                record.consecutive_hits += 1
                if record.consecutive_hits >= self._persistence_threshold:
                    record.state = AlertState.ACTIVE
                    logger.info(
                        "Track %s: CANDIDATE → ACTIVE (hits=%d)",
                        track_id,
                        record.consecutive_hits,
                    )
                    return self._make_alert(record, event)
                logger.debug(
                    "Track %s: CANDIDATE stays CANDIDATE (hits=%d)",
                    track_id,
                    record.consecutive_hits,
                )
            else:
                record.state = AlertState.INACTIVE
                record.consecutive_hits = 0
                record.consecutive_misses = 0
                logger.debug("Track %s: CANDIDATE → INACTIVE (miss)", track_id)

        elif record.state == AlertState.ACTIVE:
            if is_danger:
                record.consecutive_misses = 0
                logger.debug("Track %s: ACTIVE stays ACTIVE (danger)", track_id)
            else:
                record.consecutive_misses += 1
                if record.consecutive_misses >= self._resolve_threshold:
                    record.state = AlertState.RESOLVED
                    record.consecutive_hits = 0
                    logger.debug(
                        "Track %s: ACTIVE → RESOLVED (misses=%d)",
                        track_id,
                        record.consecutive_misses,
                    )
                else:
                    logger.debug(
                        "Track %s: ACTIVE stays ACTIVE (misses=%d, resolve_threshold=%d)",
                        track_id,
                        record.consecutive_misses,
                        self._resolve_threshold,
                    )

        elif record.state == AlertState.RESOLVED:
            if is_danger:
                record.consecutive_hits = 1
                record.consecutive_misses = 0
                if record.consecutive_hits >= self._persistence_threshold:
                    record.state = AlertState.ACTIVE
                    logger.info(
                        "Track %s: RESOLVED → ACTIVE (hits=%d)",
                        track_id,
                        record.consecutive_hits,
                    )
                    return self._make_alert(record, event)
                record.state = AlertState.CANDIDATE
                logger.debug(
                    "Track %s: RESOLVED → CANDIDATE (hits=%d)",
                    track_id,
                    record.consecutive_hits,
                )
            else:
                record.state = AlertState.INACTIVE
                record.consecutive_hits = 0
                record.consecutive_misses = 0
                logger.debug("Track %s: RESOLVED → INACTIVE (miss)", track_id)

        return None

    @staticmethod
    def _make_alert(
        record: TrackAlertRecord,
        event: ActionEvent,
    ) -> AlertRaisedEvent:
        """Construct an AlertRaisedEvent from the current record and event.

        Args:
            record: Current track alert record (state must already be ACTIVE).
            event: ActionEvent that triggered the transition.

        Returns:
            Populated AlertRaisedEvent.
        """
        return AlertRaisedEvent(
            track_id=event.track_id,
            label=event.label,
            state=AlertState.ACTIVE,
            consecutive_hits=record.consecutive_hits,
            triggering_event=event,
        )


__all__ = [
    "AlertRaisedEvent",
    "AlertState",
    "AlertStateMachine",
    "TrackAlertRecord",
]