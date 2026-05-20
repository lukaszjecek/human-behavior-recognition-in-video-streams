"""Business event aggregation for action event streams.

Groups consecutive ActionEvents with the same label for the same track
into higher-level BusinessEvents with explicit start, end, and duration.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from src.inference.action_event import ActionEvent

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BusinessEvent:
    """Aggregated business-level event spanning multiple detection windows.

    Attributes:
        track_id: Optional tracking ID for multi-object tracking scenarios.
        label: Action label shared by all windows in this event.
        start_frame_index: Frame index from the first ActionEvent in the group.
        end_frame_index: Frame index from the last ActionEvent in the group.
        duration_windows: Number of ActionEvent windows aggregated into this event.
        start_timestamp: Timestamp from the first ActionEvent, if available.
        end_timestamp: Timestamp from the last ActionEvent, if available.
        mean_confidence: Average confidence across all windows in the group.
    """

    track_id: Optional[int]
    label: str
    start_frame_index: int
    end_frame_index: int
    duration_windows: int
    start_timestamp: Optional[float]
    end_timestamp: Optional[float]
    mean_confidence: float


@dataclass
class _TrackBuffer:
    """Internal mutable buffer for an open event on one track.

    Attributes:
        label: The action label currently being accumulated.
        events: Ordered list of ActionEvents for the open event.
    """

    label: str
    events: list[ActionEvent]


class EventAggregator:
    """Per-track aggregator that groups consecutive same-label ActionEvents.

    An open event is closed (a BusinessEvent is emitted) when:

    - The incoming event has a different label, or
    - There is a real gap between windows
      (``event.start_frame_index > previous.end_frame_index + 1``).

    Overlapping and adjacent windows are treated as continuous:

    - Adjacent:   ``[0-15], [16-31]`` → ``16 <= 15 + 1`` ✓
    - Overlapping: ``[0-15], [8-23]`` → ``8 <= 15 + 1`` ✓

    Per-track isolation: every ``track_id`` (including ``None``) has its own
    independent buffer, consistent with ``MajorityVoteSmoother`` and
    ``AlertStateMachine`` conventions.

    Example:
        >>> aggregator = EventAggregator()
        >>> e1 = ActionEvent(0, 15, "walk", 0.9)
        >>> e2 = ActionEvent(16, 31, "walk", 0.8)
        >>> aggregator.update(e1)  # None — event open
        >>> aggregator.update(e2)  # None — same label, continuous
        >>> e3 = ActionEvent(32, 47, "fight", 0.95)
        >>> result = aggregator.update(e3)  # BusinessEvent for "walk"
    """

    def __init__(self) -> None:
        """Initialize the aggregator with empty per-track buffers."""
        self._buffers: dict[Optional[int], _TrackBuffer] = {}
        logger.debug("EventAggregator initialized")

    def update(self, event: ActionEvent) -> Optional[BusinessEvent]:
        """Add event to the per-track buffer.

        Returns a closed BusinessEvent when continuity breaks, None otherwise.

        Args:
            event: Incoming ActionEvent to process.

        Returns:
            A BusinessEvent if the arrival of this event closes an existing open
            event (label change or frame gap), ``None`` otherwise.

        Raises:
            TypeError: If ``event`` is not an ``ActionEvent`` instance.
        """
        if not isinstance(event, ActionEvent):
            raise TypeError(
                f"event must be an ActionEvent instance, got {type(event).__name__}"
            )

        track_id = event.track_id

        if track_id not in self._buffers:
            self._buffers[track_id] = _TrackBuffer(label=event.label, events=[event])
            logger.debug("Track %s: opened new event label=%r", track_id, event.label)
            return None

        buf = self._buffers[track_id]
        prev = buf.events[-1]
        label_changed = event.label != buf.label
        gap_detected = event.start_frame_index > prev.end_frame_index + 1

        if label_changed or gap_detected:
            closed = self._close_buffer(track_id, buf)
            logger.debug(
                "Track %s: closed event label=%r windows=%d (label_changed=%s, gap=%s)",
                track_id,
                closed.label,
                closed.duration_windows,
                label_changed,
                gap_detected,
            )
            self._buffers[track_id] = _TrackBuffer(label=event.label, events=[event])
            logger.debug("Track %s: opened new event label=%r", track_id, event.label)
            return closed

        buf.events.append(event)
        logger.debug(
            "Track %s: extended event label=%r windows=%d",
            track_id,
            buf.label,
            len(buf.events),
        )
        return None

    def flush(self, track_id: Optional[int] = None) -> list[BusinessEvent]:
        """Force-close open events and clear their buffered state.

        Args:
            track_id: If a non-``None`` integer is provided, flush only that
                track's open event.  If ``None`` (the default), flush all open
                tracks.

        Returns:
            List of closed BusinessEvents.  When flushing all tracks, events
            are returned in deterministic order: integer track_ids sorted
            ascending, ``None`` last.
        """
        if track_id is not None:
            buf = self._buffers.pop(track_id, None)
            if buf is None:
                return []
            result = self._close_buffer(track_id, buf)
            logger.debug(
                "Track %s: flushed event label=%r windows=%d",
                track_id,
                result.label,
                result.duration_windows,
            )
            return [result]

        if not self._buffers:
            return []

        int_ids = sorted(tid for tid in self._buffers if tid is not None)
        ordered: list[Optional[int]] = int_ids + ([None] if None in self._buffers else [])

        results: list[BusinessEvent] = []
        for tid in ordered:
            buf = self._buffers[tid]
            be = self._close_buffer(tid, buf)
            results.append(be)
            logger.debug(
                "Track %s: flushed event label=%r windows=%d",
                tid,
                be.label,
                be.duration_windows,
            )

        self._buffers.clear()
        return results

    def reset(self, track_id: Optional[int] = None) -> None:
        """Discard buffered state without emitting BusinessEvents.

        Args:
            track_id: If a non-``None`` integer is provided, reset only that
                track's buffer.  If ``None`` (the default), reset all tracks.
        """
        if track_id is not None:
            self._buffers.pop(track_id, None)
            logger.debug("EventAggregator: track %s buffer discarded", track_id)
        else:
            self._buffers.clear()
            logger.debug("EventAggregator: all track buffers discarded")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _close_buffer(track_id: Optional[int], buf: _TrackBuffer) -> BusinessEvent:
        """Build a BusinessEvent from a completed buffer.

        Args:
            track_id: Track identifier for the resulting event.
            buf: Completed track buffer containing at least one event.

        Returns:
            Frozen BusinessEvent summarising all buffered windows.
        """
        first = buf.events[0]
        last = buf.events[-1]
        mean_conf = sum(e.confidence for e in buf.events) / len(buf.events)
        return BusinessEvent(
            track_id=track_id,
            label=buf.label,
            start_frame_index=first.start_frame_index,
            end_frame_index=last.end_frame_index,
            duration_windows=len(buf.events),
            start_timestamp=first.start_timestamp,
            end_timestamp=last.end_timestamp,
            mean_confidence=mean_conf,
        )


__all__ = ["BusinessEvent", "EventAggregator"]
