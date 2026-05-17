"""Temporal smoothing for action predictions using majority vote.

Provides a per-track sliding-window majority vote smoother that reduces
noise in sequential model outputs before they reach the alert state machine.
"""

import logging
from collections import Counter, deque
from typing import Optional

from src.inference.action_event import ActionEvent

logger = logging.getLogger(__name__)


class MajorityVoteSmoother:
    """Per-track majority-vote smoother for action prediction sequences.

    Maintains a sliding window of the last ``window_size`` events for each
    track.  When the window is full the most-frequent label is selected as
    the smoothed output.  Confidence is averaged over only the events that
    voted for the winning label.

    Tie-breaking: when two or more labels share the top vote count, the label
    whose most recent occurrence in the buffer is latest wins.

    Per-track isolation mirrors the convention used in ``AlertStateMachine``:
    every ``track_id`` value (including ``None``) maintains an independent
    buffer.

    Example:
        >>> smoother = MajorityVoteSmoother(window_size=3)
        >>> evt = ActionEvent(0, 1, "fight", 0.9)
        >>> smoother.update(evt)   # 1/3 → None
        >>> smoother.update(evt)   # 2/3 → None
        >>> result = smoother.update(evt)  # 3/3 → ActionEvent("fight", …)
    """

    def __init__(self, window_size: int = 5) -> None:
        """Initialize the smoother.

        Args:
            window_size: Number of most-recent events to consider per track.
                Must be >= 1.

        Raises:
            TypeError: If ``window_size`` is not an integer.
            ValueError: If ``window_size`` is less than 1.
        """
        if not isinstance(window_size, int) or isinstance(window_size, bool):
            raise TypeError("window_size must be an integer")
        if window_size < 1:
            raise ValueError("window_size must be >= 1")

        self._window_size = window_size
        self._buffers: dict[Optional[int], deque[ActionEvent]] = {}
        logger.debug("MajorityVoteSmoother initialized with window_size=%d", window_size)

    def update(self, event: ActionEvent) -> Optional[ActionEvent]:
        """Add event to the per-track buffer and return a smoothed event when ready.

        Args:
            event: Incoming ActionEvent to add to the per-track buffer.

        Returns:
            Smoothed ActionEvent when the buffer reaches ``window_size``,
            ``None`` otherwise.  The returned event carries metadata
            (frame indices, timestamps, track_id, context) from the most
            recent buffered event, with label and confidence replaced by
            the majority-vote result.

        Raises:
            TypeError: If ``event`` is not an ActionEvent instance.
        """
        if not isinstance(event, ActionEvent):
            raise TypeError("event must be an ActionEvent instance")

        track_id = event.track_id
        if track_id not in self._buffers:
            self._buffers[track_id] = deque(maxlen=self._window_size)

        buf = self._buffers[track_id]
        buf.append(event)
        logger.debug(
            "Track %s: buffer updated (%d/%d)", track_id, len(buf), self._window_size
        )

        if len(buf) < self._window_size:
            return None

        return self._compute_smoothed(buf, track_id)

    def reset(self, track_id: Optional[int] = None) -> None:
        """Reset buffer for a specific track, or all tracks if no argument given.

        Args:
            track_id: Track identifier whose buffer should be cleared.
                Pass ``None`` (default) to reset all tracks simultaneously.
        """
        if track_id is None:
            self._buffers.clear()
            logger.debug("MajorityVoteSmoother: all track buffers reset")
        else:
            self._buffers.pop(track_id, None)
            logger.debug("MajorityVoteSmoother: buffer for track %s reset", track_id)

    def is_ready(self, track_id: Optional[int] = None) -> bool:
        """Return True when the buffer for the given track is full.

        Args:
            track_id: Track identifier to query.  Defaults to ``None``.

        Returns:
            ``True`` if the buffer for the track contains exactly
            ``window_size`` events, ``False`` otherwise (including when
            the track has never been seen).
        """
        buf = self._buffers.get(track_id)
        return buf is not None and len(buf) == self._window_size

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_smoothed(
        self,
        buf: deque[ActionEvent],
        track_id: Optional[int],
    ) -> ActionEvent:
        """Compute the majority-vote smoothed event from a full buffer.

        Args:
            buf: Full deque of ActionEvent objects for one track.
            track_id: Track identifier (used for debug logging only).

        Returns:
            New ActionEvent with the winning label, averaged confidence over
            votes cast for the winner, and metadata from the most recent event.
        """
        label_counts: Counter[str] = Counter(evt.label for evt in buf)
        max_count = max(label_counts.values())
        top_labels = {lbl for lbl, cnt in label_counts.items() if cnt == max_count}

        most_recent = buf[-1]

        if len(top_labels) == 1:
            winning_label = next(iter(top_labels))
        else:
            # Tie-break: among tied labels, pick the one whose last occurrence
            # in the buffer is the most recent.
            winning_label = most_recent.label  # safe fallback
            for evt in reversed(buf):
                if evt.label in top_labels:
                    winning_label = evt.label
                    break

        winning_events = [evt for evt in buf if evt.label == winning_label]
        avg_confidence = sum(evt.confidence for evt in winning_events) / len(winning_events)

        logger.debug(
            "Track %s: majority=%s (votes=%d/%d) confidence=%.3f",
            track_id,
            winning_label,
            len(winning_events),
            self._window_size,
            avg_confidence,
        )

        return ActionEvent(
            start_frame_index=most_recent.start_frame_index,
            end_frame_index=most_recent.end_frame_index,
            label=winning_label,
            confidence=avg_confidence,
            start_timestamp=most_recent.start_timestamp,
            end_timestamp=most_recent.end_timestamp,
            track_id=most_recent.track_id,
            context=most_recent.context,
        )


__all__ = ["MajorityVoteSmoother"]
