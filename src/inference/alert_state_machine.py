"""Per-track alert state machine for behavior detection pipelines.

Transitions action events through INACTIVE → CANDIDATE → ACTIVE → RESOLVED
states with configurable persistence and resolve thresholds.
"""

import logging
from dataclasses import dataclass
from enum import Enum, auto

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