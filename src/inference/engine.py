import collections
from typing import Any, Optional, Tuple
from dataclasses import dataclass
from threading import Lock
import logging
import time

from src.inference.buffer import FrameBuffer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InferenceResult:
    """
    Stores metadata and prediction for an inference step.
    """
    window: Tuple[Any, ...]

    start_frame_index: int
    end_frame_index: int

    start_timestamp: Optional[float]
    end_timestamp: Optional[float]

    prediction: Optional[Any]


class InferenceEngine:
    """
    Production-grade inference engine with:

    - configurable stride
    - frame + timestamp tracking
    - deterministic windowing
    - thread safety
    - logging
    - metrics
    """

    def __init__(
        self,
        window_size: int = 16,
        stride: int = 1,
        model: Optional[Any] = None
    ):
        """
        Args:
            window_size:
                Number of frames per inference window.

            stride:
                Frames between inference triggers.

            model:
                Optional model object (callable PyTorch model or object with predict()).
        """

        if window_size <= 0:
            raise ValueError("window_size must be > 0")

        if stride <= 0:
            raise ValueError("stride must be > 0")

        self.buffer = FrameBuffer(window_size=window_size)
        self._stride = stride
        self.model = model

        # Thread safety
        self._lock = Lock()

        # Frame tracking
        self.frame_count: int = 0

        # Timestamp tracking
        self._timestamps = collections.deque(maxlen=window_size)

        # Last inference location
        self._last_inference_frame: Optional[int] = None

        # Result tracking
        self._latest_result: Optional[InferenceResult] = None
        self._unread_result: bool = False  # Track if there is a new, unconsumed result

        # Metrics
        self.total_inferences: int = 0
        self.total_frames_processed: int = 0
        self.total_frames_skipped: int = 0

        logger.debug(
            "InferenceEngine initialized "
            "(window=%d stride=%d)",
            window_size,
            stride
        )

    # ========================
    # Public properties
    # ========================

    @property
    def window_size(self) -> int:
        """
        Single Source of Truth for window_size, delegated to the buffer.
        """
        return self.buffer.window_size

    @property
    def stride(self) -> int:
        return self._stride

    def set_stride(self, stride: int):
        """
        Dynamically updates stride.
        """
        if stride <= 0:
            raise ValueError("stride must be > 0")

        with self._lock:
            logger.info(
                "Stride changed from %d to %d",
                self._stride,
                stride
            )
            self._stride = stride

    # ========================
    # Core processing
    # ========================

    def process_frame(
        self,
        frame: Any,
        timestamp: Optional[float] = None
    ) -> Optional[InferenceResult]:
        """
        Processes a single frame.

        Returns:
            InferenceResult if triggered
            None otherwise
        """
        if timestamp is None:
            timestamp = time.time()

        with self._lock:

            self.frame_count += 1
            self.total_frames_processed += 1

            self.buffer.append(frame)
            self._timestamps.append(timestamp)

            if not self.buffer.is_full():
                return None

            if not self._should_trigger_inference():
                self.total_frames_skipped += 1
                return None

            # Prepare immutable snapshot for the model
            window_snapshot = tuple(self.buffer.get_window())
            start_frame_snap = self.frame_count - self.window_size + 1
            end_frame_snap = self.frame_count
            start_ts_snap = self._timestamps[0]
            end_ts_snap = self._timestamps[-1]

            self._last_inference_frame = self.frame_count
            self.total_inferences += 1

        prediction = None

        if self.model is not None:
            if callable(self.model):
                prediction = self.model(window_snapshot)
            elif hasattr(self.model, "predict"):
                prediction = self.model.predict(window_snapshot)
            else:
                logger.warning(
                    "Model is neither callable nor has a predict() method"
                )

        result = InferenceResult(
            window=window_snapshot,
            start_frame_index=start_frame_snap,
            end_frame_index=end_frame_snap,
            start_timestamp=start_ts_snap,
            end_timestamp=end_ts_snap,
            prediction=prediction
        )

        with self._lock:
            self._latest_result = result
            self._unread_result = True

            logger.debug(
                "Inference completed (frames %d-%d)",
                start_frame_snap,
                end_frame_snap
            )

        return result

    # ========================
    # Internal logic
    # ========================

    def _should_trigger_inference(self) -> bool:

        if self._last_inference_frame is None:
            return True

        frames_since_last = (
            self.frame_count
            - self._last_inference_frame
        )

        return frames_since_last >= self._stride

    # ========================
    # API helpers
    # ========================

    def get_latest_result(
        self
    ) -> Optional[InferenceResult]:

        with self._lock:
            self._unread_result = False  # Result consumed
            return self._latest_result

    def has_new_result(self) -> bool:

        with self._lock:
            return self._unread_result

    def peek_next_trigger_frame(
        self
    ) -> Optional[int]:
        """
        Predicts next inference frame index.
        """

        with self._lock:

            if self._last_inference_frame is None:

                if self.buffer.is_full():
                    return self.frame_count

                return self.window_size

            return (
                self._last_inference_frame
                + self._stride
            )

    def get_metrics(self) -> dict:

        with self._lock:

            return {
                "total_frames_processed":
                    self.total_frames_processed,

                "total_inferences":
                    self.total_inferences,

                "total_frames_skipped":
                    self.total_frames_skipped
            }

    def reset(self):

        with self._lock:

            logger.info("Engine reset")

            self.buffer.clear()

            self.frame_count = 0

            self._timestamps.clear()

            self._last_inference_frame = None

            self._latest_result = None
            self._unread_result = False

            self.total_inferences = 0
            self.total_frames_processed = 0
            self.total_frames_skipped = 0
