"""Bounding-box detection module for inferring object locations in video frames.

Implements issue #119: detects objects on a representative frame from an
inference window, filters to action-relevant classes, and attaches results as
``ActionEvent.bboxes``.

Conceptual flow::

    BBoxEnricher.__call__(event, result)
        → select representative frame from result.window
        → ObjectDetector.detect(frame)          # returns list[RawDetection]
        → filter by ACTION_LABEL_TO_OBJECT_CLASSES[event.label]
        → convert to BoundingBox and attach to ActionEvent
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np

from src.app.schemas.action_event import ActionEvent, BoundingBox
from src.inference.engine import InferenceResult

logger = logging.getLogger(__name__)


ACTION_LABEL_TO_OBJECT_CLASSES: dict[str, list[str]] = {
    "car_drops_off_person": ["car", "person"],
    "car_makes_u_turn": ["car"],
    "motorcycle_makes_u_turn": ["motorcycle"],
    "motorcycle_turns_right": ["motorcycle"],
    "person_sits_down": ["person"],
}


@dataclass(frozen=True)
class RawDetection:
    """One raw detection candidate from the underlying detector, before filtering."""

    label: str
    confidence: float
    x_min: float
    y_min: float
    x_max: float
    y_max: float


class ObjectDetector(Protocol):
    """Abstraction over a pretrained object detector backend."""

    def detect(self, frame: np.ndarray) -> list[RawDetection]:
        """Run detection on a single BGR frame.

        Args:
            frame: Raw BGR numpy array of shape (H, W, 3).

        Returns:
            List of raw detection candidates.
        """
        ...


class YoloObjectDetector:
    """Wraps a pretrained YOLO model (e.g. via the ``ultralytics`` package).

    Lazily imports ``ultralytics`` inside ``__init__`` so that environments
    without the dependency installed (e.g. CI) can still import this module
    and use a mocked detector instead.
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.4,
    ) -> None:
        """Initialize the YOLO detector.

        Args:
            model_name: YOLO model identifier or path to a local weights file.
            confidence_threshold: Minimum confidence score for a detection to be
                included in results.  Must be in ``[0.0, 1.0]``.

        Raises:
            ValueError: If ``confidence_threshold`` is outside ``[0.0, 1.0]``.
            RuntimeError: If the ``ultralytics`` package is not installed.
        """
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(
                f"confidence_threshold must be in [0.0, 1.0], got {confidence_threshold}"
            )

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is required for YoloObjectDetector. "
                "Install it with: pip install ultralytics"
            ) from exc

        self._model = YOLO(model_name)
        self._confidence_threshold = float(confidence_threshold)
        logger.debug(
            "YoloObjectDetector initialised (model=%r confidence_threshold=%.2f)",
            model_name,
            self._confidence_threshold,
        )

    def detect(self, frame: np.ndarray) -> list[RawDetection]:
        """Run YOLO detection on a single BGR frame.

        Args:
            frame: Raw BGR numpy array of shape (H, W, 3).

        Returns:
            List of raw detections whose confidence exceeds the threshold.
        """
        results = self._model(frame, verbose=False)
        detections: list[RawDetection] = []
        for result in results:
            for box in result.boxes:
                confidence = float(box.conf[0])
                if confidence < self._confidence_threshold:
                    continue
                cls_id = int(box.cls[0])
                label = result.names[cls_id]
                x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
                detections.append(
                    RawDetection(
                        label=label,
                        confidence=confidence,
                        x_min=x_min,
                        y_min=y_min,
                        x_max=x_max,
                        y_max=y_max,
                    )
                )
        logger.debug("YoloObjectDetector: %d detections on frame", len(detections))
        return detections


class BBoxEnricher:
    """Production component implementing the BBoxHook contract for issue #119.

    Detects objects on a representative frame, filters detections to only
    those classes relevant to the event's action label, and attaches the
    result as ``ActionEvent.bboxes``.
    """

    def __init__(
        self,
        detector: ObjectDetector,
        label_to_classes: Optional[dict[str, list[str]]] = None,
        frame_selector: str = "middle",
    ) -> None:
        """Initialize the BBoxEnricher.

        Args:
            detector: Any object implementing the ``ObjectDetector`` protocol.
            label_to_classes: Mapping from action label to allowed object class
                names.  When *None*, defaults to
                ``ACTION_LABEL_TO_OBJECT_CLASSES``.
            frame_selector: Which frame from the inference window to use for
                detection.  One of ``"first"``, ``"middle"``, or ``"last"``.

        Raises:
            ValueError: If ``frame_selector`` is not one of the allowed values.
        """
        if frame_selector not in {"first", "middle", "last"}:
            raise ValueError(
                f"frame_selector must be 'first', 'middle', or 'last', got {frame_selector!r}"
            )
        self._detector = detector
        self._label_to_classes: dict[str, list[str]] = (
            label_to_classes
            if label_to_classes is not None
            else ACTION_LABEL_TO_OBJECT_CLASSES
        )
        self._frame_selector = frame_selector

    def __call__(self, event: ActionEvent, result: InferenceResult) -> ActionEvent:
        """Implement the BBoxHook contract.

        Steps:
            1. Look up allowed object classes for ``event.label``.  If the label
               has no mapping entry → return *event* unchanged (``bboxes`` stays
               ``None``).
            2. Select the representative frame from ``result.window`` per
               ``frame_selector``.
            3. Call ``detector.detect(frame)`` → ``list[RawDetection]``.
            4. Filter detections to only those whose label is in the allowed set.
            5. If no detections remain after filtering → return *event* unchanged
               (``bboxes`` stays ``None``, **not** an empty list).
            6. Convert surviving ``RawDetection`` objects into ``BoundingBox``
               instances with ``box_format="xyxy"`` and
               ``coordinate_space="source_pixels"``.
            7. Return ``event.model_copy(update={"bboxes": [...]})``.

        Args:
            event: ``ActionEvent`` enriched with context, ready for bbox attachment.
            result: ``InferenceResult`` providing the raw frame window.

        Returns:
            ``ActionEvent`` with ``bboxes`` populated, or the original event
            unchanged when no relevant detections are found.
        """
        # Step 1: label mapping
        allowed = self._label_to_classes.get(event.label)
        if allowed is None:
            logger.debug("BBoxEnricher: no mapping for label=%r, skipping", event.label)
            return event

        # Step 2: select frame
        window = result.window
        if not window:
            logger.warning(
                "BBoxEnricher: empty window for event label=%r, returning event unchanged",
                event.label,
            )
            return event

        if self._frame_selector == "first":
            offset = 0
        elif self._frame_selector == "last":
            offset = len(window) - 1
        else:  # "middle"
            offset = len(window) // 2

        frame: np.ndarray = window[offset]  # type: ignore[assignment]

        # Step 3: detect objects
        detections = self._detector.detect(frame)
        logger.debug(
            "BBoxEnricher: %d raw detections for label=%r", len(detections), event.label
        )

        # Step 4: filter to relevant classes
        allowed_set = set(allowed)
        relevant = [d for d in detections if d.label in allowed_set]

        # Step 5: no relevant detections → return unchanged
        if not relevant:
            logger.debug("BBoxEnricher: no relevant detections for label=%r", event.label)
            return event

        # Step 6: convert to BoundingBox
        frame_index = int(result.start_frame_index + offset)
        h, w = frame.shape[:2]
        bboxes = [
            BoundingBox(
                box_format="xyxy",
                coordinate_space="source_pixels",
                frame_index=frame_index,
                source_width=int(w),
                source_height=int(h),
                x_min=float(d.x_min),
                y_min=float(d.y_min),
                x_max=float(d.x_max),
                y_max=float(d.y_max),
                label=d.label,
                confidence=float(d.confidence),
            )
            for d in relevant
        ]

        # Step 7: return updated event (never mutate in place)
        return event.model_copy(update={"bboxes": bboxes})


__all__ = [
    "ACTION_LABEL_TO_OBJECT_CLASSES",
    "BBoxEnricher",
    "ObjectDetector",
    "RawDetection",
    "YoloObjectDetector",
]
