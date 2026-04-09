"""
JSON writer for converting inference results to action event records.

Handles serialization of InferenceResult objects to ActionEvent format
with confidence score extraction and error handling.
"""

from typing import Optional, Any
import logging
import numpy as np
import torch

from src.inference.action_event import ActionEvent, ActionEventLog
from src.inference.engine import InferenceResult

logger = logging.getLogger(__name__)


class ActionEventWriter:
    """
    Converts InferenceResult objects to ActionEvent records and manages logging.

    Supports multiple prediction formats:
    - Logits (raw model outputs)
    - Probabilities
    - Dictionaries with 'class' and 'confidence'
    - Single confidence values
    """

    def __init__(self, class_labels: Optional[list[str]] = None):
        """
        Initialize the writer.

        Args:
            class_labels: Optional list of class labels indexed by prediction index.
                         If None, uses numeric class indices.
        """
        self.class_labels = class_labels or []
        self.log = ActionEventLog()

    def process_inference_result(
        self,
        result: InferenceResult,
        track_id: Optional[int] = None,
    ) -> Optional[ActionEvent]:
        """
        Convert InferenceResult to ActionEvent.

        Args:
            result: InferenceResult from the inference engine.
            track_id: Optional tracking ID for multi-object scenarios.

        Returns:
            ActionEvent if prediction is valid, None otherwise.
        """
        if result.prediction is None:
            logger.debug("Skipping result with None prediction")
            return None

        try:
            label, confidence = self._extract_label_confidence(result.prediction)
        except (ValueError, IndexError) as e:
            logger.warning(f"Could not extract label/confidence from prediction: {e}")
            return None

        if confidence is None or label is None:
            logger.debug("Skipping result with invalid label or confidence")
            return None

        try:
            event = ActionEvent(
                start_frame_index=result.start_frame_index,
                end_frame_index=result.end_frame_index,
                start_timestamp=result.start_timestamp,
                end_timestamp=result.end_timestamp,
                label=label,
                confidence=confidence,
                track_id=track_id,
            )
            return event
        except ValueError as e:
            logger.error(f"Failed to create ActionEvent: {e}")
            return None

    def add_result(
        self,
        result: InferenceResult,
        track_id: Optional[int] = None,
    ) -> bool:
        """
        Process and add an inference result to the log.

        Args:
            result: InferenceResult from the inference engine.
            track_id: Optional tracking ID.

        Returns:
            True if event was added, False otherwise.
        """
        event = self.process_inference_result(result, track_id)
        if event is not None:
            self.log.add_event(event)
            return True
        return False

    def add_results(
        self,
        results: list[InferenceResult],
        track_ids: Optional[list[int]] = None,
    ) -> int:
        """
        Process and add multiple inference results.

        Args:
            results: List of InferenceResult objects.
            track_ids: Optional list of tracking IDs (must match results length).

        Returns:
            Number of events successfully added.
        """
        if track_ids is None:
            track_ids = [None] * len(results)

        if len(track_ids) != len(results):
            logger.warning(
                f"track_ids length ({len(track_ids)}) != results length ({len(results)})"
            )
            track_ids = [None] * len(results)

        count = 0
        for result, track_id in zip(results, track_ids):
            if self.add_result(result, track_id):
                count += 1

        return count

    def get_log(self) -> ActionEventLog:
        """Get the current action event log."""
        return self.log

    def save(self, filepath: str) -> None:
        """Save the action event log to a JSON file."""
        self.log.save_to_file(filepath)
        logger.info(f"Saved {len(self.log.events)} action events to {filepath}")

    def clear(self) -> None:
        """Clear all recorded events."""
        self.log.clear()

    def _extract_label_confidence(
        self,
        prediction: Any,
    ) -> tuple[Optional[str], Optional[float]]:
        """
        Extract label and confidence from various prediction formats.

        Args:
            prediction: Can be logits, probabilities, dict, or single float.

        Returns:
            Tuple of (label, confidence) or (None, None) if extraction fails.

        Raises:
            ValueError: If prediction format is unsupported.
        """
        # Handle dictionary format: {"class": int, "confidence": float}
        if isinstance(prediction, dict):
            if "class" in prediction and "confidence" in prediction:
                class_idx = prediction["class"]
                confidence = float(prediction["confidence"])
                label = self._get_label(class_idx)
                return label, confidence
            if "label" in prediction and "confidence" in prediction:
                label = str(prediction["label"])
                confidence = float(prediction["confidence"])
                return label, confidence

        # Handle single confidence value (assume positive class)
        if isinstance(prediction, (int, float)):
            confidence = float(prediction)
            label = "action" if confidence > 0.5 else "no_action"
            return label, abs(confidence)

        # Handle numpy/torch tensors or lists
        if isinstance(prediction, (list, np.ndarray, torch.Tensor)):
            return self._extract_from_logits_or_probs(prediction)

        raise ValueError(f"Unsupported prediction format: {type(prediction)}")

    def _extract_from_logits_or_probs(
        self,
        prediction: Any,
    ) -> tuple[Optional[str], Optional[float]]:
        """
        Extract label and confidence from logits or probability distributions.

        Args:
            prediction: Logits, probabilities as list, numpy array, or torch tensor.

        Returns:
            Tuple of (label, confidence) or (None, None) if invalid.
        """
        try:
            # Convert to numpy if needed
            if isinstance(prediction, torch.Tensor):
                pred_array = prediction.detach().cpu().numpy()
            elif isinstance(prediction, list):
                pred_array = np.array(prediction)
            else:
                pred_array = prediction

            # Ensure it's 1D
            pred_array = np.atleast_1d(pred_array).flatten()

            if len(pred_array) == 0:
                return None, None

            # Single value: treat as confidence
            if len(pred_array) == 1:
                confidence = float(pred_array[0])
                # Clamp to [0, 1] in case it's logits
                confidence = max(0.0, min(1.0, confidence))
                label = "action"
                return label, confidence

            # Multiple values: apply softmax and get argmax
            # Heuristic: if sum is close to 1.0 and all values in [0,1], treat as probs
            array_sum = np.sum(pred_array)
            is_likely_probs = (
                0.9 <= array_sum <= 1.1
                and np.all(pred_array >= 0)
                and np.all(pred_array <= 1)
            )
            if is_likely_probs:
                # Likely probabilities
                probs = pred_array
            else:
                # Likely logits, apply softmax
                probs = self._softmax(pred_array)

            class_idx = int(np.argmax(probs))
            confidence = float(np.max(probs))
            label = self._get_label(class_idx)

            return label, confidence

        except Exception as e:
            raise ValueError(f"Failed to extract from logits/probs: {e}")

    def _get_label(self, class_idx: int) -> str:
        """Get label for class index."""
        if self.class_labels and 0 <= class_idx < len(self.class_labels):
            return self.class_labels[class_idx]
        return f"class_{class_idx}"

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Compute softmax of array."""
        x_shifted = x - np.max(x)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x)
