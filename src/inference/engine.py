import copy
from typing import Any, Optional, List
from src.inference.buffer import FrameBuffer


class InferenceEngine:
    """
    Core engine for processing real-time video streams and predicting human behavior.
    """

    def __init__(self, window_size: int = 16, stride: int = 1, model: Optional[Any] = None):
        """
        Initializes the InferenceEngine.

        Args:
            window_size (int): The number of frames required to make a prediction.
            stride (int): The number of frames to skip before triggering the next inference.
            model (Any, optional): The behavior recognition model (e.g., PyTorch nn.Module).
        """
        if stride <= 0:
            raise ValueError("Stride must be greater than 0")

        if window_size <= 0:
            raise ValueError("window_size must be greater than 0")

        self.buffer = FrameBuffer(window_size=window_size)
        self.stride = stride
        self.model = model

        # Counter to track the total number of processed frames
        self.frame_count: int = 0

        # Stores the exact window of frames used in the most recent valid inference
        self._latest_inference_window: Optional[List[Any]] = None

    def process_frame(self, frame: Any) -> Optional[Any]:
        """
        Adds a new frame to the temporal buffer and triggers inference based on the stride logic.

        Args:
            frame (Any): A single video frame.

        Returns:
            Optional[Any]: The prediction result if inference is triggered, otherwise None.
        """
        self.frame_count += 1
        self.buffer.append(frame)

        # We only trigger inference if the buffer is completely full
        if self.buffer.is_full():
            # Calculate how many frames have passed since the buffer first filled up
            frames_since_full = self.frame_count - self.buffer.window_size

            # Trigger inference on the very first full window, and then exactly every `stride` frames
            if frames_since_full % self.stride == 0:
                self._latest_inference_window = copy.deepcopy(
                    self.buffer.get_window())

                if self.model is not None:
                    # TODO: Implement actual tensor preprocessing and model forward pass
                    # window = self.get_latest_window()
                    # return self.model(window)
                    return "prediction_stub"

        return None

    def get_latest_window(self) -> Optional[List[Any]]:
        """
        Returns the most recent temporal window on which inference was triggered.
        This provides a clear API to retrieve valid frames regardless of the current buffer state.

        Returns:
            Optional[List[Any]]: A list of frames, or None if no inference has been triggered yet.
        """
        return self._latest_inference_window
