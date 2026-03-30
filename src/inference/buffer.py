import collections
from typing import Any, List


class FrameBuffer:
    """
    Fixed-size FIFO data structure for buffering video frames 
    for real-time inference.
    """

    def __init__(self, window_size: int = 16):
        """
        Initializes the buffer.

        Args:
            window_size (int): Maximum number of frames stored in the window.
        """
        if window_size <= 0:
            raise ValueError(
                "Window size must be greater than 0.")

        self.window_size = window_size
        # deque with maxlen automatically removes the oldest elements upon overflow
        self.buffer = collections.deque(maxlen=window_size)

    def append(self, frame: Any) -> None:
        """
        Appends a new frame to the buffer. If the buffer is full, 
        the oldest frame is automatically removed.
        """
        self.buffer.append(frame)

    def get_window(self) -> List[Any]:
        """
        Returns the current buffer content as a list.
        """
        return list(self.buffer)

    def is_full(self) -> bool:
        """
        Checks if the buffer has reached its target size.
        """
        return len(self.buffer) == self.window_size

    def clear(self) -> None:
        """
        Clears the buffer content.
        """
        self.buffer.clear()

    @property
    def current_size(self) -> int:
        """
        Returns the current number of frames in the buffer.
        """
        return len(self.buffer)
