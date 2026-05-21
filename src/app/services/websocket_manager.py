"""WebSocket connection manager for real-time client communication.

Provides thread-safe event broadcasting to all connected WebSocket clients.
"""

import asyncio
import logging
import threading
from typing import List, Optional

from fastapi import WebSocket

from src.app.schemas.action_event import EventPayload

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Manages WebSocket connections and provides a thread-safe broadcast mechanism."""

    def __init__(self) -> None:
        """Initialize the WebSocket manager with empty connections and loop reference."""
        self.active_connections: List[WebSocket] = []
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self._state_lock = threading.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        """Register a new WebSocket connection."""
        await websocket.accept()
        # Store the current running loop to allow thread-safe scheduling of broadcasts
        active_count = 0
        with self._state_lock:
            self.active_connections.append(websocket)
            try:
                current_loop = asyncio.get_running_loop()
                if self.loop is None or self.loop.is_closed() or self.loop is not current_loop:
                    self.loop = current_loop
            except RuntimeError:
                pass
            active_count = len(self.active_connections)
        logger.info(
            f"WebSocket client connected. "
            f"Active connections: {active_count}"
        )

    def disconnect(self, websocket: WebSocket) -> None:
        """Unregister a disconnected WebSocket connection."""
        with self._state_lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
            if not self.active_connections:
                self.loop = None
            active_count = len(self.active_connections)
        logger.info(
            f"WebSocket client disconnected. "
            f"Active connections: {active_count}"
        )

    async def broadcast(self, message: dict) -> None:
        """Broadcast a message (as JSON) to all connected clients."""
        disconnected = []
        with self._state_lock:
            connections = list(self.active_connections)
        for connection in connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send message to connection: {e}")
                disconnected.append(connection)

        # Cleanup any failed connections
        for conn in disconnected:
            self.disconnect(conn)

    def broadcast_sync(self, payload: EventPayload) -> None:
        """Thread-safe entrypoint to broadcast a payload from a background thread."""
        try:
            # Serialize payload on the calling thread to reduce work on the main event loop
            message = payload.model_dump(mode="json")
            with self._state_lock:
                loop = self.loop

            if loop is None or loop.is_closed():
                return

            future = asyncio.run_coroutine_threadsafe(self.broadcast(message), loop)

            def _log_future_exception(submitted_future: asyncio.Future) -> None:
                try:
                    submitted_future.result()
                except Exception:
                    logger.exception("WebSocket broadcast failed after scheduling")

            future.add_done_callback(_log_future_exception)
        except Exception as e:
            logger.error(f"Failed to submit WebSocket broadcast: {e}")


websocket_manager = WebSocketManager()
