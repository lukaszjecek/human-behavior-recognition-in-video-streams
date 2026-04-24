"""Placeholder WebSocket routes for the backend.

Provides a namespace for future websocket endpoints.
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()


@router.get("/ping", summary="Websocket ping placeholder")
async def ws_ping() -> dict[str, str]:
    """HTTP placeholder endpoint for websocket namespace."""
    return {"message": "websocket placeholder"}


@router.websocket("/echo")
async def websocket_echo(ws: WebSocket) -> None:
    """Simple echo websocket that sends back received text prefixed with 'echo: '."""
    await ws.accept()
    try:
        while True:
            data = await ws.receive_text()
            await ws.send_text(f"echo: {data}")
    except WebSocketDisconnect:
        # Normalne rozłączenie klienta - nie traktujemy tego jako błąd
        pass
    except Exception:
        # Obsługa nieoczekiwanych błędów
        pass
    finally:
        # Upewniamy się, że połączenie zostanie zamknięte, jeśli jeszcze nie jest
        try:
            await ws.close()
        except RuntimeError:
            # Rzucane, jeśli socket został już zamknięty przez klienta
            pass