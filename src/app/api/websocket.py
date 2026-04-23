"""Placeholder WebSocket routes for the backend.

Provides a namespace for future websocket endpoints.
"""
from fastapi import APIRouter, WebSocket

router = APIRouter()


@router.get("/ping", summary="Websocket ping placeholder")
async def ws_ping():
    return {"message": "websocket placeholder"}


@router.websocket("/echo")
async def websocket_echo(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_text()
            await ws.send_text(f"echo: {data}")
    except Exception:
        await ws.close()