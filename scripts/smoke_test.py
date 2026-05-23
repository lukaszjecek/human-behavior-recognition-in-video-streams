#!/usr/bin/env python3
"""End-to-end smoke test script for Sprint 3.

Verifies the integration chain: source -> inference -> alert/event -> API/WebSocket -> DB.
Can be executed on the host (if dependencies are present) or within the API container.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

# Setup sys.path to find the src package
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

# Try to import crucial dependencies or print run-in-docker instructions
try:
    import cv2
    import httpx
    import numpy as np
    import torch
    import websockets
except ImportError as e:
    print(f"[SMOKE TEST] Error: Missing dependency '{e.name}'.")
    print(
        "[SMOKE TEST] To run this script, it is highly recommended to "
        "execute it inside the API container:"
    )
    print("  docker compose exec api python scripts/smoke_test.py")
    sys.exit(1)

# Configuration from environment or defaults
# When running inside the api container, API is at localhost:8000.
# The database url is set in the api container env, so the API server accesses it automatically.
API_URL = os.getenv("API_URL", "http://localhost:8000")
WS_URL = os.getenv("WS_URL", API_URL.replace("http://", "ws://").replace("https://", "wss://"))

RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
LOGS_DIR = ROOT_DIR / "data" / "logs"
CHECKPOINT_DIR = LOGS_DIR / "checkpoints"
CONFIG_PATH = ROOT_DIR / "configs" / "data_pipeline.yml"

VIDEO_NAME = "smoke_sample.mp4"
VIDEO_PATH = RAW_DATA_DIR / VIDEO_NAME
CHECKPOINT_NAME = "dummy_checkpoint.pth"
CHECKPOINT_PATH = CHECKPOINT_DIR / CHECKPOINT_NAME

# The path sent in the POST request to the API.
# Inside the container, uvicorn runs at /app, and volumes are mounted to /app.
CONTAINER_VIDEO_PATH = "/app/data/raw/smoke_sample.mp4"
CONTAINER_CHECKPOINT_PATH = "/app/data/logs/checkpoints/dummy_checkpoint.pth"
CONTAINER_CONFIG_PATH = "/app/configs/data_pipeline.yml"


def create_dummy_checkpoint() -> None:
    """Create a dummy PyTorch model checkpoint compatible with DummyBehaviorModel."""
    from src.models.dummy import DummyBehaviorModel
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 5 classes matches the configs/data_pipeline.yml class_labels count
    model = DummyBehaviorModel(num_classes=5)
    checkpoint = {
        "model_name": "dummy",
        "model_state_dict": model.state_dict(),
    }
    torch.save(checkpoint, CHECKPOINT_PATH)
    print(f"[SMOKE TEST] Generated dummy checkpoint at: {CHECKPOINT_PATH}")


def create_dummy_video() -> None:
    """Create a dummy 40-frame MP4 video using OpenCV."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Simple checkerboard/gradient video using OpenCV
    out = cv2.VideoWriter(
        str(VIDEO_PATH),
        cv2.VideoWriter_fourcc(*'mp4v'),
        10,  # 10 FPS
        (224, 224)  # matches pipeline.target_resolution [224, 224] in configs
    )
    
    if not out.isOpened():
        print(
            "[SMOKE TEST] Error: Could not open VideoWriter. "
            "Make sure ffmpeg/plugins are installed.",
            file=sys.stderr
        )
        sys.exit(1)

    for i in range(40):
        frame = np.zeros((224, 224, 3), dtype=np.uint8)
        # Create a dynamic moving text element to make the frames distinct
        x_pos = int(10 + i * 3)
        y_pos = int(50 + i * 2)
        cv2.putText(
            frame,
            f"Frame {i}",
            (x_pos, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )
        # Draw a moving square
        cv2.rectangle(
            frame,
            (x_pos + 10, y_pos + 10),
            (x_pos + 40, y_pos + 40),
            (0, 0, 255),
            -1
        )
        out.write(frame)
    out.release()
    print(f"[SMOKE TEST] Generated dummy video at: {VIDEO_PATH}")


async def listen_ws_events(received_events: list[Any], stop_event: asyncio.Event) -> None:
    """WebSocket listener to verify live streaming of detections and alerts."""
    uri = f"{WS_URL}/ws/live"
    print(f"[SMOKE TEST] Connecting to WebSocket: {uri}")
    try:
        async with websockets.connect(uri) as websocket:
            print("[SMOKE TEST] WebSocket connected successfully. Listening for live events...")
            while not stop_event.is_set():
                try:
                    # Read websocket messages with timeout
                    msg = await asyncio.wait_for(websocket.recv(), timeout=0.5)
                    event_data = json.loads(msg)
                    received_events.append(event_data)
                    print(
                        f"[SMOKE TEST] WS Event Received: {event_data.get('event_type')} "
                        f"- ID: {event_data.get('event_id')} "
                        f"- Camera: {event_data.get('camera_id')}"
                    )
                except asyncio.TimeoutError:
                    continue
                except websockets.exceptions.ConnectionClosed:
                    print("[SMOKE TEST] WebSocket connection closed by server.")
                    break
    except Exception as e:
        if not stop_event.is_set():
            print(f"[SMOKE TEST] WebSocket listener encountered error: {e}", file=sys.stderr)


async def check_api_health(client: httpx.AsyncClient) -> bool:
    """Wait for FastAPI API /health endpoint to be ready."""
    print(f"[SMOKE TEST] Checking API health at {API_URL}/health...")
    for _ in range(20):
        try:
            response = await client.get(f"{API_URL}/health", timeout=2.0)
            if response.status_code == 200 and response.json().get("status") == "ok":
                print("[SMOKE TEST] API is healthy and responding.")
                return True
        except Exception:
            pass
        await asyncio.sleep(1.0)
    print("[SMOKE TEST] API did not become healthy in time.", file=sys.stderr)
    return False


async def run_smoke_test() -> None:
    """Main smoke test orchestration flow."""
    print("=" * 70)
    print("STARTING SPRINT 3 INTEGRATION SMOKE TEST")
    print("=" * 70)

    # 1. Generate local files (will map to /app inside container)
    create_dummy_checkpoint()
    create_dummy_video()

    # 2. Setup Async HTTP client and verify API is up
    async with httpx.AsyncClient() as client:
        if not await check_api_health(client):
            print("[SMOKE TEST] Smoke test aborted: API is not healthy.", file=sys.stderr)
            sys.exit(1)

        # 3. Spin up WebSocket live listener in the background
        received_events: list[Any] = []
        stop_ws = asyncio.Event()
        ws_task = asyncio.create_task(listen_ws_events(received_events, stop_ws))
        
        # Give WS a moment to handshake
        await asyncio.sleep(1.0)

        # 4. Trigger Session creation via POST /api/sessions/
        session_request = {
            "video_path": CONTAINER_VIDEO_PATH,
            "checkpoint_path": CONTAINER_CHECKPOINT_PATH,
            "config_path": CONTAINER_CONFIG_PATH,
            "device": "cpu"
        }
        
        print("[SMOKE TEST] Starting session via POST /api/sessions/")
        try:
            resp = await client.post(
                f"{API_URL}/api/sessions/",
                json=session_request,
                timeout=5.0
            )
        except Exception as e:
            print(f"[SMOKE TEST] HTTP Request failed: {e}", file=sys.stderr)
            stop_ws.set()
            await ws_task
            sys.exit(1)

        if resp.status_code != 201:
            print(
                f"[SMOKE TEST] Session creation failed with status {resp.status_code}: {resp.text}",
                file=sys.stderr
            )
            stop_ws.set()
            await ws_task
            sys.exit(1)

        session_data = resp.json()
        session_id = session_data.get("id")
        print(f"[SMOKE TEST] Session created successfully. ID: {session_id}")

        # 5. Monitor Session status
        print("[SMOKE TEST] Monitoring session status...")
        session_completed = False
        for _ in range(60):  # Wait up to 60 seconds
            await asyncio.sleep(1.0)
            status_resp = await client.get(f"{API_URL}/api/sessions/{session_id}")
            if status_resp.status_code != 200:
                print(
                    f"[SMOKE TEST] Failed to retrieve status: {status_resp.text}",
                    file=sys.stderr
                )
                break
                
            status_data = status_resp.json()
            status = status_data.get("status")
            print(f"[SMOKE TEST] Session Status: {status}")
            
            if status == "completed":
                session_completed = True
                break
            elif status in ("failed", "stopped"):
                print(
                    f"[SMOKE TEST] Session terminated early. Status: {status}. "
                    f"Error: {status_data.get('error')}",
                    file=sys.stderr
                )
                break

        # Stop WebSocket listener
        await asyncio.sleep(1.0)  # wait a moment for late WS frames
        stop_ws.set()
        await ws_task

        if not session_completed:
            print(
                "[SMOKE TEST] Smoke test failed: Session did not complete successfully.",
                file=sys.stderr
            )
            sys.exit(1)

        # 6. Verify Database Persistence via REST API GET /api/events/
        print(
            f"[SMOKE TEST] Verifying database persistence via "
            f"GET /api/events/?session_id={session_id}"
        )
        try:
            events_resp = await client.get(
                f"{API_URL}/api/events/?session_id={session_id}",
                timeout=5.0
            )
        except Exception as e:
            print(f"[SMOKE TEST] Failed to query persisted events: {e}", file=sys.stderr)
            sys.exit(1)

        if events_resp.status_code != 200:
            print(
                f"[SMOKE TEST] Querying events returned status {events_resp.status_code}: "
                f"{events_resp.text}",
                file=sys.stderr
            )
            sys.exit(1)

        db_events = events_resp.json()
        print(f"[SMOKE TEST] Retrieved {len(db_events)} events from database.")

        # 7. Assertions
        if len(db_events) == 0:
            print(
                "[SMOKE TEST] FAILURE: DB is empty. Events were not persisted to database!",
                file=sys.stderr
            )
            sys.exit(1)

        if len(received_events) == 0:
            print(
                "[SMOKE TEST] FAILURE: No live events were broadcasted over WebSocket!",
                file=sys.stderr
            )
            sys.exit(1)

        print("=" * 70)
        print("VERIFICATION SUMMARY:")
        print(" - API liveness check:                      PASSED")
        print(" - Asynchronous Session initiation:          PASSED")
        print(" - In-process model inference processing:   PASSED (40 frames)")
        print(f" - Live event broadcasting (WebSocket):     PASSED ({len(received_events)} events)")
        print(f" - Event/Alert persistence (DB):            PASSED ({len(db_events)} events)")
        print("=" * 70)
        print("[SMOKE TEST] SUCCESS: Sprint 3 Integrated System Smoke Path is verified end-to-end!")
        print("=" * 70)


def main() -> None:
    """Entry point for the smoke test."""
    try:
        asyncio.run(run_smoke_test())
    except KeyboardInterrupt:
        print("\n[SMOKE TEST] Aborted by user.")
        sys.exit(1)


if __name__ == "__main__":
    main()