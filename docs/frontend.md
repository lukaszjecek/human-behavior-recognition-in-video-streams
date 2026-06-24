# Frontend Dashboard

## Author: [Filip Wasiel](https://github.com/Filipwasiel) (Updated & Optimized)

[Back to README](../README.md)

This directory contains the React-based frontend dashboard for the **Human Behavior Recognition (HBR) in Video Streams** application. It provides operator interfaces, streaming video feeds with real-time detection bounding box overlays, live event logging, and contextual scene telemetry.

---

## 1. High-Level Architecture

The frontend is built on **React 19** and **Vite 8** using **Tailwind CSS v4** coupled with custom Vanilla CSS for premium micro-interactions. The dashboard is designed around a unified state model where events from the backend (live cameras, active files, alerts) are consumed and reduced to populate visual overlays and event logs.



---

## 2. Dashboard Modes

The player workspace (`VideoPlayer.jsx`) acts as a wrapper that coordinates the configuration parameters (model checkpoint path, pipeline YAML config path, and device selection) and allows the operator to toggle between two operational modes.

### 2.1 Live Camera Mode (`WebcamPlayer.jsx`)
This is the primary demonstration path. It captures frames from the browser operator's local webcam and streams them to the backend in real-time.

*   **Camera Access:** Utilizes the HTML5 media API `navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } })`.
*   **Real-time Binary Streaming:** Frames are rendered onto a hidden canvas scaled to `224x224` (matching the model input requirement), compressed to JPEG (`quality: 0.8`), and transmitted as binary frames over a dedicated WebSocket connection (`/ws/camera`).
*   **Backpressure & Latency Protection:** The sending loop checks `ws.bufferedAmount`. If the outbound network buffer is backed up, frames are skipped to maintain real-time synchronization and avoid socket connection drops.
*   **Zero-Copy Send Optimization:** The frame is sent directly as a browser-native `Blob` using `ws.send(blob)`. This skips the JavaScript heap allocation and serialization overhead of converting the blob to an `ArrayBuffer`.
*   **Response Handling:** Detections (bounding boxes, classifications, scene context) are received on the same socket and updated inside the `WebSocketContext` state immediately for rendering.

### 2.2 MP4 Session Mode (`FilePlayer.jsx`)
This is the secondary/fallback demonstration path. It uploads an operator-selected MP4 to backend storage, runs an offline batch inference session on that backend-visible copy, and plays the browser-local preview with synchronized bounding boxes.

*   **Uploaded Videos Path:** Operator-selected MP4 files are sent to `POST /api/videos/upload` as multipart form data. The backend stores them under `/app/data/uploads/` using a generated UUID filename and returns a stable `video_id`.
*   **Demo/Raw Videos Path:** Existing backend-visible demo or raw files can still be started by server path when they are mounted under the configured backend data directories.
*   **Lifecycle API Integration:**
    1.  Operator selects an `.mp4` file from their device. The frontend creates a browser object URL only for local preview.
    2.  Frontend uploads the MP4 to `POST /api/videos/upload` using multipart field `file`.
    3.  After upload succeeds, frontend triggers a session on the backend: `POST /api/sessions/` with the returned `video_id`, checkpoint path, and YAML configuration path.
    4.  Frontend displays a blurred scanning screen and progress bar (`Processed X / Y frames`) while polling `GET /api/sessions/{id}` at a 1-second interval.
    5.  If the session is running or pending, video scrubbing, seeking, and drawing are disabled to protect the browser decoder from crashing.
    6.  Upon successful completion (`completed`), the frontend triggers synchronized video playback from the beginning and fetches the generated events database entries: `GET /api/events/sessions/{id}`.
*   **Synchronized Overlay Render Loop:**
    *   To prevent flickering and $O(N)$ lookup delays on every frame render tick (which is extremely slow for long videos), the session events are pre-indexed into a `Map` using `useMemo`:
        $$\text{Frame Index} \rightarrow \text{Array of active ActionEvents}$$
    *   During playback, the `requestAnimationFrame` loop performs a direct $O(1)$ map lookup: `eventsMap.get(Math.floor(currentTime * videoFps))`.
    *   Corner brackets, translucent bounding boxes, and label badges are dynamically drawn on top of the `<video>` element using an overlay `<canvas>`.

---

## 3. Shared State & Contexts

### 3.1 WebSocket Context (`WebSocketContext.jsx`)
Handles communication with the backend live channels and maintains the central event store.
*   **Connection Lifecycle:** Establishes a persistent connection to `/ws/live` on startup. Includes an **exponential backoff reconnect system** (starts at 1s, doubles on consecutive failures, capped at 30s) to survive container restarts.
*   **State Reduction:** Integrates live webcam detections, alerts, and historical database sessions into a unified reducer state (`initialState.alerts`, `initialState.sessionEvents`, `initialState.currentDetection`).
*   **Auto-fetch Alert History:** Upon establishing the WebSocket connection, queries `GET /api/events/?event_type=ALERT` to populate the event feed.

### 3.2 Scene Context (`SceneContext.jsx`)
Tracks the current environmental environment telemetry (e.g., `indoor`, `outdoor`, `vehicle_setting`).
*   **Operator Override:** Allows the operator to toggle an override switch to bypass backend scene contexts. It exposes a custom tag selector and confidence slider to test UI response behaviors during prototyping.

---

## 4. UI/UX Components

*   **`AlertLog.jsx`:** Displays active alarms (classified by severity: `danger` in red, `warning` in amber, `normal` in blue). Operators can search, filter by tab, and acknowledge (ACK) alarms.
*   **`ContextSettings.jsx`:** Displays system context status. Implements responsive CSS rules to render as a sidebar component on desktop screens and a sliding bottom-sheet settings drawer on mobile layouts.
*   **`StatusBar.jsx`:** Placed in the bottom-right. Displays a pulsing LED indicating the state of the socket connection (`connected` in green, `connecting` in amber, `disconnected` in red).
*   **`Header.jsx`:** Shows application branding and houses the connection status warnings.

---

## 5. Performance & Safety Optimizations

To ensure the client remains responsive during intensive, long-duration video processing, the following optimizations were implemented:

1.  **Call Stack Safe Event Parsing:**
    Instead of calculating progress bounds using spread operators (e.g., `Math.max(...events.map(e => e.end_frame_index))`), which throws a `RangeError: Maximum call stack size exceeded` in JS engines when processing more than ~65,000 frames, the progress is calculated using a linear loop wrapped in `useMemo`.
2.  **Throttling & Change-Detection:**
    React state updates from the high-frequency canvas drawing loop are strictly limited. The `SceneContext` is only updated when the context tag changes, or the confidence level shifts by more than $\pm 0.01$.
3.  **Frame Rate Limit Protection:**
    Webcam sending is throttled to 4 FPS to prevent CPU backpressure. The frame-sending interval includes a `ws.bufferedAmount` check to skip frames when the websocket buffer is full.

---

## 6. Technical Stack & Commands

### Prerequisites
*   **Node.js** v18+
*   **npm** v9+

### Package Commands
Inside the `frontend` folder:

```bash
# Install dependencies
npm install

# Start local dev server (default: port 5173)
npm run dev

# Build production assets (outputs to frontend/dist)
npm run build

# Run ESLint validation checks
npm run lint
```

### Dev Server Proxy Configuration
Vite dev server proxies traffic to avoid CORS issues. This is configured in `vite.config.js`:
*   `http://localhost:5173/api` $\rightarrow$ `http://localhost:8000/api` (REST API)
*   `ws://localhost:5173/ws` $\rightarrow$ `ws://localhost:8000/ws` (WebSockets)

---

## 7. Operator & User Manual (Instrukcja Użytkowania)

### 7.1 Initial Setup & Verification
1. Open your web browser and navigate to the dashboard URL (default: `http://localhost:5173`).
2. Verify that the system has successfully established connection to the backend. The status bar in the bottom right corner should display a pulsing green indicator: `WebSocket: Connected`.
3. *(Optional)* Click the **Inference Paths** button to configure or override the backend model checkpoint file path or the pipeline YAML configuration file.

### 7.2 Running Live Camera Inference (Webcam)
1. Select the **Webcam** tab in the main player view.
2. Click **Wybierz video** (which triggers the camera initialization workflow).
3. Grant camera access permission when prompted by the web browser.
4. The live webcam feed will start. The frontend will begin streaming frame data to the backend at 4 FPS.
5. Bounding boxes with high-fidelity corner brackets will draw automatically over detected humans, and the recognized actions (e.g. `person_sits_down`) will appear in the top center overlay and the sidebar alert feed.
6. Click **Pause** or change tabs to stop the webcam stream and close the hardware camera session.

### 7.3 Running Offline Video Inference (MP4 Session)
1. Select the **MP4 Session** tab in the main player view.
2. Click **Wybierz video** and select an `.mp4` file from your device.
3. The dashboard uploads the selected MP4 to the backend first. Browser-local file paths are not sent to the backend and are not expected to be readable by the backend container.
4. After the upload succeeds, the dashboard automatically starts a background analysis session using the returned `video_id`.
5. A scanning overlay will appear displaying `Analyzing Video...` along with a progress bar and the current frame count status.
6. You can abort the processing at any time by clicking **Stop Running Session**.
7. When the server completes the analysis, the overlay clears, and the video starts playing automatically from the beginning.
8. Bounding boxes and behavior labels will display in real-time sync with the video timeline. You can use the **Play/Pause** button to control playback.

### 7.4 Handling Alarms & Scene Context
*   **Acknowledge Alarms:** Review generated events in the **Live Event Log** sidebar. Click the **ACK** button on any critical alert to mark it as read and acknowledge the event.
*   **Scene Context Override:** Use the **Context Status** panel to view model-predicted scene context (e.g., `outdoor`, `indoor`). For simulation and testing, turn on the **Operator Override** toggle to manually set a custom scene tag and confidence slider.

