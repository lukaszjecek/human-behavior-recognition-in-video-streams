# Frontend
## Author: [Filip Wasiel](https://github.com/Filipwasiel)

[Back to README](../README.md)

This directory contains the React-based frontend dashboard for the **Human Behavior Recognition (HBR) in Video Streams** application. It provides operator interfaces, streaming video feeds with real-time detection bounding box overlays, live event logging, and contextual scene telemetry.

## Key Features

### 1. Camera Feed & Object Bounding Boxes (`VideoPlayer.jsx`)
- Supports loading local video streams (`sample.mp4` by default) and drawing simulated behavior classifications.
- Renders high-fidelity corner brackets, bounding box fills, label badges, and prediction confidence scores dynamically.
- Automatically handles video dimensions scaling.

### 2. Live Alerts & Event Log (`AlertLog.jsx`)
- Displays historical and simulated behavior events.
- Features search filtering, severity tags (`danger` in red, `warning` in amber, `info` in blue), and interactive alert acknowledgment (ACK).
- Responsive tabs to show filtered views.

### 3. Scene Context Status Panel (`ContextSettings.jsx`)
- **System Inference Status:** Displays the current active scene context tag (`indoor`, `outdoor`, `vehicle_setting`, `unknown`) conforming to the Sprint 3 context contract, along with confidence progress bars.
- **Local Prototyping Override:** Provides a simulated operator override switch to prototype dashboard behaviors under different scene contexts without replacing backend logic. Includes a custom scene tag selector and mock confidence slider.
- **Unified State Management (`SceneContext.jsx`):** Employs React Context to coordinate backend contexts and manual overrides globally between the video timeline and sidebar widget views.
- **Adaptive Layout Support:** The `ContextSettings` component automatically transitions between a desktop sidebar widget and a mobile bottom-sheet settings modal dynamically based on props and CSS queries.

---

## Technical Stack & Architecture

- **Core:** [React 19](https://react.dev/) + [Vite 8](https://vite.dev/)
- **Styling:** [Tailwind CSS v4](https://tailwindcss.com/) + Custom Vanilla CSS for rich micro-interactions and custom toggles.
- **Performance Optimization:** Implements ref-based throttling inside the `requestAnimationFrame` loop in `VideoPlayer.jsx` to limit state-updating frequency, preventing performance degradation from 30fps React re-renders.

---

## Getting Started

### Prerequisites
Make sure you have [Node.js](https://nodejs.org/) (v18+) installed.

### Installation
Run the following command inside the `frontend` folder to install dependencies:
```bash
npm install
```

### Run Locally (Development Mode)
Start the hot-reloading development server:
```bash
npm run dev
```

### Production Build
Compile the optimized production assets:
```bash
npm run build
```

### Linting
Validate the codebase against ESLint rules and React hooks validation:
```bash
npm run lint
```
