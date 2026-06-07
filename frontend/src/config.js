// Dynamic configuration helper for API and WebSocket connections.
// In development, the Vite proxy handles relative requests for '/api' and '/ws'.
// Build-time overrides can be provided via VITE_API_BASE_URL and VITE_WS_URL.

export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '';

export const getWsUrl = () => {
  if (import.meta.env.VITE_WS_URL) {
    return import.meta.env.VITE_WS_URL;
  }
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  // Use relative path through Vite proxy / server config mapping to /ws/live
  return `${protocol}//${window.location.host}/ws/live`;
};
