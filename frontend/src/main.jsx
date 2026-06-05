import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { SceneContextProvider } from './context/SceneContext'
import { WebSocketProvider } from './context/WebSocketContext'
import './index.css'
import App from './App.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <SceneContextProvider>
      <WebSocketProvider>
        <App />
      </WebSocketProvider>
    </SceneContextProvider>
  </StrictMode>,
)
