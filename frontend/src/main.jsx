import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { SceneContextProvider } from './context/SceneContext'
import './index.css'
import App from './App.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <SceneContextProvider>
      <App />
    </SceneContextProvider>
  </StrictMode>,
)
