import Header from './components/Header'
import VideoPlayer from './components/VideoPlayer'
import AlertLog from './components/AlertLog'
import ContextSettings from './components/ContextSettings'
import StatusBar from './components/StatusBar'
import { useWebSocket } from './context/WebSocketContext'
import './App.css'

function App() {
  const { connectionStatus } = useWebSocket()

  return (
    <div className="shell">
      {connectionStatus === 'disconnected' && (
        <div className="bg-red/15 border-b border-red/25 text-red px-4 py-2 text-center text-xs font-mono flex items-center justify-center gap-2 select-none">
          <span className="w-2 h-2 rounded-full bg-red animate-pulse" />
          CRITICAL: Connection to backend lost. Retrying connection...
        </div>
      )}
      {connectionStatus === 'connecting' && (
        <div className="bg-amber/15 border-b border-amber/25 text-amber px-4 py-2 text-center text-xs font-mono flex items-center justify-center gap-2 select-none">
          <span className="w-2 h-2 rounded-full bg-amber animate-pulse" />
          Connecting to backend...
        </div>
      )}
      <Header />

      <main className="dashboard">
        <VideoPlayer />

        <div className="sidebar">
          <AlertLog />
          <ContextSettings />
        </div>
      </main>

      <StatusBar />
    </div>
  )
}

export default App

