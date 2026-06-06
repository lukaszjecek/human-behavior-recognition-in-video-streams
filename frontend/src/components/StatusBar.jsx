import { useWebSocket } from '../context/WebSocketContext'

export default function StatusBar() {
  const { connectionStatus } = useWebSocket()

  const getStatusConfig = () => {
    switch (connectionStatus) {
      case 'connected':
        return {
          colorClass: 'bg-green shadow-[0_0_8px_rgba(52,211,153,0.5)]',
          label: 'Connected',
          pulse: false
        }
      case 'connecting':
        return {
          colorClass: 'bg-amber shadow-[0_0_8px_rgba(245,166,35,0.5)]',
          label: 'Connecting...',
          pulse: true
        }
      case 'disconnected':
      default:
        return {
          colorClass: 'bg-red shadow-[0_0_8px_rgba(240,83,101,0.5)]',
          label: 'Disconnected',
          pulse: true
        }
    }
  }

  const { colorClass, label, pulse } = getStatusConfig()

  return (
    <footer className="flex items-center justify-between px-5 py-1.5 border-t border-border text-[11px] text-text-faint bg-surface/50">
      <span className="font-mono">HBR Dashboard v0.1.0</span>
      <div className="flex items-center gap-2 font-mono">
        <span className="text-text-dim">WebSocket:</span>
        <span className={`w-2 h-2 rounded-full ${colorClass} ${pulse ? 'animate-pulse' : ''}`} />
        <span className="text-text font-medium">{label}</span>
      </div>
    </footer>
  )
}

