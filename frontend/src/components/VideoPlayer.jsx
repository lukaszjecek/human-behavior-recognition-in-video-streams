import { useState } from 'react'
import { useWebSocket } from '../context/WebSocketContext'
import WebcamPlayer from './WebcamPlayer'
import FilePlayer from './FilePlayer'

export default function VideoPlayer() {
  const { dispatch } = useWebSocket()

  const [mode, setMode] = useState('webcam') // 'webcam' | 'mp4'
  const [showSettings, setShowSettings] = useState(false)

  // Inference Settings (Shared between players via props)
  const [checkpointPath, setCheckpointPath] = useState('data/logs/checkpoints/baseline_epoch_50.pth')
  const [configPath, setConfigPath] = useState('configs/data_pipeline.yml')
  const [device, setDevice] = useState('auto')
  const [serverVideoPath, setServerVideoPath] = useState('data/raw/smoke_sample.mp4')

  // Mode change handler to clean state transitions cleanly
  function handleModeChange(newMode) {
    if (newMode === mode) return
    setMode(newMode)
    dispatch({ type: 'CLEAR_ALERTS' })
  }

  return (
    <section className="panel flex flex-col" id="video-player">
      <div className="panel-header flex items-center justify-between border-b border-border">
        <h2 className="font-mono flex items-center gap-2">
          <span className={`w-2 h-2 rounded-full ${mode === 'webcam' ? 'bg-green' : 'bg-text-dim'}`} />
          {mode === 'webcam' ? 'Live Camera Feed' : 'MP4 Session Feed'}
        </h2>

        <div className="flex gap-2">
          {/* Mode Selector */}
          <button
            onClick={() => handleModeChange('webcam')}
            className={`px-2 py-0.5 text-[11px] font-mono rounded border ${
              mode === 'webcam'
                ? 'border-border bg-surface text-text font-bold'
                : 'border-transparent text-text-dim hover:bg-surface-alt/50'
            } cursor-pointer transition-all`}
          >
            Webcam
          </button>
          <button
            onClick={() => handleModeChange('mp4')}
            className={`px-2 py-0.5 text-[11px] font-mono rounded border ${
              mode === 'mp4'
                ? 'border-border bg-surface text-text font-bold'
                : 'border-transparent text-text-dim hover:bg-surface-alt/50'
            } cursor-pointer transition-all`}
          >
            MP4 Session
          </button>

          <button
            onClick={() => setShowSettings(!showSettings)}
            className={`px-2 py-0.5 text-[11px] font-mono rounded border border-border bg-surface-alt hover:bg-border cursor-pointer text-text transition-colors flex items-center gap-1 ${showSettings ? 'bg-border' : ''}`}
          >
            <svg className="w-3 h-3 text-text-dim" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="3" />
              <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
            </svg>
            Inference Paths
          </button>
        </div>
      </div>

      {/* Collapsible config settings */}
      {showSettings && (
        <div className="p-3 border-b border-border bg-surface-alt/40 grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs font-mono">
          <div className="flex flex-col gap-1">
            <label className="text-[10px] text-text-dim">Model Checkpoint Path</label>
            <input
              type="text"
              value={checkpointPath}
              onChange={(e) => setCheckpointPath(e.target.value)}
              className="bg-surface border border-border rounded px-2 py-1 text-text text-xs focus:outline-none focus:border-text-dim"
            />
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-[10px] text-text-dim">Pipeline YAML Path</label>
            <input
              type="text"
              value={configPath}
              onChange={(e) => setConfigPath(e.target.value)}
              className="bg-surface border border-border rounded px-2 py-1 text-text text-xs focus:outline-none focus:border-text-dim"
            />
          </div>

          {mode === 'mp4' && (
            <div className="flex flex-col gap-1 col-span-1 sm:col-span-2">
              <label className="text-[10px] text-text-dim">Server Video Path</label>
              <input
                type="text"
                value={serverVideoPath}
                onChange={(e) => setServerVideoPath(e.target.value)}
                className="bg-surface border border-border rounded px-2 py-1 text-text text-xs focus:outline-none focus:border-text-dim"
              />
            </div>
          )}
        </div>
      )}

      {/* Render Subcomponents based on active mode */}
      <div className="flex-1 flex flex-col min-h-0">
        {mode === 'webcam' ? (
          <WebcamPlayer
            checkpointPath={checkpointPath}
            configPath={configPath}
            device={device}
          />
        ) : (
          <FilePlayer
            checkpointPath={checkpointPath}
            configPath={configPath}
            device={device}
            serverVideoPath={serverVideoPath}
            setServerVideoPath={setServerVideoPath}
          />
        )}
      </div>
    </section>
  )
}
