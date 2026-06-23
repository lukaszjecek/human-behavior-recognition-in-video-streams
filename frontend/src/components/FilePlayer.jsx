import { useState, useEffect, useRef, useMemo } from 'react'
import { useSceneContext } from '../context/SceneContext'
import { useWebSocket } from '../context/WebSocketContext'
import { API_BASE_URL } from '../config'

export default function FilePlayer({
  checkpointPath,
  configPath,
  device,
  serverVideoPath,
  setServerVideoPath,
}) {
  const { setBackendContext } = useSceneContext()
  const { state, dispatch } = useWebSocket()

  const [videoSrc, setVideoSrc] = useState(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [videoDimensions, setVideoDimensions] = useState({ width: 0, height: 0 })
  const [videoDuration, setVideoDuration] = useState(0)

  // Offline Session state
  const [sessionId, setSessionId] = useState(null)
  const [sessionStatus, setSessionStatus] = useState('idle') // 'idle', 'pending', 'running', 'completed', 'failed', 'stopped'
  const [statusMessage, setStatusMessage] = useState('')
  const [errorMessage, setErrorMessage] = useState('')

  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const fileInputRef = useRef(null)
  const animationFrameId = useRef(null)
  const lastReportedContext = useRef({ scene_tag: '', confidence: -1 })
  const lastTimeRef = useRef(-1)
  const autoPlayTriggeredRef = useRef(null)

  // Pre-index session events by frame index when events list changes using useMemo for fast O(1) lookup
  const eventsMap = useMemo(() => {
    const map = new Map()
    const events = [...state.sessionEvents]
    
    // Sort events by start_frame_index to ensure chronological ordering
    events.sort((a, b) => (a.start_frame_index || 0) - (b.start_frame_index || 0))

    events.forEach((event, index) => {
      let start = event.start_frame_index || 0
      // Extend the very first event's start to 0 to avoid a brief blink at the beginning of the video
      if (index === 0) {
        start = 0
      }
      
      let end = event.end_frame_index || 0
      
      // Look ahead to the next event to extend the current event's range and fill the stride gap
      const nextEvent = events[index + 1]
      if (nextEvent && typeof nextEvent.start_frame_index === 'number') {
        end = Math.max(end, nextEvent.start_frame_index - 1)
      } else {
        // For the last event, extend by a default stride buffer (e.g. 32 frames)
        end = end + 32
      }

      for (let f = start; f <= end; f++) {
        if (!map.has(f)) {
          map.set(f, [])
        }
        map.get(f).push(event)
      }
    })
    return map
  }, [state.sessionEvents])

  // Calculate actual video FPS dynamically when events or video source changes using useMemo
  const videoFps = useMemo(() => {
    if (sessionStatus === 'completed' && state.sessionEvents.length > 0 && videoDuration > 0) {
      const maxFrame = Math.max(...state.sessionEvents.map(e => e.end_frame_index || 0))
      if (maxFrame > 0) {
        return maxFrame / videoDuration
      }
    }
    return 30
  }, [state.sessionEvents, sessionStatus, videoDuration])

  async function fetchSessionEvents(sId) {
    try {
      const response = await fetch(`${API_BASE_URL}/api/events/sessions/${sId}?limit=10000`)
      if (response.ok) {
        const data = await response.json()
        dispatch({ type: 'SET_SESSION_EVENTS', payload: data })
        setStatusMessage(`Loaded ${data.length} events for session.`)
      }
    } catch (err) {
      console.error('Failed to load session events:', err)
      setErrorMessage('Failed to fetch session event database records.')
    }
  }

  // Offline MP4 Session controls
  async function startOfflineSession(videoPathOverride) {
    setErrorMessage('')
    setSessionStatus('pending')
    setStatusMessage('Spawning background inference session on server...')

    const targetVideoPath = (typeof videoPathOverride === 'string') ? videoPathOverride : serverVideoPath

    try {
      const response = await fetch(`${API_BASE_URL}/api/sessions/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          video_path: targetVideoPath,
          checkpoint_path: checkpointPath,
          config_path: configPath,
          device: device
        })
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.detail || `Server returned error status ${response.status}`)
      }

      const data = await response.json()
      setSessionId(data.id)
      setSessionStatus(data.status)
      setStatusMessage(`Session started. ID: ${data.id}`)

      dispatch({ type: 'SET_SESSION_EVENTS', payload: [] })

    } catch (err) {
      console.error('Session create failed:', err)
      setErrorMessage(err.message || 'Failed to trigger background video session.')
      setSessionStatus('failed')
    }
  }

  async function stopOfflineSession() {
    if (!sessionId) return
    setStatusMessage('Stopping background session...')
    try {
      const response = await fetch(`${API_BASE_URL}/api/sessions/${sessionId}/stop`, {
        method: 'POST'
      })
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.detail || 'Stop request failed.')
      }
      const data = await response.json()
      setSessionStatus(data.status)
      setStatusMessage('Session stopped by operator request.')
    } catch (err) {
      console.error('Session stop failed:', err)
      setErrorMessage(err.message || 'Failed to stop session.')
    }
  }

  // Cleanup on unmount or videoSrc change
  useEffect(() => {
    return () => {
      if (videoSrc && videoSrc.startsWith('blob:')) {
        URL.revokeObjectURL(videoSrc)
      }
    }
  }, [videoSrc])

  // Poll running session status
  useEffect(() => {
    if (!sessionId || (sessionStatus !== 'pending' && sessionStatus !== 'running')) {
      return
    }

    const interval = setInterval(async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/sessions/${sessionId}`)
        if (response.ok) {
          const data = await response.json()
          setSessionStatus(data.status)

          if (data.status === 'completed') {
            setStatusMessage('Session completed successfully. Loading results...')
            fetchSessionEvents(sessionId)
            clearInterval(interval)
          } else if (data.status === 'failed' || data.status === 'stopped') {
            setStatusMessage(`Session terminated. Status: ${data.status}`)
            if (data.error) setErrorMessage(data.error)
            clearInterval(interval)
          }
        }
      } catch (err) {
        console.error('Failed to poll session status:', err)
      }
    }, 1000)

    return () => clearInterval(interval)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId, sessionStatus])

  // Reset auto-play trigger ref when session ID changes
  useEffect(() => {
    autoPlayTriggeredRef.current = null
  }, [sessionId])

  // Intelligently auto-play the video only when the background inference session is fully COMPLETED.
  useEffect(() => {
    if (sessionId && videoRef.current && sessionStatus === 'completed') {
      if (autoPlayTriggeredRef.current !== sessionId) {
        autoPlayTriggeredRef.current = sessionId
        console.log(`Backend session ${sessionId} is completed. Starting synchronized video playback.`);
        videoRef.current.currentTime = 0 // Reset to the beginning for perfect sync
        videoRef.current.play().catch(err => {
          console.warn('Auto-playback failed or was blocked by browser:', err)
        })
        setIsPlaying(true)
      }
    }
  }, [sessionStatus, sessionId])

  const handleFileChange = (e) => {
    const file = e.target.files?.[0]
    if (file) {
      if (videoSrc && videoSrc.startsWith('blob:')) {
        URL.revokeObjectURL(videoSrc)
      }
      const objectURL = URL.createObjectURL(file)
      setVideoSrc(objectURL)
      setVideoDuration(0)
      const targetPath = `data/raw/${file.name}`
      setServerVideoPath(targetPath)
      setIsPlaying(false)
      // Auto-start background inference session on the server for this video file
      startOfflineSession(targetPath)
    }
  }

  const triggerFileInput = () => {
    fileInputRef.current?.click()
  }

  const handleLoadedMetadata = () => {
    const video = videoRef.current
    const canvas = canvasRef.current
    if (video && canvas) {
      canvas.width = video.videoWidth || video.width || 640
      canvas.height = video.videoHeight || video.height || 480
      setVideoDimensions({ width: canvas.width, height: canvas.height })
      setVideoDuration(video.duration || 0)
    }
  }

  // Canvas Render Loop for MP4 Video Detections
  useEffect(() => {
    const video = videoRef.current
    const canvas = canvasRef.current

    if (!canvas) {
      if (animationFrameId.current) cancelAnimationFrame(animationFrameId.current)
      return
    }

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Reset last time to force a redraw on new dependencies
    lastTimeRef.current = -1

    const renderLoop = () => {
      const W = canvas.width
      const H = canvas.height

      let activeLabel = 'unknown'
      let activeConfidence = 0.0
      let activeSceneTag = 'unknown'
      let activeSceneConfidence = 0.0
      let activeBboxes = []

      if (video) {
        // Find events corresponding to current playing timestamp
        const time = video.currentTime
        if (time === lastTimeRef.current) {
          // Playback time has not progressed, skip recalculations and canvas clears
          animationFrameId.current = requestAnimationFrame(renderLoop)
          return
        }
        lastTimeRef.current = time

        const frameIndex = Math.floor(time * videoFps)
        const activeEvents = eventsMap.get(frameIndex) || []

        if (activeEvents.length > 0) {
          activeEvents.sort((a, b) => b.confidence - a.confidence)
          activeLabel = activeEvents[0].label
          activeConfidence = activeEvents[0].confidence
          activeBboxes = activeEvents[0].bboxes || []
          if (activeEvents[0].context) {
            activeSceneTag = activeEvents[0].context.scene_tag || 'unknown'
            activeSceneConfidence = activeEvents[0].context.confidence || 0.0
          }
        }
      }

      ctx.clearRect(0, 0, W, H)

      // Sync SceneContext
      if (
        lastReportedContext.current.scene_tag !== activeSceneTag ||
        Math.abs(lastReportedContext.current.confidence - activeSceneConfidence) > 0.01
      ) {
        lastReportedContext.current = { scene_tag: activeSceneTag, confidence: activeSceneConfidence }
        setBackendContext({ scene_tag: activeSceneTag, confidence: activeSceneConfidence })
      }

      // Draw overall Action Label header overlay on canvas
      if (activeLabel && activeLabel !== 'unknown') {
        const bannerText = `DETECTED ACTION: ${activeLabel} [${(activeConfidence * 100).toFixed(1)}%]`
        ctx.font = 'bold 12px "Fira Mono", ui-monospace, monospace'
        const textWidth = ctx.measureText(bannerText).width

        ctx.fillStyle = 'rgba(0, 0, 0, 0.7)'
        ctx.beginPath()
        ctx.roundRect(W / 2 - textWidth / 2 - 10, 10, textWidth + 20, 22, 4)
        ctx.fill()

        ctx.fillStyle = '#f05365'
        ctx.textBaseline = 'middle'
        ctx.fillText(bannerText, W / 2 - textWidth / 2, 21)
      }

      // Draw bounding boxes if present in payload
      if (activeBboxes.length > 0) {
        activeBboxes.forEach(box => {
          let xMin = box.x_min
          let yMin = box.y_min
          let xMax = box.x_max
          let yMax = box.y_max

          if (xMin === undefined || yMin === undefined || xMax === undefined || yMax === undefined) {
            return
          }

          const isNormalized = box.coordinate_space === 'normalized' ||
            (!box.coordinate_space && xMin <= 1.0 && yMin <= 1.0 && xMax <= 1.0 && yMax <= 1.0)

          if (isNormalized) {
            xMin *= W
            yMin *= H
            xMax *= W
            yMax *= H
          } else {
            const sw = box.source_width || 640
            const sh = box.source_height || 480
            xMin *= (W / sw)
            yMin *= (H / sh)
            xMax *= (W / sw)
            yMax *= (H / sh)
          }

          // Draw translucent box fill
          ctx.fillStyle = 'rgba(240, 83, 101, 0.08)'
          ctx.fillRect(xMin, yMin, xMax - xMin, yMax - yMin)

          // Draw thin box border
          ctx.strokeStyle = 'rgba(240, 83, 101, 0.4)'
          ctx.lineWidth = 1
          ctx.strokeRect(xMin, yMin, xMax - xMin, yMax - yMin)

          // Draw premium high-fidelity corner brackets
          ctx.strokeStyle = '#f05365'
          ctx.lineWidth = 2.5
          const cornerLen = Math.min(12, (xMax - xMin) / 4, (yMax - yMin) / 4)

          // Top-Left corner
          ctx.beginPath()
          ctx.moveTo(xMin + cornerLen, yMin)
          ctx.lineTo(xMin, yMin)
          ctx.lineTo(xMin, yMin + cornerLen)
          ctx.stroke()

          // Top-Right corner
          ctx.beginPath()
          ctx.moveTo(xMax - cornerLen, yMin)
          ctx.lineTo(xMax, yMin)
          ctx.lineTo(xMax, yMin + cornerLen)
          ctx.stroke()

          // Bottom-Left corner
          ctx.beginPath()
          ctx.moveTo(xMin + cornerLen, yMax)
          ctx.lineTo(xMin, yMax)
          ctx.lineTo(xMin, yMax - cornerLen)
          ctx.stroke()

          // Bottom-Right corner
          ctx.beginPath()
          ctx.moveTo(xMax - cornerLen, yMax)
          ctx.lineTo(xMax, yMax)
          ctx.lineTo(xMax, yMax - cornerLen)
          ctx.stroke()

          // Draw label badge on top-left of the bounding box
          if (box.label) {
            const confText = box.confidence !== undefined ? ` ${(box.confidence * 100).toFixed(0)}%` : ''
            const labelText = `${box.label}${confText}`.toUpperCase()
            
            ctx.font = 'bold 9px "Fira Mono", ui-monospace, monospace'
            const textMetrics = ctx.measureText(labelText)
            const badgeW = textMetrics.width + 8
            const badgeH = 14

            // Position badge above box, or inside box top-left if there is no space above
            const badgeX = xMin
            const badgeY = yMin - badgeH >= 0 ? yMin - badgeH : yMin

            ctx.fillStyle = 'rgba(240, 83, 101, 0.85)'
            ctx.beginPath()
            ctx.roundRect(badgeX, badgeY, badgeW, badgeH, 2)
            ctx.fill()

            ctx.fillStyle = '#ffffff'
            ctx.textBaseline = 'middle'
            ctx.fillText(labelText, badgeX + 4, badgeY + badgeH / 2)
          }
        })
      }

      animationFrameId.current = requestAnimationFrame(renderLoop)
    }

    renderLoop()

    return () => {
      if (animationFrameId.current) {
        cancelAnimationFrame(animationFrameId.current)
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [videoSrc, state.sessionEvents, eventsMap, videoFps])

  const maxFrameProcessed = state.sessionEvents.length > 0
    ? Math.max(...state.sessionEvents.map(e => e.end_frame_index || 0))
    : 0
  const totalFrames = videoDuration ? Math.round(videoDuration * videoFps) : 0
  const progressPercent = totalFrames > 0 ? Math.min(Math.round((maxFrameProcessed / totalFrames) * 100), 100) : 0

  // Scrub video to show the latest processed frame in real-time
  useEffect(() => {
    if ((sessionStatus === 'running' || sessionStatus === 'pending') && videoRef.current && videoFps > 0 && maxFrameProcessed > 0) {
      videoRef.current.currentTime = maxFrameProcessed / videoFps
    }
  }, [maxFrameProcessed, sessionStatus, videoFps])

  return (
    <div className="w-full h-full flex flex-col">
      {/* Status Bar / Error message logs */}
      {(statusMessage || errorMessage) && (
        <div className="px-4 py-2 border-b border-border text-xs font-mono flex flex-col gap-1 w-full bg-surface-alt/25">
          {statusMessage && <div className="text-text-dim">Status: {statusMessage}</div>}
          {errorMessage && <div className="text-red font-semibold bg-red/10 border border-red/25 px-2 py-1 rounded">Error: {errorMessage}</div>}
        </div>
      )}

      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileChange}
        accept="video/*"
        className="hidden"
      />

      <div className="flex-1 relative flex items-center justify-center bg-black overflow-hidden min-h-[350px]">
        {videoSrc ? (
          <div className="w-full h-full relative flex items-center justify-center">
            <video
              ref={videoRef}
              src={videoSrc}
              onLoadedMetadata={handleLoadedMetadata}
              onPlay={() => setIsPlaying(true)}
              onPause={() => setIsPlaying(false)}
              className="w-full h-full object-contain"
              loop
              muted
              playsInline
            />
            <canvas
              ref={canvasRef}
              className="absolute top-0 left-0 w-full h-full object-contain pointer-events-none"
            />

            {/* SCI-FI SCANNING OVERLAY */}
            {(sessionStatus === 'running' || sessionStatus === 'pending') && (
              <div className="absolute inset-0 bg-black/65 backdrop-blur-xs flex flex-col items-center justify-center p-6 z-10 select-none">
                <div className="w-10 h-10 rounded-full border-2 border-red border-t-transparent animate-spin mb-4" />
                <h3 className="text-sm font-semibold text-text mb-1 font-mono tracking-wide uppercase">
                  Analyzing Video...
                </h3>
                <p className="text-xs text-text-dim mb-3 font-mono">
                  Processed {maxFrameProcessed} / {totalFrames} frames ({progressPercent}%)
                </p>
                <div className="w-64 h-1.5 bg-surface-alt border border-border rounded-full overflow-hidden">
                  <div
                    className="h-full bg-red transition-all duration-300"
                    style={{ width: `${progressPercent}%` }}
                  />
                </div>
              </div>
            )}

            {videoDimensions.width > 0 && (
              <div className="absolute bottom-3 left-3 px-2 py-0.5 rounded bg-black/60 backdrop-blur-xs text-[10px] font-mono text-text-dim pointer-events-none select-none">
                {videoDimensions.width}×{videoDimensions.height} px
              </div>
            )}

            <div className="absolute bottom-3 right-3 flex gap-2">
              {(sessionStatus === 'running' || sessionStatus === 'pending') && (
                <button
                  onClick={stopOfflineSession}
                  className="px-2.5 py-1 text-xs font-mono rounded border border-red/45 bg-red/10 hover:bg-red text-white cursor-pointer transition-all animate-pulse"
                >
                  Stop Running Session
                </button>
              )}
              <button
                disabled={sessionStatus === 'running' || sessionStatus === 'pending'}
                onClick={() => {
                  const video = videoRef.current
                  if (video) {
                    if (isPlaying) {
                      video.pause()
                      setIsPlaying(false)
                    } else {
                      video.play().catch(err => console.log('Playback failed:', err))
                      setIsPlaying(true)
                    }
                  }
                }}
                className={`px-2.5 py-1 text-xs font-mono rounded border ${
                  sessionStatus === 'running' || sessionStatus === 'pending'
                    ? 'border-border/40 bg-surface-alt/40 text-text-dim/40 cursor-not-allowed'
                    : 'border-border bg-surface-alt hover:bg-border cursor-pointer text-text flex items-center gap-1'
                } transition-colors`}
              >
                {isPlaying ? 'Pause' : 'Play'}
              </button>
              <button
                disabled={sessionStatus === 'running' || sessionStatus === 'pending'}
                onClick={triggerFileInput}
                className={`px-2.5 py-1 text-xs font-mono rounded border ${
                  sessionStatus === 'running' || sessionStatus === 'pending'
                    ? 'border-border/40 bg-surface-alt/40 text-text-dim/40 cursor-not-allowed'
                    : 'border-border bg-surface-alt hover:bg-border cursor-pointer text-text transition-colors'
                }`}
              >
                Wybierz video
              </button>
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center p-8 text-center max-w-md mx-auto">
            <div className="w-14 h-14 rounded-2xl bg-surface-alt border border-border flex items-center justify-center mb-4 text-text-dim">
              <svg className="w-6 h-6" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path strokeLinecap="round" strokeLinejoin="round" d="M3.375 19.5h17.25m-17.25 0a1.125 1.125 0 01-1.125-1.125M3.375 19.5h7.5c.621 0 1.125-.504 1.125-1.125v-1.5c0-.621-.504-1.125-1.125-1.125h-7.5M2.25 18.75V6.75A2.25 2.25 0 014.5 4.5h15a2.25 2.25 0 012.25 2.25v12m-18 0A2.25 2.25 0 004.5 21h15a2.25 2.25 0 002.25-2.25m-18 0v-3.75a2.25 2.25 0 012.25-2.25h13.5a2.25 2.25 0 012.25 2.25v3.75m-16.5-7.5h15" />
              </svg>
            </div>
            <h3 className="text-sm font-semibold text-text mb-1 font-mono">Backend MP4 Video Session</h3>
            <p className="text-xs text-text-dim mb-4 leading-relaxed">
              Trigger an offline inference session on the server, load the corresponding local video to see overlays, or watch events populate.
            </p>

            <div className="flex flex-col gap-2 w-full">
              <div className="flex gap-2 justify-center">
                {(sessionStatus === 'running' || sessionStatus === 'pending') && (
                  <button
                    onClick={stopOfflineSession}
                    className="px-4 py-2 text-xs font-mono font-medium rounded-lg border border-red/45 bg-red/10 hover:bg-red text-white cursor-pointer shadow transition-all animate-pulse"
                  >
                    Stop Running Session
                  </button>
                )}

                <button
                  onClick={triggerFileInput}
                  className="px-4 py-2 text-xs font-mono font-medium rounded-lg border border-border bg-surface-alt hover:bg-border cursor-pointer text-text shadow-sm hover:shadow transition-all"
                >
                  Wybierz video
                </button>
              </div>

              {sessionId && (
                <div className="text-[10px] font-mono text-text-dim mt-2 bg-surface-alt p-2 rounded border border-border">
                  <div>Session ID: {sessionId}</div>
                  <div>Status: <span className="font-bold text-text uppercase">{sessionStatus}</span></div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
