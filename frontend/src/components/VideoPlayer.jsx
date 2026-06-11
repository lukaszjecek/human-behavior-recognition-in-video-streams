import { useState, useEffect, useRef } from 'react'
import { useSceneContext } from '../context/SceneContext'
import { useWebSocket } from '../context/WebSocketContext'
import { API_BASE_URL } from '../config'

export default function VideoPlayer() {
  const { setBackendContext } = useSceneContext()
  const { state, dispatch } = useWebSocket()

  const [mode, setMode] = useState('webcam') // 'webcam' | 'mp4'
  const [webcamActive, setWebcamActive] = useState(false)
  const [videoSrc, setVideoSrc] = useState(null)
  const [fileName, setFileName] = useState('')
  const [isPlaying, setIsPlaying] = useState(false)
  const [videoDimensions, setVideoDimensions] = useState({ width: 0, height: 0 })
  const [showSettings, setShowSettings] = useState(false)

  // Inference Settings
  const [serverVideoPath, setServerVideoPath] = useState('data/raw/smoke_sample.mp4')
  const [checkpointPath, setCheckpointPath] = useState('data/logs/checkpoints/baseline_epoch_50.pth')
  const [configPath, setConfigPath] = useState('configs/data_pipeline.yml')
  const [device, setDevice] = useState('cpu')

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

  // Pre-indexed events and FPS for fast O(1) lookup
  const [eventsMap, setEventsMap] = useState(new Map())
  const [videoFps, setVideoFps] = useState(30)

  // Pre-index session events by frame index when events list changes
  useEffect(() => {
    const map = new Map()
    state.sessionEvents.forEach(event => {
      const start = event.start_frame_index || 0
      const end = event.end_frame_index || 0
      for (let f = start; f <= end; f++) {
        if (!map.has(f)) {
          map.set(f, [])
        }
        map.get(f).push(event)
      }
    })
    setEventsMap(map)
  }, [state.sessionEvents])

  // Calculate actual video FPS dynamically when events or video source changes
  useEffect(() => {
    if (state.sessionEvents.length > 0 && videoRef.current?.duration) {
      const maxFrame = Math.max(...state.sessionEvents.map(e => e.end_frame_index || 0))
      if (maxFrame > 0) {
        setVideoFps(maxFrame / videoRef.current.duration)
      }
    } else {
      setVideoFps(30)
    }
  }, [state.sessionEvents, videoSrc])
  const webcamWsRef = useRef(null)
  const sendIntervalRef = useRef(null)

  // Construct WebSocket URL for camera
  function getCameraWsUrl() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    return `${protocol}//${window.location.host}/ws/camera`
  }

  // Webcam controls
  function stopWebcam() {
    if (sendIntervalRef.current) {
      clearInterval(sendIntervalRef.current)
      sendIntervalRef.current = null
    }

    if (webcamWsRef.current) {
      if (webcamWsRef.current.readyState === WebSocket.OPEN) {
        webcamWsRef.current.send('stop')
        webcamWsRef.current.close()
      }
      webcamWsRef.current = null
    }

    if (videoRef.current && videoRef.current.srcObject) {
      videoRef.current.srcObject.getTracks().forEach(track => track.stop())
      videoRef.current.srcObject = null
    }

    setWebcamActive(false)
    dispatch({ type: 'RESET_DETECTION' })
  }

  function startFrameSending() {
    if (sendIntervalRef.current) clearInterval(sendIntervalRef.current)

    const scaleCanvas = document.createElement('canvas')
    scaleCanvas.width = 224
    scaleCanvas.height = 224
    const scaleCtx = scaleCanvas.getContext('2d')

    sendIntervalRef.current = setInterval(() => {
      const video = videoRef.current
      const ws = webcamWsRef.current
      if (video && ws && ws.readyState === WebSocket.OPEN) {
        // Skip frame if the websocket outbound queue has backed up (prevents disconnections & latency lag)
        if (ws.bufferedAmount > 0) {
          console.warn('Webcam socket buffer backup detected, skipping frame to maintain realtime sync.');
          return;
        }

        scaleCtx.drawImage(video, 0, 0, scaleCanvas.width, scaleCanvas.height)
        scaleCanvas.toBlob((blob) => {
          if (blob && ws && ws.readyState === WebSocket.OPEN) {
            blob.arrayBuffer().then((buf) => {
              if (ws.readyState === WebSocket.OPEN) {
                ws.send(buf)
              }
            })
          }
        }, 'image/jpeg', 0.8)
      }
    }, 250) // 4 FPS, slightly more conservative to prevent backpressure on CPU inference
  }

  async function startWebcam() {
    setErrorMessage('')
    setStatusMessage('Requesting camera access...')
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 }
      })

      if (videoRef.current) {
        videoRef.current.srcObject = stream
        videoRef.current.play().catch(e => console.log('Playback failed:', e))
      }

      setStatusMessage('Connecting to backend camera socket...')
      const wsUrl = getCameraWsUrl()
      console.log(`Connecting to camera WS: ${wsUrl}`)
      const ws = new WebSocket(wsUrl)
      webcamWsRef.current = ws

      ws.onopen = () => {
        setStatusMessage('Initializing inference pipeline...')
        ws.send(JSON.stringify({
          checkpoint_path: checkpointPath,
          config_path: configPath,
          device: device
        }))
      }

      ws.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data)
          if (payload.message_type === 'STATUS') {
            setStatusMessage(`Backend: ${payload.message}`)
            if (payload.status === 'initialized') {
              setWebcamActive(true)
              startFrameSending()
            } else if (payload.status === 'initialization_failed' || payload.status === 'failed') {
              setErrorMessage(payload.error || payload.message)
              stopWebcam()
            }
          } else {
            dispatch({ type: 'PROCESS_EVENT', payload })
          }
        } catch (err) {
          console.error('Webcam WebSocket message error:', err)
        }
      }

      ws.onclose = (e) => {
        console.log(`Camera WebSocket closed. Code: ${e.code}, Reason: ${e.reason || 'None'}`)
        setStatusMessage(`Camera WebSocket connection closed (Code: ${e.code}).`)
        if (e.code === 4000) {
          setErrorMessage('Pipeline initialization failed. Verify paths and checkpoint validity.')
        }
        stopWebcam()
      }

      ws.onerror = (err) => {
        console.error('Camera WebSocket error:', err)
        setErrorMessage('Camera WebSocket encountered an error.')
        stopWebcam()
      }

    } catch (err) {
      console.error('Camera access failed:', err)
      setErrorMessage(err.name === 'NotAllowedError' ? 'Webcam permission denied.' : err.message)
      setStatusMessage('')
    }
  }

  async function fetchSessionEvents(sId) {
    try {
      const response = await fetch(`${API_BASE_URL}/api/events/sessions/${sId}`)
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

  // Mode change handler to clean state transitions cleanly without hook triggers
  function handleModeChange(newMode) {
    if (newMode === mode) return
    setMode(newMode)
    stopWebcam()
    if (videoSrc && videoSrc.startsWith('blob:')) {
      URL.revokeObjectURL(videoSrc)
    }
    setVideoSrc(null)
    setFileName('')
    setIsPlaying(false)
    setSessionId(null)
    setSessionStatus('idle')
    setStatusMessage('')
    setErrorMessage('')
    dispatch({ type: 'CLEAR_ALERTS' })
  }

  // Cleanup on unmount or videoSrc change
  useEffect(() => {
    const videoNode = videoRef.current
    return () => {
      if (sendIntervalRef.current) {
        clearInterval(sendIntervalRef.current)
      }
      if (webcamWsRef.current) {
        if (webcamWsRef.current.readyState === WebSocket.OPEN) {
          webcamWsRef.current.send('stop')
        }
        webcamWsRef.current.close()
      }
      if (videoNode && videoNode.srcObject) {
        videoNode.srcObject.getTracks().forEach(track => track.stop())
      }
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

  // Intelligently auto-play the video as soon as the first detection/alert event for the current session is received from the backend
  useEffect(() => {
    if (mode === 'mp4' && sessionId && videoRef.current) {
      const hasEventsForSession = state.sessionEvents.some(
        event => event.session_id === sessionId
      )

      if (hasEventsForSession && autoPlayTriggeredRef.current !== sessionId) {
        autoPlayTriggeredRef.current = sessionId
        console.log(`Backend session ${sessionId} is ready (received events). Starting video playback.`);
        videoRef.current.play().catch(err => {
          console.warn('Auto-playback failed or was blocked by browser:', err)
        })
        setIsPlaying(true)
      }
    }
  }, [state.sessionEvents, sessionId, mode])

  const handleFileChange = (e) => {
    const file = e.target.files?.[0]
    if (file) {
      if (videoSrc && videoSrc.startsWith('blob:')) {
        URL.revokeObjectURL(videoSrc)
      }
      const objectURL = URL.createObjectURL(file)
      setVideoSrc(objectURL)
      setFileName(file.name)
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
    }
  }

  // Parse Hex colors to RGBA dynamically based on theme CSS variables
  const getThemeColorWithAlpha = (variableName, alpha) => {
    const color = getComputedStyle(document.documentElement).getPropertyValue(variableName).trim()
    if (color.startsWith('#')) {
      const r = parseInt(color.slice(1, 3), 16)
      const g = parseInt(color.slice(3, 5), 16)
      const b = parseInt(color.slice(5, 7), 16)
      return `rgba(${r}, ${g}, ${b}, ${alpha})`
    }
    return variableName === '--color-red'
      ? `rgba(240, 83, 101, ${alpha})`
      : `rgba(91, 141, 249, ${alpha})`
  }

  const drawBboxes = (ctx, bboxes, W, H) => {
    bboxes.forEach(bbox => {
      let x, y, boxWidth, boxHeight

      const isNormalized = bbox.coordinate_space === 'normalized' || !bbox.coordinate_space
      if (isNormalized) {
        x = bbox.x_min * W
        y = bbox.y_min * H
        boxWidth = (bbox.x_max - bbox.x_min) * W
        boxHeight = (bbox.y_max - bbox.y_min) * H
      } else {
        const srcWidth = bbox.source_width || W
        const srcHeight = bbox.source_height || H
        const scaleX = W / srcWidth
        const scaleY = H / srcHeight
        x = bbox.x_min * scaleX
        y = bbox.y_min * scaleY
        boxWidth = (bbox.x_max - bbox.x_min) * scaleX
        boxHeight = (bbox.y_max - bbox.y_min) * scaleY
      }

      const label = bbox.label || 'object'
      const conf = bbox.confidence !== undefined ? bbox.confidence : 1.0
      const themeColorVar = '--color-red'
      const primaryColor = getThemeColorWithAlpha(themeColorVar, 1.0)
      const semiTransColor = getThemeColorWithAlpha(themeColorVar, 0.12)
      const borderTransColor = getThemeColorWithAlpha(themeColorVar, 0.4)

      // 1. Draw Bounding Box semi-transparent fill and border
      ctx.fillStyle = semiTransColor
      ctx.fillRect(x, y, boxWidth, boxHeight)

      ctx.strokeStyle = borderTransColor
      ctx.lineWidth = 1.5
      ctx.strokeRect(x, y, boxWidth, boxHeight)

      // 2. Draw Corner Brackets
      ctx.strokeStyle = primaryColor
      ctx.lineWidth = 4
      ctx.lineCap = 'round'
      const L = Math.min(boxWidth * 0.2, 24)

      // Top Left Corner
      ctx.beginPath()
      ctx.moveTo(x + L, y)
      ctx.lineTo(x, y)
      ctx.lineTo(x, y + L)
      ctx.stroke()

      // Top Right Corner
      ctx.beginPath()
      ctx.moveTo(x + boxWidth - L, y)
      ctx.lineTo(x + boxWidth, y)
      ctx.lineTo(x + boxWidth, y + L)
      ctx.stroke()

      // Bottom Left Corner
      ctx.beginPath()
      ctx.moveTo(x + L, y + boxHeight)
      ctx.lineTo(x, y + boxHeight)
      ctx.lineTo(x, y + boxHeight - L)
      ctx.stroke()

      // Bottom Right Corner
      ctx.beginPath()
      ctx.moveTo(x + boxWidth - L, y + boxHeight)
      ctx.lineTo(x + boxWidth, y + boxHeight)
      ctx.lineTo(x + boxWidth, y + boxHeight - L)
      ctx.stroke()

      // 3. Draw Badge
      const badgeLabel = `${label} [${(conf * 100).toFixed(1)}%]`
      ctx.font = '500 11px "Fira Mono", ui-monospace, monospace'
      const textMetrics = ctx.measureText(badgeLabel)
      const badgeWidth = textMetrics.width + 12
      const badgeHeight = 18

      const badgeX = x
      const badgeY = y - badgeHeight - 4
      const finalBadgeY = badgeY < 4 ? y + 4 : badgeY

      ctx.fillStyle = primaryColor
      ctx.beginPath()
      ctx.roundRect(badgeX, finalBadgeY, badgeWidth, badgeHeight, 3)
      ctx.fill()

      ctx.fillStyle = '#ffffff'
      ctx.textBaseline = 'middle'
      ctx.fillText(badgeLabel, badgeX + 6, finalBadgeY + badgeHeight / 2 + 0.5)
    })
  }

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

      let activeEvents = []
      let activeLabel = 'unknown'
      let activeConfidence = 0.0
      let activeSceneTag = 'unknown'
      let activeSceneConfidence = 0.0

      if (mode === 'mp4' && video) {
        // Find events corresponding to current playing timestamp
        const time = video.currentTime
        if (time === lastTimeRef.current) {
          // Playback time has not progressed, skip recalculations and canvas clears
          animationFrameId.current = requestAnimationFrame(renderLoop)
          return
        }
        lastTimeRef.current = time

        const frameIndex = Math.floor(time * videoFps)
        activeEvents = eventsMap.get(frameIndex) || []

        if (activeEvents.length > 0) {
          activeEvents.sort((a, b) => b.confidence - a.confidence)
          activeLabel = activeEvents[0].label
          activeConfidence = activeEvents[0].confidence
          if (activeEvents[0].context) {
            activeSceneTag = activeEvents[0].context.scene_tag || 'unknown'
            activeSceneConfidence = activeEvents[0].context.confidence || 0.0
          }
        }
      } else if (mode === 'webcam') {
        activeLabel = state.currentDetection.label
        activeConfidence = state.currentDetection.confidence
        activeSceneTag = state.currentDetection.scene_tag
        activeSceneConfidence = state.currentDetection.scene_confidence
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

      // Draw bounding boxes (commented out to save thread cycles as backend doesn't output coordinates yet)
      /*
      if (W > 0 && H > 0) {
        if (mode === 'mp4') {
          const bboxes = activeEvents.flatMap(e => e.bboxes || [])
          drawBboxes(ctx, bboxes, W, H)
        } else if (mode === 'webcam' && webcamActive) {
          const bboxes = state.currentDetection.bboxes || []
          drawBboxes(ctx, bboxes, W, H)
        }
      }
      */

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

      animationFrameId.current = requestAnimationFrame(renderLoop)
    }

    renderLoop()

    return () => {
      if (animationFrameId.current) {
        cancelAnimationFrame(animationFrameId.current)
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [videoSrc, mode, webcamActive, state.sessionEvents, state.currentDetection, eventsMap, videoFps])

  return (
    <section className="panel flex flex-col" id="video-player">
      <div className="panel-header flex items-center justify-between border-b border-border">
        <h2 className="font-mono flex items-center gap-2">
          <span className={`w-2 h-2 rounded-full ${webcamActive || sessionStatus === 'running' ? 'bg-green animate-pulse' : 'bg-text-dim'}`} />
          {mode === 'webcam' ? 'Live Camera Feed' : 'MP4 Session Feed'} {fileName && `— ${fileName}`}
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
            <div className="flex flex-col gap-1">
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

      {/* Control bar / status message log */}
      {(statusMessage || errorMessage) && (
        <div className="px-4 py-2 border-b border-border text-xs font-mono flex flex-col gap-1">
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

      <div className="panel-body flex-1 relative flex items-center justify-center bg-black overflow-hidden min-h-[350px]">
        {/* WEBCAM FLOW ACTIVE VIEW */}
        {mode === 'webcam' && (
          <div className="w-full h-full relative flex items-center justify-center">
            <video
              ref={videoRef}
              onLoadedMetadata={handleLoadedMetadata}
              className={`w-full h-full object-contain ${webcamActive ? 'block' : 'hidden'}`}
              muted
              playsInline
            />
            <canvas
              ref={canvasRef}
              className={`absolute top-0 left-0 w-full h-full object-contain pointer-events-none ${webcamActive ? 'block' : 'hidden'}`}
            />

            {!webcamActive && (
              <div className="flex flex-col items-center justify-center p-8 text-center max-w-md mx-auto">
                <div className="w-14 h-14 rounded-2xl bg-surface-alt border border-border flex items-center justify-center mb-4 text-text-dim">
                  <svg className="w-6 h-6 animate-pulse" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 10.5l4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 002.25-2.25v-9a2.25 2.25 0 00-2.25-2.25h-9A2.25 2.25 0 002.25 7.5v9a2.25 2.25 0 002.25 2.25z" />
                  </svg>
                </div>
                <h3 className="text-sm font-semibold text-text mb-1 font-mono">Live Camera Streaming</h3>
                <p className="text-xs text-text-dim mb-4 leading-relaxed">
                  Start the browser camera stream to send live video frames to the backend for behavior detection.
                </p>
                <button
                  onClick={startWebcam}
                  className="px-4 py-2 text-xs font-mono font-medium rounded-lg border border-border bg-surface-alt hover:bg-border cursor-pointer text-text shadow-sm hover:shadow transition-all"
                >
                  Start Webcam
                </button>
              </div>
            )}

            {webcamActive && (
              <button
                onClick={stopWebcam}
                className="absolute bottom-4 right-4 px-3 py-1.5 text-xs font-mono rounded-lg border border-red/45 bg-red/10 hover:bg-red text-white cursor-pointer shadow transition-all"
              >
                Stop Webcam
              </button>
            )}
          </div>
        )}

        {/* MP4 SESSION ACTIVE VIEW */}
        {mode === 'mp4' && (
          <div className="w-full h-full relative flex items-center justify-center">
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

                {videoDimensions.width > 0 && (
                  <div className="absolute bottom-3 left-3 px-2 py-0.5 rounded bg-black/60 backdrop-blur-xs text-[10px] font-mono text-text-dim pointer-events-none select-none">
                    {videoDimensions.width}×{videoDimensions.height} px
                  </div>
                )}

                <div className="absolute bottom-3 right-3 flex gap-2">
                  {(sessionStatus === 'idle' || sessionStatus === 'completed' || sessionStatus === 'failed' || sessionStatus === 'stopped') ? (
                    <button
                      onClick={startOfflineSession}
                      className="px-2.5 py-1 text-xs font-mono rounded border border-border bg-surface-alt hover:bg-border cursor-pointer text-text transition-colors"
                    >
                      Start Session on Server
                    </button>
                  ) : (
                    <button
                      onClick={stopOfflineSession}
                      className="px-2.5 py-1 text-xs font-mono rounded border border-red/45 bg-red/10 hover:bg-red text-white cursor-pointer transition-all animate-pulse"
                    >
                      Stop Running Session
                    </button>
                  )}
                  <button
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
                    className="px-2.5 py-1 text-xs font-mono rounded border border-border bg-surface-alt hover:bg-border cursor-pointer text-text transition-colors flex items-center gap-1"
                  >
                    {isPlaying ? 'Pause' : 'Play'}
                  </button>
                  <button
                    onClick={triggerFileInput}
                    className="px-2.5 py-1 text-xs font-mono rounded border border-border bg-surface-alt hover:bg-border cursor-pointer text-text transition-colors"
                  >
                    Change Local File
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
                    {(sessionStatus === 'idle' || sessionStatus === 'completed' || sessionStatus === 'failed' || sessionStatus === 'stopped') ? (
                      <button
                        onClick={startOfflineSession}
                        className="px-4 py-2 text-xs font-mono font-medium rounded-lg border border-border bg-surface-alt hover:bg-border cursor-pointer text-text shadow-sm hover:shadow transition-all"
                      >
                        Start Session on Server
                      </button>
                    ) : (
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
                      Load Local Video File
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
        )}
      </div>
    </section>
  )
}
