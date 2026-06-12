import { useState, useEffect, useRef } from 'react'
import { useSceneContext } from '../context/SceneContext'
import { useWebSocket } from '../context/WebSocketContext'

export default function WebcamPlayer({ checkpointPath, configPath, device }) {
  const { setBackendContext } = useSceneContext()
  const { state, dispatch } = useWebSocket()

  const [webcamActive, setWebcamActive] = useState(false)
  const [statusMessage, setStatusMessage] = useState('')
  const [errorMessage, setErrorMessage] = useState('')

  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const animationFrameId = useRef(null)
  const lastReportedContext = useRef({ scene_tag: '', confidence: -1 })

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

  const handleLoadedMetadata = () => {
    const video = videoRef.current
    const canvas = canvasRef.current
    if (video && canvas) {
      canvas.width = video.videoWidth || video.width || 640
      canvas.height = video.videoHeight || video.height || 480
    }
  }

  // Cleanup on unmount
  useEffect(() => {
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
      if (videoRef.current && videoRef.current.srcObject) {
        videoRef.current.srcObject.getTracks().forEach(track => track.stop())
      }
    }
  }, [])

  // Canvas Render Loop for Webcam Detections
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !webcamActive) {
      if (animationFrameId.current) cancelAnimationFrame(animationFrameId.current)
      return
    }

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const renderLoop = () => {
      const W = canvas.width
      const H = canvas.height

      const activeLabel = state.currentDetection.label
      const activeConfidence = state.currentDetection.confidence
      const activeSceneTag = state.currentDetection.scene_tag
      const activeSceneConfidence = state.currentDetection.scene_confidence

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

      animationFrameId.current = requestAnimationFrame(renderLoop)
    }

    renderLoop()

    return () => {
      if (animationFrameId.current) {
        cancelAnimationFrame(animationFrameId.current)
      }
    }
  }, [webcamActive, state.currentDetection, setBackendContext])

  return (
    <div className="w-full h-full flex flex-col">
      {/* Status bar */}
      {(statusMessage || errorMessage) && (
        <div className="px-4 py-2 border-b border-border text-xs font-mono flex flex-col gap-1 w-full bg-surface-alt/25">
          {statusMessage && <div className="text-text-dim">Status: {statusMessage}</div>}
          {errorMessage && <div className="text-red font-semibold bg-red/10 border border-red/25 px-2 py-1 rounded">Error: {errorMessage}</div>}
        </div>
      )}

      <div className="flex-1 relative flex items-center justify-center bg-black overflow-hidden min-h-[350px]">
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
            className="absolute bottom-4 right-4 px-3 py-1.5 text-xs font-mono rounded-lg border border-red/45 bg-red/10 hover:bg-red text-white cursor-pointer shadow transition-all animate-none hover:shadow-md"
          >
            Stop Webcam
          </button>
        )}
      </div>
    </div>
  )
}
