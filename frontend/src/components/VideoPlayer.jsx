import { useState, useEffect, useRef } from 'react'
import mockCoordinates from './mockCoordinates.json'
import { useSceneContext } from '../context/SceneContext'

export default function VideoPlayer() {
  const { setBackendContext } = useSceneContext()
  const [videoSrc, setVideoSrc] = useState('/sample.mp4')
  const [fileName, setFileName] = useState('sample.mp4')
  const [isPlaying, setIsPlaying] = useState(true)
  const [videoDimensions, setVideoDimensions] = useState({ width: 0, height: 0 })

  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const fileInputRef = useRef(null)
  const animationFrameId = useRef(null)
  const lastReportedContext = useRef({ scene_tag: '', confidence: -1 })

  // Revoke object URL on source change or unmount to avoid memory leaks
  useEffect(() => {
    return () => {
      if (videoSrc && videoSrc.startsWith('blob:')) {
        URL.revokeObjectURL(videoSrc)
      }
    }
  }, [videoSrc])

  const handleFileChange = (e) => {
    const file = e.target.files?.[0]
    if (file) {
      if (videoSrc && videoSrc.startsWith('blob:')) {
        URL.revokeObjectURL(videoSrc)
      }
      const objectURL = URL.createObjectURL(file)
      setVideoSrc(objectURL)
      setFileName(file.name)
      setIsPlaying(false)
    }
  }

  const triggerFileInput = () => {
    fileInputRef.current?.click()
  }

  const handleLoadedMetadata = () => {
    const video = videoRef.current
    const canvas = canvasRef.current
    if (video && canvas) {
      canvas.width = video.videoWidth
      canvas.height = video.videoHeight
      setVideoDimensions({ width: video.videoWidth, height: video.videoHeight })
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
    // Fallback based on default palette variables
    return variableName === '--color-red'
      ? `rgba(240, 83, 101, ${alpha})`
      : `rgba(91, 141, 249, ${alpha})`
  }

  useEffect(() => {
    const video = videoRef.current
    const canvas = canvasRef.current

    if (!videoSrc || !video || !canvas) {
      if (animationFrameId.current) {
        cancelAnimationFrame(animationFrameId.current)
      }
      return
    }

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const renderLoop = () => {
      if (!video || !canvas) return

      const W = canvas.width
      const H = canvas.height

      ctx.clearRect(0, 0, W, H)

      let activeTag = 'unknown'
      let activeConfidence = 0.0

      if (W > 0 && H > 0 && fileName === 'sample.mp4') {
        const fps = 30
        const frameIndex = Math.floor(video.currentTime * fps)

        // Determine context tag and confidence score based on simulated timeline
        if (frameIndex < 20) {
          activeTag = 'unknown'
          activeConfidence = 0.0
        } else if (frameIndex < 90) {
          activeTag = 'vehicle_setting'
          activeConfidence = 0.92 + Math.sin(frameIndex * 0.15) * 0.015
        } else if (frameIndex < 160) {
          activeTag = 'outdoor'
          activeConfidence = 0.88 + Math.cos(frameIndex * 0.1) * 0.01
        } else if (frameIndex < 220) {
          activeTag = 'outdoor'
          activeConfidence = 0.95 + Math.sin(frameIndex * 0.05) * 0.008
        } else {
          activeTag = 'outdoor'
          activeConfidence = 0.91 + Math.cos(frameIndex * 0.08) * 0.02
        }
      }

      // Throttle React state updates to avoid rendering bottleneck
      const roundedConf = Math.round(activeConfidence * 1000) / 1000
      if (
        lastReportedContext.current.scene_tag !== activeTag ||
        Math.abs(lastReportedContext.current.confidence - roundedConf) > 0.005
      ) {
        lastReportedContext.current = { scene_tag: activeTag, confidence: roundedConf }
        setTimeout(() => {
          setBackendContext({ scene_tag: activeTag, confidence: roundedConf })
        }, 0)
      }

      if (W > 0 && H > 0 && fileName === 'sample.mp4') {
        const fps = 30
        const frameIndex = Math.floor(video.currentTime * fps)

        if (frameIndex >= 20) {
          let x, y, boxWidth, boxHeight
          let actionLabel = 'Standing'
          let confidence = 97.4
          let themeColorVar = '--color-red'


          if (mockCoordinates && mockCoordinates.length > 0) {
            const safeFrameIndex = Math.min(Math.max(0, frameIndex), mockCoordinates.length - 1)
            const frameData = mockCoordinates[safeFrameIndex]

            const rawBox = frameData.box

            // Scale coordinates to the current canvas dimensions dynamically
            const scaleX = W / 512.0
            const scaleY = H / 512.0

            x = rawBox[0] * scaleX
            y = rawBox[1] * scaleY
            boxWidth = rawBox[2] * scaleX
            boxHeight = rawBox[3] * scaleY

            // Determine action label and theme color dynamically based on video timeline
            if (safeFrameIndex > 220) {
              actionLabel = 'Closes car door'
              confidence = 96.8 + Math.sin(safeFrameIndex * 0.05) * 1.2
              // themeColorVar = '--color-red'
            } else if (safeFrameIndex > 160) {
              actionLabel = 'Standing'
              confidence = 96.8 + Math.sin(safeFrameIndex * 0.05) * 1.2
              // themeColorVar = '--color-red'
            } else if (safeFrameIndex > 90) {
              actionLabel = 'Exiting Car'
              confidence = 93.5 + Math.cos(safeFrameIndex * 0.1) * 1.8
              // themeColorVar = '--color-amber'
            } else {
              actionLabel = 'Sitting'
              confidence = 97.8 + Math.sin(safeFrameIndex * 0.15) * 0.8
              // themeColorVar = '--color-blue'
            }
          }
          const primaryColor = getThemeColorWithAlpha(themeColorVar, 1.0)
          const semiTransColor = getThemeColorWithAlpha(themeColorVar, 0.12)
          const borderTransColor = getThemeColorWithAlpha(themeColorVar, 0.4)

          // 1. Draw Bounding Box semi-transparent fill and light border
          ctx.fillStyle = semiTransColor
          ctx.fillRect(x, y, boxWidth, boxHeight)

          ctx.strokeStyle = borderTransColor
          ctx.lineWidth = 1.5
          ctx.strokeRect(x, y, boxWidth, boxHeight)

          // 2. Draw Thick Corner Brackets
          ctx.strokeStyle = primaryColor
          ctx.lineWidth = 4
          ctx.lineCap = 'round'
          const L = Math.min(boxWidth * 0.2, 24) // Length of corner indicators

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

          // 3. Draw Labeled Badge above the bounding box
          const badgeLabel = `${actionLabel} [${confidence.toFixed(1)}%]`
          ctx.font = '500 12px "Fira Mono", ui-monospace, monospace'
          const textMetrics = ctx.measureText(badgeLabel)
          const badgeWidth = textMetrics.width + 16
          const badgeHeight = 22

          const badgeX = x
          const badgeY = y - badgeHeight - 6

          // Ensure badge stays visible inside screen boundaries
          const finalBadgeY = badgeY < 6 ? y + 6 : badgeY

          // Draw Badge Background with subtle rounded corners
          ctx.fillStyle = primaryColor
          ctx.beginPath()
          ctx.roundRect(badgeX, finalBadgeY, badgeWidth, badgeHeight, 4)
          ctx.fill()

          // Draw Badge text
          ctx.fillStyle = '#ffffff'
          ctx.textBaseline = 'middle'
          ctx.fillText(badgeLabel, badgeX + 8, finalBadgeY + badgeHeight / 2 + 0.5)
        }
      }

      animationFrameId.current = requestAnimationFrame(renderLoop)
    }

    renderLoop()

    return () => {
      if (animationFrameId.current) {
        cancelAnimationFrame(animationFrameId.current)
      }
    }
  }, [videoSrc, fileName, setBackendContext])

  return (
    <section className="panel flex flex-col" id="video-player">
      <div className="panel-header flex items-center justify-between">
        <h2 className="font-mono flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-green animate-pulse" />
          Camera Feed {fileName && `— ${fileName}`}
        </h2>

        <div className="flex gap-2">
          {videoSrc && (
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
              {isPlaying ? (
                <>
                  <span className="w-1.5 h-3 border-l-2 border-r-2 border-text inline-block" /> Pause
                </>
              ) : (
                <>
                  <span className="w-0 h-0 border-y-4 border-y-transparent border-l-6 border-l-text inline-block" /> Play
                </>
              )}
            </button>
          )}

          <button
            onClick={triggerFileInput}
            className="px-2.5 py-1 text-xs font-mono rounded border border-border bg-surface-alt hover:bg-border cursor-pointer text-text transition-colors"
          >
            {videoSrc ? 'Change Stream' : 'Load Stream'}
          </button>
        </div>
      </div>

      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileChange}
        accept="video/*"
        className="hidden"
      />

      <div className="panel-body flex-1 relative flex items-center justify-center bg-black overflow-hidden min-h-[300px]">
        {videoSrc ? (
          <div className="relative w-full h-full flex items-center justify-center">
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
              autoPlay
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
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center p-8 text-center max-w-md mx-auto">
            <div className="w-14 h-14 rounded-2xl bg-surface-alt border border-border flex items-center justify-center mb-4 text-text-dim animate-bounce duration-3000">
              <svg className="w-6 h-6" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 10.5l4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 002.25-2.25v-9a2.25 2.25 0 00-2.25-2.25h-9A2.25 2.25 0 002.25 7.5v9a2.25 2.25 0 002.25 2.25z" />
              </svg>
            </div>
            <h3 className="text-sm font-semibold text-text mb-1 font-mono">No Video Stream Active</h3>
            <p className="text-xs text-text-dim mb-4 leading-relaxed">
              Load a local video file from your disk to begin simulated HBR tracking overlays and behaviour recognition analysis.
            </p>
            <button
              onClick={triggerFileInput}
              className="px-4 py-2 text-xs font-mono font-medium rounded-lg border border-border bg-surface-alt hover:bg-border cursor-pointer text-text shadow-sm hover:shadow transition-all"
            >
              Select Video File
            </button>
          </div>
        )}
      </div>
    </section>
  )
}
