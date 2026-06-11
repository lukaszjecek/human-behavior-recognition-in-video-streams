import { useEffect } from 'react'
import { useSceneContext } from '../context/SceneContext'

/**
 * Unified Scene Context component.
 * It behaves as:
 * - A desktop sidebar panel when rendered normally
 * - A mobile bottom-sheet settings modal when passed `isModal={true}` and `open/onClose` props.
 */
export default function ContextSettings({ isModal = false, open = false, onClose = null }) {
  const { backendContext } = useSceneContext()

  // Handle ESC key press to close the modal
  useEffect(() => {
    if (!isModal || !open || !onClose) return
    const handleKey = (e) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', handleKey)
    return () => window.removeEventListener('keydown', handleKey)
  }, [isModal, open, onClose])

  // Format machine tags to user-friendly titles
  const getContextLabel = (tag) => {
    switch (tag) {
      case 'indoor': return 'Indoor Scene'
      case 'outdoor': return 'Outdoor Scene'
      case 'vehicle_setting': return 'Vehicle Interior'
      case 'unknown': return 'Unknown Context'
      default: return tag
    }
  }

  // Render the core scene context status
  const renderBody = () => (
    <div className="panel-body flex flex-col p-4 gap-4 overflow-y-auto">
      {/* 1. CURRENT ACTIVE STATUS SUMMARY CARD */}
      <div className="p-4 rounded-xl border transition-all duration-300 flex flex-col gap-2 bg-green/5 border-green/25 shadow-[0_0_12px_rgba(52,211,153,0.06)]">
        <div className="flex items-center justify-between">
          <span className="text-[10px] font-mono tracking-wider text-text-dim uppercase">
            Effective Scene Context
          </span>
          <span className="px-2 py-0.5 rounded-full text-[9px] font-mono font-bold border text-green bg-green/10 border-green/30">
            System Decided
          </span>
        </div>

        <div className="flex items-center gap-3 mt-1.5">
          <div className="flex flex-col min-w-0">
            <h3 className="text-sm font-semibold text-text font-mono truncate leading-none">
              {getContextLabel(backendContext.scene_tag)}
            </h3>
            <span className="text-xs text-text-dim font-mono mt-1">
              Confidence: {(backendContext.confidence * 100).toFixed(1)}%
            </span>
          </div>
        </div>
      </div>

      {/* 2. SYSTEM INFERENCE (SOURCE OF TRUTH) */}
      <div className="p-3.5 rounded-xl border border-border bg-surface-alt/45 flex flex-col gap-3">
        <div className="flex items-center justify-between">
          <span className="text-[10px] font-mono font-semibold tracking-wider text-text-dim uppercase flex items-center gap-1.5">
            System Context Inference
          </span>
        </div>

        <div className="flex items-center justify-between border-b border-border/40 pb-2.5">
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono font-semibold text-text">
              {backendContext.scene_tag}
            </span>
          </div>
          <span className="text-xs font-mono text-text-dim font-bold">
            {(backendContext.confidence * 100).toFixed(1)}%
          </span>
        </div>

        {/* Confidence progress bar */}
        <div className="flex flex-col gap-1.5">
          <div className="h-1.5 w-full bg-border rounded-full overflow-hidden">
            <div
              className="h-full bg-green rounded-full transition-all duration-300"
              style={{ width: `${backendContext.confidence * 100}%` }}
            />
          </div>
        </div>
      </div>
    </div>
  )

  // 1. MODAL RENDERING (Used on mobile bottom-sheet)
  if (isModal) {
    if (!open) return null
    return (
      <div
        className="settings-modal-overlay"
        onClick={onClose}
        role="dialog"
        aria-modal="true"
        aria-label="Scene Context Settings"
      >
        <div
          className="settings-modal"
          onClick={(e) => e.stopPropagation()}
        >
          <div className="settings-modal-header">
            <span className="font-mono text-sm font-semibold">Scene Context</span>
            <button
              id="settings-modal-close"
              onClick={onClose}
              aria-label="Close settings"
              className="w-7 h-7 rounded-md border border-border bg-surface-alt flex items-center justify-center cursor-pointer hover:bg-border transition-colors"
            >
              <svg
                className="w-3.5 h-3.5 text-text-dim"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2.5"
                strokeLinecap="round"
              >
                <line x1="18" y1="6" x2="6" y2="18" />
                <line x1="6" y1="6" x2="18" y2="18" />
              </svg>
            </button>
          </div>
          {renderBody()}
        </div>
      </div>
    )
  }

  // 2. DESKTOP SIDEBAR PANEL RENDERING
  return (
    <section className="panel" id="context-settings">
      <div className="panel-header">
        <h2 className="font-mono flex items-center gap-2">
          Scene Context
        </h2>
      </div>
      {renderBody()}
    </section>
  )
}
