import { useEffect } from 'react'
import { ContextSettingsContent } from './ContextSettings'

/**
 * Bottom-sheet modal z ustawieniami – widoczny tylko na mobile.
 *
 * @param {boolean}  open    - czy modal jest widoczny
 * @param {function} onClose - callback zamknięcia
 */
export default function SettingsModal({ open, onClose }) {
  /* Zamknij modal klawiszem Escape */
  useEffect(() => {
    if (!open) return
    const handleKey = (e) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', handleKey)
    return () => window.removeEventListener('keydown', handleKey)
  }, [open, onClose])

  if (!open) return null

  return (
    <div
      className="settings-modal-overlay"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-label="Settings"
    >
      <div
        className="settings-modal"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="settings-modal-header">
          <span className="font-mono text-sm font-semibold">Settings</span>
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

        <ContextSettingsContent />
      </div>
    </div>
  )
}
