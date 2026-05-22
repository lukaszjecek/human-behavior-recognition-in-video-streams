import { useState, useEffect, useRef } from 'react'

const ALL_MOCK_ALERTS = [
  {
    id: 1,
    time: '20:18:45',
    severity: 'danger',
    message: 'Aggression Detected (fight simulation)',
    camera: 'CAM',
    acknowledged: false,
    delay: 5000 // appears at 5 seconds
  },
  {
    id: 2,
    time: '20:17:12',
    severity: 'warning',
    message: 'Unusual running speed detected',
    camera: 'CAM',
    acknowledged: false,
    delay: 4000 // appears at 4 seconds
  },
  {
    id: 3,
    time: '20:15:30',
    severity: 'normal',
    message: 'Person exiting vehicle',
    camera: 'CAM',
    acknowledged: false,
    delay: 3000 // appears at 3 seconds
  },
  {
    id: 4,
    time: '20:12:05',
    severity: 'normal',
    message: 'Vehicle doors opened',
    camera: 'CAM',
    acknowledged: false,
    delay: 2000 // appears at 2 seconds
  },
  {
    id: 5,
    time: '20:10:50',
    severity: 'warning',
    message: 'Person loitering near restricted boundary',
    camera: 'CAM',
    acknowledged: false,
    delay: 1000 // appears at 1 second
  },
  {
    id: 6,
    time: '20:08:15',
    severity: 'normal',
    message: 'Camera connection established',
    camera: 'CAM',
    acknowledged: true,
    delay: 0 // instantly visible
  }
]

export default function AlertLog() {
  const [alerts, setAlerts] = useState(() => ALL_MOCK_ALERTS.filter(a => a.delay === 0))
  const [filter, setFilter] = useState('all') // 'all', 'danger', 'warning', 'normal'
  const [searchQuery, setSearchQuery] = useState('')

  const timersRef = useRef([])

  const clearTimers = () => {
    timersRef.current.forEach(clearTimeout)
    timersRef.current = []
  }

  const triggerSimulation = () => {
    clearTimers()
    setAlerts(ALL_MOCK_ALERTS.filter(a => a.delay === 0))

    ALL_MOCK_ALERTS.forEach(alert => {
      if (alert.delay > 0) {
        const tId = setTimeout(() => {
          setAlerts(prev => {
            // Prevent duplicate entries in strict mode / duplicate runs
            if (prev.some(a => a.id === alert.id)) return prev
            return [alert, ...prev]
          })
        }, alert.delay)
        timersRef.current.push(tId)
      }
    })
  }

  // Set up the asynchronous simulation timers on component mount.
  // We schedule only the delayed timers (delay > 0) here because the initial state
  // is already synchronously set to the delay === 0 items via the useState lazy initializer.
  // This avoids calling setState synchronously in the effect, fully satisfying ESLint rules.
  useEffect(() => {
    ALL_MOCK_ALERTS.forEach(alert => {
      if (alert.delay > 0) {
        const tId = setTimeout(() => {
          setAlerts(prev => {
            if (prev.some(a => a.id === alert.id)) return prev
            return [alert, ...prev]
          })
        }, alert.delay)
        timersRef.current.push(tId)
      }
    })
    return clearTimers
  }, [])

  const handleAcknowledge = (id) => {
    setAlerts(prev =>
      prev.map(alert => alert.id === id ? { ...alert, acknowledged: true } : alert)
    )
  }

  const handleClearAll = () => {
    clearTimers()
    setAlerts([])
  }

  const handleReset = () => {
    triggerSimulation()
  }

  // Filter & Search alerts
  const filteredAlerts = alerts.filter(alert => {
    const matchesFilter = filter === 'all' || alert.severity === filter
    const matchesSearch = alert.message.toLowerCase().includes(searchQuery.toLowerCase()) ||
                          alert.camera.toLowerCase().includes(searchQuery.toLowerCase())
    return matchesFilter && matchesSearch
  })

  // Group counts for severity badge bubbles
  const getCountBySeverity = (severity) => {
    return alerts.filter(a => a.severity === severity && !a.acknowledged).length
  }

  const activeDangerCount = getCountBySeverity('danger')
  const activeAlertsCount = alerts.filter(a => !a.acknowledged).length

  return (
    <section className="panel flex flex-col" id="alert-log">
      <div className="panel-header flex items-center justify-between border-b border-border px-4 py-2.5">
        <h2 className="font-mono flex items-center gap-2 text-sm font-semibold">
          <span className={`w-2 h-2 rounded-full ${activeDangerCount > 0 ? 'bg-red animate-pulse' : 'bg-text-dim'}`} />
          Event Log
          {activeAlertsCount > 0 && (
            <span className="px-1.5 py-0.5 rounded-full text-[10px] font-mono bg-surface-alt border border-border text-text">
              {activeAlertsCount} active
            </span>
          )}
        </h2>

        <div className="flex gap-2">
          {alerts.length > 0 ? (
            <button
              onClick={handleClearAll}
              className="px-2 py-1 text-[10px] font-mono rounded border border-border bg-surface-alt hover:bg-red hover:text-white cursor-pointer transition-colors"
            >
              Clear Log
            </button>
          ) : (
            <button
              onClick={handleReset}
              className="px-2 py-1 text-[10px] font-mono rounded border border-border bg-surface-alt hover:bg-blue hover:text-white cursor-pointer transition-colors"
            >
              Start Simulation
            </button>
          )}
        </div>
      </div>

      {/* Control Area: Filtering Tabs & Search Input */}
      <div className="p-3 border-b border-border bg-surface-alt/30 flex flex-col gap-2.5">
        <div className="flex items-center bg-surface-alt/75 border border-border rounded-lg px-2 py-1">
          <svg className="w-3.5 h-3.5 text-text-dim mr-2 shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
            <circle cx="11" cy="11" r="8" />
            <line x1="21" y1="21" x2="16.65" y2="16.65" />
          </svg>
          <input
            type="text"
            placeholder="Search alerts or cameras..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Escape') setSearchQuery('')
            }}
            className="w-full bg-transparent border-none text-xs text-text focus:outline-none placeholder-text-faint font-mono"
          />
          {searchQuery && (
            <button onClick={() => setSearchQuery('')} className="text-[10px] text-text-dim hover:text-text cursor-pointer px-1 font-mono">
              CLEAR
            </button>
          )}
        </div>

        <div className="flex gap-1">
          {['all', 'danger', 'warning', 'normal'].map((tab) => {
            const count = tab === 'all' ? activeAlertsCount : getCountBySeverity(tab)
            const isActive = filter === tab
            return (
              <button
                key={tab}
                onClick={() => setFilter(tab)}
                className={`flex-1 py-1 text-[10px] font-mono rounded border transition-all cursor-pointer text-center capitalize ${
                  isActive
                    ? 'border-border bg-surface-alt text-text font-semibold shadow-sm'
                    : 'border-transparent text-text-dim hover:text-text hover:bg-surface-alt/30'
                }`}
              >
                {tab === 'normal' ? 'info' : tab}
                {count > 0 && (
                  <span className={`ml-1 px-1 rounded-full text-[9px] ${
                    tab === 'danger' ? 'bg-red/20 text-red border border-red/35' :
                    tab === 'warning' ? 'bg-amber/20 text-amber border border-amber/35' :
                    tab === 'normal' ? 'bg-blue/20 text-blue border border-blue/35' :
                    'bg-surface border border-border text-text-dim'
                  }`}>
                    {count}
                  </span>
                )}
              </button>
            )
          })}
        </div>
      </div>

      {/* Main Alert List */}
      <div className="panel-body flex-1 overflow-y-auto min-h-0 bg-surface">
        {filteredAlerts.length > 0 ? (
          <div className="flex flex-col divide-y divide-border/60">
            {filteredAlerts.map((alert) => {
              const isDanger = alert.severity === 'danger'
              const isWarning = alert.severity === 'warning'

              return (
                <div
                  key={alert.id}
                  className={`flex flex-col gap-1.5 p-3.5 transition-all group ${
                    alert.acknowledged
                      ? 'opacity-40 bg-surface'
                      : isDanger
                      ? 'bg-red/3 hover:bg-red/6 border-l-3 border-red'
                      : isWarning
                      ? 'bg-amber/3 hover:bg-amber/6 border-l-3 border-amber'
                      : 'bg-blue/3 hover:bg-blue/6 border-l-3 border-blue'
                  }`}
                >
                  <div className="flex items-center justify-between gap-2">
                    <div className="flex items-center gap-1.5 min-w-0">
                      {/* Pulsing indicator for active warnings */}
                      <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${
                        alert.acknowledged ? 'bg-text-faint' :
                        isDanger ? 'bg-red animate-pulse shadow-[0_0_8px_rgba(240,83,101,0.6)]' :
                        isWarning ? 'bg-amber' : 'bg-blue'
                      }`} />

                      <span className={`text-[10px] font-mono font-bold uppercase tracking-wider px-1 py-0.2 rounded border ${
                        isDanger ? 'text-red bg-red/10 border-red/20' :
                        isWarning ? 'text-amber bg-amber/10 border-amber/20' :
                        'text-blue bg-blue/10 border-blue/20'
                      }`}>
                        {alert.severity === 'normal' ? 'info' : alert.severity}
                      </span>

                      <span className="text-[10px] font-mono text-text-dim">
                        {alert.time}
                      </span>
                    </div>

                    <div className="flex items-center gap-2">
                      <span className="text-[9px] font-mono px-1 py-0.2 bg-surface-alt border border-border rounded text-text-dim uppercase">
                        {alert.camera}
                      </span>

                      {!alert.acknowledged && (
                        <button
                          onClick={() => handleAcknowledge(alert.id)}
                          className="opacity-0 group-hover:opacity-100 focus:opacity-100 px-1.5 py-0.5 text-[9px] font-mono rounded border border-border bg-surface hover:bg-border cursor-pointer transition-opacity"
                          title="Acknowledge Alert"
                        >
                          ACK
                        </button>
                      )}
                    </div>
                  </div>

                  <p className={`text-xs font-mono break-words leading-relaxed ${
                    alert.acknowledged ? 'text-text-faint line-through' : 'text-text'
                  }`}>
                    {alert.message}
                  </p>
                </div>
              )
            })}
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center p-8 text-center h-full min-h-[200px]">
            <div className="w-10 h-10 rounded-full bg-surface-alt border border-border flex items-center justify-center mb-3 text-text-faint">
              <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
            </div>
            <h4 className="text-xs font-semibold text-text mb-0.5 font-mono">No Matching Events</h4>
            <p className="text-[11px] text-text-dim max-w-[200px] leading-normal font-mono">
              Adjust filters or clear query to locate alerts.
            </p>
          </div>
        )}
      </div>
    </section>
  )
}
