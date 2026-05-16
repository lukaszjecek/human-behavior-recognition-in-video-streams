import { useState, useEffect } from 'react'

export default function Header() {
  const [time, setTime] = useState(new Date())

  useEffect(() => {
    const id = setInterval(() => setTime(new Date()), 1000)
    return () => clearInterval(id)
  }, [])

  return (
    <header className="flex items-center justify-between px-5 py-2.5 border-b border-border bg-surface/70 backdrop-blur-sm">
      <div className="flex items-center gap-2.5">
        <span className="text-sm font-semibold text-text tracking-tight font-mono">HBR Dashboard</span>
      </div>

      <span className="text-xs font-mono text-text-dim tabular-nums">
        {time.toLocaleTimeString('en-GB')}
      </span>
    </header>
  )
}
