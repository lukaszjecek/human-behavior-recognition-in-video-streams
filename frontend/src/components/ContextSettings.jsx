/** Treść ustawień – reużywalna w panelu i w modalu */
export function ContextSettingsContent() {
  return (
    <div className="panel-body flex items-center justify-center">
      <p className="text-sm text-text-faint font-mono">Settings placeholder</p>
    </div>
  )
}

/** Panel ustawień widoczny w sidebarze (tylko desktop) */
export default function ContextSettings() {
  return (
    <section className="panel" id="context-settings">
      <div className="panel-header">
        <h2 className="font-mono">Settings</h2>
      </div>
      <ContextSettingsContent />
    </section>
  )
}
