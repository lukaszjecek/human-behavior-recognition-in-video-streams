/* eslint-disable react-refresh/only-export-components */
import { createContext, useContext, useState, useMemo } from 'react'

const SceneContext = createContext(null)

/**
 * SceneContextProvider maintains the global state for the
 * backend-provided scene context telemetry.
 */
export function SceneContextProvider({ children }) {
  const [backendContext, setBackendContext] = useState({
    scene_tag: 'unknown',
    confidence: 0.0
  })

  const value = useMemo(() => ({
    backendContext,
    setBackendContext
  }), [backendContext])

  return (
    <SceneContext.Provider value={value}>
      {children}
    </SceneContext.Provider>
  )
}

/**
 * Hook to consume the Scene Context
 */
export function useSceneContext() {
  const context = useContext(SceneContext)
  if (!context) {
    throw new Error('useSceneContext must be used within a SceneContextProvider')
  }
  return context
}
