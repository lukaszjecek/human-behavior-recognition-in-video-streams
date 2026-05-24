/* eslint-disable react-refresh/only-export-components */
import { createContext, useContext, useState, useMemo } from 'react'

const SceneContext = createContext(null)

/**
 * SceneContextProvider maintains the global state for:
 * 1. Backend-provided context (simulated during video playback)
 * 2. Local manual overrides for demonstration/prototyping
 * 3. Unified effective context that active widgets should display
 */
export function SceneContextProvider({ children }) {
  const [backendContext, setBackendContext] = useState({
    scene_tag: 'unknown',
    confidence: 0.0
  })

  const [isOverrideEnabled, setIsOverrideEnabled] = useState(false)
  const [overrideTag, setOverrideTag] = useState('indoor')
  const [overrideConfidence, setOverrideConfidence] = useState(0.85)

  // Resolve active state dynamically
  const effectiveContext = useMemo(() => {
    if (isOverrideEnabled) {
      return {
        scene_tag: overrideTag,
        confidence: overrideConfidence,
        isOverride: true
      }
    }
    return {
      scene_tag: backendContext.scene_tag,
      confidence: backendContext.confidence,
      isOverride: false
    }
  }, [isOverrideEnabled, overrideTag, overrideConfidence, backendContext])

  const value = useMemo(() => ({
    backendContext,
    setBackendContext,
    isOverrideEnabled,
    setIsOverrideEnabled,
    overrideTag,
    setOverrideTag,
    overrideConfidence,
    setOverrideConfidence,
    effectiveContext
  }), [backendContext, isOverrideEnabled, overrideTag, overrideConfidence, effectiveContext])

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
