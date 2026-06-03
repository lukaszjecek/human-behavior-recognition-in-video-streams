/* eslint-disable react-refresh/only-export-components */
import { createContext, useContext, useState, useEffect, useRef } from 'react'
import { API_BASE_URL, getWsUrl } from '../config'
import { useSceneContext } from './SceneContext'

const WebSocketContext = createContext(null)

export function WebSocketProvider({ children }) {
  const { setBackendContext } = useSceneContext()
  const [connectionStatus, setConnectionStatus] = useState('disconnected') // 'connected' | 'connecting' | 'disconnected'
  const [alerts, setAlerts] = useState([])
  const socketRef = useRef(null)
  const reconnectTimeoutRef = useRef(null)
  const reconnectDelayRef = useRef(1000) // Start reconnect delay at 1 second

  // Fetch initial event/alert history from REST API
  const fetchHistory = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/events/?event_type=ALERT&limit=100`)
      if (response.ok) {
        const data = await response.json()
        // Map backend EventPayload format to UI alert object format
        const mappedAlerts = data.map(payload => ({
          id: payload.event_id,
          timestamp: payload.timestamp, // Store original ISO timestamp string
          time: new Date(payload.timestamp).toLocaleTimeString('en-GB'),
          severity: payload.data?.severity ? payload.data.severity.toLowerCase() : 'normal',
          message: payload.data?.message || 'Unknown event',
          camera: payload.camera_id || 'CAM',
          acknowledged: false,
        }))
        
        // Merge with existing state, deduplicate, and sort by timestamp descending
        setAlerts(prev => {
          const combined = [...prev, ...mappedAlerts]
          const unique = []
          const seen = new Set()
          for (const alert of combined) {
            if (!seen.has(alert.id)) {
              seen.add(alert.id)
              unique.push(alert)
            }
          }
          unique.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))
          return unique
        })
      }
    } catch (err) {
      console.error('Failed to fetch historical alerts:', err)
    }
  }

  const connect = () => {
    // Clear any existing connection and reconnect timers
    if (socketRef.current) {
      socketRef.current.close()
    }
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
    }

    setConnectionStatus('connecting')
    const wsUrl = getWsUrl()
    console.log(`Connecting to WebSocket: ${wsUrl}`)
    
    try {
      const ws = new WebSocket(wsUrl)
      socketRef.current = ws

      ws.onopen = () => {
        console.log('WebSocket connection established')
        setConnectionStatus('connected')
        reconnectDelayRef.current = 1000 // Reset reconnection delay on successful connection
        fetchHistory() // Fetch history upon successful connection
      }

      ws.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data)
          
          // 1. Handle ALERT events
          if (payload.event_type === 'ALERT') {
            const newAlert = {
              id: payload.event_id,
              timestamp: payload.timestamp, // Store original ISO timestamp string
              time: new Date(payload.timestamp).toLocaleTimeString('en-GB'),
              severity: payload.data?.severity ? payload.data.severity.toLowerCase() : 'normal',
              message: payload.data?.message || 'Unknown event',
              camera: payload.camera_id || 'CAM',
              acknowledged: false,
            }
            // Add and deduplicate
            setAlerts(prev => {
              if (prev.some(a => a.id === newAlert.id)) return prev
              const combined = [newAlert, ...prev]
              combined.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))
              return combined
            })

            // Update Scene Context if alert payload contains context
            if (payload.data?.action_event?.context) {
              setBackendContext({
                scene_tag: payload.data.action_event.context.scene_tag,
                confidence: payload.data.action_event.context.confidence,
              })
            }
          }
          // 2. Handle DETECTION events
          else if (payload.event_type === 'DETECTION') {
            if (payload.data?.context) {
              setBackendContext({
                scene_tag: payload.data.context.scene_tag,
                confidence: payload.data.context.confidence,
              })
            }
          }
        } catch (err) {
          console.error('Error parsing WebSocket message data:', err)
        }
      }

      ws.onclose = () => {
        console.log('WebSocket connection closed')
        setConnectionStatus('disconnected')
        socketRef.current = null
        
        // Trigger auto-reconnect with exponential backoff
        const delay = reconnectDelayRef.current
        reconnectDelayRef.current = Math.min(delay * 2, 30000) // Cap at 30 seconds
        
        console.log(`Reconnecting to WebSocket in ${delay}ms...`)
        reconnectTimeoutRef.current = setTimeout(() => {
          connect()
        }, delay)
      }

      ws.onerror = (err) => {
        console.error('WebSocket encountered an error:', err)
        ws.close() // Close triggers the onclose logic
      }
    } catch (err) {
      console.error('Failed to create WebSocket client:', err)
      setConnectionStatus('disconnected')
      
      // Retry connection
      const delay = reconnectDelayRef.current
      reconnectDelayRef.current = Math.min(delay * 2, 30000)
      reconnectTimeoutRef.current = setTimeout(() => {
        connect()
      }, delay)
    }
  }

  useEffect(() => {
    connect()

    return () => {
      if (socketRef.current) {
        socketRef.current.close()
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const value = {
    connectionStatus,
    alerts,
    setAlerts,
    reconnect: connect,
  }

  return (
    <WebSocketContext.Provider value={value}>
      {children}
    </WebSocketContext.Provider>
  )
}

export function useWebSocket() {
  const context = useContext(WebSocketContext)
  if (!context) {
    throw new Error('useWebSocket must be used within a WebSocketProvider')
  }
  return context
}
