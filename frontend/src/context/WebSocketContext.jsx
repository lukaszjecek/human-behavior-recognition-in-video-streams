/* eslint-disable react-refresh/only-export-components */
import { createContext, useContext, useReducer, useEffect, useRef, useState } from 'react'
import { API_BASE_URL, getWsUrl } from '../config'
import { useSceneContext } from './SceneContext'

const WebSocketContext = createContext(null)

const initialState = {
  alerts: [],
  sessionEvents: [], // accumulated events for active MP4 session
  currentDetection: {
    label: 'unknown',
    confidence: 0.0,
    scene_tag: 'unknown',
    scene_confidence: 0.0,
    bboxes: []
  }
}

const mapSeverity = (severity) => {
  const s = (severity || 'normal').toLowerCase();
  if (s === 'high' || s === 'critical' || s === 'danger') return 'danger';
  if (s === 'medium' || s === 'warn' || s === 'warning') return 'warning';
  return 'normal';
}

const eventReducer = (state, action) => {
  switch (action.type) {
    case 'PROCESS_EVENT': {
      const payload = action.payload
      if (!payload || !payload.event_type) return state

      let nextDetection = { ...state.currentDetection }
      let newAlerts = [...state.alerts]
      let newSessionEvents = [...state.sessionEvents]

      if (payload.event_type === 'ALERT') {
        const alertData = payload.data
        const newAlert = {
          id: payload.event_id,
          timestamp: payload.timestamp,
          time: new Date(payload.timestamp).toLocaleTimeString('en-GB'),
          severity: mapSeverity(alertData?.severity),
          message: alertData?.message || 'Unknown event',
          camera: payload.camera_id || 'CAM',
          acknowledged: false,
          session_id: payload.session_id || null,
        }

        if (!newAlerts.some(a => a.id === newAlert.id)) {
          newAlerts = [newAlert, ...newAlerts]
          newAlerts.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))
        }

        if (alertData?.action_event) {
          const ae = alertData.action_event
          nextDetection = {
            label: ae.label || 'unknown',
            confidence: ae.confidence || 0.0,
            scene_tag: ae.context?.scene_tag || 'unknown',
            scene_confidence: ae.context?.confidence || 0.0,
            bboxes: ae.bboxes || [],
            timestamp: payload.timestamp,
            start_timestamp: ae.start_timestamp,
            end_timestamp: ae.end_timestamp,
            start_frame_index: ae.start_frame_index,
            end_frame_index: ae.end_frame_index,
          }

          if (payload.session_id) {
            const sessEvent = {
              event_id: payload.event_id,
              event_type: 'ALERT',
              session_id: payload.session_id,
              ...ae
            }
            if (!newSessionEvents.some(e => e.event_id === sessEvent.event_id)) {
              newSessionEvents = [...newSessionEvents, sessEvent]
            }
          }
        }
      } else if (payload.event_type === 'DETECTION') {
        const detectionData = payload.data
        nextDetection = {
          label: detectionData?.label || 'unknown',
          confidence: detectionData?.confidence || 0.0,
          scene_tag: detectionData?.context?.scene_tag || 'unknown',
          scene_confidence: detectionData?.context?.confidence || 0.0,
          bboxes: detectionData?.bboxes || [],
          timestamp: payload.timestamp,
          start_timestamp: detectionData?.start_timestamp,
          end_timestamp: detectionData?.end_timestamp,
          start_frame_index: detectionData?.start_frame_index,
          end_frame_index: detectionData?.end_frame_index,
        }

        if (payload.session_id) {
          const sessEvent = {
            event_id: payload.event_id,
            event_type: 'DETECTION',
            session_id: payload.session_id,
            ...detectionData
          }
          if (!newSessionEvents.some(e => e.event_id === sessEvent.event_id)) {
            newSessionEvents = [...newSessionEvents, sessEvent]
          }
        }
      }

      return {
        ...state,
        alerts: newAlerts,
        sessionEvents: newSessionEvents,
        currentDetection: nextDetection,
      }
    }

    case 'SET_ALERTS': {
      return {
        ...state,
        alerts: action.payload,
      }
    }

    case 'ACKNOWLEDGE_ALERT': {
      return {
        ...state,
        alerts: state.alerts.map(a => a.id === action.payload ? { ...a, acknowledged: true } : a),
      }
    }

    case 'CLEAR_ALERTS': {
      return {
        ...state,
        alerts: [],
        sessionEvents: [],
        currentDetection: {
          label: 'unknown',
          confidence: 0.0,
          scene_tag: 'unknown',
          scene_confidence: 0.0,
          bboxes: []
        }
      }
    }

    case 'SET_SESSION_EVENTS': {
      const flattened = (action.payload || []).map(payload => {
        if (payload.event_type === 'ALERT') {
          return {
            event_id: payload.event_id,
            event_type: 'ALERT',
            session_id: payload.session_id,
            ...payload.data?.action_event
          }
        } else if (payload.event_type === 'DETECTION') {
          return {
            event_id: payload.event_id,
            event_type: 'DETECTION',
            session_id: payload.session_id,
            ...payload.data
          }
        }
        return payload
      })
      // Sort chronologically by start_frame_index
      flattened.sort((a, b) => (a.start_frame_index || 0) - (b.start_frame_index || 0))
      return {
        ...state,
        sessionEvents: flattened,
      }
    }

    case 'RESET_DETECTION': {
      return {
        ...state,
        currentDetection: {
          label: 'unknown',
          confidence: 0.0,
          scene_tag: 'unknown',
          scene_confidence: 0.0,
          bboxes: []
        }
      }
    }

    default:
      return state
  }
}

export function WebSocketProvider({ children }) {
  const { setBackendContext } = useSceneContext()
  const [connectionStatus, setConnectionStatus] = useState('disconnected') // 'connected' | 'connecting' | 'disconnected'
  const [state, dispatch] = useReducer(eventReducer, initialState)
  const socketRef = useRef(null)
  const reconnectTimeoutRef = useRef(null)
  const reconnectDelayRef = useRef(1000)

  // Sync SceneContext with latest currentDetection scene tag & confidence
  const currentSceneTag = state.currentDetection.scene_tag
  const currentSceneConfidence = state.currentDetection.scene_confidence
  useEffect(() => {
    if (currentSceneTag) {
      setBackendContext({
        scene_tag: currentSceneTag,
        confidence: currentSceneConfidence,
      })
    }
  }, [currentSceneTag, currentSceneConfidence, setBackendContext])

  // Fetch initial event/alert history from REST API
  const fetchHistory = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/events/?event_type=ALERT&limit=100`)
      if (response.ok) {
        const data = await response.json()
        const mappedAlerts = data.map(payload => ({
          id: payload.event_id,
          timestamp: payload.timestamp,
          time: new Date(payload.timestamp).toLocaleTimeString('en-GB'),
          severity: mapSeverity(payload.data?.severity),
          message: payload.data?.message || 'Unknown event',
          camera: payload.camera_id || 'CAM',
          acknowledged: false,
          session_id: payload.session_id || null,
        }))
        
        mappedAlerts.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))
        dispatch({ type: 'SET_ALERTS', payload: mappedAlerts })
      }
    } catch (err) {
      console.error('Failed to fetch historical alerts:', err)
    }
  }

  const connect = () => {
    if (socketRef.current) {
      socketRef.current.onclose = null
      socketRef.current.onerror = null
      socketRef.current.onmessage = null
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
        reconnectDelayRef.current = 1000
        fetchHistory()
      }

      ws.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data)
          dispatch({ type: 'PROCESS_EVENT', payload })
        } catch (err) {
          console.error('Error parsing WebSocket message data:', err)
        }
      }

      ws.onclose = () => {
        console.log('WebSocket connection closed')
        setConnectionStatus('disconnected')
        socketRef.current = null
        
        const delay = reconnectDelayRef.current
        reconnectDelayRef.current = Math.min(delay * 2, 30000)
        
        console.log(`Reconnecting to WebSocket in ${delay}ms...`)
        reconnectTimeoutRef.current = setTimeout(() => {
          connect()
        }, delay)
      }

      ws.onerror = (err) => {
        console.error('WebSocket encountered an error:', err)
        ws.close()
      }
    } catch (err) {
      console.error('Failed to create WebSocket client:', err)
      setConnectionStatus('disconnected')
      
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
        socketRef.current.onclose = null
        socketRef.current.onerror = null
        socketRef.current.onmessage = null
        socketRef.current.close()
        socketRef.current = null
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const value = {
    connectionStatus,
    state,
    dispatch,
    reconnect: connect,
    alerts: state.alerts,
    setAlerts: (payload) => dispatch({ type: 'SET_ALERTS', payload }),
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

