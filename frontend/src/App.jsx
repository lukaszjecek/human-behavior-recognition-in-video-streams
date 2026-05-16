import Header from './components/Header'
import VideoPlayer from './components/VideoPlayer'
import AlertLog from './components/AlertLog'
import ContextSettings from './components/ContextSettings'
import StatusBar from './components/StatusBar'
import './App.css'

function App() {
  return (
    <div className="shell">
      <Header />

      <main className="dashboard">
        <VideoPlayer />

        <div className="sidebar">
          <AlertLog />
          <ContextSettings />
        </div>
      </main>

      <StatusBar />
    </div>
  )
}

export default App
