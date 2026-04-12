import { useState } from 'react'
import ConsentScreen from './ConsentScreen'
import RecordingScreen from './RecordingScreen'

function App() {
  // Tracks which screen the user is on
  const [screen, setScreen] = useState('consent')

  // Stores consent data from screen 1
  const [consentData, setConsentData] = useState(null)

  // Stores all recordings from screen 2
  const [recordings, setRecordings] = useState([])

  // Called when user proceeds from consent screen
  const handleConsentProceed = (data) => {
    setConsentData(data)
    setScreen('recording')
  }

  // Called when all 5 questions are recorded
  const handleRecordingComplete = (data) => {
    setRecordings(data)
    setScreen('processing')
  }

  return (
    <div className="min-h-screen bg-gray-950 text-white">
      {screen === 'consent' && (
        <ConsentScreen onProceed={handleConsentProceed} />
      )}
      {screen === 'recording' && (
        <RecordingScreen
          consentData={consentData}
          onComplete={handleRecordingComplete}
        />
      )}
      {/* Placeholder for next screens */}
      {screen === 'processing' && (
        <div className="flex items-center justify-center min-h-screen">
          <p className="text-gray-400">Processing screen coming next...</p>
        </div>
      )}
    </div>
  )
}

export default App