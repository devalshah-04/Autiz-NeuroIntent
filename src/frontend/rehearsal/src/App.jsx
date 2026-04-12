import { useState } from 'react'
import ConsentScreen from './ConsentScreen'
import RecordingScreen from './RecordingScreen'
import ProcessingScreen from './ProcessingScreen'

function App() {
  const [screen, setScreen] = useState('consent')
  const [consentData, setConsentData] = useState(null)
  const [recordings, setRecordings] = useState([])
  const [results, setResults] = useState([])

  const handleConsentProceed = (data) => {
    setConsentData(data)
    setScreen('recording')
  }

  const handleRecordingComplete = (data) => {
    setRecordings(data)
    setScreen('processing')
  }

  // Called when all answers are processed — moves to reflection report
  const handleProcessingComplete = (data) => {
    setResults(data)
    setScreen('report')
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
      {screen === 'processing' && (
        <ProcessingScreen
          recordings={recordings}
          consentData={consentData}
          onComplete={handleProcessingComplete}
        />
      )}
      {/* Placeholder for reflection report */}
      {screen === 'report' && (
        <div className="flex items-center justify-center min-h-screen">
          <p className="text-gray-400">Reflection report coming next...</p>
        </div>
      )}
    </div>
  )
}

export default App