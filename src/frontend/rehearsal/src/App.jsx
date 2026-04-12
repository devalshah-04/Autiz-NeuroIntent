import { useState } from 'react'
import ConsentScreen from './ConsentScreen'
import RecordingScreen from './RecordingScreen'
import ProcessingScreen from './ProcessingScreen'
import ReportScreen from './ReportScreen'

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
      {screen === 'report' && (
        <ReportScreen
          recordings={recordings}
          results={results}
        />
      )}
    </div>
  )
}

export default App