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
  const [errors, setErrors] = useState([])

  const handleConsentProceed = (data) => {
    setConsentData(data)
    setScreen('recording')
  }

  const handleRecordingComplete = (data) => {
    setRecordings(data)
    setScreen('processing')
  }

  const handleProcessingComplete = (data, errorList = []) => {
    setResults(data)
    setErrors(errorList)
    setScreen('report')
  }

  // Withdrawing consent drops everything held in memory and returns to the consent screen.
  // Leaving a screen unmounts it, which also stops the microphone and cancels requests in flight.
  const handleWithdraw = () => {
    setConsentData(null)
    setRecordings([])
    setResults([])
    setErrors([])
    setScreen('consent')
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
          onWithdraw={handleWithdraw}
        />
      )}
      {screen === 'processing' && (
        <ProcessingScreen
          recordings={recordings}
          onComplete={handleProcessingComplete}
          onWithdraw={handleWithdraw}
        />
      )}
      {screen === 'report' && (
        <ReportScreen
          recordings={recordings}
          results={results}
          errors={errors}
          onWithdraw={handleWithdraw}
        />
      )}
    </div>
  )
}

export default App
