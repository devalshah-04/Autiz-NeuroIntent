import { useState } from 'react'
import LoginScreen from './LoginScreen'
import PortalScreen from './PortalScreen'

function App() {
  // Tracks whether evaluator is logged in
  const [loggedIn, setLoggedIn] = useState(false)

  return (
    <div className="min-h-screen bg-gray-950 text-white">
      <div
        role="status"
        className="fixed bottom-0 left-0 right-0 z-50 bg-yellow-900 border-t border-yellow-600 text-yellow-100 text-xs font-semibold text-center py-2"
      >
        Prototype: not connected to live data
      </div>
      {/* Show login first, then portal after authentication */}
      {!loggedIn ? (
        <LoginScreen onLogin={() => setLoggedIn(true)} />
      ) : (
        <PortalScreen />
      )}
    </div>
  )
}

export default App