import { useState } from 'react'
import LoginScreen from './LoginScreen'
import PortalScreen from './PortalScreen'

function App() {
  // Tracks whether evaluator is logged in
  const [loggedIn, setLoggedIn] = useState(false)

  return (
    <div className="min-h-screen bg-gray-950 text-white">
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