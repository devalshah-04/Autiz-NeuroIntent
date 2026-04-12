// Import the overlay card component
import OverlayCard from './OverlayCard'

function App() {
  return (
    // Dark background simulates a live interview screen behind the overlay
    <div className="min-h-screen bg-gray-800">
      <OverlayCard />
    </div>
  )
}

export default App