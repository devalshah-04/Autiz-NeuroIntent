// Import the overlay card component
import OverlayCard from './OverlayCard'

function App() {
  return (
    // Dark background simulates a live interview screen behind the overlay
    <div className="min-h-screen bg-gray-800">
      <OverlayCard />
      <div
        role="status"
        className="fixed bottom-0 left-0 right-0 z-50 bg-yellow-900 border-t border-yellow-600 text-yellow-100 text-xs font-semibold text-center py-2"
      >
        Prototype: not connected to live data
      </div>
    </div>
  )
}

export default App