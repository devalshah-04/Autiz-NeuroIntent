// Import React hooks — no external WebSocket library needed
import { useState, useEffect, useRef } from 'react'

// Read backend URL from environment variable
const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'ws://localhost:8000'

function OverlayCard() {
    // Controls whether recruiter has opted in — default is OFF per design rules
    const [isOn, setIsOn] = useState(false)

    // Stores the latest data received from WebSocket
    const [data, setData] = useState({
        content_quality_score: null,
        prosody_summary: null,
        misread_flag: false,
        misread_description: null,
    })

    // Tracks connection status for display
    const [status, setStatus] = useState('Off')

    // Holds the WebSocket instance across renders
    const wsRef = useRef(null)

    // Open or close WebSocket based on toggle state
    useEffect(() => {
        // If turned OFF, close any existing connection
        if (!isOn) {
            if (wsRef.current) {
                wsRef.current.close()
                wsRef.current = null
            }
            setStatus('Off')
            return
        }

        // Open new WebSocket connection when turned ON
        setStatus('Connecting...')
        const ws = new WebSocket(`${BACKEND_URL}/stream`)
        wsRef.current = ws

        ws.onopen = () => setStatus('Live')

        ws.onmessage = (event) => {
            try {
                const msg = JSON.parse(event.data)
                // Destructure only allowed fields — intent_label is intentionally ignored
                const { content_quality_score, prosody_summary, misread_flag, misread_description } = msg
                setData({ content_quality_score, prosody_summary, misread_flag, misread_description })
            } catch (e) {
                console.error('Invalid message from backend:', e)
            }
        }

        ws.onerror = () => setStatus('Error')
        ws.onclose = () => setStatus('Disconnected')

        // Cleanup: close WebSocket when component unmounts
        return () => {
            ws.close()
        }
    }, [isOn])

    return (
        // Outer card — fixed position, top-right corner, dark background
        <div className="fixed top-4 right-4 w-80 bg-gray-900 border border-gray-700 rounded-2xl shadow-2xl p-4 z-50">

            {/* Header row — title, research badge, and ON/OFF toggle */}
            <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                    <span className="text-white font-semibold text-sm">NeuroIntent</span>
                    {/* research_pilot badge — always visible per design rules */}
                    <span className="bg-indigo-700 text-indigo-100 text-xs px-2 py-0.5 rounded-full">
                        research_pilot
                    </span>
                </div>

                {/* ON/OFF toggle — default OFF, recruiter opts in */}
                <button
                    onClick={() => setIsOn(prev => !prev)}
                    className={`relative w-12 h-6 rounded-full transition-colors duration-300 ${isOn ? 'bg-green-500' : 'bg-gray-600'
                        }`}
                >
                    <span
                        className={`absolute top-1 w-4 h-4 bg-white rounded-full shadow transition-transform duration-300 ${isOn ? 'translate-x-7' : 'translate-x-1'
                            }`}
                    />
                </button>
            </div>

            {/* Show this message when overlay is OFF */}
            {!isOn && (
                <div className="text-gray-500 text-xs text-center py-4">
                    Toggle ON to begin live monitoring
                </div>
            )}

            {/* Show live data only when overlay is ON */}
            {isOn && (
                <div className="space-y-3">

                    {/* Connection status indicator */}
                    <div className="flex items-center gap-1.5">
                        <span className={`w-2 h-2 rounded-full ${status === 'Live' ? 'bg-green-400' : 'bg-yellow-400'}`} />
                        <span className="text-gray-400 text-xs">{status}</span>
                    </div>

                    {/* Content quality score — number + progress bar */}
                    <div>
                        <div className="flex justify-between items-center mb-1">
                            <span className="text-gray-300 text-xs font-medium">Content Quality Score</span>
                            <span className="text-white text-sm font-bold">
                                {data.content_quality_score !== null
                                    ? data.content_quality_score.toFixed(2)
                                    : '—'}
                            </span>
                        </div>
                        {/* Progress bar fills based on score value */}
                        <div className="w-full bg-gray-700 rounded-full h-2">
                            <div
                                className="bg-indigo-500 h-2 rounded-full transition-all duration-500"
                                style={{ width: `${(data.content_quality_score ?? 0) * 100}%` }}
                            />
                        </div>
                    </div>

                    {/* Prosody summary — 2-word string from backend */}
                    <div>
                        <span className="text-gray-300 text-xs font-medium">Prosody Pattern</span>
                        <p className="text-white text-sm mt-0.5">
                            {data.prosody_summary ?? '—'}
                        </p>
                    </div>

                    {/* Misread flag — only shown when backend sends misread_flag: true */}
                    {data.misread_flag && (
                        <div className="bg-amber-900 border border-amber-600 rounded-lg px-3 py-2">
                            <p className="text-amber-300 text-xs font-semibold">⚠ Possible Misread Detected</p>
                            {data.misread_description && (
                                <p className="text-amber-200 text-xs mt-0.5">{data.misread_description}</p>
                            )}
                        </div>
                    )}

                </div>
            )}

            {/* Footer — always visible per design rules, never shows intent labels */}
            <div className="mt-3 pt-3 border-t border-gray-700">
                <p className="text-gray-500 text-xs text-center leading-tight">
                    Showing content quality measurement — full analysis after session
                </p>
            </div>

        </div>
    )
}

export default OverlayCard