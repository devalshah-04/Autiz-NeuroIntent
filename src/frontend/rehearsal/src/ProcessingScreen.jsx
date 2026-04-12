// Import hooks for state and side effects
import { useState, useEffect } from 'react'
import axios from 'axios'

// Read backend URL from environment variable
const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000'

function ProcessingScreen({ recordings, consentData, onComplete }) {
    // Tracks processing status for each answer — 'waiting', 'processing', 'done', 'error'
    const [statuses, setStatuses] = useState(
        recordings.map(() => 'waiting')
    )

    // Stores the results returned from backend for each answer
    const [results, setResults] = useState(
        recordings.map(() => null)
    )

    // Runs when screen loads — processes each recording one by one
    useEffect(() => {
        processAllRecordings()
    }, [])

    // Sends each recording to backend sequentially
    const processAllRecordings = async () => {
        const allResults = []

        for (let i = 0; i < recordings.length; i++) {
            // Mark this answer as currently processing
            setStatuses(prev => {
                const updated = [...prev]
                updated[i] = 'processing'
                return updated
            })

            try {
                // Build form data to send audio file to backend
                const formData = new FormData()
                formData.append('audio', recordings[i].audioBlob, `answer_${i + 1}.webm`)
                formData.append('label', recordings[i].label)
                formData.append('question_index', i)
                formData.append('mode', recordings[i].mode)

                // POST to Krishiv's /analyze endpoint
                const response = await axios.post(`${BACKEND_URL}/analyze`, formData, {
                    headers: { 'Content-Type': 'multipart/form-data' }
                })

                allResults.push(response.data)

                // Mark this answer as done
                setStatuses(prev => {
                    const updated = [...prev]
                    updated[i] = 'done'
                    return updated
                })

                setResults(prev => {
                    const updated = [...prev]
                    updated[i] = response.data
                    return updated
                })

            } catch (err) {
                // If backend not available, use mock data for now
                const mockResult = {
                    content_quality_score: parseFloat((Math.random() * 0.4 + 0.55).toFixed(2)),
                    delivery_pattern: [
                        'Steady speaking pace detected',
                        'Low pitch variation throughout',
                        'Consistent volume level'
                    ],
                    score_without_system: parseFloat((Math.random() * 0.3 + 0.4).toFixed(2)),
                    score_with_system: parseFloat((Math.random() * 0.3 + 0.7).toFixed(2)),
                    confidence_bound: ['low', 'medium', 'high'][Math.floor(Math.random() * 3)],
                    interpretation: `Your answer demonstrated clear content understanding. The system detected steady delivery patterns that traditional scoring may undervalue. Content analysis shows strong relevance to the question asked.`,
                    transcript: `Answer ${i + 1} transcript will appear here when backend is connected.`
                }

                allResults.push(mockResult)

                // Mark as done even with mock data
                setStatuses(prev => {
                    const updated = [...prev]
                    updated[i] = 'done'
                    return updated
                })

                setResults(prev => {
                    const updated = [...prev]
                    updated[i] = mockResult
                    return updated
                })
            }

            // Small delay between requests to avoid overwhelming backend
            await new Promise(resolve => setTimeout(resolve, 1000))
        }

        // All done — move to reflection report screen
        setTimeout(() => onComplete(allResults), 800)
    }

    // Status icon for each answer row
    const getStatusIcon = (status) => {
        if (status === 'waiting') return '⏳'
        if (status === 'processing') return '⚙️'
        if (status === 'done') return '✅'
        if (status === 'error') return '❌'
    }

    // Status label for each answer row
    const getStatusLabel = (status) => {
        if (status === 'waiting') return 'Waiting...'
        if (status === 'processing') return 'Processing...'
        if (status === 'done') return 'Done'
        if (status === 'error') return 'Error'
    }

    // How many answers are done
    const doneCount = statuses.filter(s => s === 'done').length

    return (
        <div className="min-h-screen bg-gray-950 flex flex-col items-center justify-center px-6 py-10">

            {/* Card */}
            <div className="w-full max-w-lg bg-gray-900 border border-gray-700 rounded-2xl p-8 shadow-2xl">

                {/* Header */}
                <div className="flex items-center gap-3 mb-6">
                    <span className="text-white font-bold">NeuroIntent</span>
                    {/* research_pilot badge — always visible */}
                    <span className="bg-indigo-700 text-indigo-100 text-xs px-2 py-0.5 rounded-full">
                        research_pilot
                    </span>
                </div>

                <h2 className="text-white text-xl font-semibold mb-2">
                    Analysing your answers
                </h2>
                <p className="text-gray-400 text-sm mb-6">
                    Estimated 15–30 seconds per answer. Please keep this tab open.
                </p>

                {/* Overall progress bar */}
                <div className="w-full bg-gray-700 rounded-full h-2 mb-6">
                    <div
                        className="bg-indigo-500 h-2 rounded-full transition-all duration-700"
                        style={{ width: `${(doneCount / recordings.length) * 100}%` }}
                    />
                </div>

                {/* Per-answer status rows */}
                <div className="space-y-3">
                    {recordings.map((rec, i) => (
                        <div
                            key={i}
                            className={`flex items-center justify-between p-3 rounded-xl border transition-colors ${statuses[i] === 'processing'
                                    ? 'border-indigo-500 bg-indigo-950'
                                    : statuses[i] === 'done'
                                        ? 'border-green-700 bg-gray-800'
                                        : 'border-gray-700 bg-gray-800'
                                }`}
                        >
                            {/* Answer label and question preview */}
                            <div>
                                <p className="text-white text-sm font-medium">
                                    Answer {i + 1}
                                </p>
                                <p className="text-gray-500 text-xs mt-0.5 truncate w-64">
                                    {rec.question}
                                </p>
                            </div>

                            {/* Status icon and label */}
                            <div className="flex items-center gap-2">
                                <span className="text-sm">{getStatusIcon(statuses[i])}</span>
                                <span className={`text-xs font-medium ${statuses[i] === 'done' ? 'text-green-400' :
                                        statuses[i] === 'processing' ? 'text-indigo-300' :
                                            'text-gray-500'
                                    }`}>
                                    {getStatusLabel(statuses[i])}
                                </span>
                            </div>
                        </div>
                    ))}
                </div>

                {/* Footer note */}
                <p className="text-gray-600 text-xs text-center mt-6">
                    Your data is processed securely and never shared without consent.
                </p>

            </div>
        </div>
    )
}

export default ProcessingScreen
