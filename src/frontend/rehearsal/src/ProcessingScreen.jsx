// Import hooks for state and side effects
import { useState, useEffect, useRef } from 'react'
import { analyzeRecording, describeError } from './analyze'

function ProcessingScreen({ recordings, consentData, onComplete }) {
    // Tracks processing status for each answer — 'waiting', 'processing', 'done', 'error'
    const [statuses, setStatuses] = useState(
        recordings.map(() => 'waiting')
    )

    // Error message for each answer that failed (null if none)
    const [errors, setErrors] = useState(
        recordings.map(() => null)
    )

    // True once a pass over the answers has finished with at least one failure
    const [needsAttention, setNeedsAttention] = useState(false)

    // Real backend results per answer. A failed answer stays null — it is never replaced by invented data.
    const resultsRef = useRef(recordings.map(() => null))
    // Latest error message per answer, handed to the report so it can explain failures
    const errorsRef = useRef(recordings.map(() => null))

    // Runs when screen loads — processes each recording one by one
    useEffect(() => {
        processRecordings(recordings.map((_, i) => i))
    }, [])

    // Sends the recordings at the given indices to the backend sequentially
    const processRecordings = async (indices) => {
        setNeedsAttention(false)

        for (const i of indices) {
            // Mark this answer as currently processing
            errorsRef.current[i] = null
            setStatuses(prev => {
                const updated = [...prev]
                updated[i] = 'processing'
                return updated
            })
            setErrors(prev => {
                const updated = [...prev]
                updated[i] = null
                return updated
            })

            try {
                // POST this answer to the /analyze endpoint
                resultsRef.current[i] = await analyzeRecording(recordings[i], i)

                setStatuses(prev => {
                    const updated = [...prev]
                    updated[i] = 'done'
                    return updated
                })
            } catch (err) {
                // Surface the failure — no placeholder scores are ever shown
                resultsRef.current[i] = null
                errorsRef.current[i] = describeError(err)
                setStatuses(prev => {
                    const updated = [...prev]
                    updated[i] = 'error'
                    return updated
                })
                setErrors(prev => {
                    const updated = [...prev]
                    updated[i] = describeError(err)
                    return updated
                })
            }

            // Small delay between requests to avoid overwhelming backend
            await new Promise(resolve => setTimeout(resolve, 1000))
        }

        if (resultsRef.current.every(r => r !== null)) {
            // All answers analysed — move to reflection report screen
            setTimeout(() => onComplete([...resultsRef.current], [...errorsRef.current]), 800)
        } else {
            setNeedsAttention(true)
        }
    }

    // Re-send only the answers that failed
    const retryFailed = () => {
        const failed = resultsRef.current
            .map((r, i) => (r === null ? i : -1))
            .filter(i => i >= 0)
        processRecordings(failed)
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
                            className={`p-3 rounded-xl border transition-colors ${statuses[i] === 'processing'
                                    ? 'border-indigo-500 bg-indigo-950'
                                    : statuses[i] === 'done'
                                        ? 'border-green-700 bg-gray-800'
                                        : statuses[i] === 'error'
                                            ? 'border-red-700 bg-gray-800'
                                            : 'border-gray-700 bg-gray-800'
                                }`}
                        >
                            <div className="flex items-center justify-between">
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
                                                statuses[i] === 'error' ? 'text-red-400' :
                                                    'text-gray-500'
                                        }`}>
                                        {getStatusLabel(statuses[i])}
                                    </span>
                                </div>
                            </div>

                            {/* Why this answer failed */}
                            {statuses[i] === 'error' && errors[i] && (
                                <p className="text-red-400 text-xs mt-2 break-words">
                                    {errors[i]}
                                </p>
                            )}
                        </div>
                    ))}
                </div>

                {/* Shown after a pass in which at least one answer failed */}
                {needsAttention && (
                    <div className="mt-6 space-y-2">
                        <p className="text-red-300 text-sm">
                            {doneCount === 0
                                ? 'None of your answers could be analysed. No results are available.'
                                : 'Some answers could not be analysed.'}
                        </p>
                        <button
                            onClick={retryFailed}
                            className="w-full py-2.5 bg-indigo-600 hover:bg-indigo-500 text-white rounded-xl text-sm font-semibold transition-colors"
                        >
                            Retry failed answers
                        </button>
                        {doneCount > 0 && (
                            <button
                                onClick={() => onComplete([...resultsRef.current], [...errorsRef.current])}
                                className="w-full py-2.5 bg-gray-700 hover:bg-gray-600 text-gray-200 rounded-xl text-sm font-medium transition-colors"
                            >
                                View report for the answers that worked
                            </button>
                        )}
                    </div>
                )}

                {/* Footer note */}
                <p className="text-gray-600 text-xs text-center mt-6">
                    Your data is processed securely and never shared without consent.
                </p>

            </div>
        </div>
    )
}

export default ProcessingScreen
