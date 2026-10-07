// Import hooks for state and side effects
import { useState, useEffect, useRef, useCallback } from 'react'
import { analyzeRecording } from './api'

function ProcessingScreen({ recordings, onComplete, onWithdraw }) {
    // Tracks processing status for each answer — 'waiting', 'processing', 'done', 'error'
    const [statuses, setStatuses] = useState(
        recordings.map(() => 'waiting')
    )

    // Typed error ({ userMessage, detail }) for each answer that failed (null if none)
    const [errors, setErrors] = useState(
        recordings.map(() => null)
    )

    // True once a pass over the answers has finished with at least one failure
    const [needsAttention, setNeedsAttention] = useState(false)

    // Real backend results per answer. A failed answer stays null — it is never replaced by invented data.
    const resultsRef = useRef(recordings.map(() => null))
    // Latest error per answer, handed to the report so it can explain failures
    const errorsRef = useRef(recordings.map(() => null))

    // Cancels requests in flight when this screen is left (withdrawing consent, or the report opening)
    const abortRef = useRef(null)

    // Always call the latest onComplete without re-running the effect below
    const onCompleteRef = useRef(onComplete)
    useEffect(() => {
        onCompleteRef.current = onComplete
    }, [onComplete])

    // Sends the recordings at the given indices to the backend sequentially
    const processRecordings = useCallback(async (indices, signal) => {
        setNeedsAttention(false)

        for (const i of indices) {
            if (signal.aborted) return

            // Mark this answer as currently processing
            errorsRef.current[i] = null
            setStatuses(prev => prev.map((s, j) => (j === i ? 'processing' : s)))
            setErrors(prev => prev.map((e, j) => (j === i ? null : e)))

            try {
                // POST this answer to the /analyze endpoint
                resultsRef.current[i] = await analyzeRecording(recordings[i], i, { signal })
                setStatuses(prev => prev.map((s, j) => (j === i ? 'done' : s)))
            } catch (err) {
                if (signal.aborted) return
                // Surface the failure — no placeholder scores are ever shown
                const failure = { userMessage: err.userMessage || err.message || 'Unknown error', detail: err.detail || null }
                resultsRef.current[i] = null
                errorsRef.current[i] = failure
                setStatuses(prev => prev.map((s, j) => (j === i ? 'error' : s)))
                setErrors(prev => prev.map((e, j) => (j === i ? failure : e)))
            }
        }

        if (signal.aborted) return
        if (resultsRef.current.every(r => r !== null)) {
            // All answers analysed — move to reflection report screen
            onCompleteRef.current([...resultsRef.current], [...errorsRef.current])
        } else {
            setNeedsAttention(true)
        }
    }, [recordings])

    // Runs when the screen loads — processes each recording one by one
    useEffect(() => {
        const controller = new AbortController()
        abortRef.current = controller
        // Started from a timer so state is only set from a callback; leaving the screen cancels it
        const timer = setTimeout(() => processRecordings(recordings.map((_, i) => i), controller.signal), 0)
        return () => {
            clearTimeout(timer)
            controller.abort()
        }
    }, [recordings, processRecordings])

    // Re-send only the answers that failed
    const retryFailed = () => {
        const failed = resultsRef.current
            .map((r, i) => (r === null ? i : -1))
            .filter(i => i >= 0)
        processRecordings(failed, abortRef.current.signal)
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
                <div className="flex items-center justify-between mb-6">
                    <div className="flex items-center gap-3">
                        <span className="text-white font-bold">NeuroIntent</span>
                        {/* research_pilot badge — always visible */}
                        <span className="bg-indigo-700 text-indigo-100 text-xs px-2 py-0.5 rounded-full">
                            research_pilot
                        </span>
                    </div>
                    <button
                        onClick={onWithdraw}
                        className="text-gray-400 text-xs hover:text-gray-200 transition-colors"
                    >
                        Withdraw consent
                    </button>
                </div>

                <h2 className="text-white text-xl font-semibold mb-2">
                    Analysing your answers
                </h2>
                <p className="text-gray-400 text-sm mb-6">
                    The first answer can take a minute or more while the server loads its models; later answers are
                    faster. Please keep this tab open.
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
                                <div className="mt-2">
                                    <p className="text-red-400 text-xs break-words">
                                        {errors[i].userMessage}
                                    </p>
                                    {errors[i].detail && errors[i].detail !== errors[i].userMessage && (
                                        <p className="text-gray-500 text-xs mt-1 break-words">
                                            Raw detail: {errors[i].detail}
                                        </p>
                                    )}
                                </div>
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
                <p className="text-gray-500 text-xs text-center mt-6">
                    Nothing is stored. Your audio is deleted right after analysis.
                </p>

            </div>
        </div>
    )
}

export default ProcessingScreen
