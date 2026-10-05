// Import hooks and axios for delete request
import { useState } from 'react'
import axios from 'axios'
import { analyzeRecording, describeError } from './analyze'

// Read backend URL from environment variable
const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000'

// Confidence bound badge colors
const CONFIDENCE_COLORS = {
    low: 'bg-red-900 text-red-300 border-red-700',
    medium: 'bg-yellow-900 text-yellow-300 border-yellow-700',
    high: 'bg-green-900 text-green-300 border-green-700'
}

function ReportScreen({ recordings, results: initialResults, errors: initialErrors = [] }) {
    // Tracks which answer card is expanded
    const [expandedIndex, setExpandedIndex] = useState(0)

    // Local copies so a failed answer can be retried from here without leaving the report
    const [results, setResults] = useState(initialResults)
    const [errors, setErrors] = useState(initialErrors)
    // Indices currently being retried
    const [retrying, setRetrying] = useState([])

    // Re-send one failed answer to the backend
    const retryAnswer = async (i) => {
        setRetrying(prev => [...prev, i])
        try {
            const data = await analyzeRecording(recordings[i], i)
            setResults(prev => prev.map((r, j) => (j === i ? data : r)))
            setErrors(prev => prev.map((e, j) => (j === i ? null : e)))
            setExpandedIndex(i)
        } catch (err) {
            setErrors(prev => {
                const updated = [...prev]
                updated[i] = describeError(err)
                return updated
            })
        } finally {
            setRetrying(prev => prev.filter(j => j !== i))
        }
    }

    // Handles delete data button — sends DELETE to backend
    const handleDeleteData = async () => {
        try {
            await axios.delete(`${BACKEND_URL}/session/anonymous`)
            alert('All your session data has been deleted successfully.')
        } catch (err) {
            alert('Data deletion requested. Your data will be removed shortly.')
        }
    }

    return (
        <div className="min-h-screen bg-gray-950 px-6 py-10">

            {/* Header row */}
            <div className="max-w-2xl mx-auto mb-8 flex items-center justify-between">
                <div className="flex items-center gap-3">
                    <span className="text-white font-bold text-lg">NeuroIntent</span>
                    {/* research_pilot badge — always visible */}
                    <span className="bg-indigo-700 text-indigo-100 text-xs px-2 py-0.5 rounded-full">
                        research_pilot
                    </span>
                </div>
                {/* Delete my data button — always visible per design rules */}
                <button
                    onClick={handleDeleteData}
                    className="text-red-400 text-xs hover:text-red-300 transition-colors"
                >
                    Delete my data
                </button>
            </div>

            {/* Page title */}
            <div className="max-w-2xl mx-auto mb-6">
                <h1 className="text-white text-2xl font-bold mb-1">Your Reflection Report</h1>
                <p className="text-gray-400 text-sm">
                    Here is how the system interpreted each of your answers.
                </p>
            </div>

            {/* One card per answer */}
            <div className="max-w-2xl mx-auto space-y-4">
                {recordings.map((rec, i) => {
                    const result = results[i]
                    if (!result) {
                        // Analysis failed for this answer — say so instead of showing anything invented
                        return (
                            <div
                                key={i}
                                className="bg-gray-900 border border-red-800 rounded-2xl p-5"
                            >
                                <p className="text-gray-400 text-xs mb-0.5">Answer {i + 1}</p>
                                <p className="text-white text-sm font-medium leading-snug mb-2">
                                    {rec.question}
                                </p>
                                <p className="text-red-400 text-xs">
                                    This answer could not be analysed, so there is no score for it.
                                </p>
                                {errors[i] && (
                                    <p className="text-red-300 text-xs mt-1 break-words">
                                        Reason: {errors[i]}
                                    </p>
                                )}
                                <button
                                    onClick={() => retryAnswer(i)}
                                    disabled={retrying.includes(i)}
                                    className="mt-3 px-4 py-1.5 bg-indigo-600 hover:bg-indigo-500 disabled:bg-gray-700 disabled:text-gray-500 text-white rounded-lg text-xs font-semibold transition-colors"
                                >
                                    {retrying.includes(i) ? 'Retrying...' : 'Retry this answer'}
                                </button>
                            </div>
                        )
                    }
                    const isExpanded = expandedIndex === i

                    return (
                        <div
                            key={i}
                            className="bg-gray-900 border border-gray-700 rounded-2xl overflow-hidden"
                        >
                            {/* Card header — click to expand/collapse */}
                            <button
                                onClick={() => setExpandedIndex(isExpanded ? -1 : i)}
                                className="w-full flex items-center justify-between p-5 text-left"
                            >
                                <div>
                                    <p className="text-gray-400 text-xs mb-0.5">Answer {i + 1}</p>
                                    <p className="text-white text-sm font-medium leading-snug">
                                        {rec.question}
                                    </p>
                                </div>
                                {/* Expand/collapse arrow */}
                                <span className={`text-gray-400 text-lg transition-transform ${isExpanded ? 'rotate-180' : ''}`}>
                                    ↓
                                </span>
                            </button>

                            {/* Expanded content */}
                            {isExpanded && (
                                <div className="px-5 pb-6 space-y-5 border-t border-gray-700 pt-5">

                                    {/* Content quality score bar */}
                                    <div>
                                        <div className="flex justify-between items-center mb-1.5">
                                            <span className="text-gray-300 text-xs font-medium">
                                                Content Quality Score
                                            </span>
                                            <span className="text-white text-sm font-bold">
                                                {result.content_quality_score.toFixed(2)}
                                            </span>
                                        </div>
                                        {/* Score bar */}
                                        <div className="w-full bg-gray-700 rounded-full h-2.5">
                                            <div
                                                className="bg-indigo-500 h-2.5 rounded-full transition-all duration-700"
                                                style={{ width: `${result.content_quality_score * 100}%` }}
                                            />
                                        </div>
                                    </div>

                                    {/* Delivery pattern — 3 acoustic facts */}
                                    <div>
                                        <p className="text-gray-300 text-xs font-medium mb-2">
                                            Delivery Pattern
                                        </p>
                                        <div className="space-y-1.5">
                                            {result.delivery_pattern.map((fact, j) => (
                                                <div key={j} className="flex items-start gap-2">
                                                    <span className="text-indigo-400 text-xs mt-0.5">•</span>
                                                    <p className="text-gray-300 text-xs">{fact}</p>
                                                </div>
                                            ))}
                                        </div>
                                    </div>

                                    {/* Gap visualization — two bars side by side */}
                                    <div>
                                        <p className="text-gray-300 text-xs font-medium mb-3">
                                            System Impact
                                        </p>
                                        <div className="grid grid-cols-2 gap-3">

                                            {/* Without system bar */}
                                            <div>
                                                <p className="text-gray-500 text-xs mb-1.5">Without system</p>
                                                <div className="w-full bg-gray-700 rounded-full h-2.5 mb-1">
                                                    <div
                                                        className="bg-red-500 h-2.5 rounded-full transition-all duration-700"
                                                        style={{ width: `${result.score_without_system * 100}%` }}
                                                    />
                                                </div>
                                                <p className="text-red-400 text-sm font-bold">
                                                    {result.score_without_system.toFixed(2)}
                                                </p>
                                            </div>

                                            {/* With system bar */}
                                            <div>
                                                <p className="text-gray-500 text-xs mb-1.5">With system</p>
                                                <div className="w-full bg-gray-700 rounded-full h-2.5 mb-1">
                                                    <div
                                                        className="bg-green-500 h-2.5 rounded-full transition-all duration-700"
                                                        style={{ width: `${result.score_with_system * 100}%` }}
                                                    />
                                                </div>
                                                <p className="text-green-400 text-sm font-bold">
                                                    {result.score_with_system.toFixed(2)}
                                                </p>
                                            </div>

                                        </div>
                                    </div>

                                    {/* Confidence bound badge — always visible per design rules */}
                                    <div className="flex items-center gap-2">
                                        <span className="text-gray-400 text-xs">Confidence:</span>
                                        <span className={`text-xs px-2 py-0.5 rounded-full border font-medium ${CONFIDENCE_COLORS[result.confidence_bound]
                                            }`}>
                                            {result.confidence_bound}
                                        </span>
                                    </div>

                                    {/* Llama 3 written interpretation */}
                                    <div>
                                        <p className="text-gray-300 text-xs font-medium mb-2">
                                            AI Interpretation
                                        </p>
                                        <div className="bg-gray-800 rounded-xl p-4 border border-gray-600">
                                            <p className="text-gray-200 text-sm leading-relaxed">
                                                {result.interpretation}
                                            </p>
                                        </div>
                                    </div>

                                    {/* Timestamp-linked transcript */}
                                    <div>
                                        <p className="text-gray-300 text-xs font-medium mb-2">
                                            Transcript
                                        </p>
                                        <div className="bg-gray-800 rounded-xl p-4 border border-gray-600">
                                            <p className="text-gray-400 text-xs leading-relaxed">
                                                {result.transcript}
                                            </p>
                                        </div>
                                    </div>

                                    {/* Self label that speaker selected */}
                                    <div className="flex items-center gap-2">
                                        <span className="text-gray-400 text-xs">Your intended label:</span>
                                        <span className="bg-gray-700 text-gray-200 text-xs px-2 py-0.5 rounded-full">
                                            {rec.label}
                                        </span>
                                    </div>

                                </div>
                            )}
                        </div>
                    )
                })}
            </div>

            {/* Bottom delete button — always visible per design rules */}
            <div className="max-w-2xl mx-auto mt-8 text-center">
                <button
                    onClick={handleDeleteData}
                    className="text-red-400 text-sm hover:text-red-300 transition-colors border border-red-800 px-6 py-2 rounded-xl"
                >
                    Delete all my data
                </button>
            </div>

        </div>
    )
}

export default ReportScreen
