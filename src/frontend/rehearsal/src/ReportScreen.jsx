// Import hooks
import { useState, useEffect, useRef } from 'react'
import { analyzeRecording } from './api'

// True only for real numbers; every optional field below is checked before it is shown.
const isNum = (v) => typeof v === 'number' && Number.isFinite(v)
const isText = (v) => typeof v === 'string' && v.length > 0
const clamp01 = (v) => Math.min(1, Math.max(0, v))

// One labelled block inside an answer card
function Section({ title, children }) {
    return (
        <div>
            <p className="text-gray-300 text-xs font-medium mb-2">{title}</p>
            {children}
        </div>
    )
}

function SmokeBanner() {
    return (
        <div role="alert" className="max-w-2xl mx-auto mb-6 bg-red-900 border-2 border-red-500 text-red-100 rounded-xl px-4 py-3 text-sm font-bold text-center">
            Smoke-test artifacts, not results
        </div>
    )
}

// Everything the server returned for one answer. Nothing here is invented: a missing field hides its block.
function ResultBody({ result }) {
    const baselineShown = isNum(result.prosody_only_baseline_score) && isText(result.prosody_only_baseline_label)
    const explanation = result.explanation && typeof result.explanation === 'object' ? result.explanation : null
    const topFeatures = explanation && Array.isArray(explanation.top_features) ? explanation.top_features : []
    const methodLabel = explanation ? (explanation.method === 'shap' ? 'SHAP' : explanation.method === 'proxy' ? 'proxy' : null) : null
    const delivery = Array.isArray(result.delivery_pattern) ? result.delivery_pattern : []

    return (
        <div className="px-5 pb-6 space-y-5 border-t border-gray-700 pt-5">

            {/* Content score */}
            <Section title="Content score">
                {result.content_score_source === 'proxy' && (
                    <p className="text-yellow-300 text-xs font-semibold mb-2">
                        {isText(result.content_scorer_stamp) ? result.content_scorer_stamp : 'Proxy score, not a trained scorer'}
                    </p>
                )}
                {isNum(result.content_score) ? (
                    <>
                        <div className="flex justify-between items-center mb-1.5">
                            <span className="text-gray-400 text-xs">0 to 1</span>
                            <span className="text-white text-sm font-bold">{result.content_score.toFixed(2)}</span>
                        </div>
                        <div className="w-full bg-gray-700 rounded-full h-2.5">
                            <div
                                className="bg-indigo-500 h-2.5 rounded-full transition-all duration-700"
                                style={{ width: `${clamp01(result.content_score) * 100}%` }}
                            />
                        </div>
                    </>
                ) : (
                    <p className="text-gray-400 text-xs">The server did not return a content score.</p>
                )}
                <p className="text-gray-400 text-xs mt-2">
                    This score is computed from the transcript only; your voice and delivery are not an input to it.
                </p>
                <details className="mt-2">
                    <summary className="text-gray-500 text-xs cursor-pointer">Technical details</summary>
                    <div className="text-gray-400 text-xs mt-1 space-y-1">
                        {isNum(result.content_score_raw) && (
                            <p>Unclipped score (content_score_raw): {result.content_score_raw.toFixed(4)}. The score above is this value limited to the range 0 to 1.</p>
                        )}
                        {isText(result.content_score_source) && <p>Score source: {result.content_score_source}</p>}
                        {isText(result.content_label_source) && (
                            <p>
                                Trained against: {result.content_label_source}
                                {result.content_label_source === 'chalearn' && ' (human impressions of video clips, a weak stand-in for the quality of the words alone)'}
                            </p>
                        )}
                    </div>
                </details>
            </Section>

            {/* Transcript */}
            {isText(result.transcript) && (
                <Section title="Transcript">
                    <div className="bg-gray-800 rounded-xl p-4 border border-gray-600">
                        <p className="text-gray-400 text-xs leading-relaxed">{result.transcript}</p>
                    </div>
                </Section>
            )}

            {/* Delivery measurements: plain measured values, no verdicts */}
            {delivery.length > 0 && (
                <Section title="Delivery measurements (not used in the content score)">
                    <div className="space-y-1.5">
                        {delivery.map((fact, j) => (
                            <div key={j} className="flex items-start gap-2">
                                <span className="text-indigo-400 text-xs mt-0.5">•</span>
                                <p className="text-gray-300 text-xs">{String(fact)}</p>
                            </div>
                        ))}
                    </div>
                </Section>
            )}

            {/* Baseline comparison, with the API's own label */}
            {baselineShown && (
                <Section title="Baseline comparison">
                    <p className="text-gray-400 text-xs mb-1">{result.prosody_only_baseline_label}</p>
                    <p className="text-white text-sm font-bold">{result.prosody_only_baseline_score.toFixed(2)}</p>
                    <p className="text-gray-500 text-xs mt-1">
                        This model looks at the voice measurements only. Its number is in standard-deviation units of the
                        training data, not on the 0 to 1 scale of the content score.
                    </p>
                </Section>
            )}

            {/* Explanation of the baseline, labelled with the method the API reports */}
            {methodLabel && topFeatures.length > 0 && (
                <Section title={`What drove the baseline number (method: ${methodLabel})`}>
                    <div className="space-y-1">
                        {topFeatures.map((f, j) => (
                            isText(f && f.feature) && isNum(f.contribution) ? (
                                <div key={j} className="flex justify-between gap-3 text-xs">
                                    <span className="text-gray-300 break-all">{f.feature}</span>
                                    <span className="text-gray-400 font-mono">{f.contribution.toFixed(2)}</span>
                                </div>
                            ) : null
                        ))}
                    </div>
                    {isText(explanation.note) && <p className="text-gray-500 text-xs mt-2">{explanation.note}</p>}
                </Section>
            )}

            {/* Interpretation text */}
            {isText(result.interpretation) && (
                <Section title="Interpretation">
                    <div className="bg-gray-800 rounded-xl p-4 border border-gray-600">
                        <p className="text-gray-200 text-sm leading-relaxed">{result.interpretation}</p>
                    </div>
                    {result.interpretation_source === 'template' ? (
                        <p className="text-gray-500 text-xs mt-1">This text is filled in from a template. It is not written by a model.</p>
                    ) : isText(result.interpretation_source) ? (
                        <p className="text-gray-500 text-xs mt-1">Source: {result.interpretation_source}</p>
                    ) : null}
                </Section>
            )}

            {/* Intent: no intent model exists */}
            {result.intent_label === null || result.intent_label === undefined ? (
                <p className="text-gray-500 text-xs">Intent label: not part of this milestone.</p>
            ) : (
                <p className="text-gray-400 text-xs">Intent label: {String(result.intent_label)}</p>
            )}

            {/* Mode echo */}
            {isText(result.mode) && (
                <p className="text-gray-500 text-xs">
                    Mode sent: {result.mode}.{isText(result.mode_effect) ? ` Effect: ${result.mode_effect}.` : ''}
                </p>
            )}

        </div>
    )
}

function ReportScreen({ recordings, results: initialResults, errors: initialErrors = [], onWithdraw }) {
    // Tracks which answer card is expanded
    const [expandedIndex, setExpandedIndex] = useState(0)

    // Local copies so a failed answer can be retried from here without leaving the report
    const [results, setResults] = useState(initialResults)
    const [errors, setErrors] = useState(initialErrors)
    // Indices currently being retried
    const [retrying, setRetrying] = useState([])

    // Cancels retries in flight when this screen is left (for example by withdrawing consent)
    const abortRef = useRef(null)
    useEffect(() => {
        const controller = new AbortController()
        abortRef.current = controller
        return () => controller.abort()
    }, [])

    // Re-send one failed answer to the backend
    const retryAnswer = async (i) => {
        setRetrying(prev => [...prev, i])
        try {
            const data = await analyzeRecording(recordings[i], i, { signal: abortRef.current.signal })
            setResults(prev => prev.map((r, j) => (j === i ? data : r)))
            setErrors(prev => prev.map((e, j) => (j === i ? null : e)))
            setExpandedIndex(i)
        } catch (err) {
            if (abortRef.current.signal.aborted) return
            const failure = { userMessage: err.userMessage || err.message || 'Unknown error', detail: err.detail || null }
            setErrors(prev => prev.map((e, j) => (j === i ? failure : e)))
        } finally {
            setRetrying(prev => prev.filter(j => j !== i))
        }
    }

    const anySmoke = results.some(r => r && r.smoke_artifacts === true)

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
                {/* Withdraw consent — returns to the consent screen */}
                <button
                    onClick={onWithdraw}
                    className="text-gray-400 text-xs hover:text-gray-200 transition-colors"
                >
                    Withdraw consent
                </button>
            </div>

            {anySmoke && <SmokeBanner />}

            {/* Page title */}
            <div className="max-w-2xl mx-auto mb-6">
                <h1 className="text-white text-2xl font-bold mb-1">Your Reflection Report</h1>
                <p className="text-gray-400 text-sm">
                    Here is what the system returned for each of your answers.
                </p>
                <p className="text-gray-500 text-xs mt-1">
                    Nothing is stored. Your audio is deleted right after analysis.
                </p>
            </div>

            {/* One card per answer */}
            <div className="max-w-2xl mx-auto space-y-4">
                {recordings.map((rec, i) => {
                    const result = results[i]
                    if (!result) {
                        // Analysis failed for this answer — say so instead of showing anything invented
                        const failure = errors[i]
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
                                {failure && (
                                    <>
                                        <p className="text-red-300 text-xs mt-1 break-words">
                                            {failure.userMessage}
                                        </p>
                                        {failure.detail && failure.detail !== failure.userMessage && (
                                            <p className="text-gray-500 text-xs mt-1 break-words">
                                                Raw detail: {failure.detail}
                                            </p>
                                        )}
                                    </>
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

                            {isExpanded && <ResultBody result={result} />}
                        </div>
                    )
                })}
            </div>

        </div>
    )
}

export default ReportScreen
