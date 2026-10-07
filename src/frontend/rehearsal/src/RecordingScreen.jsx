// Import hooks for state and side effects
import { useState, useRef, useEffect } from 'react'

// The 5 interview questions shown one at a time
const QUESTIONS = [
    "Tell me about yourself and your background.",
    "Describe a challenging problem you solved and how you approached it.",
    "What are your greatest strengths and how do they help you at work?",
    "Tell me about a time you worked in a team under pressure.",
    "Where do you see yourself in 5 years and what are your goals?"
]

function RecordingScreen({ consentData, onComplete, onWithdraw }) {
    // Tracks which question we are on (0 to 4)
    const [questionIndex, setQuestionIndex] = useState(0)

    // Tracks recording state — 'idle', 'recording', 'done'
    const [recordingState, setRecordingState] = useState('idle')

    // Stores all accepted recordings
    const [recordings, setRecordings] = useState([])

    // Why the microphone could not be used, if it could not
    const [micError, setMicError] = useState(null)

    // Stores waveform bar heights for visualization
    const [waveformBars, setWaveformBars] = useState(Array(40).fill(4))

    // Holds MediaRecorder instance
    const mediaRecorderRef = useRef(null)

    // Holds recorded audio chunks
    const audioChunksRef = useRef([])

    // Holds audio analyser for waveform
    const analyserRef = useRef(null)

    // Holds animation frame for waveform
    const animationRef = useRef(null)

    // Holds audio context
    const audioContextRef = useRef(null)

    // Releases the microphone and the waveform loop. Safe to call more than once.
    const releaseMicrophone = () => {
        cancelAnimationFrame(animationRef.current)
        if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
            audioContextRef.current.close()
        }
        audioContextRef.current = null
        analyserRef.current = null
        const recorder = mediaRecorderRef.current
        if (recorder) {
            if (recorder.state !== 'inactive') recorder.stop()
            recorder.stream.getTracks().forEach(t => t.stop())
        }
    }

    // Leaving this screen (including withdrawing consent) must switch the microphone off
    useEffect(() => {
        return () => {
            cancelAnimationFrame(animationRef.current)
            if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
                audioContextRef.current.close()
            }
            const recorder = mediaRecorderRef.current
            if (recorder) {
                if (recorder.state !== 'inactive') recorder.stop()
                recorder.stream.getTracks().forEach(t => t.stop())
            }
        }
    }, [])

    // Starts recording from microphone
    const startRecording = async () => {
        setMicError(null)
        try {
            // Request microphone access from browser
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true })

            // Set up audio context for waveform visualization
            audioContextRef.current = new AudioContext()
            const source = audioContextRef.current.createMediaStreamSource(stream)
            analyserRef.current = audioContextRef.current.createAnalyser()
            analyserRef.current.fftSize = 128
            source.connect(analyserRef.current)

            // Start animating waveform bars
            animateWaveform()

            // Set up MediaRecorder to capture audio
            mediaRecorderRef.current = new MediaRecorder(stream)
            audioChunksRef.current = []

            // Collect audio data as it comes in
            mediaRecorderRef.current.ondataavailable = (e) => {
                audioChunksRef.current.push(e.data)
            }

            mediaRecorderRef.current.start()
            setRecordingState('recording')
        } catch (err) {
            setMicError(
                err && err.name === 'NotAllowedError'
                    ? 'Microphone access was denied. Please allow microphone access in your browser and try again.'
                    : `The microphone could not be used (${err && err.message ? err.message : 'unknown error'}).`
            )
        }
    }

    // Animates waveform bars using audio analyser data
    const animateWaveform = () => {
        if (!analyserRef.current) return
        const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount)

        const draw = () => {
            if (!analyserRef.current) return
            analyserRef.current.getByteFrequencyData(dataArray)
            // Map frequency data to bar heights
            const bars = Array.from(dataArray).slice(0, 40).map(val =>
                Math.max(4, (val / 255) * 60)
            )
            setWaveformBars(bars)
            animationRef.current = requestAnimationFrame(draw)
        }
        draw()
    }

    // Stops recording and offers to keep or redo the answer
    const stopRecording = () => {
        releaseMicrophone()
        setWaveformBars(Array(40).fill(4))
        setRecordingState('done')
    }

    // Throws the take away so the answer can be recorded again
    const handleReRecord = () => {
        audioChunksRef.current = []
        setRecordingState('idle')
    }

    // Keeps the current answer and moves to the next question (or on to processing after the last one)
    const handleAcceptAnswer = () => {
        // Build audio blob from recorded chunks
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' })

        const newRecording = {
            questionIndex,
            question: QUESTIONS[questionIndex],
            audioBlob,
            // Send mode to backend based on checkbox B from consent screen (both modes run the same pipeline)
            mode: consentData.asdConsent ? 'speaker_declared' : 'universal_fairness'
        }

        const updated = [...recordings, newRecording]
        setRecordings(updated)

        // If all 5 questions done, move to processing screen
        if (questionIndex === QUESTIONS.length - 1) {
            onComplete(updated)
            return
        }

        setQuestionIndex(prev => prev + 1)
        setRecordingState('idle')
    }

    const isLastQuestion = questionIndex === QUESTIONS.length - 1

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
                    {/* Withdraw consent — returns to the consent screen */}
                    <button
                        onClick={onWithdraw}
                        className="text-gray-400 text-xs hover:text-gray-200 transition-colors"
                    >
                        Withdraw consent
                    </button>
                </div>

                {/* Progress indicator */}
                <div className="flex gap-1.5 mb-6">
                    {QUESTIONS.map((_, i) => (
                        <div
                            key={i}
                            className={`h-1 flex-1 rounded-full transition-colors ${i < questionIndex
                                    ? 'bg-indigo-500'
                                    : i === questionIndex
                                        ? 'bg-indigo-300'
                                        : 'bg-gray-700'
                                }`}
                        />
                    ))}
                </div>

                {/* Question counter */}
                <p className="text-gray-500 text-xs mb-2">
                    Question {questionIndex + 1} of {QUESTIONS.length}
                </p>

                {/* Question text */}
                <h2 className="text-white text-lg font-semibold mb-6 leading-snug">
                    {QUESTIONS[questionIndex]}
                </h2>

                {/* Waveform visualization — shown while recording */}
                {recordingState === 'recording' && (
                    <div className="flex items-end justify-center gap-0.5 h-16 mb-6">
                        {waveformBars.map((height, i) => (
                            <div
                                key={i}
                                className="w-1.5 bg-indigo-500 rounded-full transition-all duration-75"
                                style={{ height: `${height}px` }}
                            />
                        ))}
                    </div>
                )}

                {/* Microphone problem */}
                {micError && (
                    <p className="text-red-400 text-xs mb-4 break-words">{micError}</p>
                )}

                {/* Record / Stop button */}
                {recordingState === 'idle' && (
                    <button
                        onClick={startRecording}
                        className="w-full py-3 bg-indigo-600 hover:bg-indigo-500 text-white rounded-xl text-sm font-semibold transition-colors mb-4"
                    >
                        🎙 Start Recording
                    </button>
                )}

                {recordingState === 'recording' && (
                    <button
                        onClick={stopRecording}
                        className="w-full py-3 bg-red-600 hover:bg-red-500 text-white rounded-xl text-sm font-semibold transition-colors mb-4"
                    >
                        ⏹ Stop Recording
                    </button>
                )}

                {/* Keep or redo — shown immediately after recording stops */}
                {recordingState === 'done' && (
                    <div className="space-y-2">
                        <button
                            onClick={handleAcceptAnswer}
                            className="w-full py-3 bg-indigo-600 hover:bg-indigo-500 text-white rounded-xl text-sm font-semibold transition-colors"
                        >
                            {isLastQuestion ? 'Use this answer and finish' : 'Use this answer and continue'}
                        </button>
                        <button
                            onClick={handleReRecord}
                            className="w-full py-2.5 bg-gray-700 hover:bg-gray-600 text-gray-200 rounded-xl text-sm font-medium transition-colors"
                        >
                            Record this answer again
                        </button>
                    </div>
                )}

                {/* Privacy line */}
                <p className="text-gray-500 text-xs text-center mt-6">
                    Nothing is stored. Your audio is deleted right after analysis.
                </p>

            </div>
        </div>
    )
}

export default RecordingScreen
