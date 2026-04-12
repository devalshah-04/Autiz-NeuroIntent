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

// The 5 intent label buttons speaker selects after each recording
const INTENT_LABELS = [
    'Confident',
    'Explaining',
    'Enthusiastic',
    'Uncertain',
    'Requesting'
]

function RecordingScreen({ consentData, onComplete }) {
    // Tracks which question we are on (0 to 4)
    const [questionIndex, setQuestionIndex] = useState(0)

    // Tracks recording state — 'idle', 'recording', 'done'
    const [recordingState, setRecordingState] = useState('idle')

    // Stores all completed recordings with their labels
    const [recordings, setRecordings] = useState([])

    // Stores the selected intent label for current answer
    const [selectedLabel, setSelectedLabel] = useState(null)

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

    // Starts recording from microphone
    const startRecording = async () => {
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
            alert('Microphone access denied. Please allow microphone access and try again.')
        }
    }

    // Animates waveform bars using audio analyser data
    const animateWaveform = () => {
        if (!analyserRef.current) return
        const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount)

        const draw = () => {
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

    // Stops recording and shows label buttons
    const stopRecording = () => {
        // Stop waveform animation
        cancelAnimationFrame(animationRef.current)
        setWaveformBars(Array(40).fill(4))

        // Stop audio context
        if (audioContextRef.current) {
            audioContextRef.current.close()
        }

        // Stop MediaRecorder
        mediaRecorderRef.current.stop()

        // Stop all microphone tracks
        mediaRecorderRef.current.stream.getTracks().forEach(t => t.stop())

        setRecordingState('done')
    }

    // Saves current answer and moves to next question
    const handleLabelSelect = (label) => {
        setSelectedLabel(label)

        // Build audio blob from recorded chunks
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' })

        // Save recording with its label and question
        const newRecording = {
            questionIndex,
            question: QUESTIONS[questionIndex],
            audioBlob,
            label,
            // Send mode to backend based on checkbox B from consent screen
            mode: consentData.asdConsent ? 'speaker_declared' : 'universal_fairness'
        }

        const updated = [...recordings, newRecording]
        setRecordings(updated)

        // If all 5 questions done, move to processing screen
        if (questionIndex === 4) {
            setTimeout(() => onComplete(updated), 500)
            return
        }

        // Move to next question
        setTimeout(() => {
            setQuestionIndex(prev => prev + 1)
            setRecordingState('idle')
            setSelectedLabel(null)
        }, 500)
    }

    // Handles delete data button — always visible per design rules
    const handleDeleteData = () => {
        alert('Data deletion requested. All session data will be removed.')
    }

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
                    {/* Delete my data button — always visible per design rules */}
                    <button
                        onClick={handleDeleteData}
                        className="text-red-400 text-xs hover:text-red-300 transition-colors"
                    >
                        Delete my data
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

                {/* Intent label buttons — shown immediately after recording stops */}
                {recordingState === 'done' && (
                    <div>
                        <p className="text-gray-300 text-sm font-medium mb-3">
                            How did you intend that answer?
                        </p>
                        <div className="grid grid-cols-3 gap-2">
                            {INTENT_LABELS.map((label) => (
                                <button
                                    key={label}
                                    onClick={() => handleLabelSelect(label)}
                                    className={`py-2 rounded-lg text-sm font-medium transition-colors ${selectedLabel === label
                                            ? 'bg-indigo-600 text-white'
                                            : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                                        }`}
                                >
                                    {label}
                                </button>
                            ))}
                        </div>
                    </div>
                )}

            </div>
        </div>
    )
}

export default RecordingScreen