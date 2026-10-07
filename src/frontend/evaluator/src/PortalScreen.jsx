// Import hooks
import { useState, useRef } from 'react'

// The 5 intent label buttons
const INTENT_LABELS = [
    'Confident',
    'Explaining',
    'Enthusiastic',
    'Uncertain',
    'Requesting'
]

// Mock audio clips for testing — Krishiv will replace with real API
const MOCK_CLIPS = Array.from({ length: 10 }, (_, i) => ({
    id: `clip_${i + 1}`,
    url: null // no real audio yet — placeholder
}))

function PortalScreen() {
    // Tracks which clip we are on
    const [clipIndex, setClipIndex] = useState(0)

    // Tracks selected label for current clip
    const [selectedLabel, setSelectedLabel] = useState(null)

    // Tracks confidence slider value 1-5
    const [confidence, setConfidence] = useState(3)

    // Tracks if audio is playing
    const [isPlaying, setIsPlaying] = useState(false)

    // Tracks if clip has been played at least once
    const [hasPlayed, setHasPlayed] = useState(false)

    // Tracks total clips labeled so far
    const [labeled, setLabeled] = useState(0)

    // Tracks if all clips are done
    const [allDone, setAllDone] = useState(false)

    // Stores submission status message
    const [submitStatus, setSubmitStatus] = useState('')

    // Holds audio element reference
    const audioRef = useRef(null)

    // Total number of clips
    const totalClips = MOCK_CLIPS.length

    // Handles audio play button
    const handlePlay = () => {
        if (!audioRef.current) return
        if (isPlaying) {
            audioRef.current.pause()
            setIsPlaying(false)
        } else {
            audioRef.current.play()
            setIsPlaying(true)
            setHasPlayed(true)
        }
    }

    // Handles submit button — loads the next clip. Prototype: the label is not sent anywhere or stored.
    const handleSubmit = () => {
        if (!selectedLabel) return

        const newLabeled = labeled + 1
        setLabeled(newLabeled)
        setSubmitStatus('Not stored (prototype)')

        // Check if all clips are done
        if (clipIndex === totalClips - 1) {
            setAllDone(true)
            return
        }

        // Move to next clip after short delay
        setTimeout(() => {
            setClipIndex(prev => prev + 1)
            setSelectedLabel(null)
            setConfidence(3)
            setHasPlayed(false)
            setIsPlaying(false)
            setSubmitStatus('')
        }, 600)
    }

    // All done screen
    if (allDone) {
        return (
            <div className="min-h-screen bg-gray-950 flex items-center justify-center px-6">
                <div className="w-full max-w-sm bg-gray-900 border border-gray-700 rounded-2xl p-8 text-center shadow-2xl">
                    <span className="text-4xl mb-4 block">🎉</span>
                    <h2 className="text-white text-xl font-semibold mb-2">All done!</h2>
                    <p className="text-gray-400 text-sm">
                        You have labeled {totalClips} of {totalClips} clips. Thank you.
                    </p>
                </div>
            </div>
        )
    }

    return (
        <div className="min-h-screen bg-gray-950 flex flex-col items-center justify-center px-6 py-10">

            {/* Card */}
            <div className="w-full max-w-md bg-gray-900 border border-gray-700 rounded-2xl p-8 shadow-2xl">

                {/* Header */}
                <div className="flex items-center gap-3 mb-6">
                    <span className="text-white font-bold">NeuroIntent</span>
                    {/* research_pilot badge — always visible */}
                    <span className="bg-indigo-700 text-indigo-100 text-xs px-2 py-0.5 rounded-full">
                        research_pilot
                    </span>
                </div>

                {/* Progress counter — per design rules */}
                <p className="text-gray-400 text-sm mb-6">
                    You have labeled{' '}
                    <span className="text-white font-semibold">{labeled}</span>{' '}
                    of{' '}
                    <span className="text-white font-semibold">{totalClips}</span>{' '}
                    clips. Thank you.
                </p>

                {/* Audio play button — no transcript, no speaker identity shown */}
                <div className="flex flex-col items-center mb-6">
                    <button
                        onClick={handlePlay}
                        className="w-20 h-20 rounded-full bg-indigo-600 hover:bg-indigo-500 flex items-center justify-center text-3xl transition-colors shadow-lg"
                    >
                        {isPlaying ? '⏸' : '▶'}
                    </button>
                    <p className="text-gray-500 text-xs mt-3">
                        {hasPlayed ? 'Click to replay' : 'Play the audio clip'}
                    </p>

                    {/* Hidden audio element — src will come from backend */}
                    <audio
                        ref={audioRef}
                        src={MOCK_CLIPS[clipIndex].url || ''}
                        onEnded={() => setIsPlaying(false)}
                    />
                </div>

                {/* 5 label buttons — no speaker info shown */}
                <div className="grid grid-cols-3 gap-2 mb-6">
                    {INTENT_LABELS.map((label) => (
                        <button
                            key={label}
                            onClick={() => setSelectedLabel(label)}
                            className={`py-2.5 rounded-xl text-sm font-medium transition-colors ${selectedLabel === label
                                    ? 'bg-indigo-600 text-white'
                                    : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                                }`}
                        >
                            {label}
                        </button>
                    ))}
                </div>

                {/* Confidence slider 1-5 */}
                <div className="mb-6">
                    <div className="flex justify-between items-center mb-2">
                        <span className="text-gray-300 text-xs font-medium">
                            Confidence
                        </span>
                        <span className="text-white text-sm font-bold">
                            {confidence} / 5
                        </span>
                    </div>
                    <input
                        type="range"
                        min="1"
                        max="5"
                        step="1"
                        value={confidence}
                        onChange={(e) => setConfidence(Number(e.target.value))}
                        className="w-full accent-indigo-500"
                    />
                    <div className="flex justify-between text-gray-600 text-xs mt-1">
                        <span>Not sure</span>
                        <span>Very sure</span>
                    </div>
                </div>

                {/* Submit button — disabled until label selected */}
                <button
                    onClick={handleSubmit}
                    className={`w-full py-3 rounded-xl text-sm font-semibold transition-colors ${selectedLabel
                            ? 'bg-indigo-600 hover:bg-indigo-500 text-white cursor-pointer'
                            : 'bg-gray-700 text-gray-500 cursor-not-allowed'
                        }`}
                >
                    {submitStatus || 'Submit & Next'}
                </button>

            </div>
        </div>
    )
}

export default PortalScreen