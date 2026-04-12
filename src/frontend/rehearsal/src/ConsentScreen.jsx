// Import useState for checkbox tracking
import { useState } from 'react'

function ConsentScreen({ onProceed }) {
    // Checkbox A — required to proceed
    const [researchConsent, setResearchConsent] = useState(false)

    // Checkbox B — optional, never pre-checked per design rules
    const [asdConsent, setAsdConsent] = useState(false)

    // Handles proceed button click — only works if checkbox A is checked
    const handleProceed = () => {
        if (!researchConsent) return
        // Pass consent data up to App.jsx
        onProceed({ researchConsent, asdConsent })
    }

    // Handles data withdrawal — will call backend DELETE in later step
    const handleWithdraw = () => {
        alert('Your data withdrawal request has been noted. No data has been stored yet.')
    }

    return (
        // Full screen centered layout
        <div className="min-h-screen bg-gray-950 flex flex-col items-center justify-center px-6">

            {/* Card container */}
            <div className="w-full max-w-lg bg-gray-900 border border-gray-700 rounded-2xl p-8 shadow-2xl">

                {/* Header */}
                <div className="flex items-center gap-3 mb-6">
                    <span className="text-white text-xl font-bold">NeuroIntent</span>
                    {/* research_pilot badge — always visible per design rules */}
                    <span className="bg-indigo-700 text-indigo-100 text-xs px-2 py-0.5 rounded-full">
                        research_pilot
                    </span>
                </div>

                <h1 className="text-white text-2xl font-semibold mb-2">
                    Speaker Rehearsal
                </h1>
                <p className="text-gray-400 text-sm mb-8">
                    Practice interview answers and see how your speech is interpreted.
                    Your privacy is protected throughout.
                </p>

                {/* Checkbox A — required */}
                <div
                    onClick={() => setResearchConsent(prev => !prev)}
                    className="flex items-start gap-3 mb-5 cursor-pointer group"
                >
                    {/* Custom checkbox box */}
                    <div className={`mt-0.5 w-5 h-5 rounded border-2 flex items-center justify-center flex-shrink-0 transition-colors ${researchConsent
                            ? 'bg-indigo-600 border-indigo-600'
                            : 'border-gray-500 group-hover:border-indigo-400'
                        }`}>
                        {researchConsent && (
                            <svg className="w-3 h-3 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                            </svg>
                        )}
                    </div>
                    {/* Label */}
                    <p className="text-gray-300 text-sm leading-relaxed">
                        I consent to my speech being processed and features stored for research.
                        <span className="text-red-400 ml-1">*</span>
                    </p>
                </div>

                {/* Checkbox B — optional, never pre-checked */}
                <div
                    onClick={() => setAsdConsent(prev => !prev)}
                    className="flex items-start gap-3 mb-8 cursor-pointer group"
                >
                    {/* Custom checkbox box */}
                    <div className={`mt-0.5 w-5 h-5 rounded border-2 flex items-center justify-center flex-shrink-0 transition-colors ${asdConsent
                            ? 'bg-indigo-600 border-indigo-600'
                            : 'border-gray-500 group-hover:border-indigo-400'
                        }`}>
                        {asdConsent && (
                            <svg className="w-3 h-3 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                            </svg>
                        )}
                    </div>
                    {/* Label */}
                    <p className="text-gray-300 text-sm leading-relaxed">
                        I identify as autistic / ASD and consent to ASD-aware personalized interpretation.
                        <span className="text-gray-500 ml-1">(optional)</span>
                    </p>
                </div>

                {/* Required field note */}
                <p className="text-gray-500 text-xs mb-6">
                    <span className="text-red-400">*</span> Required to proceed
                </p>

                {/* Proceed button — disabled until checkbox A is checked */}
                <button
                    onClick={handleProceed}
                    className={`w-full py-3 rounded-xl text-sm font-semibold transition-colors ${researchConsent
                            ? 'bg-indigo-600 hover:bg-indigo-500 text-white cursor-pointer'
                            : 'bg-gray-700 text-gray-500 cursor-not-allowed'
                        }`}
                >
                    Begin Rehearsal
                </button>

                {/* Withdraw anytime button — always visible per design rules */}
                <button
                    onClick={handleWithdraw}
                    className="w-full mt-3 py-3 rounded-xl text-sm text-gray-500 hover:text-gray-300 transition-colors"
                >
                    Withdraw anytime
                </button>

            </div>
        </div>
    )
}

export default ConsentScreen