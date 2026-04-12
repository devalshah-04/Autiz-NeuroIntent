// Import useState for form fields
import { useState } from 'react'

function LoginScreen({ onLogin }) {
    // Stores evaluator password input
    const [password, setPassword] = useState('')

    // Stores error message if login fails
    const [error, setError] = useState('')

    // Handles login — simple password check for now
    const handleLogin = () => {
        // Simple password gate — Krishiv will replace with proper auth later
        if (password === 'neurointent2024') {
            onLogin()
        } else {
            setError('Incorrect password. Please try again.')
        }
    }

    return (
        <div className="min-h-screen bg-gray-950 flex items-center justify-center px-6">

            {/* Login card */}
            <div className="w-full max-w-sm bg-gray-900 border border-gray-700 rounded-2xl p-8 shadow-2xl">

                {/* Header */}
                <div className="flex items-center gap-3 mb-8">
                    <span className="text-white font-bold">NeuroIntent</span>
                    {/* research_pilot badge — always visible */}
                    <span className="bg-indigo-700 text-indigo-100 text-xs px-2 py-0.5 rounded-full">
                        research_pilot
                    </span>
                </div>

                <h1 className="text-white text-xl font-semibold mb-1">
                    Evaluator Portal
                </h1>
                <p className="text-gray-400 text-sm mb-6">
                    Enter your access password to begin labeling.
                </p>

                {/* Password input — using onChange not form */}
                <input
                    type="password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleLogin()}
                    placeholder="Enter password"
                    className="w-full bg-gray-800 border border-gray-600 rounded-xl px-4 py-3 text-white text-sm placeholder-gray-500 focus:outline-none focus:border-indigo-500 mb-3"
                />

                {/* Error message */}
                {error && (
                    <p className="text-red-400 text-xs mb-3">{error}</p>
                )}

                {/* Login button */}
                <button
                    onClick={handleLogin}
                    className="w-full py-3 bg-indigo-600 hover:bg-indigo-500 text-white rounded-xl text-sm font-semibold transition-colors"
                >
                    Enter Portal
                </button>

            </div>
        </div>
    )
}

export default LoginScreen