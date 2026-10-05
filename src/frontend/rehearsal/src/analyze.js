// Shared helper for sending one recorded answer to the backend's /analyze endpoint.
// Used by ProcessingScreen (first pass) and ReportScreen (per-answer retry).
import axios from 'axios'

// Read backend URL from environment variable
const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000'

// Send one recording and return the backend's JSON. Throws on any failure.
export async function analyzeRecording(recording, index) {
    // Build form data to send audio file to backend
    const formData = new FormData()
    formData.append('audio', recording.audioBlob, `answer_${index + 1}.webm`)
    formData.append('label', recording.label)
    formData.append('question_index', index)
    formData.append('mode', recording.mode)

    const response = await axios.post(`${BACKEND_URL}/analyze`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
    })
    return response.data
}

// Turn an axios error into a short, honest message for the user.
// A missing response means the request never reached the backend (network, CORS, server down).
export function describeError(err) {
    const detail = err.response?.data?.detail
    if (typeof detail === 'string') return detail
    if (detail) return JSON.stringify(detail)
    if (err.response) return `Server returned status ${err.response.status}`
    return err.message || 'Could not reach the server'
}
