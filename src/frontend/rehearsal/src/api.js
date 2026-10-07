// The single client for the backend. Nothing else in the app talks to the network.
// It sends audio and mode as multipart form fields (no consent field, no label) and turns every failure
// into a typed error with a plain-language message. It never fills in values the server did not return.

export const BACKEND_URL = (import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000').replace(/\/+$/, '')

// The first request after the server starts can be slow while models load, so the default is generous.
export const DEFAULT_TIMEOUT_MS = 300000

export function resolveTimeoutMs(raw) {
    const n = Number.parseInt(raw, 10)
    return Number.isFinite(n) && n > 0 ? n : DEFAULT_TIMEOUT_MS
}

const TIMEOUT_MS = resolveTimeoutMs(import.meta.env.VITE_REQUEST_TIMEOUT_MS)

// Every error carries a message for the user (userMessage) and the raw server or browser text (detail).
export class ApiError extends Error {
    constructor(kind, userMessage, detail = null, status = null) {
        super(userMessage)
        this.name = kind
        this.kind = kind
        this.userMessage = userMessage
        this.detail = detail
        this.status = status
    }
}
export class ServiceNotReady extends ApiError {
    constructor(userMessage, detail, status) { super('ServiceNotReady', userMessage, detail, status) }
}
export class NoSpeech extends ApiError {
    constructor(userMessage, detail, status) { super('NoSpeech', userMessage, detail, status) }
}
export class InvalidRequest extends ApiError {
    constructor(userMessage, detail, status) { super('InvalidRequest', userMessage, detail, status) }
}
export class NetworkError extends ApiError {
    constructor(userMessage, detail) { super('Network', userMessage, detail) }
}
export class TimeoutError extends ApiError {
    constructor(userMessage, detail) { super('Timeout', userMessage, detail) }
}
// Anything else the server or the browser gave back that does not fit the kinds above (for example HTTP 500).
export class UnexpectedResponse extends ApiError {
    constructor(userMessage, detail, status) { super('UnexpectedResponse', userMessage, detail, status) }
}

// FastAPI sends detail as a string, or as a list of objects for form validation errors.
export function detailToText(detail) {
    if (typeof detail === 'string') return detail
    if (Array.isArray(detail)) {
        return detail.map(d => (d && typeof d.msg === 'string' ? d.msg : JSON.stringify(d))).join('; ')
    }
    if (detail === null || detail === undefined) return ''
    return JSON.stringify(detail)
}

// Map an HTTP error status and the server's detail to a typed error.
export function errorFromResponse(status, rawDetail) {
    const detail = detailToText(rawDetail)
    if (status === 503) {
        return new ServiceNotReady(
            `The analysis service is not ready. ${detail || 'Required model files may be missing on the server.'}`,
            detail, status)
    }
    if (status === 422 && /no speech/i.test(detail)) {
        return new NoSpeech(
            'No speech was detected in this recording. Please record the answer again, speaking close to the microphone.',
            detail, status)
    }
    if (status === 400 || status === 403 || status === 422) {
        return new InvalidRequest(`The server rejected this request. ${detail}`.trim(), detail, status)
    }
    return new UnexpectedResponse(`The server returned an unexpected error (status ${status}).`, detail, status)
}

export function networkError(backendUrl, cause) {
    return new NetworkError(
        `Could not reach the analysis server at ${backendUrl}. Check that the backend is running and that this page is on its allowed-origins list (CORS).`,
        cause && cause.message ? cause.message : String(cause))
}

export function timeoutError(timeoutMs) {
    return new TimeoutError(
        `The server did not answer within ${Math.round(timeoutMs / 1000)} seconds. The first request after the server starts can be slow while it loads its models; please try again.`,
        `timeout after ${timeoutMs} ms`)
}

// Send one recording to POST /analyze and return the parsed JSON. Throws an ApiError on any failure.
// If the caller's `signal` aborts (for example the user withdrew consent), the browser's AbortError is rethrown.
export async function analyzeRecording(recording, index, { signal, timeoutMs = TIMEOUT_MS, fetchImpl = fetch } = {}) {
    const form = new FormData()
    form.append('audio', recording.audioBlob, `answer_${index + 1}.webm`)
    form.append('mode', recording.mode)

    const controller = new AbortController()
    let timedOut = false
    const timer = setTimeout(() => { timedOut = true; controller.abort() }, timeoutMs)
    const onCallerAbort = () => controller.abort()
    if (signal) {
        if (signal.aborted) controller.abort()
        else signal.addEventListener('abort', onCallerAbort)
    }

    try {
        let response
        try {
            // No Content-Type header: the browser must add the multipart boundary itself.
            response = await fetchImpl(`${BACKEND_URL}/analyze`, { method: 'POST', body: form, signal: controller.signal })
        } catch (err) {
            if (timedOut) throw timeoutError(timeoutMs)
            if (signal && signal.aborted) throw err
            throw networkError(BACKEND_URL, err)
        }

        let body = null
        try {
            body = await response.json()
        } catch {
            body = null
        }
        if (!response.ok) throw errorFromResponse(response.status, body && body.detail)
        if (!body || typeof body.content_score !== 'number') {
            throw new UnexpectedResponse('The server answered, but the reply did not contain a content score.', null, response.status)
        }
        return body
    } finally {
        clearTimeout(timer)
        if (signal) signal.removeEventListener('abort', onCallerAbort)
    }
}
